#include "solver/ddar.hpp"
#include "predicate/eqratio.hpp"
#include "predicate/perp.hpp"

#include <algorithm>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace
{
constexpr size_t MAX_COMPLETION_REQUESTS = 100;

Dist as_distance(const TermArg &segment)
{
    const auto &ends = segment.points();
    return Dist(ends[0], ends[1]);
}

Proof *unique_missing_premise(const Application &application)
{
    Proof *missing = nullptr;
    for (auto *premise : application.hypotheses())
    {
        if (premise->is_proved())
        {
            continue;
        }
        if (missing != nullptr)
        {
            return nullptr;
        }
        missing = premise;
    }
    return missing;
}

struct DistanceProduct
{
    const TermArg &first;
    const TermArg &second;
};
}

void DDARSolver::advance_targeted_ar(const Point &last_point, size_t max_levels)
{
    for (size_t i = 0; i < max_levels && !_solved; ++i)
    {
        if (!run_level(last_point))
        {
            break;
        }
    }
}

void DDARSolver::run_targeted_ar(size_t max_levels)
{
    const auto points = _problem->points();
    if (points.empty())
    {
        return;
    }

    std::unordered_set<Proof *> requested;
    auto request_proof = [&](Proof *proof) {
        if (!proof->is_proved() && requested.size() < MAX_COMPLETION_REQUESTS)
        {
            requested.insert(proof);
        }
    };
    for (auto *goal : _goals)
    {
        request_proof(goal);
    }
    for (const auto &app : _applications)
    {
        if (requested.size() >= MAX_COMPLETION_REQUESTS)
        {
            break;
        }
        if (app.state() != ApplicationState::PENDING)
        {
            continue;
        }
        if (auto *missing = unique_missing_premise(app))
        {
            request_proof(missing);
        }
    }
    while (!_solved)
    {
        const auto before = ar_revision();
        if (transfer_perpendicular_equations(requested))
        {
            advance_targeted_ar(points.back(), max_levels);
        }

        if (!_solved && transfer_equal_distance_products(requested))
        {
            advance_targeted_ar(points.back(), max_levels);
        }
        if (ar_revision() == before)
        {
            break;
        }
    }
}

bool DDARSolver::transfer_perpendicular_equations(const std::unordered_set<Proof *> &requested)
{
    std::set<TermArg> distances;
    for (auto *request : requested)
    {
        if (!request->is_proved())
        {
            request->collect_product_distances(distances);
        }
    }
    if (distances.empty())
    {
        return false;
    }

    const auto points = _problem->points();
    const auto count = points.size();
    std::unordered_map<std::string, size_t> point_indices;
    for (size_t i = 0; i < count; ++i)
    {
        point_indices.emplace(points[i].name(), i);
    }
    std::set<std::pair<size_t, size_t>> relevant;
    for (const auto &distance : distances)
    {
        const auto &ends = distance.points();
        const auto a = point_indices.find(ends[0].name());
        const auto b = point_indices.find(ends[1].name());
        if (a != point_indices.end() && b != point_indices.end() && a->second != b->second)
        {
            relevant.emplace(std::minmax(a->second, b->second));
        }
    }
    std::set<std::tuple<size_t, size_t, size_t>> candidates;
    auto add = [&](size_t vertex, size_t a, size_t b) {
        candidates.emplace(vertex, std::min(a, b), std::max(a, b));
    };

    for (const auto &[a, b] : relevant)
    {
        for (size_t c = 0; c < count; ++c)
        {
            if (c == a || c == b)
            {
                continue;
            }
            // The requested edge can be either leg or the hypotenuse.
            add(a, b, c);
            add(b, a, c);
            add(c, a, b);
        }
    }

    std::vector<std::pair<int, Perp>> ranked;
    for (const auto &[vertex, a, b] : candidates)
    {
        const Perp relation(Slope(points[vertex], points[a]), Slope(points[vertex], points[b]));
        if (relation.check_numerically())
        {
            const int relevant_edges = relevant.count(std::minmax(vertex, a)) +
                                       relevant.count(std::minmax(vertex, b)) + relevant.count({a, b});
            ranked.emplace_back(relevant_edges, relation);
        }
    }

    // Candidates already arrive in point-index order, which breaks ties.
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto &left, const auto &right) {
        return left.first > right.first;
    });
    std::unordered_set<Proof *> checked;
    bool transferred = false;
    for (const auto &[priority, relation] : ranked)
    {
        auto *proof = insert_statement(relation.normalize());
        if (proof->is_proved() || !checked.insert(proof).second)
        {
            continue;
        }
        proof->ar(_level);
        transferred |= proof->is_proved();
    }
    return transferred;
}

bool DDARSolver::transfer_equal_distance_products(const std::unordered_set<Proof *> &requested)
{
    std::set<TermArg> seeds;
    for (auto *proof : requested)
        if (!proof->is_proved())
            proof->collect_residual_distances(seeds);
    if (seeds.empty())
        return false;

    auto neighborhood = seeds;
    for (const auto *system : {&_system_dist, &_system_distlog})
    {
        for (const auto &[head, rule] : system->rules())
        {
            std::set<TermArg> members;
            for (const auto &[term, coefficient] : rule.body().terms())
                for (const auto &[variable, exponent] : term.vars())
                {
                    const auto &points = variable.points();
                    if (points.size() == 2)
                        members.emplace(Dist(points[0], points[1]));
                }
            if (std::any_of(members.begin(), members.end(), [&](const auto &member) {
                    return seeds.count(member) != 0;
                }))
                neighborhood.insert(members.begin(), members.end());
        }
    }
    std::vector<TermArg> variables;
    for (const auto &variable : neighborhood)
    {
        if (as_distance(variable).check_nondegen())
            variables.push_back(variable);
    }

    std::vector<std::unique_ptr<Statement>> ratios;
    std::unordered_map<Polynomial, DistanceProduct> representatives;
    for (auto first = variables.begin(); first != variables.end(); ++first)
    {
        for (auto second = first; second != variables.end(); ++second)
        {
            auto normal = _system_dist.product_normal_form(Monomial(*first, 1) * Monomial(*second, 1));
            if (!normal || normal->empty())
            {
                continue;
            }
            const auto [representative, inserted] =
                representatives.emplace(std::move(*normal), DistanceProduct{*first, *second});
            if (inserted)
            {
                continue;
            }
            const auto &left = representative->second;
            EqRatio ratio(as_distance(left.first), as_distance(*first),
                          as_distance(*second), as_distance(left.second));
            if (ratio.check_numerically())
            {
                ratios.push_back(ratio.normalize());
            }
        }
    }

    bool transferred = false;
    for (const auto &ratio : ratios)
    {
        auto *proof = insert_statement(ratio);
        if (proof->is_proved())
            continue;

        proof->ar(_level);
        if (!proof->is_proved())
            continue;

        transferred = true;
        for (auto *demand : requested)
        {
            if (demand->is_proved())
                continue;
            demand->ar(_level);
            if (demand->is_proved())
                return true;
        }
    }
    return transferred;
}
