#include "predicate/cong.hpp"
#include "predicate/sameclock.hpp"
#include <array>
#include <bit>
#include <limits>
#include "matcher.hpp"
#include "numerical.hpp"
#include "theorem.hpp"
#include "problem.hpp"
#include "type/dist.hpp"
#include "type/angle.hpp"
#include "predicate/congruent_triangles.hpp"
#include "predicate/secant.hpp"
#include "predicate/eqpoint.hpp"
#include "predicate/para.hpp"
#include "predicate/perp.hpp"
#include "predicate/eqangle.hpp"
#include "predicate/eqratio.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <tuple>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <functional>
#include <iomanip>
#include <cmath>

using namespace std;

Matcher::Matcher(Problem *prob, const std::map<std::string, bool> &config)
    : _problem(prob), _config(config)
{
    match_similar_triangles();
    match_between();
    match_equal_angles();
    match_circles();
    match_orthocenters();
    // 全题 AR 候选枚举停用，定理匹配照常进行。
    // match_perps_paras();
    // match_dist_ratios();
}

namespace
{
struct MatchPredicate
{
    const Statement &raw;
    unique_ptr<Statement> canonical{};
    optional<bool> valid{};

    bool check()
    {
        if (!valid)
            valid = raw.check_numerically();
        return *valid;
    }

    const Statement &canonical_predicate()
    {
        if (!canonical)
            canonical = raw.normalize();
        return *canonical;
    }
};
}

void Matcher::on_similar_triangles(const SimilarTriangles &simtri)
{
    const auto ratio_abbc = simtri.eqratio_abbc();
    const auto ratio_abac = simtri.eqratio_abac();
    const auto ratio_bcac = simtri.eqratio_bcac();
    const auto angle_abc = simtri.eqangle_abc();
    const auto angle_bca = simtri.eqangle_bca();
    const auto angle_cab = simtri.eqangle_cab();
    const auto clock = simtri.to_sameclock();
    MatchPredicate relation{simtri}, orientation{clock};
    MatchPredicate abbc{ratio_abbc}, abac{ratio_abac}, bcac{ratio_bcac};
    MatchPredicate abc{angle_abc}, bca{angle_bca}, cab{angle_cab};
    const auto emit = [&](const char *name, const char *rule,
                          initializer_list<MatchPredicate *> hypotheses,
                          initializer_list<MatchPredicate *> conclusions)
    {
        if (!get_config(rule, false))
            return;
        for (auto *predicate : hypotheses)
            if (!predicate->check())
                return;
        for (auto *predicate : conclusions)
            if (!predicate->check())
                return;

        Theorem theorem(name, rule);
        for (auto *predicate : hypotheses)
            theorem.add_hypothesis(predicate->canonical_predicate().clone());
        for (auto *predicate : conclusions)
            theorem.add_conclusion(predicate->canonical_predicate().clone());
        _theorems.push_back(move(theorem));
    };

    const bool sameclock = simtri.sameclock();
    const char *sas_rule = sameclock ? "r62" : "r63";
    if (get_config(sas_rule, false))
    {
        constexpr const char *sas_name = "SAS Similarity of triangles";
        emit(sas_name, sas_rule, {&abbc, &abc, &orientation}, {&relation});
        const auto &l = simtri.left();
        const auto &r = simtri.right();
        const auto emit_rotated = [&](const SimilarTriangles &rotated, MatchPredicate &angle)
        {
            const auto ratio = rotated.eqratio_abbc();
            const auto direction = rotated.to_sameclock();
            MatchPredicate rotated_ratio{ratio}, rotated_direction{direction}, rotated_relation{rotated};
            emit(sas_name, sas_rule, {&rotated_ratio, &angle, &rotated_direction}, {&rotated_relation});
        };
        emit_rotated({Triangle(l.c(), l.a(), l.b()), Triangle(r.c(), r.a(), r.b()), sameclock}, cab);
        emit_rotated({Triangle(l.b(), l.c(), l.a()), Triangle(r.b(), r.c(), r.a()), sameclock}, bca);
    }

    emit("Properties of similar triangles", sameclock ? "r52" : "r53",
         {&orientation, &relation}, {&abbc, &abac, &bcac, &abc, &bca});
    emit("AA Similarity of triangles", sameclock ? "r34" : "r35", {&abc, &bca, &orientation}, {&relation});
    emit("SSS Similarity of triangles", sameclock ? "r60" : "r61", {&abbc, &abac, &orientation}, {&relation});

    const CongruentTriangles congruent(simtri.left(), simtri.right(), sameclock);
    if (!congruent.check_nondegen())
        return;
    const auto length_ab = congruent.cong_ab();
    const auto length_bc = congruent.cong_bc();
    const auto length_ac = congruent.cong_ac();
    if (!length_ab.check_equations() || !length_bc.check_equations() || !length_ac.check_equations())
        return;

    MatchPredicate ab{length_ab}, bc{length_bc}, ac{length_ac};
    ab.valid = length_ab.check_nondegen();
    bc.valid = length_bc.check_nondegen();
    ac.valid = length_ac.check_nondegen();
    MatchPredicate congruence{congruent};
    congruence.valid = true;

    const char *congruent_sas = sameclock ? "r112" : "r113";
    if (get_config(congruent_sas, false))
    {
        constexpr const char *sas_name = "SAS Congruence of triangles";
        emit(sas_name, congruent_sas, {&ab, &bc, &abc, &orientation}, {&congruence});
        const auto &l = congruent.left();
        const auto &r = congruent.right();
        const Cong length_ca(Dist(l.c(), l.a()), Dist(r.c(), r.a()));
        MatchPredicate ca{length_ca};
        const auto emit_rotated = [&](const CongruentTriangles &rotated, MatchPredicate &first,
                                      MatchPredicate &second, MatchPredicate &angle)
        {
            const auto direction = rotated.to_sameclock();
            MatchPredicate rotated_direction{direction}, rotated_relation{rotated};
            emit(sas_name, congruent_sas, {&first, &second, &angle, &rotated_direction}, {&rotated_relation});
        };
        emit_rotated({Triangle(l.c(), l.a(), l.b()), Triangle(r.c(), r.a(), r.b()), sameclock},
                     ca, ab, cab);
        emit_rotated({Triangle(l.b(), l.c(), l.a()), Triangle(r.b(), r.c(), r.a()), sameclock},
                     bc, ca, bca);
    }
    emit("SSS Congruence of triangles", sameclock ? "r110" : "r111",
         {&ab, &ac, &bc, &orientation}, {&congruence});
    emit("Properties of congruent triangles", sameclock ? "r103" : "r104",
         {&congruence}, {&ab, &ac, &bc, &abc, &bca, &cab});
}

void Matcher::match_similar_triangles()
{
    using item_type = tuple<double, double, array<size_t, 3>>;
    vector<item_type> triangles;
    const auto &points = _problem->points();
    const size_t num_pts = points.size();
    triangles.reserve(num_pts * (num_pts - 1) * (num_pts - 2) / 6);

    vector<double> distances(num_pts * num_pts);
    for (size_t a = 0; a < num_pts; ++a)
    {
        for (size_t b = 0; b < num_pts; ++b)
        {
            distances[a * num_pts + b] = Dist(points[a], points[b]).to_double();
        }
    }
    for (size_t a = 0; a < num_pts; ++a)
    {
        const auto &pt_a = points[a];
        for (size_t b = 0; b < num_pts; ++b)
        {
            const auto &pt_b = points[b];
            if (pt_a.is_close(pt_b))
            {
                continue;
            }
            const double ab = distances[a * num_pts + b];
            for (size_t c = 0; c < num_pts; ++c)
            {
                const auto &pt_c = points[c];
                if (pt_a.is_close(pt_c) || Coll(pt_a, pt_b, pt_c).check_equations())
                {
                    continue;
                }
                const double ac = distances[a * num_pts + c];
                const double bc = distances[b * num_pts + c];
                if (ab > (1 + REL_TOL) * bc || bc > (1 + REL_TOL) * ac)
                {
                    continue;
                }
                triangles.emplace_back(ab / ac,
                                 ab / bc,
                                 array<size_t, 3>{a, b, c});
            }
        }
    }

    if (triangles.empty())
    {
        return;
    }

    sort(triangles.begin(), triangles.end(),
         [](const item_type &a, const item_type &b)
         {
             double ratio1_a = get<0>(a);
             double ratio1_b = get<0>(b);
             if (fabs(ratio1_a - ratio1_b) >= 1e-8)
             {
                 return ratio1_a < ratio1_b;
             }
             else
             {
                 return get<1>(a) < get<1>(b);
             }
         });

    const auto triangle_at = [&points](const item_type &item)
    {
        const auto &idx = get<2>(item);
        return Triangle(points[idx[0]], points[idx[1]], points[idx[2]]);
    };

    const auto match_bucket = [&](size_t begin, size_t end)
    {
        for (size_t left = begin; left + 1 < end; ++left)
        {
            const auto left_triangle = triangle_at(triangles[left]);
            const double area_left = left_triangle.area();
            for (size_t right = left + 1; right < end; ++right)
            {
                const auto right_triangle = triangle_at(triangles[right]);
                const bool sameclock = (area_left > 0) == (right_triangle.area() > 0);
                on_similar_triangles({left_triangle, right_triangle, sameclock});
            }
        }
    };

    size_t begin = 0;
    for (size_t i = 1; i < triangles.size(); ++i)
    {
        const auto &previous = triangles[i - 1];
        const auto &current = triangles[i];
        if (!Numerical::close_enough(get<0>(previous), get<0>(current)) ||
            !Numerical::close_enough(get<1>(previous), get<1>(current)))
        {
            match_bucket(begin, i);
            begin = i;
        }
    }
    match_bucket(begin, triangles.size());
}

vector<tuple<double, Coll>> Matcher::all_betweens()
{
    vector<tuple<double, Coll>> res;
    const size_t num_pts = _problem->num_points();
    res.reserve(num_pts * (num_pts - 1) / 2);
    for (const auto &right : _problem->points())
    {
        for (const auto &left : _problem->points())
        {
            for (const auto &middle : _problem->points())
            {
                if (left.is_close(middle) || middle.is_close(right))
                {
                    continue;
                }
                Coll const pred(left, middle, right);
                if (!pred.is_between())
                {
                    continue;
                }
                double const dist_left = Dist(left, middle).to_double();
                double const dist_right = Dist(middle, right).to_double();
                if (dist_left <= (1 + REL_TOL) * dist_right)
                {
                    res.emplace_back(dist_left / (dist_left + dist_right), pred);
                }
            }
        }
    }
    return res;
}

vector<pair<Point, Point>> Matcher::all_eqpoints()
{
    const auto &pts = _problem->points();
    vector<pair<Point, Point>> res;
    for (size_t i = 0; i < pts.size(); i++)
    {
        for (size_t j = i + 1; j < pts.size(); j++)
        {
            if (pts[i].is_close(pts[j]))
            {
                res.push_back(make_pair(pts[i], pts[j]));
            }
        }
    }
    return res;
}

void Matcher::on_pappus(const Pappus &pappus)
{
    for (const auto &rotated : pappus.permutations())
    {
        insert_theorem(Theorem::pappus(rotated));
    }
}

void Matcher::on_between(const Coll &coll)
{
    for (const auto &rotated : coll.cyclic_rotations())
    {
        insert_theorem(Theorem::coll_of_para(rotated));
        insert_theorem(Theorem::para_of_coll(rotated));
    }
}

void Matcher::on_midpoint(const Midp &midp)
{
    insert_theorem(Theorem::coll_cong_of_midpoint(midp));
    insert_theorem(Theorem::midpoint_of_coll_cong(midp));
    insert_theorem(Theorem::midpoint_ratio_dist(midp));
}

void Matcher::on_eqratio(const Coll &left, const Coll &right)
{
    auto l = left.cyclic_rotations();
    auto r = right.cyclic_rotations();
    insert_theorem(Theorem::eqratio_of_coll(l[0], r[0]));
    insert_theorem(Theorem::eqratio_of_coll(l[1], r[1]));
    insert_theorem(Theorem::eqratio_of_coll(l[2], r[2]));
    insert_theorem(Theorem::ratio_segment_to_whole(left, right));

    if (left.a() == right.a())
    {
        insert_theorem(Theorem::thales_para_of_eqratio_with_common_point(left, right));
        insert_theorem(Theorem::thales_para_of_eqratio_with_common_point(Coll(left.a(), left.c(), left.b()), Coll(right.a(), right.c(), right.b())));
        insert_theorem(Theorem::thales_eqratio_of_para_with_common_point(left, right));
        insert_theorem(Theorem::thales_eqratio_of_para_with_common_point(Coll(left.a(), left.c(), left.b()), Coll(right.a(), right.c(), right.b())));
    }
    else if (left.b() == right.b())
    {
        insert_theorem(Theorem::thales_para_of_eqratio_with_common_point(Coll(left.b(), left.a(), left.c()), Coll(right.b(), right.a(), right.c())));
        insert_theorem(Theorem::thales_para_of_eqratio_with_common_point(Coll(left.b(), left.c(), left.a()), Coll(right.b(), right.c(), right.a())));
        insert_theorem(Theorem::thales_eqratio_of_para_with_common_point(Coll(left.b(), left.a(), left.c()), Coll(right.b(), right.a(), right.c())));
        insert_theorem(Theorem::thales_eqratio_of_para_with_common_point(Coll(left.b(), left.c(), left.a()), Coll(right.b(), right.c(), right.a())));
    }
    else if (left.c() == right.c())
    {
        insert_theorem(Theorem::thales_para_of_eqratio_with_common_point(Coll(left.c(), left.a(), left.b()), Coll(right.c(), right.a(), right.b())));
        insert_theorem(Theorem::thales_para_of_eqratio_with_common_point(Coll(left.c(), left.b(), left.a()), Coll(right.c(), right.b(), right.a())));
        insert_theorem(Theorem::thales_eqratio_of_para_with_common_point(Coll(left.c(), left.a(), left.b()), Coll(right.c(), right.a(), right.b())));
        insert_theorem(Theorem::thales_eqratio_of_para_with_common_point(Coll(left.c(), left.b(), left.a()), Coll(right.c(), right.b(), right.a())));
    }

    Thales const thales(left, right);
    if (!thales.check_numerically())
    {
        return;
    }
    for (const auto &rotated : thales.permutations())
    {
        insert_theorem(Theorem::thales_para_of_eqratio(rotated));
    }
    insert_theorem(Theorem::thales_eqratio_of_para(thales));
}

void Matcher::match_between()
{
    using item_type = tuple<double, Coll>;
    vector<item_type> betweens = all_betweens();

    vector<pair<Point, Point>> eqpoints = all_eqpoints();

    for (const auto &eq_pair : eqpoints)
    {
        const Point &p1 = eq_pair.first;
        const Point &p2 = eq_pair.second;
        vector<Coll> colls;
        for (const auto &item : betweens)
        {
            const Coll &coll = get<1>(item);
            if (coll.a() == p1 || coll.b() == p1 || coll.c() == p1)
            {
                colls.push_back(coll);
            }
        }
        set<Point> used_points;
        vector<tuple<double, Point, Point>> candidates;
        for (const auto &coll : colls)
        {
            Point a = p1 == coll.a() ? coll.b() : coll.a();
            Point b = p1 == coll.c() ? coll.b() : coll.c();
            used_points.insert(a);
            used_points.insert(b);
            candidates.emplace_back(coll.angle(), a, b);
        }
        for (const Point &pt : _problem->points())
        {
            if (pt == p1 || p1.is_close(pt))
            {
                continue;
            }
            candidates.emplace_back(Slope(p1, pt).angle(), p1, pt);
            insert_theorem(Theorem::cong_of_eqpoints(EqPoint(p1, p2), pt));
        }

        std::sort(candidates.begin(), candidates.end());

        for (size_t i = 0; i < candidates.size(); ++i)
        {
            const Point &a = std::get<1>(candidates[i]);
            const Point &b = std::get<2>(candidates[i]);
            insert_theorem(Theorem::eqpoints_of_same_ratio_on_line(p1, p2, a, b));
            for (size_t j = i + 1; j < candidates.size(); ++j)
            {
                const Point &c = std::get<1>(candidates[j]);
                const Point &d = std::get<2>(candidates[j]);
                if (Coll(a, b, d).check_numerically())
                {
                    continue;
                }
                insert_theorem(Theorem::eqpoints_of_same_intersections(p1, p2, a, b, c, d));
            }
        }
    }

    if (betweens.empty())
    {
        return;
    }

    // for (size_t i = 0; i < betweens.size(); i++)
    // {
    //     for (size_t j = i + 1; j < betweens.size(); j++)
    //     {
    //         for (size_t k = j + 1; k < betweens.size(); k++)
    //         {
    //             Coll left1 = get<1>(betweens[i]);
    //             Coll middle1 = get<1>(betweens[j]);
    //             Coll right1 = get<1>(betweens[k]);
    //             Coll left2 = left1.reverse();
    //             Coll middle2 = middle1.reverse();
    //             Coll right2 = right1.reverse();
    //             Pappus p1(left1, middle1, right1);
    //             Pappus p2(left1, middle1, right2);
    //             Pappus p3(left1, middle2, right1);
    //             Pappus p4(left1, middle2, right2);
    //             Pappus p5(left2, middle1, right1);
    //             Pappus p6(left2, middle1, right2);
    //             Pappus p7(left2, middle2, right1);
    //             Pappus p8(left2, middle2, right2);
    //             if (p1.check_numerically())
    //             {
    //                 on_pappus(p1);
    //             }
    //             if (p2.check_numerically())
    //             {
    //                 on_pappus(p2);
    //             }
    //             if (p3.check_numerically())
    //             {
    //                 on_pappus(p3);
    //             }
    //             if (p4.check_numerically())
    //             {
    //                 on_pappus(p4);
    //             }
    //             if (p5.check_numerically())
    //             {
    //                 on_pappus(p5);
    //             }
    //             if (p6.check_numerically())
    //             {
    //                 on_pappus(p6);
    //             }
    //             if (p7.check_numerically())
    //             {
    //                 on_pappus(p7);
    //             }
    //             if (p8.check_numerically())
    //             {
    //                 on_pappus(p8);
    //             }
    //         }
    //     }
    // }

    sort(betweens.begin(), betweens.end(),
         [](const item_type &a, const item_type &b)
         {
             return get<0>(a) < get<0>(b);
         });

    for (size_t i = 0; i < betweens.size(); i++)
    {
        const auto &curr = betweens[i];
        on_between(get<1>(curr));
        if (Numerical::close_enough(get<0>(curr), 0.5))
        {
            auto base = get<1>(curr);
            on_midpoint(Midp(base.b(), base.a(), base.c()));
        }
    }

    vector<item_type> bucket;
    bucket.push_back(betweens[0]);

    for (size_t i = 1; i < betweens.size(); i++)
    {
        const auto &prev = bucket.back();
        const auto &curr = betweens[i];
        if (Numerical::close_enough(get<0>(prev), get<0>(curr)))
        {
            bucket.push_back(curr);
        }
        else
        {
            if (bucket.size() > 1)
            {
                for (size_t left = 0; left < bucket.size(); left++)
                {
                    for (size_t right = left + 1; right < bucket.size(); right++)
                    {
                        on_eqratio(get<1>(bucket[left]), get<1>(bucket[right]));
                    }
                }
            }
            bucket.clear();
            bucket.push_back(curr);
        }
    }

    if (bucket.size() > 1)
    {
        for (size_t left = 0; left < bucket.size(); left++)
        {
            for (size_t right = left + 1; right < bucket.size(); right++)
            {
                on_eqratio(get<1>(bucket[left]), get<1>(bucket[right]));
            }
        }
    }
}

vector<tuple<double, Angle>> Matcher::all_angles()
{
    vector<tuple<double, Angle>> res;
    const size_t num_pts = _problem->num_points();
    res.reserve(num_pts * (num_pts - 1) * (num_pts - 2));
    for (const auto &left : _problem->points())
    {
        for (const auto &vertex : _problem->points())
        {
            for (const auto &right : _problem->points())
            {
                Angle const ang(left, vertex, right);
                if (ang.check_nondegen())
                {
                    res.emplace_back(ang.angle(), ang);
                }
            }
        }
    }
    return res;
}

void Matcher::on_cyclic(const Cyclic &cyclic)
{
    const auto permutations = cyclic.permutation();
    std::optional<Theorem> properties;
    for (size_t i = 0; i < permutations.size(); ++i)
    {
        insert_theorem(Theorem::cyclic_of_equal_angles(permutations[i]));
        if (i == 0)
        {
            auto candidate = Theorem::cyclic_properties(cyclic);
            if (candidate.check_numerically() && get_config(candidate.rule(), false))
            {
                properties.emplace(std::move(candidate).normalize());
            }
        }
        // Keep applications in their original order, including repeated instances.
        if (properties)
        {
            if (i + 1 == permutations.size())
            {
                _theorems.push_back(std::move(*properties));
            }
            else
            {
                _theorems.push_back(properties->clone());
            }
        }
    }
}

void Matcher::on_bisector(const Point &pt, const Angle &ang)
{
    insert_theorem(Theorem::triangle_bisector_of_eqratio(pt, ang));
    insert_theorem(Theorem::triangle_bisector_of_equal_angles(pt, ang));
    insert_theorem(Theorem::incenter(pt, ang));
}

void Matcher::on_eqangle(const Angle &left, const Angle &right)
{
    // _stmts.push_back(make_unique<EqAngle>(left, right));
    // ∠ABD = ∠ACD
    if (left.left() == right.left() && left.right() == right.right() && left.left() < left.right())
    {
        on_cyclic({left.vertex(), right.vertex(), left.left(), left.right()});
    }

    // ∠ABC = ∠CBD, A ≠ D
    if (left.vertex() == right.vertex())
    {
        if (left.right() == right.left() && left.left() < right.right())
        {
            on_bisector(left.right(), {left.left(), left.vertex(), right.right()});
        }
        else if (left.left() == right.right() && right.left() < left.right())
        {
            on_bisector(left.left(), {right.left(), left.vertex(), left.right()});
        }
    }
}

void Matcher::match_equal_angles()
{
    using item_type = tuple<double, Angle>;
    vector<item_type> angles = all_angles();

    if (angles.empty())
    {
        return;
    }

    sort(angles.begin(), angles.end(),
         [](const item_type &a, const item_type &b)
         {
             return get<0>(a) < get<0>(b);
         });

    // --- 新增：打印排序后的所有角度 ---
    // cout << "排序后的角度: ";
    // for (const auto& item : angles) {
    //     cout << get<1>(item) << ": " << get<0>(item) << endl;
    // }
    // cout << endl;
    // ----------------------------

    size_t hi = 0;
    for (size_t i = 0; i < angles.size(); i++)
    {
        if (hi < i + 1)
        {
            hi = i + 1;
        }
        while (hi < angles.size() &&
               Numerical::close_enough(get<0>(angles[i]), get<0>(angles[hi])))
        {
            hi++;
        }
        for (size_t j = i + 1; j < hi; j++)
        {
            on_eqangle(get<1>(angles[i]), get<1>(angles[j]));
        }
    }
}

void Matcher::on_circumcenter(const CircumCenter &circumcenter)
{
    insert_theorem(Theorem::cong_of_circumcenter(circumcenter));
    insert_theorem(Theorem::circumcenter_of_cong(circumcenter));
}

void Matcher::on_quadrangle_circumcenter(const Point &center, const Cyclic &cyc)
{
    insert_theorem(Theorem::cong_of_circumcenter_of_cyclic({center, Triangle(cyc.a(), cyc.b(), cyc.c())}, cyc.d()));
    insert_theorem(Theorem::cong_of_circumcenter_of_cyclic({center, Triangle(cyc.b(), cyc.c(), cyc.d())}, cyc.a()));
    insert_theorem(Theorem::cong_of_circumcenter_of_cyclic({center, Triangle(cyc.c(), cyc.d(), cyc.a())}, cyc.b()));
    insert_theorem(Theorem::cong_of_circumcenter_of_cyclic({center, Triangle(cyc.d(), cyc.a(), cyc.b())}, cyc.c()));
    insert_theorem(Theorem::center_of_cyclic_of_cong_of_cong(cyc, center));
    insert_theorem(Theorem::center_of_cyclic_of_cong_of_cong(Cyclic(cyc.a(), cyc.c(), cyc.b(), cyc.d()), center));
    insert_theorem(Theorem::center_of_cyclic_of_cong_of_cong(Cyclic(cyc.a(), cyc.d(), cyc.b(), cyc.c()), center));
}

void Matcher::on_circle(const Point &center, const vector<pair<double, Point>> &points)
{
    size_t const size = points.size();
    for (size_t pt_a = 0; pt_a < size; pt_a++)
    {
        for (size_t pt_b = pt_a + 1; pt_b < size; pt_b++)
        {
            if (Coll(points[pt_a].second, points[pt_b].second, center).check_numerically())
            {
                for (size_t pt_c = 0; pt_c < size; pt_c++)
                {
                    if (pt_c != pt_a && pt_c != pt_b)
                    {
                        insert_theorem(Theorem::hypotenuse_is_diameter(Midp(center, points[pt_a].second, points[pt_b].second), points[pt_c].second));
                    }
                }
            }
            else if (!points[pt_a].second.is_close(points[pt_b].second))
            {
                // 等腰三角形底角相等
                insert_theorem(Theorem::cong_of_eqangle(center, points[pt_a].second, points[pt_b].second));
                insert_theorem(Theorem::eqangle_of_cong(center, points[pt_a].second, points[pt_b].second));
            }

            for (size_t pt_c = pt_b + 1; pt_c < size; pt_c++)
            {
                on_circumcenter(CircumCenter(center, Triangle(points[pt_a].second, points[pt_b].second, points[pt_c].second)));
                for (size_t pt_d = pt_c + 1; pt_d < size; pt_d++)
                {
                    on_quadrangle_circumcenter(center, {points[pt_a].second, points[pt_b].second, points[pt_c].second, points[pt_d].second});
                }
            }
            if (points[pt_a].second.is_close(points[pt_b].second))
            {
                continue;
            }
            for (auto const &pt : _problem->points())
            {
                auto sec = Secant(center, points[pt_a].second, points[pt_b].second, pt);
                if (sec.check_numerically())
                {
                    insert_theorem(Theorem::definition_of_secant(sec));
                }
            }
        }
    }
}

void Matcher::match_circles()
{
    using item_type = pair<double, Point>;
    const size_t num_pts = _problem->num_points();
    for (const Point &center : _problem->points())
    {
        vector<item_type> pts;
        pts.reserve(num_pts - 1);
        for (const Point &pt : _problem->points())
        {
            if (!center.is_close(pt))
            {
                pts.emplace_back(Dist(center, pt).to_double(), pt);
            }
        }

        sort(pts.begin(), pts.end(), [](const item_type &a, const item_type &b)
             { return a.first < b.first; });

        vector<item_type> bucket;
        bucket.push_back(pts[0]);
        for (size_t i = 1; i < pts.size(); i++)
        {
            const auto &prev = bucket.back();
            const auto &curr = pts[i];
            if (Numerical::close_enough(prev.first, curr.first))
            {
                bucket.push_back(curr);
            }
            else
            {
                if (bucket.size() > 1)
                {
                    on_circle(center, bucket);
                }
                bucket.clear();
                bucket.push_back(curr);
            }
        }

        if (bucket.size() > 1)
        {
            on_circle(center, bucket);
        }
    }
}

void Matcher::match_orthocenters()
{
    const auto &all_pts = _problem->points();
    const size_t n = all_pts.size();

    // 使用索引遍历，避免重复组合：只考虑 a < b < c < d
    for (size_t idx_d = 0; idx_d < n; idx_d++)
    {
        for (size_t idx_c = 0; idx_c < idx_d; idx_c++)
        {
            for (size_t idx_b = 0; idx_b < idx_c; idx_b++)
            {
                for (size_t idx_a = 0; idx_a < idx_b; idx_a++)
                {
                    const auto &pt_a = all_pts[idx_a];
                    const auto &pt_b = all_pts[idx_b];
                    const auto &pt_c = all_pts[idx_c];
                    const auto &pt_d = all_pts[idx_d];

                    OrthoCenter const ortho(pt_d, Triangle(pt_a, pt_b, pt_c));
                    if (ortho.check_numerically())
                    {
                        for (const auto &rotated : ortho.cyclic_rotations())
                        {
                            insert_theorem(Theorem::orthocenter(rotated));
                        }
                    }
                }
            }
        }
    }
}

void Matcher::match_perps_paras()
{
    using item_type = pair<double, Slope>;
    vector<item_type> slopes;
    const size_t num_pt = _problem->num_points();
    slopes.reserve(num_pt * (num_pt - 1) / 2);

    for (size_t pt_a = 0; pt_a < num_pt; pt_a++)
    {
        for (size_t pt_b = pt_a + 1; pt_b < num_pt; pt_b++)
        {
            Slope slope(_problem->point(pt_a), _problem->point(pt_b));
            slopes.emplace_back(slope.angle(), slope);
        }
    }

    sort(slopes.begin(), slopes.end(), [](const item_type &a, const item_type &b)
         { return a.first < b.first; });

    for (size_t i = 0; i < slopes.size(); ++i)
    {
        for (size_t j = i + 1; j < slopes.size(); ++j)
        {
            const auto &l = slopes[i];
            const auto &r = slopes[j];

            if (Numerical::close_enough(l.first, r.first))
            {
                _stmts.push_back(std::make_unique<Para>(l.second, r.second));
            }

            if (Numerical::close_enough(r.first - l.first, M_PI / 2.0))
            {
                _stmts.push_back(std::make_unique<Perp>(l.second, r.second));
            }
        }
    }
}

vector<tuple<double, Dist>> Matcher::all_dists()
{
    vector<tuple<double, Dist>> res;
    const size_t num_pt = _problem->num_points();
    res.reserve(num_pt * (num_pt - 1) / 2);
    for (size_t pt_a = 0; pt_a < num_pt; pt_a++)
    {
        for (size_t pt_b = pt_a + 1; pt_b < num_pt; pt_b++)
        {
            Dist dist(_problem->point(pt_a), _problem->point(pt_b));
            if (!dist.check_nondegen())
            {
                continue;
            }
            res.emplace_back(dist.to_double(), dist);
        }
    }
    return res;
}

// 把"所有比值相等的线段对"作为已验证的 EqRatio 送进 _stmts，供 AR 直接使用。
// 与 match_equal_angles 对角度做的事完全对称：按数值分桶，桶内两两配对。
// 桶内组合数是 O(n^4)（n 为点数），比 all_triangles/all_angles 的 O(n^3) 更贵，
// 但和 match_perps_paras 一样只在 Matcher 构造时算一次。
void Matcher::match_dist_ratios()
{
    using item_type = tuple<double, Dist>;
    vector<item_type> dists = all_dists();

    if (dists.size() < 2)
    {
        return;
    }

    // 按 dist(a,b) / dist(c,d) 的比值分桶：先取所有二元组合 (i<j)，
    // 用 dists[i]/dists[j] 作为比值 key。
    vector<tuple<double, size_t, size_t>> ratios;
    ratios.reserve(dists.size() * (dists.size() - 1) / 2);
    for (size_t i = 0; i < dists.size(); i++)
    {
        for (size_t j = i + 1; j < dists.size(); j++)
        {
            double const a = get<0>(dists[i]);
            double const b = get<0>(dists[j]);
            if (a <= (1 + REL_TOL) * b)
            {
                ratios.emplace_back(a / b, i, j);
            }
            if (b <= (1 + REL_TOL) * a)
            {
                ratios.emplace_back(b / a, j, i);
            }
        }
    }

    sort(ratios.begin(), ratios.end(),
         [](const auto &a, const auto &b)
         { return get<0>(a) < get<0>(b); });

    size_t hi = 0;
    for (size_t i = 0; i < ratios.size(); i++)
    {
        if (hi < i + 1)
        {
            hi = i + 1;
        }
        while (hi < ratios.size() &&
               Numerical::close_enough(get<0>(ratios[i]), get<0>(ratios[hi])))
        {
            hi++;
        }
        for (size_t j = i + 1; j < hi; j++)
        {
            const Dist &up1 = get<1>(dists[get<1>(ratios[i])]);
            const Dist &down1 = get<1>(dists[get<2>(ratios[i])]);
            const Dist &up2 = get<1>(dists[get<1>(ratios[j])]);
            const Dist &down2 = get<1>(dists[get<2>(ratios[j])]);
            _stmts.push_back(make_unique<EqRatio>(up1, down1, up2, down2));
        }
    }
}

void Matcher::insert_theorem(const Theorem &thm)
{
    if (!thm.check_numerically())
    {
        // cout << "定理未通过数值检测: " << thm.name() << endl;
        // cout << "前提: " << endl;
        // for (const auto &stmt : thm.hypotheses())
        // {
        //     cout << "  " << *stmt << endl;
        // }
        // cout << "结论: " << endl;
        // for (const auto &stmt : thm.conclusions())
        // {
        //     cout << "  " << *stmt << endl;
        // }
        return;
    }
    if (!get_config(thm.rule(), false))
    {
        return;
    }
    _theorems.push_back(thm.normalize());
}

// ============================================================================
// CustomTheoremMatcher 实现 - 独立的自定义定理匹配功能
// ============================================================================

namespace
{
    // 内部工具：判断谓词是否包含尾部常量参数
    bool HasTrailingConstant(const string &pred)
    {
        return pred == "rconst" || pred == "aconst";
    }

    // 内部工具：获取谓词中“点”参数的数量
    size_t GetPointArgsCount(const Stmt &stmt)
    {
        size_t n = stmt.second.size();
        return (HasTrailingConstant(stmt.first) && n > 0) ? n - 1 : n;
    }

    // Bind points with smaller indexed predicates before checking angles or ratios.
    set<string> DeferredPredicates(const vector<Stmt> &premises)
    {
        set<string> bound;
        for (const auto &premise : premises)
        {
            const auto &pred = premise.first;
            if (pred == "midp" || pred == "coll" || pred == "cong" ||
                pred == "para" || pred == "perp")
                bound.insert(premise.second.begin(), premise.second.end());
        }
        set<string> deferred{"eqangle", "eqratio"};
        for (const auto &premise : premises)
        {
            if (!deferred.contains(premise.first))
                continue;
            set<string> unbound;
            for (const auto &point : premise.second)
                if (!bound.contains(point))
                    unbound.insert(point);
            // Scanning one free point avoids building a whole-diagram index.
            if (unbound.size() > 1)
                deferred.erase(premise.first);
        }
        return deferred;
    }

    // 内部工具：在 Mapping 中查找变量绑定的点索引
    int FindBinding(const Mapping &mapping, const string &var)
    {
        return mapping.find(var);
    }
}

void CustomRule::prepare_static()
{
    const auto deferred = DeferredPredicates(premises);
    std::unordered_map<std::string,uint64_t> bits;
    static_fits = true;
    for (const auto &s : premises)
        for (size_t i=0; i<GetPointArgsCount(s); ++i)
            if (!bits.contains(s.second[i]))
            {
                if (bits.size()==64) { static_fits=false; break; }
                bits.emplace(s.second[i],uint64_t(1)<<bits.size());
            }
    static_masks.clear(); static_deferred.clear();
    for (const auto &s : premises)
    {
        uint64_t mask=0;
        if (static_fits)
            for (size_t i=0;i<GetPointArgsCount(s);++i) mask |= bits.at(s.second[i]);
        static_masks.push_back(mask); static_deferred.push_back(deferred.contains(s.first));
    }
    early_checks.clear(); early_masks.clear(); early_valid.clear();
    for (const auto &guard : guards)
    {
        if (guard.first!="npara" && guard.first!="ncoll" && guard.first!="diff" &&
            guard.first!="angle_defined" && guard.first!="neqangle") continue;
        auto check=guard;
        if (guard.first=="diff" || guard.first=="angle_defined" || guard.first=="neqangle")
            check.first="__guard_"+guard.first;
        uint64_t mask=0; bool valid=static_fits;
        for (const auto &arg : guard.second)
        {
            auto found=bits.find(arg);
            if (found==bits.end()) valid=false;
            else mask |= found->second;
        }
        early_checks.push_back(std::move(check)); early_masks.push_back(mask); early_valid.push_back(valid);
    }
    static_prepared=true;
}

// ----------------------------------------------------------------------------
// 辅助成员函数
// ----------------------------------------------------------------------------

void CustomTheoremMatcher::DebugLog(const string &msg) const
{
    if (get_config("verbose", false))
    {
        cout << "[DEBUG] " << msg << endl;
    }
}

vector<string> CustomTheoremMatcher::collect_new_vars(const Stmt &stmt, const Mapping &current) const
{
    vector<string> new_vars;
    size_t n_point_args = GetPointArgsCount(stmt);

    for (size_t i = 0; i < n_point_args; ++i)
    {
        const string &arg = stmt.second[i];
        // 如果当前 mapping 中没有，且 new_vars 中也没加入过，则视为新变量
        if (FindBinding(current, arg) == -1 &&
            std::find(new_vars.begin(), new_vars.end(), arg) == std::end(new_vars))
        {
            new_vars.push_back(arg);
        }
    }
    return new_vars;
}

bool CustomTheoremMatcher::verify_stmt(const Stmt &stmt, const Mapping &mapping) const
{
    const size_t n_point_args = GetPointArgsCount(stmt);
    std::array<int, 8> small_indices;
    vector<int> wide_indices;
    if (n_point_args > small_indices.size()) wide_indices.reserve(n_point_args);
    uint64_t key = 0;
    for (size_t i = 0; i < n_point_args; ++i)
    {
        int index = FindBinding(mapping, stmt.second[i]);
        if (index < 0) return false;
        if (n_point_args <= small_indices.size()) small_indices[i] = index;
        else wide_indices.push_back(index);
        key = (key << 8) | uint64_t(index);
    }
    const bool cache = n_point_args <= 8 && n_point_args == stmt.second.size() &&
                       _problem->num_points() <= 256 && _memo_enabled;
    unordered_map<uint64_t, bool> *memo = nullptr;
    if (cache)
    {
        memo = &_numeric_memo[stmt.first][n_point_args];
        if (auto found = memo->find(key); found != memo->end())
        {
            ++_numeric_hits;
            return found->second;
        }
        ++_numeric_misses;
    }
    vector<int> pt_indices;
    if (n_point_args <= small_indices.size())
        pt_indices.assign(small_indices.begin(), small_indices.begin()+n_point_args);
    else
        pt_indices = std::move(wide_indices);
    vector<string> real_args;
    real_args.reserve(stmt.second.size());
    for (int index : pt_indices)
        real_args.push_back(_problem->point(index).name());
    for (size_t i = n_point_args; i < stmt.second.size(); ++i)
        real_args.push_back(stmt.second[i]);
    const auto evaluate = [&]() -> bool
    {
        if (stmt.first == "__guard_diff")
        {
            if (pt_indices.size() != 2)
                return false;
            const auto &a = _problem->point(pt_indices[0]);
            const auto &b = _problem->point(pt_indices[1]);
            return std::isfinite(std::hypot(a.x()-b.x(), a.y()-b.y())) && !a.is_close(b);
        }
        if (stmt.first == "__guard_angle_defined" || stmt.first == "__guard_neqangle")
        {
            try
            {
                auto angle = _problem->create_statement("eqangle", real_args);
                return angle && angle->check_nondegen() &&
                       (stmt.first == "__guard_angle_defined" || !angle->check_equations());
            }
            catch (...) { return false; }
        }

        try
        {
            auto s = _problem->create_statement(stmt.first, real_args);
            // 退化筛子只针对 para：其它 4 元谓词里出现重复变量是正常的
            // （例如 cong A B B C），不能一起拦掉。
            return s && s->check_numerically() &&
                   (stmt.first != "para" || para_instance_ok(pt_indices));
        }
        catch (...)
        {
            return false;
        }
    };
    const bool valid = evaluate();
    // Hitting the memory limit only causes recomputation, never pruning.
    if (memo && _numeric_memo_size < 500000)
    {
        memo->emplace(key, valid);
        ++_numeric_memo_size;
    }
    return valid;
}

bool CustomTheoremMatcher::para_instance_ok(const std::vector<int> &pts) const
{
    if (!get_config("custom_para_no_collinear", true))
        return true;
    if (pts.size() != 4)
        return true;

    // 情形 1：四点中存在重合点 —— 两条"直线"共享端点，此时平行成立 ⟺ 三点共线
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = i + 1; j < 4; ++j)
            if (pts[i] == pts[j])
            {
                if (get_config("debug_custom_match", false))
                    cout << "[para-screen] 共享端点 " << _problem->point(pts[0]).name()
                         << _problem->point(pts[1]).name() << _problem->point(pts[2]).name()
                         << _problem->point(pts[3]).name() << endl;
                return false;
            }

    // 情形 2：四点互异但共线。调用方已先做过数值平行判定，故只需查一个三点组：
    // CD ∥ AB 且 A,B,C 共线 ⟹ C 在直线 AB 上，过 C 作 AB 的平行线即 AB 自身 ⟹ D 也在其上。
    try
    {
        const Point &pa = _problem->point(pts[0]);
        const Point &pb = _problem->point(pts[1]);
        const Point &pc = _problem->point(pts[2]);
        // 用 check_equations 而非 check_numerically：Coll::check_nondegen 对重合点返回 false，
        // 这里只关心"是否共线"这一几何事实，重合点情形已在上面单独拦掉。
        if (Coll(pa, pb, pc).check_equations())
        {
            if (get_config("debug_custom_match", false))
                cout << "[para-screen] 四点共线 " << pa.name() << pb.name() << pc.name()
                     << _problem->point(pts[3]).name() << endl;
            return false;
        }
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool CustomTheoremMatcher::stmt_instance_ok(const Stmt &stmt, const std::map<std::string, int> &m_map) const
{
    if (stmt.first != "para")
        return true;

    vector<int> pts;
    size_t n_point_args = GetPointArgsCount(stmt);
    if (stmt.second.size() < n_point_args)
        return true;

    for (size_t i = 0; i < n_point_args; ++i)
    {
        auto it = m_map.find(stmt.second[i]);
        if (it == m_map.end())
            return true; // 变量尚未绑定，交给原有流程处理
        pts.push_back(it->second);
    }
    return para_instance_ok(pts);
}

// ----------------------------------------------------------------------------
// 核心逻辑：谓词索引构建
// ----------------------------------------------------------------------------

void CustomTheoremMatcher::build_triangle_indices(size_t n)
{
    struct TriangleInfo
    {
        double ratio1, ratio2; // 形状签名：ab/ac, ab/bc
        int a, b, c;           // 原始点索引
    };

    vector<TriangleInfo> triangles;
    const double eps = 1e-8;

    // 1. 遍历并过滤所有合法三角形
    for (int a = 0; a < n; ++a)
    {
        for (int b = 0; b < n; ++b)
        {
            if (a == b)
                continue;
            for (int c = 0; c < n; ++c)
            {
                if (c == a || c == b)
                    continue;

                auto pt_a = _problem->point(a);
                auto pt_b = _problem->point(b);
                auto pt_c = _problem->point(c);

                // 排除共线三角形
                if (Coll(pt_a, pt_b, pt_c).check_equations())
                    continue;

                double ab = Dist(pt_a, pt_b).to_double();
                double ac = Dist(pt_a, pt_c).to_double();
                double bc = Dist(pt_b, pt_c).to_double();

                // 使用 Canonical Order 过滤：ab ≤ bc ≤ ac 以去重并统一对应关系
                // 这里的 REL_TOL 建议定义在配置类中
                if (ab > (1 + 1e-6) * bc || bc > (1 + 1e-6) * ac)
                    continue;

                triangles.push_back({ab / ac, ab / bc, a, b, c});
            }
        }
    }

    // 2. 按形状签名排序
    std::sort(triangles.begin(), triangles.end(), [eps](const TriangleInfo &t1, const TriangleInfo &t2)
              {
        if (std::abs(t1.ratio1 - t2.ratio1) > eps) return t1.ratio1 < t2.ratio1;
        return t1.ratio2 < t2.ratio2; });

    // 3. 桶内配对
    auto &idx_simtri = _index["simtri"];
    auto &idx_simtrir = _index["simtrir"];

    for (size_t i = 0; i < triangles.size(); ++i)
    {
        for (size_t j = i + 1; j < triangles.size(); ++j)
        {
            // 检查形状是否相同（是否在同一个桶）
            if (std::abs(triangles[i].ratio1 - triangles[j].ratio1) > eps ||
                std::abs(triangles[i].ratio2 - triangles[j].ratio2) > eps)
            {
                break;
            }

            const auto &t1 = triangles[i];
            const auto &t2 = triangles[j];

            // 判断方向：利用面积正负判断是相似(simtri)还是反相似(simtrir)
            double area1 = Triangle(_problem->point(t1.a), _problem->point(t1.b), _problem->point(t1.c)).area();
            double area2 = Triangle(_problem->point(t2.a), _problem->point(t2.b), _problem->point(t2.c)).area();
            bool same_orientation = (area1 > 0) == (area2 > 0);

            auto &target_idx = same_orientation ? idx_simtri : idx_simtrir;

            // 注入所有可能的对应排列（共 12 种：2 个三角形交换 × 6 种顶点置换）
            // 这里封装一个 lambda 简化代码
            auto push_all_perms = [&](int a, int b, int c, int d, int e, int f)
            {
                target_idx.push_back({a, b, c, d, e, f});
                target_idx.push_back({b, c, a, e, f, d});
                target_idx.push_back({c, a, b, f, d, e});
                target_idx.push_back({a, c, b, d, f, e});
                target_idx.push_back({b, a, c, e, d, f});
                target_idx.push_back({c, b, a, f, e, d});
            };

            push_all_perms(t1.a, t1.b, t1.c, t2.a, t2.b, t2.c);
            push_all_perms(t2.a, t2.b, t2.c, t1.a, t1.b, t1.c);
        }
    }
}

void CustomTheoremMatcher::build_eqangle_indices(size_t n)
{
    struct AngleInfo
    {
        double value;       // 角度值（弧度）
        int p1, p2, p3, p4; // 原始点索引，表示两条线段 (p1,p2) 和 (p3,p4)
    };

    vector<AngleInfo> angles;
    const double eps = 1e-8;

    // 1. 枚举所有合法的角（两条线段：p1≠p2, p3≠p4）
    for (int p1 = 0; p1 < n; ++p1)
    {
        for (int p2 = p1 + 1; p2 < n; ++p2)
        {
            for (int p3 = 0; p3 < n; ++p3)
            {
                for (int p4 = p3 + 1; p4 < n; ++p4)
                {
                    try
                    {
                        // 计算线段 (p1,p2) 和 (p3,p4) 之间的角度
                        const Point &pt1 = _problem->point(p1);
                        const Point &pt2 = _problem->point(p2);
                        const Point &pt3 = _problem->point(p3);
                        const Point &pt4 = _problem->point(p4);

                        // 线段 (p1,p2) 的方向向量
                        double dx1 = pt2.x() - pt1.x();
                        double dy1 = pt2.y() - pt1.y();

                        // 线段 (p3,p4) 的方向向量
                        double dx2 = pt4.x() - pt3.x();
                        double dy2 = pt4.y() - pt3.y();

                        // 检查是否退化（线段长度太小）
                        double len1_sq = dx1 * dx1 + dy1 * dy1;
                        double len2_sq = dx2 * dx2 + dy2 * dy2;
                        if (len1_sq < 1e-14 || len2_sq < 1e-14)
                            continue;

                        // 计算 dot product 和 cross product
                        double dot = dx1 * dx2 + dy1 * dy2;
                        double cross = dx1 * dy2 - dy1 * dx2;

                        // 使用 atan2 计算角度
                        double val = std::atan2(cross, dot);
                        if (val < 0.0)
                        {
                            val += M_PI;
                        }

                        if (val > 1e-7 && val < M_PI - 1e-7)
                        {
                            angles.push_back({val, p1, p2, p3, p4});
                        }
                    }
                    catch (...)
                    {
                        continue;
                    }
                }
            }
        }
    }

    // 2. 按角度值排序
    std::sort(angles.begin(), angles.end(), [](const AngleInfo &a1, const AngleInfo &a2)
              { return a1.value < a2.value; });

    // 3. 桶内配对
    auto &idx_eqangle = _index["eqangle"];

    for (size_t i = 0; i < angles.size(); ++i)
    {
        for (size_t j = i + 1; j < angles.size(); ++j)
        {
            // 检查角度是否相同（是否在同一个桶）
            if (std::abs(angles[i].value - angles[j].value) > eps)
            {
                break;
            }

            const auto &ang1 = angles[i];
            const auto &ang2 = angles[j];

            // eqangle p1 p2 p3 p4 p5 p6 p7 p8 表示 ∠(p1,p2,p3,p4) = ∠(p5,p6,p7,p8)
            // 对称性：每条线段两端点可交换 (2^4 = 16)，两个角可整体交换 (×2)，共 32 种
            int a1 = ang1.p1, a2 = ang1.p2, a3 = ang1.p3, a4 = ang1.p4;
            int b1 = ang2.p1, b2 = ang2.p2, b3 = ang2.p3, b4 = ang2.p4;

            for (int s1 = 0; s1 < 2; ++s1)
                for (int s2 = 0; s2 < 2; ++s2)
                    for (int s3 = 0; s3 < 2; ++s3)
                        for (int s4 = 0; s4 < 2; ++s4)
                        {
                            int x1 = s1 ? a2 : a1, x2 = s1 ? a1 : a2;
                            int x3 = s2 ? a4 : a3, x4 = s2 ? a3 : a4;
                            int x5 = s3 ? b2 : b1, x6 = s3 ? b1 : b2;
                            int x7 = s4 ? b4 : b3, x8 = s4 ? b3 : b4;
                            idx_eqangle.push_back({x1, x2, x3, x4, x5, x6, x7, x8});
                            idx_eqangle.push_back({x5, x6, x7, x8, x1, x2, x3, x4});
                        }
        }
    }
}

void CustomTheoremMatcher::build_eqratio_indices(size_t n)
{
    struct RatioInfo
    {
        double value;   // 比值
        double up, down;
        int a, b, c, d; // 原始点索引，表示 AB/CD
    };

    vector<RatioInfo> ratios;
    double minimum_denominator = std::numeric_limits<double>::infinity();

    // 1. 遍历所有合法的距离对 (4个点，AB和CD可以共享点)
    for (int a = 0; a < n; ++a)
    {
        for (int b = a + 1; b < n; ++b)
        {
            for (int c = 0; c < n; ++c)
            {
                for (int d = c + 1; d < n; ++d)
                {

                    try
                    {
                        Dist dist_ab(_problem->point(a), _problem->point(b));
                        Dist dist_cd(_problem->point(c), _problem->point(d));

                        if (!dist_ab.check_nondegen() || !dist_cd.check_nondegen())
                            continue;

                        double len_ab = dist_ab.to_double();
                        double len_cd = dist_cd.to_double();

                        double ratio = len_ab / len_cd;
                        if (!(len_cd > 0) || !std::isfinite(ratio) || !std::isfinite(len_ab) || !std::isfinite(len_cd))
                            continue;
                        minimum_denominator = std::min(minimum_denominator, len_cd);
                        ratios.push_back({ratio, len_ab, len_cd, a, b, c, d});
                    }
                    catch (...)
                    {
                        continue;
                    }
                }
            }
        }
    }

    // 2. 按比值排序
    std::sort(ratios.begin(), ratios.end(), [](const RatioInfo &r1, const RatioInfo &r2)
              { return r1.value < r2.value; });

    // 3. 桶内配对
    auto &idx_eqratio = _index["eqratio"];

    for (size_t i = 0; i < ratios.size(); ++i)
    {
        const auto &rat1 = ratios[i];
        // Sorted ratio windows must contain both branches of the native
        // cross-product tolerance. A fixed ratio epsilon is not sufficient.
        double upper = std::max(rat1.value / (1 - Numerical::relative_tolerance()),
            rat1.value + Numerical::absolute_tolerance() / rat1.down / minimum_denominator);
        upper += 64 * std::numeric_limits<double>::epsilon() * std::max(1.0, upper);
        for (size_t j = i; j < ratios.size(); ++j)
        {
            const auto &rat2 = ratios[j];
            if (rat2.value > upper)
                break;
            if (!Numerical::close_enough(rat1.up * rat2.down, rat1.down * rat2.up))
                continue;

            // eqratio a b c d e f g h 表示 AB/CD = EF/GH
            int a1 = rat1.a, a2 = rat1.b, a3 = rat1.c, a4 = rat1.d;
            int b1 = rat2.a, b2 = rat2.b, b3 = rat2.c, b4 = rat2.d;

            for (int s1 = 0; s1 < 2; ++s1)
                for (int s2 = 0; s2 < 2; ++s2)
                    for (int s3 = 0; s3 < 2; ++s3)
                        for (int s4 = 0; s4 < 2; ++s4)
                        {
                            int x1 = s1 ? a2 : a1, x2 = s1 ? a1 : a2;
                            int x3 = s2 ? a4 : a3, x4 = s2 ? a3 : a4;
                            int x5 = s3 ? b2 : b1, x6 = s3 ? b1 : b2;
                            int x7 = s4 ? b4 : b3, x8 = s4 ? b3 : b4;
                            idx_eqratio.push_back({x1, x2, x3, x4, x5, x6, x7, x8});
                            idx_eqratio.push_back({x5, x6, x7, x8, x1, x2, x3, x4});
                        }
        }
    }
}

void CustomTheoremMatcher::build_predicate_index(const vector<CustomRule> &rules)
{
    set<string> predicates;
    for (const auto &rule : rules)
    {
        const auto deferred = DeferredPredicates(rule.premises);
        for (const auto &premise : rule.premises)
            if (!deferred.contains(premise.first))
                predicates.insert(premise.first);
    }
    auto t_total = chrono::steady_clock::now();
    const size_t n = _problem->num_points();

    // 辅助检查函数
    auto quick_check = [&](const string &pred, const vector<int> &p_indices)
    {
        vector<string> names;
        for (int i : p_indices)
            names.push_back(_problem->point(i).name());
        try
        {
            auto s = _problem->create_statement(pred, names);
            return s && s->check_numerically();
        }
        catch (...)
        {
            return false;
        }
    };

    // 1. 处理简单谓词 (midp, coll, cong, para, perp)
    auto add_simple_indices = [&](const string &pred, int arity)
    {
        auto &idx = _index[pred];
        // 这里可以根据 arity 使用通用的组合枚举逻辑，为了性能此处保持展开
        if (arity == 3)
        {
            for (int a = 0; a < n; ++a)
                for (int b = 0; b < n; ++b)
                {
                    if (a == b)
                        continue;
                    for (int c = 0; c < n; ++c)
                    {
                        if (c == a || c == b)
                            continue;
                        if (quick_check(pred, {a, b, c}))
                            idx.push_back({a, b, c});
                    }
                }
        }
        else if (arity == 4)
        {
            for (int a = 0; a < n; ++a)
                for (int b = 0; b < n; ++b)
                {
                    if (a == b)
                        continue;
                    for (int c = 0; c < n; ++c)
                        for (int d = 0; d < n; ++d)
                        {
                            if (c == d)
                                continue;
                            // para 额外要求实例非退化（索引路径不走 verify_stmt，必须在此拦）
                            if (quick_check(pred, {a, b, c, d}) &&
                                (pred != "para" || para_instance_ok({a, b, c, d})))
                                idx.push_back({a, b, c, d});
                        }
                }
        }
    };

    for (auto &p : {make_pair("midp", 3), {"coll", 3}, {"cong", 4}, {"para", 4}, {"perp", 4}})
    {
        if (predicates.contains(p.first))
            add_simple_indices(p.first, p.second);
    }

    // 2. 处理三角形相似 (simtri/simtrir) - 保持原有形状签名算法，但逻辑更紧凑
    if (predicates.contains("simtri") || predicates.contains("simtrir"))
        build_triangle_indices(n);

    // 3. 处理等角 (eqangle)
    if (predicates.contains("eqangle"))
        build_eqangle_indices(n);

    // 4. 处理等比 (eqratio)
    if (predicates.contains("eqratio"))
        build_eqratio_indices(n);

    auto ms = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - t_total).count();
    // DebugLog("Predicate cache built in " + to_string(ms) + "ms");

    // // 输出所有谓词的缓存详细信息
    // bool verbose = get_config("verbose_cache_output", false);
    // if (verbose)
    // {
    //     DebugLog("=== Predicate Cache Details ===");
    //     size_t total_cached = 0;

    //     // 辅助函数：将点索引转换为点名
    //     auto idx_to_name = [&](int idx) -> string {
    //         return _problem->point(idx).name();
    //     };

    //     for (const auto &entry : _index)
    //     {
    //         const string &pred_name = entry.first;
    //         const auto &instances = entry.second;
    //         size_t count = instances.size();
    //         total_cached += count;

    //         DebugLog("\n[" + pred_name + "] " + to_string(count) + " instances:");

    //         // 根据谓词类型输出不同格式
    //         for (size_t i = 0; i < instances.size(); ++i)
    //         {
    //             const auto &inst = instances[i];
    //             string output = "  " + to_string(i) + ": " + pred_name + " ";

    //             for (size_t j = 0; j < inst.size(); ++j)
    //             {
    //                 output += idx_to_name(inst[j]);
    //                 if (j < inst.size() - 1)
    //                     output += " ";
    //             }

    //             DebugLog(output);
    //         }
    //     }

    //     DebugLog("\n=== Summary ===");
    //     DebugLog("Total cached instances: " + to_string(total_cached));
    //     DebugLog("===============");
    // }
    // else
    // {
    //     // 简要统计信息
    //     DebugLog("=== Predicate Cache Statistics ===");
    //     size_t total_cached = 0;
    //     for (const auto &entry : _index)
    //     {
    //         size_t count = entry.second.size();
    //         total_cached += count;
    //         DebugLog("  " + entry.first + ": " + to_string(count) + " cached instances");
    //     }
    //     DebugLog("Total cached instances: " + to_string(total_cached));
    //     DebugLog("===================================");
    // }
}

// ----------------------------------------------------------------------------
// 核心逻辑：回溯匹配
// ----------------------------------------------------------------------------

void CustomTheoremMatcher::enumerate_brute_force(
    const vector<Stmt> &stmts,
    size_t stmt_idx,
    const vector<string> &new_vars,
    Mapping &current,
    vector<bool> &used,
    vector<Mapping> &results) const
{
    const size_t n_pts = _problem->num_points();
    const size_t n_vars = new_vars.size();
    bool allow_duplicate = _allow_duplicates;

    // 内部递归 Lambda 函数用于生成笛卡尔积
    std::function<void(size_t)> generate_combinations = [&](size_t v_idx)
    {
        if (results.size() >= _result_limit) return;
        // 所有新变量都已绑定
        if (v_idx == n_vars)
        {
            if (verify_stmt(stmts[stmt_idx], current))
            {
                // 当前谓词数值校验通过，进入下一条 Stmt 的匹配
                backtrack(stmts, stmt_idx + 1, current, used, results);
            }
            return;
        }

        // 为第 v_idx 个新变量尝试绑定每一个点
        for (size_t p = 0; p < n_pts; ++p)
        {
            if (!allow_duplicate && used[p])
                continue;
            current.push_back({new_vars[v_idx], (int)p});
            used[p] = true;
            generate_combinations(v_idx + 1);
            used[p] = false;
            current.pop_back();
            if (results.size() >= _result_limit) break;
        }
    };

    generate_combinations(0);
}

void CustomTheoremMatcher::backtrack(const vector<Stmt> &stmts, size_t stmt_idx,
                                     Mapping &current, vector<bool> &used,
                                     vector<Mapping> &results) const
{
    if (results.size() >= _result_limit) return;
    if (stmt_idx == stmts.size())
    {
        results.push_back(current);
        return;
    }

    const auto &stmt = stmts[stmt_idx];
    auto new_vars = collect_new_vars(stmt, current);

    // 情况 A：没有新变量，直接验证当前谓词
    if (new_vars.empty())
    {
        if (verify_stmt(stmt, current))
        {
            backtrack(stmts, stmt_idx + 1, current, used, results);
        }
        return;
    }

    // 情况 B：利用预计算索引快速匹配
    if (auto it = _index.find(stmt.first);
        it != _index.end() &&
        (it->second.empty() || it->second.front().size() == stmt.second.size()) &&
        (new_vars.size() > 1 || it->second.size() <= _problem->num_points()))
    {
        bool allow_duplicate = _allow_duplicates;

        // Compile a shared join index for the already bound argument columns.
        // This replaces repeated full relation scans without changing candidates
        // or their order, including repeated-variable and injectivity checks below.
        const vector<size_t> *selected = nullptr;
        if (_problem->num_points() <= 256 && stmt.second.size() <= 8 && _projection_enabled)
        {
            unsigned mask = 0;
            uint64_t key = 0;
            for (size_t i = 0; i < stmt.second.size(); ++i)
            {
                const int value = FindBinding(current, stmt.second[i]);
                if (value != -1)
                {
                    mask |= (1u << i);
                    key = (key << 8) | uint64_t(value);
                }
            }
            if (mask)
            {
            auto [projection, inserted] = _projections.try_emplace({stmt.first, mask});
            if (inserted)
                for (size_t row = 0; row < it->second.size(); ++row)
                {
                    uint64_t row_key = 0;
                    for (size_t column = 0; column < stmt.second.size(); ++column)
                        if (mask & (1u << column))
                            row_key = (row_key << 8) | uint64_t(it->second[row][column]);
                    projection->second[row_key].push_back(row);
                }
            auto found = projection->second.find(key);
            if (found == projection->second.end())
                return;
            selected = &found->second;
            }
        }
        const size_t candidate_count = selected ? selected->size() : it->second.size();
        for (size_t candidate = 0; candidate < candidate_count; ++candidate)
        {
            const auto &combo = it->second[selected ? (*selected)[candidate] : candidate];
            size_t added_count = 0;
            bool conflict = false;

            for (size_t i = 0; i < stmt.second.size(); ++i)
            {
                int bound_val = FindBinding(current, stmt.second[i]);
                if (bound_val != -1)
                {
                    if (bound_val != combo[i])
                    {
                        conflict = true;
                        break;
                    }
                }
                else
                {
                    if (!allow_duplicate && used[combo[i]])
                    {
                        conflict = true;
                        break;
                    }
                    current.push_back({stmt.second[i], combo[i]});
                    used[combo[i]] = true;
                    added_count++;
                }
            }

            if (!conflict)
                backtrack(stmts, stmt_idx + 1, current, used, results);

            // 回溯清理
            while (added_count--)
            {
                used[current.back().second] = false;
                current.pop_back();
            }
            if (results.size() >= _result_limit) break;
        }
    }
    // 情况 C：暴力枚举（仅针对无索引谓词）
    else
    {
        enumerate_brute_force(stmts, stmt_idx, new_vars, current, used, results);
    }
}

vector<Stmt> CustomTheoremMatcher::legacy_plan(const CustomRule &rule) const
{
    // 1. 准备并排序前提 (约束强的优先)
    // 注意：只用前提参与 backtrack 匹配；结论不加入 sorted_stmts。
    // 原来把结论也塞进来一起 sort+backtrack，会导致 backtrack 把结论当"已知事实"
    // 去 index 里查，而未被证明的结论自然不在 index 中，mapping 找不到 → raw_results
    // 为空 → 规则永远不会被应用。结论的验证在第 3 步逐条数值校验中完成。
    vector<Stmt> sorted_stmts = rule.premises;

    const auto deferred = DeferredPredicates(rule.premises);
    std::sort(sorted_stmts.begin(), sorted_stmts.end(), [this, &deferred](const Stmt &a, const Stmt &b)
              {
        if (deferred.contains(a.first) != deferred.contains(b.first))
            return deferred.contains(b.first);
        auto it_a = _index.find(a.first), it_b = _index.find(b.first);
        size_t size_a = (it_a != _index.end()) ? it_a->second.size() : 999999;
        size_t size_b = (it_b != _index.end()) ? it_b->second.size() : 999999;
        return size_a < size_b; });

    // Prefer checks whose points were bound by earlier premises. Preserve the
    // original order for ties so independent checks still use index estimates.
    unordered_map<string, uint64_t> variable_bits;
    bool fits_bits = true;
    for (const auto &stmt : sorted_stmts)
        for (size_t i = 0; i < GetPointArgsCount(stmt); ++i)
            if (!variable_bits.contains(stmt.second[i]))
            {
                if (variable_bits.size() == 64) { fits_bits = false; break; }
                variable_bits.emplace(stmt.second[i], uint64_t(1) << variable_bits.size());
            }
    if (fits_bits && get_config("custom_bitset_plan", true))
    {
        vector<uint64_t> masks;
        for (const auto &stmt : sorted_stmts)
        {
            uint64_t mask = 0;
            for (size_t j = 0; j < GetPointArgsCount(stmt); ++j)
                mask |= variable_bits.at(stmt.second[j]);
            masks.push_back(mask);
        }
        uint64_t bound_mask = 0;
        for (size_t first = 0; first < sorted_stmts.size(); ++first)
        {
            size_t best = first;
            for (size_t candidate = first + 1; candidate < sorted_stmts.size(); ++candidate)
            {
                bool candidate_deferred = deferred.contains(sorted_stmts[candidate].first);
                bool best_deferred = deferred.contains(sorted_stmts[best].first);
                if (candidate_deferred != best_deferred ? !candidate_deferred :
                    std::popcount(masks[candidate] & ~bound_mask) < std::popcount(masks[best] & ~bound_mask))
                    best = candidate;
            }
            std::rotate(sorted_stmts.begin()+first, sorted_stmts.begin()+best, sorted_stmts.begin()+best+1);
            std::rotate(masks.begin()+first, masks.begin()+best, masks.begin()+best+1);
            bound_mask |= masks[first];
        }
    }
    else
    {
        set<string> bound;
        for (auto first = sorted_stmts.begin(); first != sorted_stmts.end(); ++first)
        {
            auto unbound_count = [&](const Stmt &stmt) {
                set<string> new_points;
                for (size_t i = 0; i < GetPointArgsCount(stmt); ++i)
                    if (!bound.contains(stmt.second[i]))
                        new_points.insert(stmt.second[i]);
                return new_points.size();
            };
            auto best = std::min_element(first, sorted_stmts.end(), [&](const Stmt &a, const Stmt &b) {
                if (deferred.contains(a.first) != deferred.contains(b.first))
                    return !deferred.contains(a.first);
                return unbound_count(a) < unbound_count(b);
            });
            std::rotate(first, best, best + 1);
            for (size_t i = 0; i < GetPointArgsCount(*first); ++i)
                bound.insert(first->second[i]);
        }
    }

    // Schedule numerical inequality guards immediately after their last variable
    // is bound. They remain guards, never logical hypotheses. Original premise
    // order and candidate traversal are preserved.
    if (get_config("custom_early_guards", true))
    {
        vector<Stmt> plan;
        set<string> available;
        vector<bool> scheduled(rule.guards.size(), false);
        for (const auto &premise : sorted_stmts)
        {
            plan.push_back(premise);
            for (size_t i = 0; i < GetPointArgsCount(premise); ++i)
                available.insert(premise.second[i]);
            for (size_t i = 0; i < rule.guards.size(); ++i)
            {
                const auto &guard = rule.guards[i];
                if (scheduled[i] || (guard.first != "npara" && guard.first != "ncoll" &&
                                     guard.first != "diff" && guard.first != "angle_defined" &&
                                     guard.first != "neqangle"))
                    continue;
                if (std::all_of(guard.second.begin(), guard.second.end(),
                                [&](const string &arg) { return available.contains(arg); }))
                {
                    Stmt check = guard;
                    if (guard.first == "diff" || guard.first == "angle_defined" || guard.first == "neqangle")
                        check.first = "__guard_" + guard.first;
                    plan.push_back(std::move(check));
                    scheduled[i] = true;
                }
            }
        }
        sorted_stmts = std::move(plan);
    }

    return sorted_stmts;
}

vector<Stmt> CustomTheoremMatcher::static_plan(const CustomRule &rule) const
{
    if (!rule.static_prepared || !rule.static_fits || !get_config("custom_bitset_plan",true) ||
        !get_config("custom_early_guards",true)) return legacy_plan(rule);
    const size_t n=rule.premises.size();
    vector<size_t> order(n), sizes(n);
    for (size_t i=0;i<n;++i)
    {
        order[i]=i;
        auto found=_index.find(rule.premises[i].first);
        sizes[i]=found==_index.end() ? 999999 : found->second.size();
    }
    std::sort(order.begin(),order.end(),[&](size_t a,size_t b) {
        if (rule.static_deferred[a]!=rule.static_deferred[b]) return bool(rule.static_deferred[b]);
        return sizes[a]<sizes[b];
    });
    uint64_t bound=0;
    for (size_t first=0;first<n;++first)
    {
        size_t best=first;
        for (size_t candidate=first+1;candidate<n;++candidate)
        {
            size_t a=order[candidate], b=order[best];
            if (rule.static_deferred[a]!=rule.static_deferred[b] ? !rule.static_deferred[a] :
                std::popcount(rule.static_masks[a]&~bound)<std::popcount(rule.static_masks[b]&~bound)) best=candidate;
        }
        std::rotate(order.begin()+first,order.begin()+best,order.begin()+best+1);
        bound |= rule.static_masks[order[first]];
    }
    vector<Stmt> plan; plan.reserve(n+rule.early_checks.size());
    vector<uint8_t> scheduled(rule.early_checks.size(),0);
    bound=0;
    for (size_t i : order)
    {
        plan.push_back(rule.premises[i]); bound |= rule.static_masks[i];
        for (size_t g=0;g<scheduled.size();++g)
            if (!scheduled[g] && rule.early_valid[g] && (rule.early_masks[g]&~bound)==0)
            {
                plan.push_back(rule.early_checks[g]); scheduled[g]=1;
            }
    }
    return plan;
}

void CustomTheoremMatcher::match_rule(const CustomRule &rule)
{
    auto planning_start = chrono::steady_clock::now();
    auto sorted_stmts = _static_plans ? static_plan(rule) : legacy_plan(rule);
    if (get_config("custom_validate_plan", false) && sorted_stmts != legacy_plan(rule))
        throw std::runtime_error("Static plan differs for rule " + rule.rule);
    auto backtrack_start = chrono::steady_clock::now();
    _planning_seconds += chrono::duration<double>(backtrack_start-planning_start).count();
    // 2. 执行匹配（仅前提）
    Mapping current;
    vector<bool> used(_problem->num_points(), false);
    vector<Mapping> raw_results;
    if (get_config("custom_shared_prefix", true) && sorted_stmts.size() >= 2)
    {
        size_t depth = 0;
        set<string> prefix_variables;
        for (size_t i = 0; i < std::min(size_t(6), sorted_stmts.size()); ++i)
        {
            for (size_t j = 0; j < GetPointArgsCount(sorted_stmts[i]); ++j)
                prefix_variables.insert(sorted_stmts[i].second[j]);
            if (prefix_variables.size() > 8) break;
            ++depth;
        }
        if (depth == 0)
            backtrack(sorted_stmts, 0, current, used, raw_results);
        else
        {
            vector<Stmt> prefix(sorted_stmts.begin(), sorted_stmts.begin() + depth);
            auto cached = _prefix_matches.find(prefix);
            vector<Mapping> temporary;
            const vector<Mapping> *bindings = nullptr;
            if (cached != _prefix_matches.end())
                bindings = &cached->second;
            else
            {
                // Bound temporary materialization. An incomplete prefix is never
                // cached or used: rerun the complete original search instead.
                _result_limit = 5001;
                backtrack(prefix, 0, current, used, temporary);
                _result_limit = size_t(-1);
                if (temporary.size() >= 5001)
                    backtrack(sorted_stmts, 0, current, used, raw_results);
                else if (_prefix_binding_count + temporary.size() <= 100000)
                {
                    _prefix_binding_count += temporary.size();
                    bindings = &_prefix_matches.emplace(std::move(prefix), std::move(temporary)).first->second;
                }
                else
                    bindings = &temporary;
            }
            if (bindings)
                for (const auto &binding : *bindings)
                {
                    current = binding;
                    std::fill(used.begin(), used.end(), false);
                    for (const auto &entry : current) used[entry.second] = true;
                    backtrack(sorted_stmts, depth, current, used, raw_results);
                }
        }
    }
    else
        backtrack(sorted_stmts, 0, current, used, raw_results);

    auto emitting_start = chrono::steady_clock::now();
    _backtrack_seconds += chrono::duration<double>(emitting_start-backtrack_start).count();
    // 3. 结果转换与去重
    std::set<map<string, int>> unique_mappings;
    for (auto &m : raw_results)
    {
        map<string, int> m_map(m.begin(), m.end());
        if (!unique_mappings.insert(m_map).second)
            continue;

        Theorem thm(rule.name, rule.rule);
        auto build_stmt = [&](const Stmt &s_stmt) -> unique_ptr<Statement>
        {
            vector<string> args;
            const size_t point_args = GetPointArgsCount(s_stmt);
            for (size_t i = 0; i < point_args; ++i)
            {
                auto it = m_map.find(s_stmt.second[i]);
                if (it == m_map.end())
                {
                    // Variable not mapped to any concrete point in this
                    // problem — the rule does not apply.  Return nullptr
                    // so the caller silently skips this match.
                    return nullptr;
                }
                args.push_back(_problem->point(it->second).name());
            }
            if (point_args < s_stmt.second.size())
                args.push_back(s_stmt.second.back());
            try
            {
                return _problem->create_statement(s_stmt.first, args);
            }
            catch (const exception &)
            {
                return nullptr; // silently skip invalid statements
            }
        };

        bool premises_ok = true;
        for (const auto &s_stmt : rule.premises)
        {
            if (auto res_stmt = build_stmt(s_stmt))
            {
                thm.add_hypothesis(move(res_stmt));
            }
            else
            {
                premises_ok = false;
                break;
            }
        }
        if (!premises_ok)
            continue;

        // 前提须整体通过数值校验，否则这组匹配本身不成立。
        if (!thm.check_hypotheses_numerically())
            continue;

        // 守卫谓词：只做数值检查，不参与推理链。任一守卫不通过则跳过此匹配。
        bool guards_ok = true;
        for (const auto &g_stmt : rule.guards)
        {
            if (g_stmt.first == "angle_defined")
            {
                auto angle = build_stmt({"eqangle", g_stmt.second});
                if (angle && angle->check_nondegen())
                    continue;
                guards_ok = false;
                break;
            }
            if (g_stmt.first == "diff")
            {
                if (g_stmt.second.size() == 2)
                {
                    const auto left = m_map.find(g_stmt.second[0]);
                    const auto right = m_map.find(g_stmt.second[1]);
                    if (left != m_map.end() && right != m_map.end())
                    {
                        const auto &a = _problem->point(left->second);
                        const auto &b = _problem->point(right->second);
                        const double distance = std::hypot(a.x() - b.x(), a.y() - b.y());
                        if (std::isfinite(distance) && !a.is_close(b))
                            continue;
                    }
                }
                guards_ok = false;
                break;
            }
            const bool unequal_angle = g_stmt.first == "neqangle";
            auto g = unequal_angle ? build_stmt({"eqangle", g_stmt.second}) : build_stmt(g_stmt);
            // Invalid or degenerate angles cannot witness inequality.
            const bool satisfied = g && (unequal_angle
                ? g->check_nondegen() && !g->check_equations()
                : g->check_numerically());
            if (!satisfied || !stmt_instance_ok(g_stmt, m_map))
            {
                guards_ok = false;
                break;
            }
        }
        if (!guards_ok)
            continue;

        // 结论逐条数值校验：merge_same_premise 合并出的多结论规则，某条结论
        // 在具体坐标下可能因精度或退化而校验失败——只丢弃该结论，而非因为
        // 一条结论没过就连带丢弃同一前提下已经验证通过的其它结论。
        //
        // 结论不参与 backtrack 匹配（见上），意味着结论里出现的点必须全部来自
        // 前提绑定的 m_map；如果规则本身有问题、结论引用了前提里没出现的点
        // （如 rule_text 写成 "... => eqangle A B C D X Y X Y" 但前提只有
        // A B C D），build_stmt 会把未绑定的原始变量名当成点名传给
        // create_statement，导致该点在当前 problem 中查找失败而崩溃。跳过
        // 这类结论，而不是让整条匹配崩掉。
        for (const auto &s_stmt : rule.conclusions)
        {
            auto res_stmt = build_stmt(s_stmt);
            if (res_stmt && res_stmt->check_numerically() && stmt_instance_ok(s_stmt, m_map))
            {
                thm.add_conclusion(move(res_stmt));
            }
        }

        if (!thm.conclusions().empty())
        {
            _theorems.push_back(thm.normalize());
        }
    }
    _emitting_seconds += chrono::duration<double>(chrono::steady_clock::now()-emitting_start).count();
}

CustomTheoremMatcher::CustomTheoremMatcher(Problem *prob, const vector<CustomRule> &rules,
                                           const map<string, bool> &config)
    : _problem(prob), _config(config)
{
    auto index_start = chrono::steady_clock::now();
    _memo_enabled = get_config("custom_numeric_memo", true);
    _projection_enabled = get_config("custom_projection_index", true);
    _allow_duplicates = get_config("allow_duplicate_points", false);
    _static_plans = get_config("custom_static_plan", true);
    // 默认启用预计算索引
    bool use_cache = get_config("use_predicate_cache", true);
    if (use_cache)
        build_predicate_index(rules);

    auto matching_start = chrono::steady_clock::now();
    for (const auto &rule : rules)
    {
        auto started = chrono::steady_clock::now();
        size_t before = _theorems.size();
        match_rule(rule);
        if (get_config("profile_custom_match", false))
        {
            double seconds = chrono::duration<double>(chrono::steady_clock::now()-started).count();
            if (seconds > 0.005)
                cerr << "custom_profile " << rule.rule << " " << seconds << " " << (_theorems.size()-before) << endl;
        }
    }

    if (get_config("profile_custom_match", false))
    {
        cerr << "custom_stage index " << chrono::duration<double>(matching_start-index_start).count()
             << " matching " << chrono::duration<double>(chrono::steady_clock::now()-matching_start).count()
             << " theorems " << _theorems.size()
             << " planning " << _planning_seconds << " backtrack " << _backtrack_seconds
             << " emitting " << _emitting_seconds
             << " numeric_hits " << _numeric_hits << " numeric_misses " << _numeric_misses << endl;
        for (const auto &[name, rows] : _index)
        {
            set<vector<int>> unique(rows.begin(), rows.end());
            cerr << "custom_index " << name << " rows " << rows.size() << " unique " << unique.size() << endl;
        }
    }

    // 调试用：打印每条 custom 规则实际匹配出的定理条数
    // （config "debug_custom_match"，默认关闭）。para 退化筛子的效果可直接由条数差异看出。
    if (get_config("debug_custom_match", false))
    {
        map<string, int> by_rule;
        for (const auto &t : _theorems)
            by_rule[t.rule()]++;
        cout << "[CustomTheoremMatcher] points=" << _problem->num_points()
             << " keys_rules=" << rules.size() << " theorems=" << _theorems.size()
             << " custom_para_no_collinear="
             << (get_config("custom_para_no_collinear", true) ? 1 : 0)
             << " allow_duplicate_points="
             << (_allow_duplicates ? 1 : 0) << endl;
        for (const auto &kv : by_rule)
            cout << "    " << kv.first << " x" << kv.second << endl;
    }
}
