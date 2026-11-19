#include "predicate/similar_triangles.hpp"
#include "predicate/statement.hpp"
#include "type/triangle.hpp"
#include "type/angle.hpp"
#include "type/point.hpp"
#include "type/dist.hpp"
#include "predicate/eqratio.hpp"
#include <vector>
#include <iostream>
#include <algorithm>
#include <array>

using namespace std;

SimilarTriangles::SimilarTriangles(const Triangle &t1, const Triangle &t2, bool sameclock) : _left(t1), _right(t2), _sameclock(sameclock) {}

SimilarTriangles::SimilarTriangles(const vector<statement_arg> &args) : _left(args[0].tri), _right(args[1].tri), _sameclock(args[2].b) {}

string SimilarTriangles::name() const
{
    return _sameclock ? "simtri" : "simtrir";
}

unique_ptr<Statement> SimilarTriangles::replace(Point p, Point q) const
{
    auto left_pts = _left.points();
    Point la = left_pts[0], lb = left_pts[1], lc = left_pts[2];

    Point new_la = (la == p) ? q : la;
    Point new_lb = (lb == p) ? q : lb;
    Point new_lc = (lc == p) ? q : lc;

    Triangle new_left(new_la, new_lb, new_lc);

    auto right_pts = _right.points();
    Point ra = right_pts[0], rb = right_pts[1], rc = right_pts[2];

    Point new_ra = (ra == p) ? q : ra;
    Point new_rb = (rb == p) ? q : rb;
    Point new_rc = (rc == p) ? q : rc;

    Triangle new_right(new_ra, new_rb, new_rc);

    return make_unique<SimilarTriangles>(new_left, new_right, _sameclock);
}

vector<Point> SimilarTriangles::points() const
{
    return {_left.a(), _left.b(), _left.c(), _right.a(), _right.b(), _right.c()};
}

unique_ptr<Statement> SimilarTriangles::clone() const
{
    return make_unique<SimilarTriangles>(*this);
}

unique_ptr<Statement> SimilarTriangles::normalize() const
{
    // Match the order in Triangle::permutations(), then try the swapped pair.
    constexpr array<array<size_t, 3>, 6> orders{{
        {0, 1, 2}, {1, 2, 0}, {2, 0, 1},
        {0, 2, 1}, {2, 1, 0}, {1, 0, 2},
    }};
    const array<const Point *, 6> points{
        &_left.a(), &_left.b(), &_left.c(),
        &_right.a(), &_right.b(), &_right.c(),
    };

    array<const Point *, 6> best = points;
    for (size_t offset : {size_t{0}, size_t{3}})
    {
        const size_t other = 3 - offset;
        for (const auto &order : orders)
        {
            const array<const Point *, 6> candidate{
                points[offset + order[0]], points[offset + order[1]], points[offset + order[2]],
                points[other + order[0]], points[other + order[1]], points[other + order[2]],
            };
            // Triangle ordering advances only when point names are equal.
            const auto [candidate_point, best_point] = mismatch(
                candidate.begin(), candidate.end(), best.begin(),
                [](const Point *a, const Point *b) { return *a == *b; });
            if (candidate_point != candidate.end() && **candidate_point < **best_point)
            {
                best = candidate;
            }
        }
    }
    return make_unique<SimilarTriangles>(
        Triangle(*best[0], *best[1], *best[2]),
        Triangle(*best[3], *best[4], *best[5]), _sameclock);
}

bool SimilarTriangles::check_nondegen() const
{
    return _left.check_nondegen() && _right.check_nondegen() && (_sameclock == ((_left.area() > 0) == (_right.area() > 0))) && !_left.overlap(_right);
}

bool SimilarTriangles::check_equations() const
{
    return eqratio_abac().check_equations() && eqratio_bcac().check_equations();
}

vector<statement_arg> SimilarTriangles::args() const
{
    return {_left, _right, _sameclock};
}

SameClock SimilarTriangles::to_sameclock() const
{
    if (_sameclock)
    {
        return SameClock(_left, _right);
    }
    return SameClock(_left, Triangle(_right.a(), _right.c(), _right.b()));
}

EqRatio SimilarTriangles::eqratio_abbc() const
{
    return EqRatio(Dist(_left.a(), _left.b()), Dist(_left.b(), _left.c()), Dist(_right.a(), _right.b()), Dist(_right.b(), _right.c()));
}

EqRatio SimilarTriangles::eqratio_abac() const
{
    return EqRatio(Dist(_left.a(), _left.b()), Dist(_left.a(), _left.c()), Dist(_right.a(), _right.b()), Dist(_right.a(), _right.c()));
}

EqRatio SimilarTriangles::eqratio_bcac() const
{
    return EqRatio(Dist(_left.b(), _left.c()), Dist(_left.a(), _left.c()), Dist(_right.b(), _right.c()), Dist(_right.a(), _right.c()));
}

EqAngle SimilarTriangles::eqangle_abc() const
{
    return EqAngle(_left.angle_b(), _sameclock ? _right.angle_b() : -_right.angle_b());
}

EqAngle SimilarTriangles::eqangle_bca() const
{
    return EqAngle(_left.angle_c(), _sameclock ? _right.angle_c() : -_right.angle_c());
}

EqAngle SimilarTriangles::eqangle_acb() const
{
    return EqAngle(-_left.angle_c(), _sameclock ? -_right.angle_c() : _right.angle_c());
}

EqAngle SimilarTriangles::eqangle_cab() const
{
    return EqAngle(_left.angle_a(), _sameclock ? _right.angle_a() : -_right.angle_a());
}

vector<SimilarTriangles> SimilarTriangles::permutations() const
{
    const auto left = _left.permutations();
    const auto right = _right.permutations();
    return {
        SimilarTriangles(left[0], right[0], _sameclock),
        SimilarTriangles(left[1], right[1], _sameclock),
        SimilarTriangles(left[2], right[2], _sameclock),
        SimilarTriangles(left[3], right[3], _sameclock),
        SimilarTriangles(left[4], right[4], _sameclock),
        SimilarTriangles(left[5], right[5], _sameclock),
        SimilarTriangles(right[0], left[0], _sameclock),
        SimilarTriangles(right[1], left[1], _sameclock),
        SimilarTriangles(right[2], left[2], _sameclock),
        SimilarTriangles(right[3], left[3], _sameclock),
        SimilarTriangles(right[4], left[4], _sameclock),
        SimilarTriangles(right[5], left[5], _sameclock),
    };
}

vector<SimilarTriangles> SimilarTriangles::cyclic_rotations() const
{
    const auto left = _left.cyclic_rotations();
    const auto right = _right.cyclic_rotations();
    return {SimilarTriangles(left[0], right[0], _sameclock),
            SimilarTriangles(left[1], right[1], _sameclock),
            SimilarTriangles(left[2], right[2], _sameclock)};
}

ostream &SimilarTriangles::print(ostream &os) const
{
    return os << _left << " ∼" << (_sameclock ? " " : "r ") << _right;
}

bool SimilarTriangles::operator==(const SimilarTriangles &other) const
{
    return _left == other._left && _right == other._right;
}

bool SimilarTriangles::operator!=(const SimilarTriangles &other) const
{
    return !(*this == other);
}

bool SimilarTriangles::operator<(const SimilarTriangles &other) const
{
    if (_left == other._left)
    {
        return _right < other._right;
    }
    return _left < other._left;
}

bool SimilarTriangles::operator<=(const SimilarTriangles &other) const
{
    return *this < other || *this == other;
}

bool SimilarTriangles::operator>(const SimilarTriangles &other) const
{
    return !(*this <= other);
}

bool SimilarTriangles::operator>=(const SimilarTriangles &other) const
{
    return !(*this < other);
}
