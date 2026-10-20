#include "predicate/eqratio.hpp"
#include "typedef.hpp"
#include "type/dist.hpp"
#include "type/distlog.hpp"
#include <algorithm>
#include <iostream>

using namespace std;

EqRatio::EqRatio(Dist left_up, Dist left_down, Dist right_up, Dist right_down) : _left_up(left_up), _left_down(left_down), _right_up(right_up), _right_down(right_down)
{
}

string EqRatio::name() const
{
    return "eqratio";
}

vector<Point> EqRatio::points() const
{
    return {_left_up.left(), _left_up.right(), _left_down.left(), _left_down.right(), _right_up.left(), _right_up.right(), _right_down.left(), _right_down.right()};
}

unique_ptr<Statement> EqRatio::clone() const
{
    return make_unique<EqRatio>(*this);
}

bool EqRatio::check_nondegen() const
{
    return !_left_up.left().is_close(_left_up.right()) &&
           !_left_down.left().is_close(_left_down.right()) &&
           !_right_up.left().is_close(_right_up.right()) &&
           !_right_down.left().is_close(_right_down.right());
}

bool EqRatio::check_equations() const
{
    return Numerical::close_enough(_left_up.to_double() * _right_down.to_double(), _left_down.to_double() * _right_up.to_double());
}

vector<statement_arg> EqRatio::args() const
{
    return {_left_up, _left_down, _right_up, _right_down};
}

unique_ptr<Statement> EqRatio::replace(Point p, Point q) const
{
    auto lu_pts = _left_up.points();
    Point lu_a = lu_pts[0], lu_b = lu_pts[1];
    Point new_lu_a = (lu_a == p) ? q : lu_a;
    Point new_lu_b = (lu_b == p) ? q : lu_b;
    Dist new_left_up(new_lu_a, new_lu_b);

    auto ld_pts = _left_down.points();
    Point ld_a = ld_pts[0], ld_b = ld_pts[1];
    Point new_ld_a = (ld_a == p) ? q : ld_a;
    Point new_ld_b = (ld_b == p) ? q : ld_b;
    Dist new_left_down(new_ld_a, new_ld_b);

    auto ru_pts = _right_up.points();
    Point ru_a = ru_pts[0], ru_b = ru_pts[1];
    Point new_ru_a = (ru_a == p) ? q : ru_a;
    Point new_ru_b = (ru_b == p) ? q : ru_b;
    Dist new_right_up(new_ru_a, new_ru_b);

    auto rd_pts = _right_down.points();
    Point rd_a = rd_pts[0], rd_b = rd_pts[1];
    Point new_rd_a = (rd_a == p) ? q : rd_a;
    Point new_rd_b = (rd_b == p) ? q : rd_b;
    Dist new_right_down(new_rd_a, new_rd_b);

    return make_unique<EqRatio>(new_left_up, new_left_down, new_right_up, new_right_down);
}

unique_ptr<Statement> EqRatio::normalize() const
{
    Dist a = left_up().normalize();
    Dist b = left_down().normalize();
    Dist c = right_up().normalize();
    Dist d = right_down().normalize();

    vector<array<Dist, 4>> candidates = {
        {a, b, c, d},
        {a, c, b, d},
        {b, a, d, c},
        {b, d, a, c},
        {c, a, d, b},
        {c, d, a, b},
        {d, b, c, a},
        {d, c, b, a},
    };

    auto best = *min_element(candidates.begin(), candidates.end());

    return make_unique<EqRatio>(
        best[0], best[1], best[2], best[3]
    );
}

ostream &EqRatio::print(ostream &os) const
{
    return os << _left_up << ":" << _left_down << " = " << _right_up << ":" << _right_down;
}

vector<unique_ptr<Equation>> EqRatio::as_equation_dist(bool exp) const
{
    vector<unique_ptr<Equation>> result;
    if (exp)
    {
        result.push_back(make_unique<Equation>(Equation({Term({_left_up, _right_down}), -Term({_left_down, _right_up})})));
    }
    return result;
}

vector<unique_ptr<Equation>> EqRatio::as_equation_distlog(bool exp) const
{
    vector<unique_ptr<Equation>> result;
    result.push_back(make_unique<Equation>(Equation({Term(DistLog(_left_up)), Term(DistLog(_right_down)), -Term(DistLog(_left_down)), -Term(DistLog(_right_up))})));
    return result;
}