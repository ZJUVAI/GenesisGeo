#include "predicate/cong.hpp"
#include "type/dist.hpp"
#include "type/distlog.hpp"
#include "type/rational.hpp"
#include "ar/equation.hpp"
#include <iostream>
#include <vector>
#include <optional>

using namespace std;

string Cong::name() const
{
    return "cong";
}

vector<Point> Cong::points() const
{
    return {_left.left(), _left.right(), _right.left(), _right.right()};
}

unique_ptr<Statement> Cong::normalize() const
{
    if (_left > _right)
    {
        return make_unique<Cong>(_right.normalize(), _left.normalize());
    }
    return make_unique<Cong>(_left.normalize(), _right.normalize());
}

bool Cong::check_nondegen() const
{
    return _left.check_nondegen() && _right.check_nondegen();
}

bool Cong::check_equations() const
{
    return Numerical::close_enough(_left.to_double(), _right.to_double());
}

vector<statement_arg> Cong::args() const
{
    return {_left, _right};
}

ostream &Cong::print(ostream &os) const
{
    return os << _left << " = " << _right;
}

vector<unique_ptr<Equation>> Cong::as_equation_dist(bool exp) const
{
    if(!exp)
    {
        return {};
    }
    vector<unique_ptr<Equation>> result;
    result.push_back(make_unique<Equation>(Equation({Term(_left), -Term(_right)})));
    result.push_back(make_unique<Equation>(Equation({Term({_left, _left}), -Term({_right, _right})})));
    return result;
}

vector<unique_ptr<Equation>> Cong::as_equation_distlog(bool exp) const
{
    vector<unique_ptr<Equation>> result;
    result.push_back(make_unique<Equation>(Equation({Term(DistLog(_left)), -Term(DistLog(_right))})));
    return result;
}

unique_ptr<Statement> Cong::replace(Point p, Point q) const
{
    auto left_pts = _left.points();
    Point la = left_pts[0], lb = left_pts[1];

    Point new_la = (la == p) ? q : la;
    Point new_lb = (lb == p) ? q : lb;

    Dist new_left(new_la, new_lb);

    auto right_pts = _right.points();
    Point ra = right_pts[0], rb = right_pts[1];

    Point new_ra = (ra == p) ? q : ra;
    Point new_rb = (rb == p) ? q : rb;

    Dist new_right(new_ra, new_rb);

    return make_unique<Cong>(new_left, new_right);
}