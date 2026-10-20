#include "predicate/para.hpp"
#include "type/slope.hpp"
#include "ar/equation.hpp"
#include "typedef.hpp"
#include <iostream>

using namespace std;

Para::Para(Slope left, Slope right) : _left(left), _right(right) {}

string Para::name() const
{
    return "para";
}

vector<Point> Para::points() const
{
    return {_left.left(), _left.right(), _right.left(), _right.right()};
}

unique_ptr<Statement> Para::replace(Point p, Point q) const
{
    auto left_pts = _left.points();
    Point la = left_pts[0], lb = left_pts[1];

    Point new_la = (la == p) ? q : la;
    Point new_lb = (lb == p) ? q : lb;

    Slope new_left(new_la, new_lb);

    auto right_pts = _right.points();
    Point ra = right_pts[0], rb = right_pts[1];

    Point new_ra = (ra == p) ? q : ra;
    Point new_rb = (rb == p) ? q : rb;

    Slope new_right(new_ra, new_rb);

    return make_unique<Para>(new_left, new_right);
}

unique_ptr<Statement> Para::normalize() const
{
    if (_left > _right)
    {
        return make_unique<Para>(_right, _left);
    }
    return make_unique<Para>(*this);
}

bool Para::check_nondegen() const
{
    return _left.check_nondegen() && _right.check_nondegen();
}

bool Para::check_equations() const
{
    return Numerical::close_enough(_left.angle(), _right.angle());
}

vector<statement_arg> Para::args() const
{
    return {_left, _right};
}

unique_ptr<Statement> Para::clone() const
{
    return make_unique<Para>(_left, _right);
}

ostream &Para::print(ostream &os) const
{
    return os << _left.left() << _left.right() << " ∥ " << _right.left() << _right.right();
}

vector<unique_ptr<Equation>> Para::as_equation_slope(bool exp) const
{
    vector<unique_ptr<Equation>> result;
    result.push_back(make_unique<Equation>(Equation({Term(_left), -Term(_right)})));
    return result;
}