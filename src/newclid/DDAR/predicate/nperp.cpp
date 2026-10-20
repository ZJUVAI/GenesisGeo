#include "predicate/nperp.hpp"
#include "predicate/perp.hpp"
#include <iostream>

using namespace std;

NPerp::NPerp(Slope left, Slope right) : _left(left), _right(right) {}

string NPerp::name() const
{
    return "nperp";
}

vector<Point> NPerp::points() const
{
    return {_left.left(), _left.right(), _right.left(), _right.right()};
}

unique_ptr<Statement> NPerp::replace(Point p, Point q) const
{
    auto replace_in_slope = [p, q](const Slope &s)
    {
        Point new_left = (s.left() == p) ? q : s.left();
        Point new_right = (s.right() == p) ? q : s.right();
        return Slope(new_left, new_right);
    };
    return make_unique<NPerp>(replace_in_slope(_left), replace_in_slope(_right));
}

unique_ptr<Statement> NPerp::normalize() const
{
    return make_unique<NPerp>(_left.normalize(), _right.normalize());
}

bool NPerp::check_nondegen() const
{
    return _left.check_nondegen() && _right.check_nondegen();
}

bool NPerp::check_equations() const
{
    return !Perp(_left, _right).check_equations();
}

vector<statement_arg> NPerp::args() const
{
    return {_left, _right};
}

unique_ptr<Statement> NPerp::clone() const
{
    return make_unique<NPerp>(*this);
}

ostream &NPerp::print(ostream &os) const
{
    return os << _left << " ∦ " << _right;
}
