#include "predicate/rconst.hpp"
#include "type/rational.hpp"
#include "type/dist.hpp"
#include "ar/equation.hpp"
#include <cmath>

using namespace std;

RConst::RConst(const Dist &left, const Dist &right, const Rational &ratio) : _left(left), _right(right), _ratio(ratio) {}

string RConst::name() const { return "rconst"; }

vector<Point> RConst::points() const { return {_left.left(), _left.right(), _right.left(), _right.right()}; }

unique_ptr<Statement> RConst::clone() const
{
    return make_unique<RConst>(_left, _right, _ratio);
}

unique_ptr<Statement> RConst::normalize() const
{
    if (_left < _right)
    {
        return make_unique<RConst>(_left, _right, _ratio);
    }
    return make_unique<RConst>(swap());
}

unique_ptr<Statement> RConst::replace(Point p, Point q) const
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

    return make_unique<RConst>(new_left, new_right, _ratio);
}

bool RConst::check_nondegen() const
{
    return _left.check_nondegen() && _right.check_nondegen();
}

bool RConst::check_equations() const
{
    return Numerical::close_enough(_left.to_double(), _right.to_double() * _ratio.to_double());
}

vector<statement_arg> RConst::args() const
{
    return {_left, _right, _ratio};
}

RConst RConst::swap() const
{
    return RConst(_right, _left, Rational(1.0) / _ratio);
}

ostream &RConst::print(ostream &os) const
{
    return os << _left << ":" << _right << " = " << _ratio;
}

vector<unique_ptr<Equation>> RConst::as_equation_dist(bool exp) const
{
    vector<unique_ptr<Equation>> result;
    result.push_back(make_unique<Equation>(Equation({Term(_left), -Term(_right, _ratio)})));
    return result;
}

string RConst::to_string() const
{
    string res = name();
    for (const auto &pt : points())
    {
        res += " " + pt.name();
    }
    res += " " + _ratio.to_string();
    return res;
}

vector<string> RConst::to_tokens() const
{
    vector<string> res = {"rconst"};
    for (const auto &pt : points())
    {
        res.push_back(pt.name());
    }
    res.push_back(_ratio.to_string());
    return res;
}