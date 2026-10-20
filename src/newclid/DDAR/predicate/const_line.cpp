#include "predicate/const_line.hpp"

using namespace std;

ConstLine::ConstLine(Point p, Point q1, Point q2) : _p(p), _q1(q1), _q2(q2) {}

ConstLine::ConstLine(const vector<statement_arg> &args) : _p(args[0].point), _q1(args[1].point), _q2(args[2].point) {}

string ConstLine::name() const
{
    return "constline";
}

vector<Point> ConstLine::points() const
{
    return {_p, _q1, _q2};
}

unique_ptr<Statement> ConstLine::replace(Point p, Point q) const
{
    if (_p == p)
    {
        return std::make_unique<ConstLine>(q, _q1, _q2);
    }
    else
    {
        return clone();
    }
}

unique_ptr<Statement> ConstLine::normalize() const
{
    if (_q1 < _q2)
    {
        return clone();
    }
    return std::make_unique<ConstLine>(_p, _q2, _q1);
}

vector<statement_arg> ConstLine::args() const
{
    return {_p, _q1, _q2};
}

bool ConstLine::check_nondegen() const
{
    return !(_p.is_close(_q1)) && !(_p.is_close(_q2)) && !(_q1 == _q2);
}

bool ConstLine::check_equations() const
{
    double const lhs = ((_q1.x() - _p.x()) * (_q2.y() - _p.y()));
    double const rhs = ((_q2.x() - _p.x()) * (_q1.y() - _p.y()));
    return Numerical::close_enough(lhs, rhs);
}

ostream &ConstLine::print(ostream &out) const
{
    return out << _q2 << " ∈ " << _p << _q1;
}

vector<unique_ptr<Equation>> ConstLine::as_equation_slope(bool exp) const
{
    if(!exp)
    {
        return {};
    }
    vector<unique_ptr<Equation>> result;
    result.push_back(make_unique<Equation>(Equation({Term(Slope(_p, _q1)), -Term(Slope(_p, _q2))})));
    return result;
}