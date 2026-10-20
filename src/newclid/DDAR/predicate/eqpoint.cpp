#include "predicate/eqpoint.hpp"
#include <iostream>

using namespace std;

EqPoint::EqPoint(Point a, Point b) : _a(a), _b(b) {}

EqPoint::EqPoint(const vector<statement_arg> &args) : _a(args[0].point), _b(args[1].point) {}

string EqPoint::name() const
{
    return "eqpoint";
}

vector<Point> EqPoint::points() const
{
    return {_a, _b};
}

unique_ptr<Statement> EqPoint::normalize() const
{
    vector<Point> pts = points();
    sort(pts.begin(), pts.end());
    return make_unique<EqPoint>(pts[0], pts[1]);
}

bool EqPoint::check_nondegen() const
{
    return !(_a == _b);
}

bool EqPoint::check_equations() const
{
    return _a.is_close(_b);
}

unique_ptr<Statement> EqPoint::clone() const
{
    return make_unique<EqPoint>(*this);
}

vector<statement_arg> EqPoint::args() const
{
    return {_a, _b};
}

ostream &EqPoint::print(ostream &out) const
{
    return out << _a << " ≡ " << _b;
}

unique_ptr<Statement> EqPoint::replace(Point p, Point q) const
{
    Point new_a = (_a == p) ? q : _a;
    Point new_b = (_b == p) ? q : _b;
    return std::make_unique<EqPoint>(new_a, new_b);
}

vector<unique_ptr<Equation>> EqPoint::as_equation_dist(bool exp) const
{
    if(!exp)
    {
        return {};
    }
    vector<unique_ptr<Equation>> result;
    result.push_back(make_unique<Equation>(Equation({Term(Dist(_a, _b))})));
    return result;
}