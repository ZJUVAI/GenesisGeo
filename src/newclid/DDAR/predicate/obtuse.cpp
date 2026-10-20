#include "predicate/obtuse.hpp"
#include "numerical.hpp"
#include <iostream>

using namespace std;

Obtuse::Obtuse(Point a, Point b, Point c) : _a(a), _b(b), _c(c) {}

string Obtuse::name() const
{
    return "obtuse";
}

vector<Point> Obtuse::points() const
{
    return {_a, _b, _c};
}

unique_ptr<Statement> Obtuse::replace(Point p, Point q) const
{
    Point new_a = (_a == p) ? q : _a;
    Point new_b = (_b == p) ? q : _b;
    Point new_c = (_c == p) ? q : _c;
    return std::make_unique<Obtuse>(new_a, new_b, new_c);
}

unique_ptr<Statement> Obtuse::normalize() const
{
    // Keep vertex (_b) fixed; sort the other two
    vector<Point> others = {_a, _c};
    sort(others.begin(), others.end());
    return make_unique<Obtuse>(others[0], _b, others[1]);
}

bool Obtuse::check_nondegen() const
{
    // Vertex must be distinct from the other two points
    return !_a.is_close(_b) && !_c.is_close(_b);
}

bool Obtuse::check_equations() const
{
    // obtuse: dot product (A-B)·(C-B) < 0
    double dx1 = _a.x() - _b.x();
    double dy1 = _a.y() - _b.y();
    double dx2 = _c.x() - _b.x();
    double dy2 = _c.y() - _b.y();
    double dot = dx1 * dx2 + dy1 * dy2;
    return dot < -1e-8;
}

vector<statement_arg> Obtuse::args() const
{
    return {_a, _b, _c};
}

unique_ptr<Statement> Obtuse::clone() const
{
    return make_unique<Obtuse>(*this);
}

ostream &Obtuse::print(ostream &os) const
{
    return os << "∠(" << _a << _b << _c << ") > 90°";
}
