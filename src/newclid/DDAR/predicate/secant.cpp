#include "predicate/secant.hpp"
#include "predicate/cong.hpp"
#include "predicate/perp.hpp"
#include "type/dist.hpp"
#include <vector>

using namespace std;

Secant::Secant(const Point &o, const Point &a, const Point &b, const Point &p)
    : _o(o), _a(a), _b(b), _p(p) {}

Secant::Secant(const vector<statement_arg> &args)
    : _o(args[0].point), _a(args[1].point), _b(args[2].point), _p(args[3].point) {}

string Secant::name() const
{
    return "secant";
}

vector<Point> Secant::points() const
{
    return {_o, _a, _b, _p};
}

vector<statement_arg> Secant::args() const
{
    return {_o, _a, _b, _p};
}

unique_ptr<Statement> Secant::replace(Point p, Point q) const
{
    Point new_o = (_o == p) ? q : _o;
    Point new_a = (_a == p) ? q : _a;
    Point new_b = (_b == p) ? q : _b;
    Point new_p = (_p == p) ? q : _p;

    return make_unique<Secant>(new_o, new_a, new_b, new_p);
}

unique_ptr<Statement> Secant::clone() const
{
    return make_unique<Secant>(*this);
}

unique_ptr<Statement> Secant::normalize() const
{
    if (_a > _b)
    {
        return make_unique<Secant>(_o, _b, _a, _p);
    }
    return make_unique<Secant>(_o, _a, _b, _p);
}

bool Secant::check_nondegen() const
{
    return !(_p.is_close(_a)) && !(_p .is_close(_b)) && !(_p.is_close(_o)) && !(_o.is_close( _a)) && !(_o.is_close( _b));
}

bool Secant::check_equations() const
{
    if (_a.is_close( _b))
    {
        return cong_ab().check_equations() && Perp(Slope(_a, _o), Slope(_o, _p)).check_equations();
    }
    return cong_ab().check_equations() && coll_pab().check_equations();
}

ostream &Secant::print(ostream &os) const
{
    return os << "secant(" << _o << ", " << _p << ", " << _a << ", " << _b << ")";
}

Cong Secant::cong_ab() const
{
    return Cong(Dist(_o, _a), Dist(_o, _b));
}

Coll Secant::coll_pab() const
{
    return Coll(_p, _a, _b);
}

vector<unique_ptr<Equation>> Secant::as_equation_dist(bool exp) const
{
    vector<unique_ptr<Equation>> result;
    if (!exp)
    {
        return result;
    }

    Term pab({Dist(_p, _a), Dist(_p, _b)});
    Term oa2({Dist(_o, _a), Dist(_o, _a)});
    Term ob2({Dist(_o, _b), Dist(_o, _b)});
    Term op2({Dist(_o, _p), Dist(_o, _p)});
    if (Coll(_a, _p, _b).is_between())
    {
        result.push_back(make_unique<Equation>(Equation({oa2, -op2, -pab})));
        result.push_back(make_unique<Equation>(Equation({ob2, -op2, -pab})));
    }
    else
    {
        result.push_back(make_unique<Equation>(Equation({oa2, -op2, pab})));
        result.push_back(make_unique<Equation>(Equation({ob2, -op2, pab})));
    }

    return result;
}