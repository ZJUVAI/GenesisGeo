#include "predicate/aconst.hpp"
#include "predicate/statement.hpp"
#include "type/slope.hpp"
#include "type/angle.hpp"
#include "ar/equation.hpp"
#include <algorithm>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

using namespace std;

AConst::AConst(const Slope &s1, const Slope &s2, const Rational &rhs)
    : _s1(s1), _s2(s2), _rhs(rhs)
{
}

AConst::AConst(const Angle &ang, const Rational &rhs)
    : _s1(ang.left_side()), _s2(ang.right_side()), _rhs(rhs)
{
}

string AConst::name() const
{
    return "aconst";
}

vector<Point> AConst::points() const
{
    return {_s1.left(), _s1.right(), _s2.left(), _s2.right()};
}

unique_ptr<Statement> AConst::normalize() const
{
    // angle(s1, s2) = rhs·π ; negating the angle swaps the two slopes
    // and negates rhs. Keep the canonical orientation s1 < s2.
    Slope a = _s1.normalize();
    Slope b = _s2.normalize();
    if (a < b)
    {
        return make_unique<AConst>(a, b, _rhs);
    }
    return make_unique<AConst>(b, a, -_rhs);
}

bool AConst::check_nondegen() const
{
    return _s1.check_nondegen() && _s2.check_nondegen();
}

bool AConst::check_equations() const
{
    // angle between _s1 and _s2 should equal rhs·π
    double dx1 = _s1.right().x() - _s1.left().x();
    double dy1 = _s1.right().y() - _s1.left().y();
    double dx2 = _s2.right().x() - _s2.left().x();
    double dy2 = _s2.right().y() - _s2.left().y();

    double dot = dx1 * dx2 + dy1 * dy2;
    double cross = dx1 * dy2 - dy1 * dx2;

    double ang = std::atan2(cross, dot);
    if (ang < 0.0)
    {
        ang += M_PI;
    }

    return Numerical::close_enough(ang, _rhs.to_double() * M_PI);
}

vector<statement_arg> AConst::args() const
{
    return {_s1, _s2, _rhs};
}

ostream &AConst::print(std::ostream &out) const
{
    return out << _s2 << "-" << _s1 << " = " << _rhs << "π";
}

vector<unique_ptr<Equation>> AConst::as_equation_slope(bool exp) const
{
    vector<unique_ptr<Equation>> result;
    result.push_back(make_unique<Equation>(Equation({Term(_s2), -Term(_s1)})));
    return result;
}

vector<string> AConst::to_tokens() const
{
    vector<string> tokens = {"aconst"};
    tokens.push_back(_s1.left().name());
    tokens.push_back(_s1.right().name());
    tokens.push_back(_s2.left().name());
    tokens.push_back(_s2.right().name());
    tokens.push_back(std::to_string(_rhs.numerator()) + "/" + std::to_string(_rhs.denominator()));
    return tokens;
}

unique_ptr<Statement> AConst::replace(Point p, Point q) const
{
    auto replace_in_slope = [p, q](const Slope &s) {
        Point left = (s.left() == p) ? q : s.left();
        Point right = (s.right() == p) ? q : s.right();
        return Slope(left, right);
    };

    return make_unique<AConst>(replace_in_slope(_s1), replace_in_slope(_s2), _rhs);
}

string AConst::to_string() const
{
    string res = name();
    res += " " + _s1.left().name();
    res += " " + _s1.right().name();
    res += " " + _s2.left().name();
    res += " " + _s2.right().name();
    res += " " + _rhs.to_string();
    return res;
}
