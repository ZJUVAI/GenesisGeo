#include "predicate/eqangle.hpp"
#include "type/angle.hpp"
#include "ar/equation.hpp"
#include "typedef.hpp"
#include <algorithm>
#include <array>
#include <vector>

using namespace std;

// Constructor from 4 slopes directly
EqAngle::EqAngle(Slope s1, Slope s2, Slope s3, Slope s4)
    : _s1(s1), _s2(s2), _s3(s3), _s4(s4)
{
}

// Constructor from two angles (for backward compatibility)
EqAngle::EqAngle(Angle left, Angle right)
    : _s1(left.left_side()), _s2(left.right_side()),
      _s3(right.left_side()), _s4(right.right_side())
{
}

// Constructor from 8 points - directly construct 4 slopes
EqAngle::EqAngle(Point p1, Point p2, Point p3, Point p4, Point p5, Point p6, Point p7, Point p8)
    : _s1(p1, p2), _s2(p3, p4), _s3(p5, p6), _s4(p7, p8)
{
}

unique_ptr<Statement> EqAngle::replace(Point p, Point q) const
{
    // Replace points in each slope
    auto replace_in_slope = [p, q](const Slope &s)
    {
        Point left = (s.left() == p) ? q : s.left();
        Point right = (s.right() == p) ? q : s.right();
        return Slope(left, right);
    };

    Slope new_s1 = replace_in_slope(_s1);
    Slope new_s2 = replace_in_slope(_s2);
    Slope new_s3 = replace_in_slope(_s3);
    Slope new_s4 = replace_in_slope(_s4);

    return make_unique<EqAngle>(new_s1, new_s2, new_s3, new_s4);
}

string EqAngle::name() const
{
    return "eqangle";
}

vector<Point> EqAngle::points() const
{
    return {_s1.left(), _s1.right(), _s2.left(), _s2.right(),
            _s3.left(), _s3.right(), _s4.left(), _s4.right()};
}

unique_ptr<Statement> EqAngle::clone() const
{
    return make_unique<EqAngle>(_s1, _s2, _s3, _s4);
}

bool EqAngle::check_nondegen() const
{
    // Each slope must be non-degenerate
    if (!_s1.check_nondegen() || !_s2.check_nondegen() ||
        !_s3.check_nondegen() || !_s4.check_nondegen())
        return false;

    // Require that not all four slopes are parallel:
    // at least two of the four lines must be non-parallel.
    // If s1 is parallel to s2, s3, and s4, then all four are parallel.
    auto are_parallel = [](const Slope &a, const Slope &b) -> bool
    {
        double dx1 = a.right().x() - a.left().x();
        double dy1 = a.right().y() - a.left().y();
        double dx2 = b.right().x() - b.left().x();
        double dy2 = b.right().y() - b.left().y();
        return Numerical::close_enough(dx1 * dy2, dy1 * dx2);
    };

    if (are_parallel(_s1, _s2) && are_parallel(_s1, _s3) && are_parallel(_s1, _s4))
        return false;

    return true;
}

bool EqAngle::check_equations() const
{
    // For angle between two slopes, we compute the cross product and dot product
    // angle(s1, s2) has: cos = dot(s1, s2) / (|s1||s2|), sin = cross(s1, s2) / (|s1||s2|)

    // Left angle: angle between _s1 and _s2
    double dx1 = _s1.right().x() - _s1.left().x();
    double dy1 = _s1.right().y() - _s1.left().y();
    double dx2 = _s2.right().x() - _s2.left().x();
    double dy2 = _s2.right().y() - _s2.left().y();

    double left_dot = dx1 * dx2 + dy1 * dy2;
    double left_cross = dx1 * dy2 - dy1 * dx2;

    // Right angle: angle between _s3 and _s4
    double dx3 = _s3.right().x() - _s3.left().x();
    double dy3 = _s3.right().y() - _s3.left().y();
    double dx4 = _s4.right().x() - _s4.left().x();
    double dy4 = _s4.right().y() - _s4.left().y();

    double right_dot = dx3 * dx4 + dy3 * dy4;
    double right_cross = dx3 * dy4 - dy3 * dx4;

    // To compare angles precisely without using atan2:
    // Two angles α and β are equal iff tan(α) = tan(β) (same sign)
    // tan(α) = sin(α)/cos(α) = cross/dot
    // So: cross1/dot1 = cross2/dot2  ⟺  cross1 * dot2 = cross2 * dot1

    // Cross-multiply to avoid division (more precise)
    double lhs = left_cross * right_dot;
    double rhs = right_cross * left_dot;

    return Numerical::close_enough(lhs, rhs);
}

vector<statement_arg> EqAngle::args() const
{
    return {_s1, _s2, _s3, _s4};
}

unique_ptr<Statement> EqAngle::normalize() const
{
    // The slope equation is s1 + s4 = s2 + s3, which shares the same
    // symmetry group as eqratio's a*d = b*c with (a,b,c,d) = (s1,s2,s3,s4).
    Slope a = _s1.normalize();
    Slope b = _s2.normalize();
    Slope c = _s3.normalize();
    Slope d = _s4.normalize();

    vector<array<Slope, 4>> candidates = {
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

    return make_unique<EqAngle>(best[0], best[1], best[2], best[3]);
}

ostream &EqAngle::print(ostream &os) const
{
    return os << _s1 << "-" << _s2 << " = " << _s3 << "-" << _s4;
}

vector<EqAngle> EqAngle::permutations() const
{
    vector<EqAngle> res;
    res.reserve(4);
    res.emplace_back(_s1, _s2, _s3, _s4); // original
    res.emplace_back(_s3, _s4, _s1, _s2); // swap left and right
    res.emplace_back(_s2, _s1, _s4, _s3); // negate both angles
    res.emplace_back(_s4, _s3, _s2, _s1); // swap and negate
    return res;
}

vector<unique_ptr<Equation>> EqAngle::as_equation_slope(bool exp) const
{
    vector<unique_ptr<Equation>> result;
    result.push_back(make_unique<Equation>(Equation({Term(_s1), -Term(_s2), -Term(_s3), Term(_s4)})));
    return result;
}

bool EqAngle::operator==(const EqAngle &other) const
{
    return _s1 == other._s1 && _s2 == other._s2 &&
           _s3 == other._s3 && _s4 == other._s4;
}

bool EqAngle::operator!=(const EqAngle &other) const
{
    return !(*this == other);
}

bool EqAngle::operator<(const EqAngle &other) const
{
    if (_s1 == other._s1)
    {
        if (_s2 == other._s2)
        {
            if (_s3 == other._s3)
            {
                return _s4 < other._s4;
            }
            return _s3 < other._s3;
        }
        return _s2 < other._s2;
    }
    return _s1 < other._s1;
}

bool EqAngle::operator<=(const EqAngle &other) const
{
    return *this < other || *this == other;
}

bool EqAngle::operator>(const EqAngle &other) const
{
    return !(*this <= other);
}

bool EqAngle::operator>=(const EqAngle &other) const
{
    return !(*this < other);
}

string EqAngle::to_string() const
{
    string res = name();
    res += " " + _s1.left().name();
    res += " " + _s1.right().name();
    res += " " + _s2.left().name();
    res += " " + _s2.right().name();
    res += " " + _s3.left().name();
    res += " " + _s3.right().name();
    res += " " + _s4.left().name();
    res += " " + _s4.right().name();
    return res;
}

vector<string> EqAngle::to_tokens() const
{
    vector<string> tokens = {"eqangle"};
    tokens.push_back(_s1.left().name());
    tokens.push_back(_s1.right().name());
    tokens.push_back(_s2.left().name());
    tokens.push_back(_s2.right().name());
    tokens.push_back(_s3.left().name());
    tokens.push_back(_s3.right().name());
    tokens.push_back(_s4.left().name());
    tokens.push_back(_s4.right().name());
    return tokens;
}