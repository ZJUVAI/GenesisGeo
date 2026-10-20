#ifndef EQANGLE_HPP
#define EQANGLE_HPP

#include "predicate/statement.hpp"
#include "type/angle.hpp"
#include "type/slope.hpp"
#include "ar/equation.hpp"

class EqAngle : public Statement
{
private:
    Slope _s1;  // First slope of left angle
    Slope _s2;  // Second slope of left angle
    Slope _s3;  // First slope of right angle
    Slope _s4;  // Second slope of right angle

public:
    // Constructor from 4 slopes directly
    EqAngle(Slope s1, Slope s2, Slope s3, Slope s4);

    // Constructor from two angles (for backward compatibility)
    EqAngle(Angle left, Angle right);

    // Constructor from 8 points
    EqAngle(Point p1, Point p2, Point p3, Point p4, Point p5, Point p6, Point p7, Point p8);

    std::string name() const override;

    std::vector<Point> points() const override;

    std::unique_ptr<Statement> clone() const override;

    bool check_nondegen() const override;

    bool check_equations() const override;

    std::vector<statement_arg> args() const override;

    std::unique_ptr<Statement> replace(Point p, Point q) const override;

    std::unique_ptr<Statement> normalize() const override;

    std::ostream &print(std::ostream &os) const override;

    std::vector<EqAngle> permutations() const;

    // Accessors for slopes
    const Slope &s1() const { return _s1; }
    const Slope &s2() const { return _s2; }
    const Slope &s3() const { return _s3; }
    const Slope &s4() const { return _s4; }

    std::vector<std::unique_ptr<Equation>> as_equation_slope(bool exp) const override;

    bool numerical_only() const { return false; }

    bool trivial() const {
        return (_s1 == _s3 && _s2 == _s4) || (_s1 == _s2 && _s3 == _s4);
    }

    bool operator<(const EqAngle &other) const;

    bool operator==(const EqAngle &other) const;

    bool operator!=(const EqAngle &other) const;

    bool operator<=(const EqAngle &other) const;

    bool operator>(const EqAngle &other) const;

    bool operator>=(const EqAngle &other) const;

    std::string to_string() const override;

    std::vector<std::string> to_tokens() const;
};

#endif // EQANGLE_HPP