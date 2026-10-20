#ifndef ACONST_HPP
#define ACONST_HPP

#include <memory>
#include <string>
#include <vector>
#include "predicate/statement.hpp"
#include "type/slope.hpp"
#include "type/angle.hpp"
#include "type/rational.hpp"

class AConst : public Statement
{
public:
    // Constructor from two slopes directly: angle(s1, s2) = rhs * π
    AConst(const Slope &s1, const Slope &s2, const Rational &rhs);

    // Constructor from an angle (for backward compatibility)
    AConst(const Angle &ang, const Rational &rhs);

    std::string name() const override;

    std::vector<Point> points() const override;

    std::unique_ptr<Statement> replace(Point p, Point q) const override;

    std::unique_ptr<Statement> normalize() const override;

    bool check_nondegen() const override;

    bool check_equations() const override;

    std::vector<statement_arg> args() const override;

    const Slope &s1() const { return _s1; }

    const Slope &s2() const { return _s2; }

    const Rational &rhs() const { return _rhs; }

    std::unique_ptr<Statement> clone() const override
    {
        return std::make_unique<AConst>(*this);
    }

    std::vector<std::unique_ptr<Equation>> as_equation_slope(bool exp) const override;

    std::string to_string() const override;

    std::ostream &print(std::ostream &out) const override;

    bool numerical_only() const { return false; }

    bool trivial() const { return false; }

    std::vector<std::string> to_tokens() const;

    ~AConst() override = default;

private:
    Slope _s1;  // First slope of the angle
    Slope _s2;  // Second slope of the angle
    Rational _rhs;
};

#endif // ACONST_HPP
