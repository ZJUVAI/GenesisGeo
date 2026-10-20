#ifndef CONST_LINE_HPP
#define CONST_LINE_HPP

#include "predicate/statement.hpp"
#include "predicate/eqratio.hpp"
#include "typedef.hpp"
#include <set>

class ConstLine : public Statement
{
private:
    Point _p;
    Point _q1;
    Point _q2;

public:
    ConstLine(Point p, Point q1, Point q2);

    ConstLine(const std::vector<statement_arg> &args);

    std::string name() const override;

    std::vector<Point> points() const override;

    std::unique_ptr<Statement> replace(Point p, Point q) const override;

    std::unique_ptr<Statement> normalize() const override;

    std::vector<statement_arg> args() const override;

    bool check_nondegen() const override;

    bool check_equations() const override;

    double angle() const { return Slope(_p, _q1).angle(); }

    const Point &p() const { return _p; }

    const Point &q1() const { return _q1; }

    const Point &q2() const { return _q2; }

    std::unique_ptr<Statement> clone() const override
    {
        return std::make_unique<ConstLine>(*this);
    }

    std::ostream &print(std::ostream &out) const override;

    bool numerical_only() const { return false; }

    bool trivial() const { return false; }

    std::vector<std::unique_ptr<Equation>> as_equation_slope(bool exp) const override;
};

#endif // CONST_LINE_HPP