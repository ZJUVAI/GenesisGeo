#ifndef EQPOINT_HPP
#define EQPOINT_HPP
#include "predicate/statement.hpp"

class EqPoint : public Statement
{
private:
    Point _a;
    Point _b;

public:
    EqPoint(Point a, Point b);

    EqPoint(const std::vector<statement_arg> &args);

    std::string name() const override;

    std::vector<Point> points() const override;

    std::unique_ptr<Statement> replace(Point p, Point q) const override;

    std::unique_ptr<Statement> normalize() const override;

    bool check_nondegen() const override;

    bool check_equations() const override;

    const Point &a() const { return _a; }

    const Point &b() const { return _b; }

    std::unique_ptr<Statement> clone() const override;

    std::vector<statement_arg> args() const override;

    std::ostream &print(std::ostream &out) const override;

    bool numerical_only() const { return false; }

    bool trivial() const { return false; }

    std::vector<std::unique_ptr<Equation>> as_equation_dist(bool exp) const override;
};

std::ostream &operator<<(std::ostream &out, const Statement &stmt);

#endif // EQPOINT_HPP