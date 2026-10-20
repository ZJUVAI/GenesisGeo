#ifndef SECANT_HPP
#define SECANT_HPP

#include "predicate/statement.hpp"
#include "predicate/cong.hpp"
#include "predicate/coll.hpp"
#include "type/point.hpp"
#include "typedef.hpp"

class Secant : public Statement
{
private:
    Point _o;
    Point _a;
    Point _b;
    Point _p;

public:
    Secant(const Point &o, const Point &a, const Point &b, const Point &p);

    Secant(const std::vector<statement_arg> &args);

    std::string name() const override;

    std::vector<Point> points() const override;

    std::unique_ptr<Statement> clone() const override;

    std::unique_ptr<Statement> replace(Point p, Point q) const override;

    std::unique_ptr<Statement> normalize() const override;

    bool check_nondegen() const override;

    bool check_equations() const override;

    std::vector<statement_arg> args() const override;

    std::ostream &print(std::ostream &os) const override;

    Point o() const { return _o; }

    Point a() const { return _a; }

    Point b() const { return _b; }

    Point p() const { return _p; }

    Cong cong_ab() const;

    Coll coll_pab() const;

    std::vector<std::unique_ptr<Equation>> as_equation_dist(bool exp) const override;

    bool numerical_only() const override { return false; }

    bool trivial() const override { return false; }
};

#endif // SECANT_HPP