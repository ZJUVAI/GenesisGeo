#ifndef OBTUSE_HPP
#define OBTUSE_HPP

#include "predicate/statement.hpp"

// obtuse A B C : angle ABC > 90 degrees (numerical check only)
class Obtuse : public Statement
{
private:
    Point _a;
    Point _b;  // vertex
    Point _c;

public:
    Obtuse(Point a, Point b, Point c);

    std::string name() const override;

    std::vector<Point> points() const override;

    std::unique_ptr<Statement> replace(Point p, Point q) const override;

    std::unique_ptr<Statement> normalize() const override;

    bool check_nondegen() const override;

    bool check_equations() const override;

    std::vector<statement_arg> args() const override;

    std::unique_ptr<Statement> clone() const override;

    std::ostream &print(std::ostream &os) const override;

    const Point &a() const { return _a; }
    const Point &b() const { return _b; }
    const Point &c() const { return _c; }

    bool numerical_only() const { return true; }

    bool trivial() const { return false; }
};

#endif // OBTUSE_HPP
