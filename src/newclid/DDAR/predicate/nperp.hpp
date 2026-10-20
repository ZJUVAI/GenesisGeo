#ifndef NPERP_HPP
#define NPERP_HPP

#include "predicate/statement.hpp"

// nperp A B C D : AB is NOT perpendicular to CD (numerical check only)
class NPerp : public Statement
{
private:
    Slope _left;
    Slope _right;

public:
    NPerp(Slope left, Slope right);

    std::string name() const override;

    std::vector<Point> points() const override;

    std::unique_ptr<Statement> replace(Point p, Point q) const override;

    std::unique_ptr<Statement> normalize() const override;

    bool check_nondegen() const override;

    bool check_equations() const override;

    std::vector<statement_arg> args() const override;

    std::unique_ptr<Statement> clone() const override;

    std::ostream &print(std::ostream &os) const override;

    bool numerical_only() const { return true; }

    bool trivial() const { return false; }
};

#endif // NPERP_HPP
