#ifndef EQRATIO_HPP
#define EQRATIO_HPP

#include "predicate/statement.hpp"
#include "type/dist.hpp"
#include "typedef.hpp"

class EqRatio : public Statement
{
private:
    Dist _left_up;
    Dist _left_down;
    Dist _right_up;
    Dist _right_down;

public:
    EqRatio(Dist left_up, Dist left_down, Dist right_up, Dist right_down);

    std::string name() const override;

    std::vector<Point> points() const override;

    std::unique_ptr<Statement> clone() const override;

    bool check_nondegen() const override;

    bool check_equations() const override;

    std::vector<statement_arg> args() const override;

    std::unique_ptr<Statement> replace(Point p, Point q) const override;

    std::unique_ptr<Statement> normalize() const override;

    std::ostream &print(std::ostream &os) const override;

    const Dist &left_up() const { return _left_up; }

    const Dist &left_down() const { return _left_down; }

    const Dist &right_up() const { return _right_up; }

    const Dist &right_down() const { return _right_down; }

    bool numerical_only() const { return false; }

    bool trivial() const { return (_left_up == _left_down && _right_up == _right_down) || (_left_up == _right_up && _left_down == _right_down); }

    std::vector<std::unique_ptr<Equation>> as_equation_dist(bool exp) const override;
    std::vector<std::unique_ptr<Equation>> as_equation_distlog(bool exp) const override;
};

#endif // EQRATIO_HPP