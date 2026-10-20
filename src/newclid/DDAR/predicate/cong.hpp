#ifndef CONG_HPP
#define CONG_HPP

#include "ar/equation.hpp"
#include "type/dist.hpp"
#include "predicate/statement.hpp"
#include <optional>

class Cong : public Statement
{
private:
    Dist _left;
    Dist _right;

public:
    Cong(Dist left, Dist right) : _left(left), _right(right) {}

    std::string name() const override;

    std::vector<Point> points() const override;

    std::unique_ptr<Statement> clone() const override
    {
        return std::make_unique<Cong>(*this);
    }

    std::unique_ptr<Statement> replace(Point p, Point q) const override;

    std::unique_ptr<Statement> normalize() const override;

    bool check_nondegen() const override;

    bool check_equations() const override;

    std::vector<statement_arg> args() const override;

    std::vector<std::unique_ptr<Equation>> as_equation_dist(bool exp) const override;
    std::vector<std::unique_ptr<Equation>> as_equation_distlog(bool exp) const override;

    const Dist &left() const { return _left; }

    const Dist &right() const { return _right; }

    std::ostream &print(std::ostream &os) const override;

    bool numerical_only() const { return false; }

    bool trivial() const { return _left == _right; }
};

#endif // CONG_HPP