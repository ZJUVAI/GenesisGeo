#ifndef STATEMENT_HPP
#define STATEMENT_HPP
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include "type/point.hpp"
#include "ar/equation.hpp"
#include "typedef.hpp"

class Statement
{
public:
    virtual std::string name() const = 0;

    virtual std::vector<Point> points() const = 0;

    virtual bool contain(const Point &p) const;

    // 简化当前结论
    virtual std::unique_ptr<Statement> normalize() const = 0;

    virtual std::unique_ptr<Statement> replace(Point p, Point q) const = 0;

    // 检查当前结论是否退化
    virtual bool check_nondegen() const = 0;

    // 检查等式是否成立
    virtual bool check_equations() const = 0;

    virtual bool check_numerically() const final
    {
        return this->check_nondegen() && this->check_equations();
    }

    virtual std::unique_ptr<Statement> clone() const = 0;

    virtual ~Statement() = default;

    virtual std::vector<statement_arg> args() const = 0;

    virtual std::ostream &print(std::ostream &out) const = 0;

    virtual bool numerical_only() const = 0;

    virtual bool trivial() const = 0;

    virtual std::string to_string() const;

    virtual std::vector<std::string> to_tokens() const;

    virtual std::vector<std::unique_ptr<Equation>> as_equation_slope(bool exp) const { return {}; }
    virtual std::vector<std::unique_ptr<Equation>> as_equation_dist(bool exp) const { return {}; }
    virtual std::vector<std::unique_ptr<Equation>> as_equation_distlog(bool exp) const { return {}; }

    virtual Point max_point() const;

    virtual bool operator==(const Statement &other) const
    {
        return this->normalize()->to_string() == other.normalize()->to_string();
    }
};

std::ostream &operator<<(std::ostream &out, const Statement &stmt);

#endif // STATEMENT_HPP