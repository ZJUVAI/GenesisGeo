#ifndef POLYNOMIAL_HPP
#define POLYNOMIAL_HPP

#include <utility>
#include <vector>
#include <functional>
#include <cstddef>
#include "ar/monomial.hpp"
#include "type/rational.hpp"

class Polynomial final
{
public:
    using Terms = std::vector<std::pair<Monomial, Rational>>;

private:
    Terms _terms; // Sorted by descending monomial order.

    void prune_zero_coeffs();

public:
    Polynomial() = default;

    // 由 (单项式, 系数) 对列表构造；同类项会被合并。
    Polynomial(std::initializer_list<std::pair<Monomial, Rational>> terms);
    explicit Polynomial(std::vector<std::pair<Monomial, Rational>> terms);

    // 便捷构造：常数多项式。
    explicit Polynomial(const Rational &c);

    // 算术运算。
    Polynomial &operator+=(const Polynomial &other);
    Polynomial &operator-=(const Polynomial &other);
    Polynomial &operator*=(const Rational &r);
    Polynomial &operator*=(const Monomial &m);

    Polynomial operator+(const Polynomial &other) const;
    Polynomial operator-(const Polynomial &other) const;
    Polynomial operator*(const Rational &r) const;
    Polynomial operator*(const Monomial &m) const;
    Polynomial operator-() const;

    // 查询
    bool empty() const { return _terms.empty(); }
    size_t size() const { return _terms.size(); }
    bool is_linear() const;

    const Monomial &leading_monomial() const; // 前提：!empty()
    Rational leading_coeff() const;            // 前提：!empty()

    const Terms &terms() const { return _terms; }
    Terms::const_iterator begin() const { return _terms.begin(); }
    Terms::const_iterator end() const { return _terms.end(); }

    // 原地变换
    void make_monic();     // 整体除以首项系数（空时为空操作）
    void content_reduce(); // 除去所有项的公共单项式因子

    double to_double() const;
    std::string to_string() const;
    size_t hash() const;

    bool operator==(const Polynomial &other) const { return _terms == other._terms; }
    bool operator!=(const Polynomial &other) const { return !(*this == other); }
};

std::ostream &operator<<(std::ostream &os, const Polynomial &p);

namespace std
{
    template <>
    struct hash<Polynomial>
    {
        size_t operator()(const Polynomial &p) const noexcept { return p.hash(); }
    };
}

#endif // POLYNOMIAL_HPP
