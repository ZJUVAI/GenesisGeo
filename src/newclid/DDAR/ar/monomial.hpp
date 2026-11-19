#ifndef MONOMIAL_HPP
#define MONOMIAL_HPP

#include <utility>
#include <vector>
#include <string>
#include <cstddef>
#include "ar/term_arg.hpp"

// 单项式：变量的纯乘积，变量带整数指数。
// 不含系数——系数存放在 Polynomial 的项表中。
// hash / operator== / operator< 都只依赖 _vars。
class Monomial final
{
private:
    using Variables = std::vector<std::pair<TermArg, int>>;
    Variables _vars; // Sorted by variable, with no zero exponents.

    Variables::const_iterator find_variable(const TermArg &var) const;
    void add_exponent(const TermArg &var, int exp, bool subtract = false);

public:
    Monomial() = default; // 常数单项式（次数为 0）
    explicit Monomial(const TermArg &var, int exp = 1);
    explicit Monomial(const std::vector<TermArg> &vars); // 每个变量指数记为 +1

    // 代数运算
    Monomial operator*(const Monomial &other) const;
    Monomial operator/(const Monomial &other) const; // 前提：other.divides(*this)
    Monomial &operator*=(const Monomial &other);
    Monomial &operator/=(const Monomial &other);

    bool divides(const Monomial &other) const; // *this 是否整除 other？
    Monomial gcd(const Monomial &other) const;
    Monomial lcm(const Monomial &other) const;
    Monomial inverse() const; // 所有指数取负（洛朗单项式）

    // 查询
    int degree() const;
    bool is_constant() const { return _vars.empty(); }
    bool contains(const TermArg &var) const { return find_variable(var) != _vars.end(); }
    const Variables &vars() const { return _vars; }

    double to_double() const;
    std::string to_string() const;

    // 排序：按最大变量的 degree-lex 序（与旧 Term 排序一致）。
    // hash / == 只依赖 _vars。
    bool operator==(const Monomial &other) const { return _vars == other._vars; }
    bool operator!=(const Monomial &other) const { return !(*this == other); }
    bool operator<(const Monomial &other) const;
    bool operator>(const Monomial &other) const { return other < *this; }
    bool operator<=(const Monomial &other) const { return !(other < *this); }
    bool operator>=(const Monomial &other) const { return !(*this < other); }

    size_t hash() const;
};

std::ostream &operator<<(std::ostream &os, const Monomial &m);

namespace std
{
    template <>
    struct hash<Monomial>
    {
        size_t operator()(const Monomial &m) const noexcept { return m.hash(); }
    };
}

#endif // AR_NEW_MONOMIAL_HPP
