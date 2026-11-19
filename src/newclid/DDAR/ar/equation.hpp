#ifndef EQUATION_HPP
#define EQUATION_HPP

#include <map>
#include <vector>
#include <cstddef>
#include "ar/monomial.hpp"
#include "ar/polynomial.hpp"
#include "type/rational.hpp"

// Term 是给谓词层用的构造辅助类：一个带有理系数的单项式。
// 它的存在只是为了让谓词文件能继续写
//   Equation({Term(_left), -Term(_right)})  而无需改动。
class Term final
{
public:
    Monomial monomial;
    Rational coeff;

    Term() : coeff(1) {}
    explicit Term(const TermArg &var) : monomial(var), coeff(1) {}
    Term(const TermArg &var, const Rational &c) : monomial(var), coeff(c) {}
    explicit Term(const std::vector<TermArg> &vars) : monomial(vars), coeff(1) {}
    Term(const std::vector<TermArg> &vars, const Rational &c) : monomial(vars), coeff(c) {}
    explicit Term(const Rational &c) : coeff(c) {} // 常数项
    Term(const Monomial &m, const Rational &c) : monomial(m), coeff(c) {}

    Term operator-() const { return Term(monomial, -coeff); }

    // 数值评估:用于谓词层根据当前点位排序候选项等启发式逻辑。
    double to_double() const { return coeff.to_double() * monomial.to_double(); }
};

// Equation 表示 “body == 0” 加上一份来源证明（provenance）。
//
//   body ：代数内容（一个 Polynomial）。
//   _deps：归档索引 -> 系数多项式，表示
//          body == sum_j _deps[j] * Eq_j（即原始方程的线性组合）。
//
// 每次算术运算都会同步维护证明信息；系数被消成零多项式的项会被剪除——
// 这样在约简过程中被消去的依赖会正确地从依赖集合中消失
//（旧的 set<size_t> 无法表达这一点）。
class Equation final
{
private:
    Polynomial _body;
    std::map<size_t, Polynomial> _deps; // 归档索引 -> 系数多项式

    void prune_zero_deps();

public:
    Equation() = default;

    // 谓词层构造：Term 之和。此时尚无来源信息。
    Equation(std::initializer_list<Term> terms);
    explicit Equation(const std::vector<Term> &terms);
    explicit Equation(Polynomial body) : _body(std::move(body)) {}

    // 算术运算：body 与证明信息同步更新。
    Equation &operator+=(const Equation &other);
    Equation &operator-=(const Equation &other);
    void subtract_scaled(const Equation &other, Rational scale);
    Equation &operator*=(const Rational &r);
    Equation &operator*=(const Monomial &m);

    Equation operator+(const Equation &other) const;
    Equation operator-(const Equation &other) const;
    Equation operator*(const Rational &r) const;
    Equation operator*(const Monomial &m) const;
    Equation operator-() const;

    // body 查询（委托给 Polynomial）。
    bool empty() const { return _body.empty(); }
    bool is_linear() const { return _body.is_linear(); }
    const Polynomial &body() const { return _body; }
    const Monomial &leading_monomial() const { return _body.leading_monomial(); }
    Rational leading_coeff() const { return _body.leading_coeff(); }

    // Body and certificate are scaled together.
    void make_monic();
    Monomial content_multiplier() const;
    void content_reduce();

    // 来源信息管理。set_index 把 self-coefficient 1 累加到归档索引 idx 上;
    // 它*不会*清空已有条目 —— 调用前 _deps 里通常已经累积了 normal_form 期间
    // 用过的规则的负系数,这两部分共同构成 body 关于原始方程的线性组合证书。
    void set_index(size_t archive_index);
    const std::map<size_t, Polynomial> &dependencies() const { return _deps; }
    std::vector<size_t> dependency_indices() const;

    std::string to_string() const;

    bool operator==(const Equation &other) const { return _body == other._body; }
    bool operator!=(const Equation &other) const { return !(*this == other); }
};

std::ostream &operator<<(std::ostream &os, const Equation &eq);

namespace std
{
    // hash 仅基于 body(与 operator== 一致)。_deps(来源证书)不参与,
    // 因为同一条代数式的 Equation 在不同时刻可能依赖不同的归档下标,
    // 但作为"代数实体"应当是同一个 key。
    template <>
    struct hash<Equation>
    {
        size_t operator()(const Equation &eq) const noexcept
        {
            return std::hash<Polynomial>{}(eq.body());
        }
    };
}

#endif // EQUATION_HPP
