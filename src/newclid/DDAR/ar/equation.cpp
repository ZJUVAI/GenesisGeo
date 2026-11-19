#include "ar/equation.hpp"
#include "numerical.hpp"
#include <sstream>

using namespace std;

Equation::Equation(initializer_list<Term> terms)
{
    vector<pair<Monomial, Rational>> pairs;
    pairs.reserve(terms.size());
    for (const auto &t : terms)
    {
        pairs.emplace_back(t.monomial, t.coeff);
    }
    _body = Polynomial(std::move(pairs));
}

Equation::Equation(const vector<Term> &terms)
{
    vector<pair<Monomial, Rational>> pairs;
    pairs.reserve(terms.size());
    for (const auto &t : terms)
    {
        pairs.emplace_back(t.monomial, t.coeff);
    }
    _body = Polynomial(std::move(pairs));
}

void Equation::prune_zero_deps()
{
    for (auto it = _deps.begin(); it != _deps.end();)
    {
        if (it->second.empty())
        {
            it = _deps.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

Equation &Equation::operator+=(const Equation &other)
{
    _body += other._body;
    for (const auto &[idx, coeff] : other._deps)
    {
        _deps[idx] += coeff;
    }
    prune_zero_deps();
    return *this;
}

Equation &Equation::operator-=(const Equation &other)
{
    _body -= other._body;
    for (const auto &[idx, coeff] : other._deps)
    {
        _deps[idx] -= coeff;
    }
    prune_zero_deps();
    return *this;
}

void Equation::subtract_scaled(const Equation &other, Rational scale)
{
    _body.subtract_scaled(other._body, scale);
    if (scale != Rational(0))
    {
        for (const auto &[index, coefficient] : other._deps)
        {
            _deps[index].subtract_scaled(coefficient, scale);
        }
    }
    prune_zero_deps();
}

Equation &Equation::operator*=(const Rational &r)
{
    _body *= r;
    if (r == Rational(0))
    {
        _deps.clear();
        return *this;
    }
    for (auto &[idx, coeff] : _deps)
    {
        coeff *= r;
    }
    return *this;
}

Equation &Equation::operator*=(const Monomial &m)
{
    _body *= m;
    for (auto &[idx, coeff] : _deps)
    {
        coeff *= m;
    }
    return *this;
}

Equation Equation::operator+(const Equation &other) const
{
    Equation res = *this;
    res += other;
    return res;
}

Equation Equation::operator-(const Equation &other) const
{
    Equation res = *this;
    res -= other;
    return res;
}

Equation Equation::operator*(const Rational &r) const
{
    Equation res = *this;
    res *= r;
    return res;
}

Equation Equation::operator*(const Monomial &m) const
{
    Equation res = *this;
    res *= m;
    return res;
}

Equation Equation::operator-() const
{
    Equation res = *this;
    res *= Rational(-1);
    return res;
}

void Equation::make_monic()
{
    if (_body.empty())
    {
        return;
    }
    Rational lead = _body.leading_coeff();
    if (lead == Rational(1))
    {
        return;
    }
    *this *= (Rational(1) / lead);
}

// content_reduce 把 body 除以其所有项的公共单项式因子 f。
// 为保持证明不变量
//      body == sum_j _deps[j] * Eq_j
// 必须把右边也除以 f，即每个证明系数都乘以 f^{-1}。
// 因为 f 是非零单项式，这不会改变任何系数是零还是非零——
// 所以依赖*集合*不受影响——但能让 body 与证明保持一致的缩放，
// 从而后续 +=/-= 的相消判定仍然正确。
Monomial Equation::content_multiplier() const
{
    if (_body.empty())
    {
        return {};
    }
    auto it = _body.begin();
    Monomial common = it->first;
    ++it;
    for (; it != _body.end(); ++it)
    {
        common = common.gcd(it->first);
        if (common.is_constant())
        {
            return {};
        }
    }
    // 滤掉数值近零的变量（不要用数值为零的量去除）。
    Monomial safe;
    for (const auto &[var, exp] : common.vars())
    {
        if (!Numerical::close_enough(var.to_double(), 0.0))
        {
            safe *= Monomial(var, exp);
        }
    }
    return safe.inverse();
}

void Equation::content_reduce()
{
    const Monomial multiplier = content_multiplier();
    if (!multiplier.is_constant())
    {
        *this *= multiplier;
    }
}

void Equation::set_index(size_t archive_index)
{
    _deps[archive_index] += Polynomial(Rational(1));
    prune_zero_deps();
}

vector<size_t> Equation::dependency_indices() const
{
    vector<size_t> res;
    res.reserve(_deps.size());
    for (const auto &[idx, coeff] : _deps)
    {
        res.push_back(idx);
    }
    return res;
}

string Equation::to_string() const
{
    ostringstream oss;
    oss << _body.to_string() << " (deps:";
    for (const auto &[idx, coeff] : _deps)
    {
        oss << " " << idx << "*[" << coeff.to_string() << "]";
    }
    oss << ")";
    return oss.str();
}

ostream &operator<<(ostream &os, const Equation &eq)
{
    os << eq.to_string();
    return os;
}
