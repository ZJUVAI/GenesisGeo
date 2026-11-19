#include "ar/polynomial.hpp"
#include "numerical.hpp"
#include <algorithm>
#include <sstream>
#include <stdexcept>

using namespace std;

namespace
{
    template <typename Key>
    Rational &coefficient(Polynomial::Terms &terms, Key &&monomial)
    {
        auto position = lower_bound(terms.begin(), terms.end(), monomial,
            [](const auto &term, const Monomial &key) { return key < term.first; });
        if (position == terms.end() || position->first < monomial)
        {
            position = terms.emplace(position, std::forward<Key>(monomial), Rational());
        }
        return position->second;
    }
}

Polynomial::Polynomial(initializer_list<pair<Monomial, Rational>> terms)
{
    for (const auto &[m, c] : terms)
    {
        coefficient(_terms, m) += c;
    }
    prune_zero_coeffs();
}

Polynomial::Polynomial(vector<pair<Monomial, Rational>> terms)
{
    for (const auto &[m, c] : terms)
    {
        coefficient(_terms, m) += c;
    }
    prune_zero_coeffs();
}

Polynomial::Polynomial(const Rational &c)
{
    if (c != Rational(0))
    {
        _terms.emplace_back(Monomial(), c);
    }
}

void Polynomial::prune_zero_coeffs()
{
    _terms.erase(remove_if(_terms.begin(), _terms.end(),
        [](const auto &term) { return term.second == Rational(0); }), _terms.end());
}

Polynomial &Polynomial::operator+=(const Polynomial &other)
{
    for (const auto &[m, c] : other._terms)
    {
        coefficient(_terms, m) += c;
    }
    prune_zero_coeffs();
    return *this;
}

Polynomial &Polynomial::operator-=(const Polynomial &other)
{
    for (const auto &[m, c] : other._terms)
    {
        coefficient(_terms, m) -= c;
    }
    prune_zero_coeffs();
    return *this;
}

void Polynomial::subtract_scaled(const Polynomial &other, Rational scale)
{
    if (scale != Rational(0))
    {
        for (const auto &[monomial, value] : other._terms)
        {
            coefficient(_terms, monomial) -= value * scale;
        }
    }
    prune_zero_coeffs();
}

Polynomial &Polynomial::operator*=(const Rational &r)
{
    if (r == Rational(0))
    {
        _terms.clear();
        return *this;
    }
    for (auto &[m, c] : _terms)
    {
        c *= r;
    }
    return *this;
}

Polynomial &Polynomial::operator*=(const Monomial &m)
{
    if (m.is_constant())
    {
        return *this;
    }
    Terms shifted;
    shifted.reserve(_terms.size());
    for (const auto &[mono, c] : _terms)
    {
        coefficient(shifted, mono * m) += c;
    }
    _terms = std::move(shifted);
    prune_zero_coeffs();
    return *this;
}

Polynomial Polynomial::operator+(const Polynomial &other) const
{
    Polynomial res = *this;
    res += other;
    return res;
}

Polynomial Polynomial::operator-(const Polynomial &other) const
{
    Polynomial res = *this;
    res -= other;
    return res;
}

Polynomial Polynomial::operator*(const Rational &r) const
{
    Polynomial res = *this;
    res *= r;
    return res;
}

Polynomial Polynomial::operator*(const Monomial &m) const
{
    Polynomial res = *this;
    res *= m;
    return res;
}

Polynomial Polynomial::operator-() const
{
    Polynomial res = *this;
    for (auto &[m, c] : res._terms)
    {
        c = -c;
    }
    return res;
}

bool Polynomial::is_linear() const
{
    for (const auto &[m, c] : _terms)
    {
        if (m.degree() > 1)
        {
            return false;
        }
    }
    return true;
}

const Monomial &Polynomial::leading_monomial() const
{
    if (_terms.empty())
    {
        throw runtime_error("leading_monomial() on empty polynomial");
    }
    return _terms.begin()->first;
}

Rational Polynomial::leading_coeff() const
{
    if (_terms.empty())
    {
        throw runtime_error("leading_coeff() on empty polynomial");
    }
    return _terms.begin()->second;
}

void Polynomial::make_monic()
{
    if (_terms.empty())
    {
        return;
    }
    Rational lead = _terms.begin()->second;
    if (lead == Rational(1))
    {
        return;
    }
    *this *= (Rational(1) / lead);
}

void Polynomial::content_reduce()
{
    if (_terms.size() < 1)
    {
        return;
    }
    auto it = _terms.begin();
    Monomial common = it->first;
    ++it;
    for (; it != _terms.end(); ++it)
    {
        common = common.gcd(it->first);
        if (common.is_constant())
        {
            return;
        }
    }
    // 从公共因子中滤掉数值近零的变量。
    Monomial safe;
    for (const auto &[var, exp] : common.vars())
    {
        if (!Numerical::close_enough(var.to_double(), 0.0))
        {
            safe *= Monomial(var, exp);
        }
    }
    if (safe.is_constant())
    {
        return;
    }
    Terms shifted;
    shifted.reserve(_terms.size());
    for (const auto &[mono, c] : _terms)
    {
        coefficient(shifted, mono / safe) += c;
    }
    _terms = std::move(shifted);
    prune_zero_coeffs();
}

double Polynomial::to_double() const
{
    double res = 0.0;
    for (const auto &[m, c] : _terms)
    {
        res += c.to_double() * m.to_double();
    }
    return res;
}

string Polynomial::to_string() const
{
    if (_terms.empty())
    {
        return "0";
    }
    ostringstream oss;
    bool first = true;
    for (const auto &[m, c] : _terms)
    {
        if (!first)
        {
            oss << " + ";
        }
        oss << c.to_string();
        if (!m.is_constant())
        {
            oss << "*" << m.to_string();
        }
        first = false;
    }
    oss << " = 0";
    return oss.str();
}

ostream &operator<<(ostream &os, const Polynomial &p)
{
    os << p.to_string();
    return os;
}

size_t Polynomial::hash() const
{
    size_t seed = 0;
    for (const auto &[m, c] : _terms)
    {
        seed ^= std::hash<Monomial>{}(m) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<long long>{}(c.numerator()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<long long>{}(c.denominator()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}
