#include "ar/monomial.hpp"
#include "numerical.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <sstream>

using namespace std;

namespace
{
    template <typename Variables>
    auto lower_bound_variable(Variables &vars, const TermArg &var)
    {
        return std::lower_bound(vars.begin(), vars.end(), var,
            [](const auto &entry, const TermArg &key) { return entry.first < key; });
    }
}

Monomial::Monomial(const TermArg &var, int exp)
{
    if (exp != 0)
    {
        _vars.emplace_back(var, exp);
    }
}

Monomial::Monomial(const vector<TermArg> &vars)
{
    for (const auto &var : vars)
    {
        add_exponent(var, 1);
    }
}

Monomial::Variables::const_iterator Monomial::find_variable(const TermArg &var) const
{
    auto it = lower_bound_variable(_vars, var);
    if (it != _vars.end() && !(var < it->first))
    {
        return it;
    }
    return _vars.end();
}

void Monomial::add_exponent(const TermArg &var, int exp, bool subtract)
{
    auto it = lower_bound_variable(_vars, var);
    if (it == _vars.end() || var < it->first)
    {
        _vars.emplace(it, var, subtract ? -exp : exp);
        return;
    }
    it->second = subtract ? it->second - exp : it->second + exp;
    if (it->second == 0)
    {
        _vars.erase(it);
    }
}

Monomial Monomial::operator*(const Monomial &other) const
{
    Monomial res = *this;
    res *= other;
    return res;
}

Monomial &Monomial::operator*=(const Monomial &other)
{
    if (this == &other)
    {
        for (auto &[var, exp] : _vars)
        {
            exp += exp;
        }
        return *this;
    }

    for (const auto &[var, exp] : other._vars)
    {
        add_exponent(var, exp);
    }
    return *this;
}

Monomial Monomial::operator/(const Monomial &other) const
{
    Monomial res = *this;
    res /= other;
    return res;
}

Monomial &Monomial::operator/=(const Monomial &other)
{
    if (this == &other)
    {
        _vars.clear();
        return *this;
    }

    for (const auto &[var, exp] : other._vars)
    {
        add_exponent(var, exp, true);
    }
    return *this;
}

bool Monomial::divides(const Monomial &other) const
{
    for (const auto &[var, exp] : _vars)
    {
        const auto it = other.find_variable(var);
        if (it == other._vars.end() || it->second < exp)
        {
            return false;
        }
    }
    return true;
}

Monomial Monomial::gcd(const Monomial &other) const
{
    Monomial res;
    for (const auto &[var, exp] : _vars)
    {
        const auto it = other.find_variable(var);
        if (it != other._vars.end())
        {
            res._vars.emplace_back(var, std::min(exp, it->second));
        }
    }
    return res;
}

Monomial Monomial::lcm(const Monomial &other) const
{
    Monomial res = *this;
    for (const auto &[var, exp] : other._vars)
    {
        auto it = lower_bound_variable(res._vars, var);
        if (it == res._vars.end() || var < it->first)
        {
            res._vars.emplace(it, var, exp);
        }
        else
        {
            it->second = std::max(it->second, exp);
        }
    }
    return res;
}

Monomial Monomial::inverse() const
{
    Monomial res;
    res._vars.reserve(_vars.size());
    for (const auto &[var, exp] : _vars)
    {
        res._vars.emplace_back(var, -exp);
    }
    return res;
}

int Monomial::degree() const
{
    int res = 0;
    for (const auto &[var, exp] : _vars)
    {
        res += exp;
    }
    return res;
}

double Monomial::to_double() const
{
    double res = 1.0;
    for (const auto &[var, exp] : _vars)
    {
        res *= std::pow(var.to_double(), exp);
    }
    return res;
}

string Monomial::to_string() const
{
    if (_vars.empty())
    {
        return "1";
    }
    ostringstream oss;
    bool first = true;
    for (const auto &[var, exp] : _vars)
    {
        if (!first)
        {
            oss << "*";
        }
        oss << var.to_string();
        if (exp != 1)
        {
            oss << "^" << exp;
        }
        first = false;
    }
    return oss.str();
}

// 按最大变量的 degree-lex 风格排序，与旧 Term::operator< 一致。
// 常数单项式最小。
bool Monomial::operator<(const Monomial &other) const
{
    if (_vars.empty())
    {
        return !other._vars.empty();
    }
    if (other._vars.empty())
    {
        return false;
    }
    const auto &max1 = *_vars.rbegin();
    const auto &max2 = *other._vars.rbegin();
    if (max1.first != max2.first)
    {
        return max1.first < max2.first;
    }
    if (max1.second != max2.second)
    {
        return max1.second < max2.second;
    }
    return _vars < other._vars;
}

size_t Monomial::hash() const
{
    size_t seed = 0;
    for (const auto &[var, exp] : _vars)
    {
        seed ^= std::hash<TermArg>{}(var) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(exp) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

ostream &operator<<(ostream &os, const Monomial &m)
{
    os << m.to_string();
    return os;
}
