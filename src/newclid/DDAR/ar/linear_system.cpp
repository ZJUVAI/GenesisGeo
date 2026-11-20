#include "ar/linear_system.hpp"
#include "ar/reduced_equation.hpp"
#include "numerical.hpp"
#include "solver/proof.hpp"
#include <algorithm>
#include <iostream>
#include <deque>
#include <stdexcept>
#include <cstdlib>
#include <string>

using namespace std;

LinearSystem::LinearSystem(const LinearSystem &other)
    : _rule_version(other._rule_version), _head_revisions(other._head_revisions),
      _equations(other._equations), _rules(other._rules),
      _head_var_index(other._head_var_index), _body_var_index(other._body_var_index)
{
    // The copied index must point to equations owned by the copied rule map.
    for (auto &[variable, entry] : _head_revisions)
    {
        if (entry.linear_rule)
        {
            entry.linear_rule = &_rules.at(entry.linear_rule->leading_monomial());
        }
    }
}

LinearSystem &LinearSystem::operator=(const LinearSystem &other)
{
    if (this != &other)
    {
        LinearSystem copy(other);
        *this = std::move(copy);
    }
    return *this;
}

void LinearSystem::index_rule(const Monomial &head, const Equation &body)
{
    for (const auto &[var, exp] : head.vars())
    {
        _head_var_index[var].insert(head);
        auto &entry = _head_revisions[var];
        if (head.vars().size() == 1 && exp == 1)
        {
            entry.linear_rule = &body;
        }
    }
    for (const auto &[mono, c] : body.body().terms())
    {
        for (const auto &[var, exp] : mono.vars())
        {
            _body_var_index[var].insert(head);
        }
    }
}

void LinearSystem::deindex_rule(const Monomial &head, const Equation &body)
{
    for (const auto &[var, exp] : head.vars())
    {
        if (head.vars().size() == 1 && exp == 1)
        {
            _head_revisions.at(var).linear_rule = nullptr;
        }
        auto it = _head_var_index.find(var);
        if (it != _head_var_index.end())
        {
            it->second.erase(head);
            if (it->second.empty())
            {
                _head_var_index.erase(it);
            }
        }
    }
    for (const auto &[mono, c] : body.body().terms())
    {
        for (const auto &[var, exp] : mono.vars())
        {
            auto it = _body_var_index.find(var);
            if (it != _body_var_index.end())
            {
                it->second.erase(head);
                if (it->second.empty())
                {
                    _body_var_index.erase(it);
                }
            }
        }
    }
}

const Equation *LinearSystem::find_divisor_rule(const Monomial &m) const
{
    if (m.is_constant())
    {
        return nullptr;
    }
    if (m.vars().size() == 1 && m.vars().begin()->second == 1)
    {
        const auto entry = _head_revisions.find(m.vars().begin()->first);
        if (entry == _head_revisions.end())
        {
            return nullptr;
        }
        if (entry->second.linear_rule)
        {
            return entry->second.linear_rule;
        }
    }
    // 优先精确命中(常见的线性变量代入)。
    auto exact = _rules.find(m);
    if (exact != _rules.end())
    {
        return &exact->second;
    }
    // 一般路径:在与 m 共享变量的 head 中找整除者,选最大的(优先消高次)。
    const Equation *best = nullptr;
    const Monomial *best_head = nullptr;
    for (const auto &[var, exp] : m.vars())
    {
        auto it = _head_var_index.find(var);
        if (it == _head_var_index.end())
        {
            continue;
        }
        for (const auto &head : it->second)
        {
            if (head.divides(m))
            {
                if (best_head == nullptr || head > *best_head)
                {
                    best_head = &head;
                    best = &_rules.at(head);
                }
            }
        }
    }
    return best;
}

Equation LinearSystem::normal_form(const Equation &eq) const
{
    auto reduced = normal_form_if_changed(eq);
    return reduced ? std::move(*reduced) : eq;
}

optional<Equation> LinearSystem::normal_form_if_changed(
    const Equation &original) const
{
    if (original.empty())
    {
        return nullopt;
    }

    optional<Equation> reduced;
    const Monomial multiplier = original.content_multiplier();
    if (!multiplier.is_constant() || original.leading_coeff() != Rational(1))
    {
        reduced.emplace(original);
        *reduced *= multiplier;
        reduced->make_monic();
    }

    bool changed = true;
    while (changed)
    {
        changed = false;
        const Equation &current = reduced ? *reduced : original;
        for (const auto &[mono, coeff] : current.body().terms())
        {
            const Equation *rule = find_divisor_rule(mono);
            if (rule == nullptr)
            {
                continue;
            }
            const auto &head = rule->leading_monomial();
            if (!reduced)
            {
                reduced.emplace(original);
            }
            if (mono == head)
            {
                reduced->subtract_scaled(*rule, coeff);
            }
            else
            {
                Equation scaled_rule = *rule;
                scaled_rule *= mono / head;
                scaled_rule *= coeff;
                *reduced -= scaled_rule;
            }
            reduced->content_reduce();
            reduced->make_monic();
            changed = true;
            break;
        }
    }
    return reduced;
}

void LinearSystem::install_rule(Equation eq)
{
    // 工作队列:重约简旧规则可能改变 head,改变后重新入队即可。
    deque<Equation> work;
    work.push_back(std::move(eq));

    size_t rounds = 0;
    while (!work.empty())
    {
        ++rounds;
        Equation cur = std::move(work.front());
        work.pop_front();

        cur.make_monic();
        if (cur.empty())
        {
            continue;
        }
        Monomial head = cur.leading_monomial();

        if (_rules.count(head))
        {
            // 该 head 已存在规则:对来者再做一次约简(避免覆盖,保持规则集一致)。
            Equation reduced = normal_form(cur);
            if (reduced.empty())
            {
                continue;
            }
            work.push_back(std::move(reduced));
            continue;
        }

        // 找出 body 含新 head 变量的旧规则,它们需要重约简。
        set<Monomial> affected;
        for (const auto &[var, exp] : head.vars())
        {
            auto it = _body_var_index.find(var);
            if (it != _body_var_index.end())
            {
                for (const auto &h : it->second)
                {
                    affected.insert(h);
                }
            }
        }

        // 装入新规则。
        const auto installed = _rules.emplace(head, std::move(cur)).first;
        index_rule(installed->first, installed->second);

        for (const auto &old_head : affected)
        {
            auto it = _rules.find(old_head);
            if (it == _rules.end())
            {
                continue;
            }
            Equation old_eq = it->second;
            deindex_rule(old_head, old_eq);
            _rules.erase(it);

            Equation new_eq = normal_form(old_eq);
            if (new_eq.empty())
            {
                continue;
            }
            if (new_eq == old_eq)
            {
                const auto restored = _rules.emplace(old_head, std::move(new_eq)).first;
                index_rule(restored->first, restored->second);
                continue;
            }
            work.push_back(std::move(new_eq));
        }
    }
}

void LinearSystem::add_reduced_equation(Proof *pf, std::string type)
{
    auto eqs = pf->reduced_equations(type);
    for (auto *eq : eqs)
    {
        eq->reduce();
        if (eq->is_solved())
        {
            continue;
        }
        add_equation(eq->original_equation(), pf);
    }
}

bool LinearSystem::add_equation(const Equation &eq, Proof *proof)
{
    // 先约简
    Equation remainder = normal_form(eq);
    if (remainder.empty())
    {
        return false; // 已被现有规则蕴含
    }
    // 归档原始方程,把自指系数加到证书,装入规则集。
    size_t idx = _equations.size();
    _equations.emplace_back(eq, proof);
    remainder.set_index(idx);
    remainder.make_monic();
    install_rule(std::move(remainder));
    ++_rule_version;
    return true;
}

const Equation &LinearSystem::at(size_t index) const
{
    return pair_at(index).first;
}

const pair<Equation, Proof *> &LinearSystem::pair_at(size_t index) const
{
    if (index >= _equations.size())
    {
        throw runtime_error("LinearSystem::pair_at: index out of range");
    }
    return _equations[index];
}

void LinearSystem::print_equations() const
{
    cout << "Linear System:" << endl;
    cout << "  Archived equations (" << _equations.size() << "):" << endl;
    for (size_t i = 0; i < _equations.size(); ++i)
    {
        cout << "    [" << i << "] " << _equations[i].first;
        if (_equations[i].second != nullptr)
        {
            cout << " (proof @" << _equations[i].second << ")";
        }
        cout << endl;
    }
    cout << "  Active rules (" << _rules.size() << "):" << endl;
    for (const auto &[head, eq] : _rules)
    {
        cout << "    [" << head << "] " << eq << endl;
    }
}

void LinearSystem::verify_equations() const
{
    // 用当前点坐标对每条方程的 body 求值。语义是 body == 0,故应数值近零。
    // 求值可能抛异常(如退化点导致 DistLog 取 log(0)),逐条捕获并标记为 ERROR,
    // 避免单条求值失败掩盖其余方程的检验结果。
    cout << "Numerical verification:" << endl;
    cout << "  Archived equations (" << _equations.size() << "):" << endl;
    for (size_t i = 0; i < _equations.size(); ++i)
    {
        const Equation &eq = _equations[i].first;
        try
        {
            double val = eq.body().to_double();
            bool ok = Numerical::nearly_zero(val);
            cout << "    [" << i << "] " << (ok ? "OK  " : "FAIL")
                 << " value=" << val << " | " << eq << endl;
        }
        catch (const exception &e)
        {
            cout << "    [" << i << "] ERROR (" << e.what() << ") | " << eq << endl;
        }
    }
    cout << "  Active rules (" << _rules.size() << "):" << endl;
    for (const auto &[head, eq] : _rules)
    {
        try
        {
            double val = eq.body().to_double();
            bool ok = Numerical::nearly_zero(val);
            cout << "    [" << head << "] " << (ok ? "OK  " : "FAIL")
                 << " value=" << val << " | " << eq << endl;
        }
        catch (const exception &e)
        {
            cout << "    [" << head << "] ERROR (" << e.what() << ") | " << eq << endl;
        }
    }
}

std::optional<Polynomial> LinearSystem::product_normal_form(const Monomial &product) const
{
    Polynomial residual({{product, Rational(1)}});
    try
    {
        for (size_t steps = 0; steps < 16; ++steps)
        {
            bool substituted = false;
            for (const auto &[term, coefficient] : residual.terms())
            {
                const auto *rule = find_divisor_rule(term);
                if (rule == nullptr)
                    continue;
                const auto scale = coefficient;
                const auto factor = term / rule->leading_monomial();
                Polynomial replacement = rule->body();
                replacement *= factor;
                residual.subtract_scaled(replacement, scale);
                substituted = true;
                break;
            }
            if (!substituted)
                return residual;
        }
    }
    catch (const std::runtime_error &) {}
    return std::nullopt;
}
