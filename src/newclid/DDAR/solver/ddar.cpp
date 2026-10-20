#include "solver/ddar.hpp"
#include "problem.hpp"
#include "solver/application.hpp"
#include "solver/proof.hpp"
#include "solver/simplify.hpp"
#include "matcher.hpp"
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <type_traits>
#include <cmath>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <unordered_set>

using namespace std;

DDARSolver::DDARSolver(Problem *problem, const std::map<std::string, bool> &config)
    : _problem(problem), _config(config)
{
    Matcher matcher(problem, config);

    for (const auto &thm : matcher.theorems())
    {
        insert_application(thm.clone());
    }
    for (const auto &hyp : problem->hypotheses())
    {
        this->insert_statement(hyp->normalize())->prove_by_assumption();
    }
    for (const auto &goal : problem->goals())
    {
        _goals.push_back(this->insert_statement(goal->normalize()));
    }
    // matcher.stmts() 是纯数值匹配出的"已验证"语句（平行/垂直/比例相等等），
    // 送进 _ars 供 AR 直接复用。有 goal 时 Matcher 内部已经不做这部分匹配，
    // matcher.stmts() 天然为空，这里不用再额外判断 goal。
    for (const auto &stmt : matcher.stmts())
    {
        if (stmt->check_numerically())
        {
            _ars.push_back(this->insert_statement(stmt->normalize()));
        }
    }
}

bool DDARSolver::run_level(const Point &max_pt)
{
    size_t num_stmts = _checked_statements.size();
    _level++;

    bool changed = true;
    while (changed)
    {
        changed = false;
        for (auto it = _ars.begin(); it != _ars.end();)
        {
            auto &goal = *it;
            if (goal->max_point() > max_pt)
            {
                ++it;
                continue;
            }
            if (goal->is_proved())
            {
                it = _ars.erase(it);
                changed = true;
            }
            else
            {
                goal->ar(_level);
                if (goal->is_proved())
                {
                    it = _ars.erase(it);
                    changed = true;
                }
                else
                {
                    ++it;
                }
            }
        }
    }

    size_t const n = _applications.size();
    for (size_t i = 0; i < n; i++)
    {
        if (_applications[i].max_point() <= max_pt && _applications[i].state() == ApplicationState::PENDING)
        {
            advance_theorem(i);
        }
    }

    if (!_problem->goals().empty())
    {
        bool res = true;
        for (auto &goal : _goals)
        {
            if (!goal->is_proved())
            {
                goal->ar(_level);
                if (!goal->is_proved())
                {
                    res = false;
                }
            }
        }
        _solved = res;
    }

    return num_stmts < _checked_statements.size();
}

bool DDARSolver::run(size_t max_levels)
{
    bool has_goal = !_problem->goals().empty();

    for (Point const &pt : _problem->points())
    {
        for (size_t i = 0; i < max_levels; ++i)
        {
            if (!run_level(pt))
            {
                break;
            }
        }
        if (_solved)
        {
            return _solved;
        }
    }
    if (!has_goal)
    {
        _solved = true;
    }
    return _solved;
}

void DDARSolver::simplify_ar()
{
    // 阈值：仅对依赖数 >= SIMPLIFY_PREMISE_THRESHOLD 的 AR 语句尝试简化。
    constexpr size_t SIMPLIFY_PREMISE_THRESHOLD = 4;

    // 先收集候选 Proof*（指针在 _statement_proofs 中由 unique_ptr 持有，稳定不失效）。
    // 简化过程中 ARSimplifier 可能 insert_statement 新中间语句、改动 _checked_statements，
    // 因此不能边遍历边处理；但候选以 Proof* 为身份，无需像扁平图那样按结论重定位。
    std::vector<Proof *> candidates;
    for (Proof *pf : _checked_statements)
    {
        const ProofState st = pf->state();
        const bool is_ar = (st == ProofState::PROVED_AR_DIST ||
                            st == ProofState::PROVED_AR_SLOPE ||
                            st == ProofState::PROVED_AR_DISTLOG);
        if (is_ar && pf->get_dependencies().size() > SIMPLIFY_PREMISE_THRESHOLD)
        {
            candidates.push_back(pf);
        }
    }

    for (Proof *target : candidates)
    {
        ARSimplifier simplifier(this, target);
        std::pair<bool, std::vector<ARSimplifier::SubProof>> result = simplifier.run_new();
        if (!result.first)
        {
            // 简化失败：打印原始前提与结论，便于排查哪些 AR 推理无法被 DD 重证。
            // cout << "[simplify_ar] FAILED to simplify: "
            //      << target->statement()->to_string()
            //      << " (reason: " << target->reason() << ")" << endl;
            // cout << "    premises (" << target->get_dependencies().size() << "):" << endl;
            // for (Proof *dep : target->get_dependencies())
            // {
            //     cout << "      " << dep->statement()->to_string() << endl;
            // }
            continue;
        }

        const std::vector<ARSimplifier::SubProof> &subgraph = result.second;

        // 简化成功：先记录简化前的原始前提，便于与简化后对比。
        // cout << "[simplify_ar] SIMPLIFIED: "
        //      << target->statement()->to_string()
        //      << " (reason: " << target->reason() << ")" << endl;
        // cout << "    before (" << target->get_dependencies().size()
        //      << " premises):" << endl;
        // for (Proof *dep : target->get_dependencies())
        // {
        //     cout << "      " << dep->statement()->to_string() << endl;
        // }

        // 计算 target 原始依赖的传递依赖闭包。子图中凡是落在闭包内的结论，
        // 都是 target 所依赖的既有事实 —— 不得改写（改写会破坏原推导并可能成环）。
        // 仅更新「不被 D1-Dn (传递)依赖」的结论：即 target 本身及新引入的中间语句。
        std::unordered_set<Proof *> closure;
        {
            std::vector<Proof *> stack = target->get_dependencies();
            while (!stack.empty())
            {
                Proof *cur = stack.back();
                stack.pop_back();
                if (cur == nullptr || !closure.insert(cur).second)
                {
                    continue;
                }
                for (Proof *dep : cur->get_dependencies())
                {
                    stack.push_back(dep);
                }
            }
        }

        // cout << "[simplify_ar] simplifying: " << target->statement()->to_string()
        //      << " (orig deps: " << target->get_dependencies().size()
        //      << ", subgraph steps: " << subgraph.size() << ")" << endl;

        // 打印简化后的完整证明子图（每一步结论及其依赖），不跳过任何步骤。
        // cout << "    after (" << subgraph.size() << " steps):" << endl;
        for (const ARSimplifier::SubProof &step : subgraph)
        {
            // cout << "      " << step.conclusion->statement()->to_string() << " <-";
            // for (Proof *d : step.deps)
            // {
            //     cout << " [" << d->statement()->to_string() << "]";
            // }
            // cout << endl;
        }

        // 定位 target 在 _checked_statements 中的位置，新中间语句插在它之前以维护拓扑序。
        auto target_it = std::find(_checked_statements.begin(), _checked_statements.end(), target);

        for (const ARSimplifier::SubProof &step : subgraph)
        {
            Proof *concl = step.conclusion;

            // 闭包内的既有事实：保持不变，跳过。
            if (concl != target && closure.find(concl) != closure.end())
            {
                continue;
            }

            // 新引入的中间语句尚不在 _checked_statements 中，插到 target 之前。
            if (concl != target &&
                std::find(_checked_statements.begin(), _checked_statements.end(), concl) ==
                    _checked_statements.end())
            {
                target_it = _checked_statements.insert(target_it, concl);
                ++target_it; // 保持指向 target，后续中间语句继续插在其前
            }

            // 诊断：检测这一步 update 是否会把 concl 直接指回自身（自环），
            // 或指向本步之外某条与 concl 同规范式的依赖。node 身份以
            // normalize()->to_string() 为准，与依赖图/环检测一致。
            const std::string concl_key = concl->statement()->normalize()->to_string();
            std::vector<Proof *> clean_deps;
            clean_deps.reserve(step.deps.size());
            for (Proof *dep : step.deps)
            {
                if (dep != nullptr &&
                    dep->statement()->normalize()->to_string() == concl_key)
                {
                    // cout << "[simplify_ar][SELF-DEP] target=" << target->statement()->to_string()
                    //      << "  concl=" << concl->statement()->to_string()
                    //      << "  依赖里出现了与结论同式的项，已剔除以避免自环" << endl;
                    continue; // 剔除自依赖
                }
                clean_deps.push_back(dep);
            }

            concl->update(clean_deps);
        }
    }
}

void DDARSolver::advance_theorem(size_t index)
{
    auto &app = _applications[index];
    if (app.state() != ApplicationState::PENDING)
    {
        return;
    }

    app.advance_proof(_level);
    if (app.state() == ApplicationState::PROVED)
    {
        for (auto *p : app.conclusions())
        {
            establish_statement(p, index);
        }
    }
}

void DDARSolver::insert_application(Theorem thm)
{
    _applications.emplace_back(this, move(thm));
}

void DDARSolver::add_custom_theorems(const vector<CustomRule> &rules)
{
    CustomTheoremMatcher matcher(_problem, rules, _config);

    // cout << "\n=== Matched Custom Theorems ===" << endl;
    size_t idx = 0;
    for (const auto &thm : matcher.theorems())
    {
        // if (get_config("verbose", false))
        // {
        //     cout << "[" << idx++ << "] " << thm.name() << " (" << thm.rule() << ")" << endl;

        //     cout << "  Hypotheses:" << endl;
        //     for (const auto &hyp : thm.hypotheses())
        //         cout << "    " << hyp->to_string() << endl;

        //     cout << "  Conclusions:" << endl;
        //     for (const auto &con : thm.conclusions())
        //         cout << "    " << con->to_string() << endl;

        //     cout << endl;
        // }
        insert_application(thm.clone());
    }
    // cout << "Total: " << matcher.theorems().size() << " custom theorems added" << endl;
    // cout << "================================\n" << endl;
}

Proof *DDARSolver::insert_statement(const unique_ptr<Statement> &p)
{
    auto key = p->to_string();
    auto [it, success] = _statement_proofs.insert({key, std::make_unique<Proof>(this, p->clone())});
    if (success)
    {
        it->second->initial();
    }
    return it->second.get();
}

vector<tuple<vector<string>, vector<vector<string>>, string>> DDARSolver::dependency_graph() const
{
    vector<tuple<vector<string>, vector<vector<string>>, string>> res;

    for (const auto &pf : _checked_statements)
    {
        if (!pf->is_proved())
        {
            cout << "?????" << endl;
            continue;
        }
        // if (pf->state() == ProofState::PROVED_AR)
        // {
        //     pf->print_equations();
        // }
        vector<vector<string>> deps;
        // bool flag = false;
        for (const auto &dep : pf->get_dependencies())
        {
            deps.push_back(dep->statement()->normalize()->to_tokens());
        }
        res.push_back({pf->statement()->normalize()->to_tokens(), deps, pf->reason()});
    }

    return res;
}

size_t DDARSolver::num_applications() const
{
    return _applications.size();
}

size_t DDARSolver::push_established_statement(Proof *pf)
{
    size_t const index = _checked_statements.size();
    _checked_statements.push_back(pf);
    return index;
}

void DDARSolver::print_equations() const
{
    cout << "Slope Equations:" << endl;
    for (const auto &[eqn, red_eqn] : _equations_slope)
    {
        cout << "Equation: " << eqn << endl;
        cout << "Reduced Equation: " << red_eqn.remainder() << endl;
        cout << endl;
    }
    cout << "Dist Equations:" << endl;
    for (const auto &[eqn, red_eqn] : _equations_dist)
    {
        cout << "Equation: " << eqn << endl;
        cout << "Reduced Equation: " << red_eqn.remainder() << endl;
        cout << endl;
    }
    cout << "DistLog Equations:" << endl;
    for (const auto &[eqn, red_eqn] : _equations_distlog)
    {
        cout << "Equation: " << eqn << endl;
        cout << "Reduced Equation: " << red_eqn.remainder() << endl;
        cout << endl;
    }

    _system_dist.print_equations();

    _system_distlog.print_equations();

    _system_slope.print_equations();
}

bool DDARSolver::establish_statement(Proof *pf, size_t thm_id)
{
    if (pf->is_proved())
    {
        return false;
    }
    pf->set_theorem(thm_id, _level);
    return true;
}

vector<ReducedEquation *> DDARSolver::insert_equation(const unique_ptr<Statement> &pf, string type)
{
    vector<ReducedEquation *> res;

    if (type == "dist")
    {
        auto eqn_ptrs = pf->as_equation_dist(get_config("using_exp"));
        if (!eqn_ptrs.empty())
        {
            for (const auto &eqn_ptr : eqn_ptrs)
            {
                LinearSystem *sys = &_system_dist;
                eqns_map_type *eqns = &_equations_dist;
                if (!eqn_ptr->empty())
                {
                    Equation eqn = *eqn_ptr;
                    eqn.make_monic();
                    auto red_eq = ReducedEquation(eqn, sys);
                    res.push_back(&(eqns->insert({eqn, red_eq}).first->second));
                }
            }
        }
        return res;
    }
    if (type == "slope")
    {
        auto eqn_ptrs = pf->as_equation_slope(get_config("using_exp"));
        if (!eqn_ptrs.empty())
        {
            for (const auto &eqn_ptr : eqn_ptrs)
            {
                LinearSystem *sys = &_system_slope;
                eqns_map_type *eqns = &_equations_slope;
                if (!eqn_ptr->empty())
                {
                    Equation eqn = *eqn_ptr;
                    eqn.make_monic();
                    auto red_eq = ReducedEquation(eqn, sys);
                    res.push_back(&(eqns->insert({eqn, red_eq}).first->second));
                }
            }
        }
        return res;
    }
    if (type == "distlog")
    {
        auto eqn_ptrs = pf->as_equation_distlog(get_config("using_exp"));
        if (!eqn_ptrs.empty())
        {
            for (const auto &eqn_ptr : eqn_ptrs)
            {
                LinearSystem *sys = &_system_distlog;
                eqns_map_type *eqns = &_equations_distlog;
                if (!eqn_ptr->empty())
                {
                    Equation eqn = *eqn_ptr;
                    eqn.make_monic();
                    auto red_eq = ReducedEquation(eqn, sys);
                    res.push_back(&(eqns->insert({eqn, red_eq}).first->second));
                }
            }
        }
        return res;
    }
    return res;
}

void DDARSolver::add_established_equations(Proof *pf)
{
    _system_dist.add_reduced_equation(pf, "dist");
    _system_distlog.add_reduced_equation(pf, "distlog");
    _system_slope.add_reduced_equation(pf, "slope");
    if (pf->name() == "eqpoint")
    {
        auto pts = pf->statement()->points();
        Point p = pts[0];
        Point q = pts[1];
        std::vector<Proof *> to_process_pq;
        std::vector<Proof *> to_process_qp;
        for (const auto &old_pf : _checked_statements)
        {
            bool has_p = old_pf->statement()->contain(p);
            bool has_q = old_pf->statement()->contain(q);
            if (has_p && !has_q)
                to_process_pq.push_back(old_pf);
            else if (has_q && !has_p)
                to_process_qp.push_back(old_pf);
        }
        for (auto &old_pf : to_process_pq)
        {
            auto stmt = old_pf->statement()->replace(p, q);
            string key = stmt->normalize()->to_string();
            auto itr = _statement_proofs.find(key);
            if (itr != _statement_proofs.end())
            {
                Proof *new_pf = itr->second.get();
                if (!new_pf->is_proved())
                    new_pf->set_proved(pf, old_pf);
            }
        }
        for (auto &old_pf : to_process_qp)
        {
            auto stmt = old_pf->statement()->replace(q, p);
            string key = stmt->normalize()->to_string();
            auto itr = _statement_proofs.find(key);
            if (itr != _statement_proofs.end())
            {
                Proof *new_pf = itr->second.get();
                if (!new_pf->is_proved())
                    new_pf->set_proved(pf, old_pf);
            }
        }
    }
}
