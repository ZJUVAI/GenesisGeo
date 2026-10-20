#ifndef DDAR_HPP
#define DDAR_HPP

#include <vector>
#include <map>
#include <string>
#include <memory>
#include <unordered_map>
#include "problem.hpp"
#include "matcher.hpp"
#include "ar/linear_system.hpp"
#include "ar/reduced_equation.hpp"
#include "typedef.hpp"
#include "theorem.hpp"
#include "solver/application.hpp"
#include "solver/proof.hpp"
#include "solver/object_table.hpp"

class DDARSolver
{
private:
    Problem *_problem;

    size_t _level{0};

    std::vector<Application> _applications;

    std::unordered_map<std::string, std::unique_ptr<Proof>> _statement_proofs;

    std::vector<Proof *> _goals;

    std::vector<Proof *> _ars;

    std::vector<Proof *> _checked_statements;

    bool _solved{false};

    std::map<std::string, bool> _config;

    bool get_config(const std::string &key, bool default_val = false) const
    {
        auto it = _config.find(key);
        return it != _config.end() ? it->second : default_val;
    }

    LinearSystem _system_slope;
    LinearSystem _system_dist;
    LinearSystem _system_distlog;

    using eqns_map_type = std::unordered_map<Equation, ReducedEquation>;

    eqns_map_type _equations_slope;
    eqns_map_type _equations_dist;
    eqns_map_type _equations_distlog;

public:
    bool run_level(const Point &max_pt);

    bool run(size_t max_levels);

    // 对前提数量过多的 AR 推导语句尝试简化（用若干步 DD 推理替代单步 AR）。
    // 在 run() 收尾时调用；逐个候选交由 ARSimplifier 处理，成功则改写其依赖。
    void simplify_ar();

    size_t get_level() const { return _level; }

    bool is_solved() const { return _solved; }

    const std::vector<Proof *> &goals() const { return _goals; }

    const std::vector<Proof *> &checked_statements() const { return _checked_statements; }

    const Problem &problem() const { return *_problem; }

    Proof *insert_statement(const std::unique_ptr<Statement> &p);

    const std::vector<Application> &applications() const { return _applications; }

    size_t num_applications() const;

    size_t push_established_statement(Proof *pf);

    DDARSolver(Problem *problem, const std::map<std::string, bool> &config = {});

    const std::map<std::string, bool> &config() const { return _config; }

    void advance_theorem(size_t index);

    void insert_application(Theorem thm);

    void add_custom_theorems(const std::vector<CustomRule> &rules);

    bool establish_statement(Proof *pf, size_t thm_id);

    void add_established_equations(Proof *pf);

    void print_equations() const;

    std::vector<std::tuple<std::vector<std::string>, std::vector<std::vector<std::string>>, std::string>> dependency_graph() const;

    std::vector<ReducedEquation *> insert_equation(const std::unique_ptr<Statement> &pf, std::string type);
};

#endif // DDAR_HPP