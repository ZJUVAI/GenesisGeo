#ifndef MATCHER_HPP
#define MATCHER_HPP

#include <vector>
#include <cstdint>
#include <array>
#include <unordered_map>
#include <map>
#include <string>
#include <span>
#include "type/triangle.hpp"
#include "type/dist.hpp"
#include "theorem.hpp"
#include "problem.hpp"

#define REL_TOL 0.001
#define EPS 1E-12

// 在构建对象时完成匹配，匹配结果存放在_theorems中
class Matcher
{
private:
    Problem *_problem;

    std::map<std::string, bool> _config;

    std::vector<Theorem> _theorems;

    std::vector<std::unique_ptr<Statement>> _stmts;

    bool get_config(const std::string &key, bool default_val = false) const
    {
        auto it = _config.find(key);
        return it != _config.end() ? it->second : default_val;
    }

    void match_similar_triangles();

    void match_between();

    void match_equal_angles();

    void match_circles();

    void match_orthocenters();

    void match_perps_paras();

    void match_dist_ratios();


    std::vector<std::tuple<double, Coll>> all_betweens();

    std::vector<std::pair<Point, Point>> all_eqpoints();

    std::vector<std::tuple<double, Angle>> all_angles();

    std::vector<std::tuple<double, Dist>> all_dists();

    void on_similar_triangles(const SimilarTriangles &simtri);

    void on_pappus(const Pappus &pappus);

    void on_between(const Coll &coll);

    void on_midpoint(const Midp &midp);

    void on_eqratio(const Coll &left, const Coll &right);

    void on_cyclic(const Cyclic &cyclic);

    void on_bisector(const Point &pt, const Angle &ang);

    void on_eqangle(const Angle &left, const Angle &right);

    void on_circle(const Point &center, const std::vector<std::pair<double, Point>> &points);

    void on_circumcenter(const CircumCenter &circumcenter);

    void on_quadrangle_circumcenter(const Point &center, const Cyclic &cyc);

    void insert_theorem(const Theorem &thm);

public:
    Matcher(Problem *prob, const std::map<std::string, bool> &config = {});

    std::vector<Theorem> take_theorems() && { return std::move(_theorems); }

    // 旧 AR 候选访问接口。
    // const std::vector<std::unique_ptr<Statement>> &stmts() const { return _stmts; }
};

// 自定义定理匹配器 - 独立功能，不影响原有Matcher

// Stmt: (谓词类型, [点代号列表])，例如 ("coll", ["A","B","C"])
using Stmt = std::pair<std::string, std::vector<std::string>>;

// Mapping: 定理点代号 -> Problem中点的索引
class Mapping
{
    std::vector<std::pair<std::string, int>> _values;
    // Normalized discovery variables use A..Z. Other names keep exact lookup.
    std::array<int, 26> _single;
public:
    Mapping() { _single.fill(-1); }
    void push_back(std::pair<std::string, int> value)
    {
        if (value.first.size()==1 && value.first[0]>='A' && value.first[0]<='Z')
            _single[value.first[0]-'A'] = value.second;
        _values.push_back(std::move(value));
    }
    void pop_back()
    {
        const auto &name = _values.back().first;
        if (name.size()==1 && name[0]>='A' && name[0]<='Z')
            _single[name[0]-'A'] = -1;
        _values.pop_back();
    }
    int find(const std::string &name) const
    {
        if (name.size()==1 && name[0]>='A' && name[0]<='Z')
            return _single[name[0]-'A'];
        for (const auto &[key, value] : _values)
            if (key==name) return value;
        return -1;
    }
    auto begin() const { return _values.begin(); }
    auto end() const { return _values.end(); }
    const auto &back() const { return _values.back(); }
};

// Rule: (前提列表, 结论列表, 定理名, 规则名)
struct CustomRule
{
    std::vector<Stmt> premises;
    std::vector<Stmt> conclusions;
    std::vector<Stmt> guards; // 数值守卫谓词：只 check_numerically，不参与推理链
    std::string name;
    std::string rule;
    bool static_prepared = false, static_fits = false;
    std::vector<uint64_t> static_masks;
    std::vector<uint8_t> static_deferred;
    std::vector<Stmt> early_checks;
    std::vector<uint64_t> early_masks;
    std::vector<uint8_t> early_valid;
    void prepare_static();
};

class CustomTheoremMatcher
{
private:
    Problem *_problem;
    std::vector<Theorem> _theorems;
    std::map<std::string, bool> _config;

    // 预计算索引：谓词名 -> 所有满足的点索引组合（包含所有排列）
    // 例如 _index["coll"] = [[0,1,2], [1,0,2], [2,1,0], ...]
    std::unordered_map<std::string, std::vector<std::vector<int>>> _index;

    // Share bound-column lookup tables across all rules in this problem.
    // The numerical predicate index is immutable during custom matching.
    // Reuse exact premise prefixes across rules; bindings retain traversal order.
    std::map<std::vector<Stmt>, std::vector<Mapping>> _prefix_matches;
    size_t _prefix_binding_count = 0;
    double _planning_seconds = 0, _backtrack_seconds = 0, _emitting_seconds = 0;
    // Exact numerical results on immutable coordinates, keyed by ordered points.
    mutable std::unordered_map<std::string, std::array<std::unordered_map<uint64_t, bool>, 9>> _numeric_memo;
    mutable size_t _numeric_memo_size = 0;
    mutable size_t _numeric_hits = 0, _numeric_misses = 0;
    bool _memo_enabled = true, _projection_enabled = true, _allow_duplicates = false;
    bool _static_plans = true;
    size_t _result_limit = size_t(-1);

    using ProjectionIndex = std::unordered_map<uint64_t, std::vector<size_t>>;
    mutable std::map<std::pair<std::string, unsigned>, ProjectionIndex> _projections;

    // 预计算常见谓词的所有满足实例
    void build_predicate_index(const std::vector<CustomRule> &rules);

    // 递归匹配 stmts[idx..] 在当前 mapping 下的所有合法扩展
    void backtrack(
        const std::vector<Stmt> &stmts,
        size_t idx,
        Mapping &current,
        std::vector<bool> &used,
        std::vector<Mapping> &out) const;

    // 对一条 rule 的所有前提匹配，生成 Theorem 并插入 _theorems
    void match_rule(const CustomRule &rule);
    std::vector<Stmt> legacy_plan(const CustomRule &rule) const;
    std::vector<Stmt> static_plan(const CustomRule &rule) const;

    bool get_config(const std::string &key, bool default_val = false) const
    {
        auto it = _config.find(key);
        return it != _config.end() ? it->second : default_val;
    }

    void DebugLog(const std::string &msg) const;

    bool verify_stmt(const Stmt &stmt, const Mapping &mapping) const;
    bool verify_bound_stmt(const Stmt &stmt, std::span<const int> points) const;

    // custom 规则专用：para 实例的退化判据（由 config "custom_para_no_collinear" 控制，
    // 默认开启）。返回 true 表示该实例可用。拒绝两种情形：
    //   1) 四点中存在重合点 —— 两条"直线"共享端点，此时平行成立 ⟺ 三点共线；
    //   2) 四点互异但全部共线 —— 两条"直线"实为同一条直线。
    // 注意：只作用于 CustomTheoremMatcher，内置 Matcher 与 AR/简化路径不受影响
    // （内置 r82 正是刻意从共线推出平行，阻断它会使 jgex_231 从 160/231 掉到 21/231）。
    bool para_instance_ok(const std::vector<int> &pts) const;

    // Apply the same degeneracy check to a bound statement.
    bool stmt_instance_ok(const Stmt &stmt, const Mapping &mapping) const;

    void enumerate_brute_force(const std::vector<Stmt> &stmts, size_t stmt_idx, std::span<const std::string *const> new_vars, Mapping &current, std::vector<bool> &used, std::vector<Mapping> &results) const;

    void build_triangle_indices(size_t n);

    void build_eqangle_indices(size_t n);

    void build_eqratio_indices(size_t n);

public:
    // rules: 每条规则包含前提、结论、名称、规则id
    // config: 配置选项，支持 "use_predicate_cache" (默认true) 控制是否启用预计算索引
    CustomTheoremMatcher(Problem *prob, const std::vector<CustomRule> &rules,
                         const std::map<std::string, bool> &config = {});

    std::vector<Theorem> take_theorems() && { return std::move(_theorems); }
};

#endif // MATCHER_HPP
