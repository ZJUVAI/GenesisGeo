#ifndef LINEAR_SYSTEM_HPP
#define LINEAR_SYSTEM_HPP

#include <map>
#include <set>
#include <vector>
#include <functional>
#include <unordered_map>
#include <cstddef>
#include "ar/monomial.hpp"
#include "ar/polynomial.hpp"
#include "ar/equation.hpp"
#include "ar/term_arg.hpp"

class Proof;

class LinearSystem final
{
private:
    size_t _rule_version = 0;
    // 已归档的原始方程,append-only;archive index 即下标。
    std::vector<std::pair<Equation, Proof *>> _equations;

    // 活跃规则集,按 leading monomial 索引(降序)。
    std::map<Monomial, Equation, std::greater<Monomial>> _rules;

    // var -> 含该变量的规则 head 集合(用于按整除性查找规则)。
    std::unordered_map<TermArg, std::set<Monomial>> _head_var_index;
    // var -> body 中含该变量的规则 head 集合(用于新规则触发的反向 inter-reduce)。
    std::unordered_map<TermArg, std::set<Monomial>> _body_var_index;

    void index_rule(const Monomial &head, const Equation &body);
    void deindex_rule(const Monomial &head, const Equation &body);

    // 找到一条 head 整除 m 的规则;无则返回 nullptr。
    const Equation *find_divisor_rule(const Monomial &m) const;

    // 把 (monic 的)方程作为新规则装入,并对受影响的旧规则做 inter-reduce。
    void install_rule(Equation eq);

public:
    LinearSystem() = default;

    // 把 pf 的某一类方程加入系统(dist / slope / distlog),并立刻 reduce。
    // 名称、签名与原版保持一致。
    void add_reduced_equation(Proof *pf, std::string type);

    // 直接加入一条方程的简化入口(给单元测试或不走 ReducedEquation 的调用方用)。
    // 返回 true 表示该方程未被现有规则集蕴含,已作为新规则装入。
    bool add_equation(const Equation &eq, Proof *proof);

    // 把任意方程约简到当前规则集下的范式(body 与证书同步更新)。
    // 给 ReducedEquation 调用。
    Equation normal_form(Equation eq) const;

    // archive 访问,签名与原版一致。
    const Equation &at(size_t index) const;
    const std::pair<Equation, Proof *> &pair_at(size_t index) const;
    size_t rule_version() const { return _rule_version; }
    size_t size() const { return _equations.size(); }

    // 调试打印,签名与原版一致。
    void print_equations() const;

    // 数值检验:用当前点坐标对每条归档方程与活跃规则的 body 求值。
    // 每个方程语义是 body == 0,故求值应数值接近 0;偏离者标记为 FAIL。
    void verify_equations() const;

    // 规则集只读访问(供调试/测试)。
    const std::map<Monomial, Equation, std::greater<Monomial>> &rules() const { return _rules; }
};

#endif // LINEAR_SYSTEM_HPP
