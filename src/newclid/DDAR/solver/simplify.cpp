#include "solver/simplify.hpp"
#include "solver/ddar.hpp"
#include "solver/proof.hpp"
#include "problem.hpp"
#include "predicate/para.hpp"
#include "predicate/perp.hpp"
#include "predicate/cong.hpp"
#include "predicate/eqangle.hpp"
#include "predicate/eqratio.hpp"
#include "predicate/aconst.hpp"
#include "predicate/rconst.hpp"
#include "predicate/statement.hpp"
#include "type/slope.hpp"
#include "type/dist.hpp"
#include "type/rational.hpp"

#include <vector>
#include <string>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <algorithm>
#include <functional>
#include <optional>

using namespace std;

ARSimplifier::ARSimplifier(DDARSolver *solver, Proof *target)
    : _solver(solver), _target(target)
{
}

namespace
{
    // 一条线/线段的身份键：归一化后两端点名拼接。
    // Slope 与 Dist 接口一致（left/right/normalize/name），模板统一处理。
    template <typename SegT>
    string seg_key(const SegT &s)
    {
        auto n = s.normalize();
        return n.left().name() + "," + n.right().name();
    }

    // 取 eqangle/eqratio 的四条线/段，按 {左差的两段, 右差的两段} 排列：
    //   eqangle: angle(s1,s2) = angle(s3,s4)  -> {s1,s2, s3,s4}
    //   eqratio: lu/ld = ru/rd                -> {lu,ld, ru,rd}
    // 差对象 (first,second) 的含义：eqangle 是 d(second)-d(first)，eqratio 是 ℓ(first)-ℓ(second)。
    array<Slope, 4> pred_quad(const EqAngle &e)
    {
        return {e.s1(), e.s2(), e.s3(), e.s4()};
    }
    array<Dist, 4> pred_quad(const EqRatio &e)
    {
        return {e.left_up(), e.left_down(), e.right_up(), e.right_down()};
    }

    // 计算 roots 的传递依赖闭包（含 roots 自身）。
    unordered_set<Proof *> dependency_closure(const vector<Proof *> &roots)
    {
        unordered_set<Proof *> closure;
        deque<Proof *> stack(roots.begin(), roots.end());
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
                if (closure.find(dep) == closure.end())
                {
                    stack.push_back(dep);
                }
            }
        }
        return closure;
    }

    // 等价类管理：节点 = 字符串键（线键/段键）。
    // 每个节点维护 reason —— 从该节点到所在类 base 的路径上所有桥接事实(Proof*)的集合。
    // 两节点 reason 的对称差，即它们之间路径上的事实集合（公共前缀被消去）。
    class EquivClasses
    {
    public:
        // 是否已知该节点
        bool has(const string &x) const { return _root.count(x) > 0; }

        // 引入一个孤立节点（若已存在则忽略）
        void ensure(const string &x)
        {
            if (_root.find(x) == _root.end())
            {
                _root[x] = x;
                _reason[x] = {};
                _off[x] = 0;
                _members[x] = {x};
            }
        }

        bool same_class(const string &a, const string &b) const
        {
            auto ia = _root.find(a);
            auto ib = _root.find(b);
            return ia != _root.end() && ib != _root.end() && ia->second == ib->second;
        }

        // 同类时，a 相对 b 的偏移（模 180）。即方向群里 dir(a) - dir(b)。
        // 纯等价关系（cong）下恒为 0。要求 same_class(a,b)。
        int offset_between(const string &a, const string &b) const
        {
            return mod180(_off.at(a) - _off.at(b));
        }

        // 记录一条事实 via：连接 a、b 两节点（二者必须已 ensure）。
        // label = dir(a) - dir(b)（para:0, perp:90），纯等价关系传 0。
        // 按四种情况维护等价类、reason 与偏移。
        void connect(const string &a, const string &b, Proof *via, int label = 0)
        {
            ensure(a);
            ensure(b);
            const string ra = _root[a];
            const string rb = _root[b];

            if (ra == rb)
            {
                // 同类：忽略（保持生成树，不引入冗余边）
                return;
            }

            // 不同类：把较小类并入较大类。
            // 设把 src 类并入 dst 类，连接处为 (a 在某类, b 在另一类)。
            // 需要保证：a 属于 src 类、b 属于 dst 类（这样 a 经 via 接到 b 一侧）。
            string a_side = a, b_side = b;
            if (_members[ra].size() > _members[rb].size())
            {
                // 让 src(被并) = rb 类，dst(保留) = ra 类
                swap(a_side, b_side);
            }
            const string src_root = _root[a_side]; // 被并入、需要更新的类
            const string dst_root = _root[b_side]; // 保留的类（新 base 不变）

            // label 给的是 dir(a)-dir(b)。若上面 swap 了 a/b，方向反过来，偏移取负。
            int edge = (a_side == a) ? label : -label;

            // 接口点 b_side 到其类 base(dst) 的路径事实/偏移
            const unordered_set<Proof *> b_path = _reason[b_side];
            const int b_off = _off[b_side];
            // 接口点 a_side 到其(旧)类 base(src) 的路径事实/偏移
            const unordered_set<Proof *> a_path = _reason[a_side];
            const int a_off = _off[a_side];

            // src 类中每个元素 w：
            //   新 reason(w) = 旧 reason(w) △ a_path  (先把 w 重定向到 a_side)
            //                  ∪ {via} ∪ b_path        (再从 a_side 经 via 接到 dst base)
            //   新 off(w)    = (off(w) - a_off)        (w 相对 a_side: dir(w)-dir(a_side))
            //                  + edge                  (a_side 相对 b_side: dir(a_side)-dir(b_side))
            //                  + b_off                 (b_side 相对 dst base)
            for (const string &w : _members[src_root])
            {
                unordered_set<Proof *> nr = sym_diff(_reason[w], a_path);
                nr.insert(via);
                for (Proof *p : b_path)
                {
                    nr.insert(p);
                }
                _reason[w] = std::move(nr);
                _off[w] = mod180(_off[w] - a_off + edge + b_off);
                _root[w] = dst_root;
            }
            // 合并成员表
            auto &dst_members = _members[dst_root];
            for (const string &w : _members[src_root])
            {
                dst_members.push_back(w);
            }
            _members.erase(src_root);
        }

        // a、b 之间路径上的事实集合（无序）。要求 same_class(a,b)。
        unordered_set<Proof *> path_facts(const string &a, const string &b) const
        {
            return sym_diff(_reason.at(a), _reason.at(b));
        }

    private:
        // 角度归一到 [0,180)
        static int mod180(int v)
        {
            int r = v % 180;
            return r < 0 ? r + 180 : r;
        }

        static unordered_set<Proof *> sym_diff(const unordered_set<Proof *> &x,
                                               const unordered_set<Proof *> &y)
        {
            unordered_set<Proof *> res;
            for (Proof *p : x)
            {
                if (y.find(p) == y.end())
                {
                    res.insert(p);
                }
            }
            for (Proof *p : y)
            {
                if (x.find(p) == x.end())
                {
                    res.insert(p);
                }
            }
            return res;
        }

        unordered_map<string, string> _root;                  // 节点 -> 所在类 base
        unordered_map<string, unordered_set<Proof *>> _reason; // 节点 -> 到 base 的路径事实集
        unordered_map<string, int> _off;                       // 节点 -> 相对 base 的偏移(度,模180)
        unordered_map<string, vector<string>> _members;        // base -> 成员列表
    };

    // 差对象图上从 src 到 dst 走出有序链，并二元化成子证明图。
    // 用于 Tier-2（eqangle/eqratio）链，以及诱导事实里证明中间 Tier-2 事实。
    //   eq/adj/rep: 已建好的差对象等价类、邻接表(to,via)、差对象->4点名
    //   pred_name:  中间语句谓词名 ("eqangle"/"eqratio")
    //   final_override: 非空则最后一步结论用它（target 复用）；空则造新中间语句
    //   out: 追加生成的子证明步骤（按拓扑序）
    // 返回：证明 pred(src_obj,dst_obj) 的 Proof*；失败返回 nullptr。
    //   注：链长 k==1 时不产生步骤，直接返回那条已有事实的 Proof*。
    Proof *build_pair_chain(
        DDARSolver *solver,
        const string &pred_name,
        const EquivClasses &eq,
        const unordered_map<string, vector<pair<string, Proof *>>> &adj,
        const unordered_map<string, array<string, 4>> &rep,
        const string &src, const string &dst,
        Proof *final_override,
        vector<ARSimplifier::SubProof> &out)
    {
        if (!eq.has(src) || !eq.has(dst) || !eq.same_class(src, dst))
        {
            return nullptr;
        }
        unordered_set<Proof *> remaining = eq.path_facts(src, dst);
        if (remaining.empty())
        {
            return nullptr;
        }

        vector<string> nodes = {src};
        vector<Proof *> edges;
        string cur = src;
        while (cur != dst)
        {
            bool advanced = false;
            auto it = adj.find(cur);
            if (it != adj.end())
            {
                for (const auto &e : it->second)
                {
                    if (remaining.erase(e.second))
                    {
                        edges.push_back(e.second);
                        nodes.push_back(e.first);
                        cur = e.first;
                        advanced = true;
                        break;
                    }
                }
            }
            if (!advanced)
            {
                return nullptr;
            }
        }

        const size_t k = edges.size();
        if (k == 0)
        {
            return nullptr;
        }
        if (k == 1)
        {
            // 中间事实本就是一条已有的 Tier-2 事实，无需链。
            return edges[0];
        }

        const Problem &problem = solver->problem();
        auto make_proof = [&](const string &obj_a, const string &obj_b) -> Proof *
        {
            const auto &ra = rep.at(obj_a);
            const auto &rb = rep.at(obj_b);
            vector<string> args = {ra[0], ra[1], ra[2], ra[3], rb[0], rb[1], rb[2], rb[3]};
            unique_ptr<Statement> stmt = problem.create_statement(pred_name, args);
            return solver->insert_statement(stmt->normalize());
        };

        Proof *chain = nullptr;
        for (size_t i = 1; i < k; ++i)
        {
            vector<Proof *> deps = (i == 1) ? vector<Proof *>{edges[0], edges[1]}
                                            : vector<Proof *>{chain, edges[i]};
            Proof *conclusion = (i + 1 == k && final_override != nullptr)
                                    ? final_override
                                    : make_proof(nodes[0], nodes[i + 1]);
            out.push_back({conclusion, deps});
            chain = conclusion;
        }
        return chain;
    }

    // 诱导的 Tier-1 事实：两个差对象共享一个分量 -> 另一对分量满足某 Tier-1 关系。
    //   eqratio: (c,d)≡(c,e) -> cong(d,e)；(d,c)≡(e,c) -> cong(d,e)
    //   eqangle: (x,y)≡(x,z) -> para(y,z)；(y,x)≡(z,x) -> para(y,z)
    struct InducedFact
    {
        string ka, kb;                          // 结论涉及的两段/线键
        pair<string, string> pa, pb;            // 两段/线的点名
        Proof *fact;                            // 物化出的 Tier-1 Proof（已 update 成立）
        vector<ARSimplifier::SubProof> support; // 证明该 fact 的子图（含中间 Tier-2 步骤 + 最后一步）
    };

    // 用闭包内的 Tier-2 事实（PredT=EqAngle/EqRatio）物化诱导的 Tier-1 事实。
    //   tier1_pred: 诱导结论谓词名 ("cong"/"para")
    // 仅物化「结论尚不在 closure 内」的诱导事实，避免与既有事实重复/成环。
    template <typename PredT, typename SegT>
    vector<InducedFact> collect_induced(
        DDARSolver *solver, Proof *target,
        const unordered_set<Proof *> &closure,
        const string &tier1_pred)
    {
        auto obj_key = [](const SegT &a, const SegT &b)
        { return seg_key(a) + "|" + seg_key(b); };

        EquivClasses eq;
        unordered_map<string, vector<pair<string, Proof *>>> adj;
        unordered_map<string, array<string, 4>> rep;
        // 差对象 -> 其两分量各为 (段/线键, 点名对)
        unordered_map<string,
                      pair<pair<string, pair<string, string>>,
                           pair<string, pair<string, string>>>>
            comp;

        auto remember = [&](const SegT &a, const SegT &b)
        {
            string k = obj_key(a, b);
            if (rep.find(k) == rep.end())
            {
                rep[k] = {a.left().name(), a.right().name(), b.left().name(), b.right().name()};
                comp[k] = {{seg_key(a), {a.left().name(), a.right().name()}},
                           {seg_key(b), {b.left().name(), b.right().name()}}};
            }
        };
        auto link = [&](const SegT &a, const SegT &b, const SegT &c, const SegT &d, Proof *via)
        {
            remember(a, b);
            remember(c, d);
            const string u = obj_key(a, b), v = obj_key(c, d);
            if (u == v)
                return;
            eq.connect(u, v, via);
            adj[u].push_back({v, via});
            adj[v].push_back({u, via});
        };

        vector<string> objs;
        for (Proof *pf : closure)
        {
            if (pf == target || !pf->is_proved())
                continue;
            const PredT *p = dynamic_cast<const PredT *>(pf->statement().get());
            if (p == nullptr)
                continue;
            const array<SegT, 4> q = pred_quad(*p);
            link(q[0], q[1], q[2], q[3], pf);
            link(q[1], q[0], q[3], q[2], pf);
        }
        for (const auto &kv : rep)
            objs.push_back(kv.first);

        vector<InducedFact> induced;
        unordered_set<string> emitted; // 去重：同一对 (ka,kb)
        const Problem &problem = solver->problem();

        // 枚举同类、共享一个分量的差对象对。
        for (size_t i = 0; i < objs.size(); ++i)
        {
            for (size_t j = i + 1; j < objs.size(); ++j)
            {
                const string &oi = objs[i];
                const string &oj = objs[j];
                if (!eq.same_class(oi, oj))
                    continue;

                const auto &ci = comp[oi];
                const auto &cj = comp[oj];

                // 找出共享分量与「另一对」分量
                string ka, kb;          // 结论两段键
                pair<string, string> pa, pb;
                bool ok = false;
                if (ci.first.first == cj.first.first) // 共享第一分量
                {
                    ka = ci.second.first; pa = ci.second.second;
                    kb = cj.second.first; pb = cj.second.second;
                    ok = true;
                }
                else if (ci.second.first == cj.second.first) // 共享第二分量
                {
                    ka = ci.first.first; pa = ci.first.second;
                    kb = cj.first.first; pb = cj.first.second;
                    ok = true;
                }
                if (!ok || ka == kb)
                    continue;

                // 去重键（无序）
                string dedup = (ka < kb) ? ka + "#" + kb : kb + "#" + ka;
                if (!emitted.insert(dedup).second)
                    continue;

                // 物化 Tier-1 结论 Proof
                vector<string> args = {pa.first, pa.second, pb.first, pb.second};
                unique_ptr<Statement> cstmt = problem.create_statement(tier1_pred, args);
                Proof *fact = solver->insert_statement(cstmt->normalize());

                // 跳过已在闭包内的（即已是既有事实，无需诱导）
                if (closure.count(fact))
                    continue;

                // 构造证明：先证中间 Tier-2 事实 (oi≡oj)，再一步诱导出 fact。
                vector<ARSimplifier::SubProof> support;
                const string t2pred = (tier1_pred == "cong") ? "eqratio" : "eqangle";
                Proof *mid = build_pair_chain(solver, t2pred, eq, adj, rep, oi, oj, nullptr, support);
                if (mid == nullptr)
                    continue;
                // 最后一步：fact <- mid（共享分量消去 -> Tier-1 关系）
                support.push_back({fact, {mid}});

                induced.push_back({ka, kb, pa, pb, fact, std::move(support)});
            }
        }
        return induced;
    }
}

std::pair<bool, std::vector<ARSimplifier::SubProof>> ARSimplifier::run()
{
    using SubProof = ARSimplifier::SubProof;

    // 方向侧：para/perp 统一在方向等价类上处理。
    if (dynamic_cast<const Para *>(_target->statement().get()) ||
        dynamic_cast<const Perp *>(_target->statement().get()))
    {
        return simplify_direction();
    }
    // 长度侧：cong 纯等价类。
    if (dynamic_cast<const Cong *>(_target->statement().get()))
    {
        return simplify_transitive<Cong, Dist>("cong");
    }
    // Tier 2：差对象等价类。eqangle 在角对象上，eqratio 在比对象上。
    if (dynamic_cast<const EqAngle *>(_target->statement().get()))
    {
        return simplify_pair<EqAngle, Slope>("eqangle");
    }
    if (dynamic_cast<const EqRatio *>(_target->statement().get()))
    {
        return simplify_pair<EqRatio, Dist>("eqratio");
    }
    return std::make_pair(false, std::vector<SubProof>{});
}

// 方向侧（para/perp）链式简化。
// para 边偏移 0，perp 边偏移 90；目标 para 要求 src->dst 总偏移 ≡0，perp 要求 ≡90。
// 二元化时每步结论谓词由累计偏移决定。
std::pair<bool, std::vector<ARSimplifier::SubProof>>
ARSimplifier::simplify_direction()
{
    using SubProof = ARSimplifier::SubProof;

    // 取目标的两条线与所需总偏移。
    const Slope *gl;
    const Slope *gr;
    int goal_off;
    if (const Para *p = dynamic_cast<const Para *>(_target->statement().get()))
    {
        gl = &p->left();
        gr = &p->right();
        goal_off = 0;
    }
    else if (const Perp *p = dynamic_cast<const Perp *>(_target->statement().get()))
    {
        gl = &p->left();
        gr = &p->right();
        goal_off = 90;
    }
    else
    {
        return std::make_pair(false, std::vector<SubProof>{});
    }

    const string src = seg_key(*gl);
    const string dst = seg_key(*gr);
    if (src == dst)
    {
        return std::make_pair(false, std::vector<SubProof>{});
    }

    const vector<Proof *> orig_deps = _target->get_dependencies();
    unordered_set<Proof *> closure = dependency_closure(orig_deps);

    unordered_map<string, pair<string, string>> rep;
    auto remember = [&](const Slope &s)
    {
        string k = seg_key(s);
        if (rep.find(k) == rep.end())
        {
            rep[k] = {s.left().name(), s.right().name()};
        }
    };

    // 带偏移的方向等价类 + 带 label 的邻接表。
    EquivClasses eq;
    struct Edge
    {
        string to;
        Proof *via;
        int label; // dir(from) - dir(to)
    };
    unordered_map<string, vector<Edge>> adj;

    // 收集闭包内的 para / perp 事实。
    auto absorb = [&](Proof *pf, const Slope &l, const Slope &r, int label)
    {
        const string a = seg_key(l);
        const string b = seg_key(r);
        if (a == b)
        {
            return;
        }
        remember(l);
        remember(r);
        eq.connect(a, b, pf, label);
        adj[a].push_back({b, pf, label});
        adj[b].push_back({a, pf, -label});
    };

    for (Proof *pf : closure)
    {
        if (pf == _target || !pf->is_proved())
        {
            continue;
        }
        if (const Para *p = dynamic_cast<const Para *>(pf->statement().get()))
        {
            absorb(pf, p->left(), p->right(), 0);
        }
        else if (const Perp *p = dynamic_cast<const Perp *>(pf->statement().get()))
        {
            absorb(pf, p->left(), p->right(), 90);
        }
    }
    remember(*gl);
    remember(*gr);

    // 诱导事实：用闭包内的 eqangle 物化额外的 para（共享分量 -> 另一对平行）。
    // 当成偏移 0 的方向边加入图；记录支撑子图，路径用到时回填。
    unordered_map<Proof *, vector<SubProof>> induced_support;
    {
        vector<InducedFact> induced =
            collect_induced<EqAngle, Slope>(_solver, _target, closure, "para");
        for (InducedFact &f : induced)
        {
            if (rep.find(f.ka) == rep.end())
                rep[f.ka] = f.pa;
            if (rep.find(f.kb) == rep.end())
                rep[f.kb] = f.pb;
            eq.connect(f.ka, f.kb, f.fact, 0);
            adj[f.ka].push_back({f.kb, f.fact, 0});
            adj[f.kb].push_back({f.ka, f.fact, 0});
            induced_support[f.fact] = std::move(f.support);
        }
    }

    // src、dst 必须连通，且总偏移要与目标谓词一致。
    if (!eq.has(src) || !eq.has(dst) || !eq.same_class(src, dst))
    {
        return std::make_pair(false, std::vector<SubProof>{});
    }
    if (eq.offset_between(src, dst) != goal_off)
    {
        // 连通但角度不符（理论上 target 成立则应相符）；保险起见放弃。
        return std::make_pair(false, std::vector<SubProof>{});
    }

    unordered_set<Proof *> facts = eq.path_facts(src, dst);
    if (facts.empty())
    {
        return std::make_pair(false, std::vector<SubProof>{});
    }

    // 在 facts 子图上从 src 走到 dst，得到节点序列、边序列、各边 label。
    unordered_set<Proof *> remaining = facts;
    vector<string> nodes = {src};
    vector<Proof *> edges;
    vector<int> labels;
    string cur = src;
    while (cur != dst)
    {
        bool advanced = false;
        auto it = adj.find(cur);
        if (it != adj.end())
        {
            for (const Edge &e : it->second)
            {
                if (remaining.erase(e.via))
                {
                    edges.push_back(e.via);
                    labels.push_back(e.label);
                    nodes.push_back(e.to);
                    cur = e.to;
                    advanced = true;
                    break;
                }
            }
        }
        if (!advanced)
        {
            return std::make_pair(false, std::vector<SubProof>{});
        }
    }

    const size_t k = edges.size();
    if (k < 2)
    {
        return std::make_pair(false, std::vector<SubProof>{});
    }

    // 二元化：每步结论 = dir(nodes[0]) - dir(nodes[i+1]) 的累计偏移 -> para/perp。
    const Problem &problem = _solver->problem();
    auto mod180 = [](int v)
    { int r = v % 180; return r < 0 ? r + 180 : r; };
    auto make_dir_proof = [&](const string &line_a, const string &line_b, int off) -> Proof *
    {
        const auto &pa = rep.at(line_a);
        const auto &pb = rep.at(line_b);
        vector<string> args = {pa.first, pa.second, pb.first, pb.second};
        const string pred = (off == 0) ? "para" : "perp";
        unique_ptr<Statement> stmt = problem.create_statement(pred, args);
        return _solver->insert_statement(stmt->normalize());
    };

    std::vector<SubProof> subgraph;
    // 路径若用到诱导 para 边，先把支撑子图按拓扑序放最前。
    {
        unordered_set<Proof *> seen;
        for (Proof *e : edges)
        {
            auto it = induced_support.find(e);
            if (it != induced_support.end() && seen.insert(e).second)
            {
                for (const SubProof &sp : it->second)
                    subgraph.push_back(sp);
            }
        }
    }
    Proof *chain = nullptr; // 已累积的 pred(nodes[0], nodes[i])
    int acc = labels[0];    // dir(nodes[0]) - dir(nodes[i]) 的累计
    for (size_t i = 1; i < k; ++i)
    {
        vector<Proof *> deps = (i == 1) ? vector<Proof *>{edges[0], edges[1]}
                                        : vector<Proof *>{chain, edges[i]};
        acc = mod180(acc + labels[i]);

        Proof *conclusion;
        if (i + 1 == k)
        {
            conclusion = _target; // 最后一步即 target（acc 应 == goal_off）
        }
        else
        {
            conclusion = make_dir_proof(nodes[0], nodes[i + 1], acc);
        }

        subgraph.push_back({conclusion, deps});
        chain = conclusion;
    }

    return std::make_pair(true, subgraph);
}

// 对一类「传递性等价关系」谓词（cong）做链式简化。
//   PredT: 谓词类型 (Cong)，需有 left()/right() 返回 SegT
//   SegT:  线段类型 (Dist)，需有 left()/right()/normalize()/name()
//   pred_name: 构造中间语句用的谓词名 ("para" / "cong")
template <typename PredT, typename SegT>
std::pair<bool, std::vector<ARSimplifier::SubProof>>
ARSimplifier::simplify_transitive(const std::string &pred_name)
{
    using SubProof = ARSimplifier::SubProof;

    const PredT *goal = dynamic_cast<const PredT *>(_target->statement().get());
    if (goal == nullptr)
    {
        return std::make_pair(false, std::vector<SubProof>{});
    }

    const string src = seg_key(goal->left());
    const string dst = seg_key(goal->right());
    if (src == dst)
    {
        return std::make_pair(false, std::vector<SubProof>{});
    }

    // target 的原始依赖闭包：建等价类只用闭包内的同类事实 —— 杜绝指向 target 的回边。
    const vector<Proof *> orig_deps = _target->get_dependencies();
    unordered_set<Proof *> closure = dependency_closure(orig_deps);

    // 记录每个键的代表点对（构造中间语句用）。
    unordered_map<string, pair<string, string>> rep;
    auto remember = [&](const SegT &s)
    {
        string k = seg_key(s);
        if (rep.find(k) == rep.end())
        {
            rep[k] = {s.left().name(), s.right().name()};
        }
    };

    // 用闭包内的同类事实构建等价类。
    EquivClasses eq;
    // 同时保留邻接表，便于把对称差得到的事实集排成有序链。
    struct Edge
    {
        string to;
        Proof *via;
    };
    unordered_map<string, vector<Edge>> adj;

    for (Proof *pf : closure)
    {
        if (pf == _target || !pf->is_proved())
        {
            continue;
        }
        const PredT *p = dynamic_cast<const PredT *>(pf->statement().get());
        if (p == nullptr)
        {
            continue;
        }
        const string a = seg_key(p->left());
        const string b = seg_key(p->right());
        if (a == b)
        {
            continue;
        }
        remember(p->left());
        remember(p->right());
        eq.connect(a, b, pf);
        adj[a].push_back({b, pf});
        adj[b].push_back({a, pf});
    }
    remember(goal->left());
    remember(goal->right());

    // 诱导事实：用闭包内的 eqratio 物化额外的 cong（共享分量 -> 另一对相等）。
    // 把这些 cong 当普通边加入图；记录其支撑子图，路径用到时再回填。
    unordered_map<Proof *, vector<SubProof>> induced_support;
    if (pred_name == "cong")
    {
        vector<InducedFact> induced =
            collect_induced<EqRatio, Dist>(_solver, _target, closure, "cong");
        for (InducedFact &f : induced)
        {
            if (rep.find(f.ka) == rep.end())
                rep[f.ka] = f.pa;
            if (rep.find(f.kb) == rep.end())
                rep[f.kb] = f.pb;
            eq.connect(f.ka, f.kb, f.fact);
            adj[f.ka].push_back({f.kb, f.fact});
            adj[f.kb].push_back({f.ka, f.fact});
            induced_support[f.fact] = std::move(f.support);
        }
    }

    // src、dst 不在同一等价类 -> 无法用该类事实链证明，放弃。
    if (!eq.has(src) || !eq.has(dst) || !eq.same_class(src, dst))
    {
        return std::make_pair(false, std::vector<SubProof>{});
    }

    // 取出 src->dst 路径上的事实集（已自动去公共项），再用邻接表排成有序链。
    unordered_set<Proof *> facts = eq.path_facts(src, dst);
    if (facts.empty())
    {
        return std::make_pair(false, std::vector<SubProof>{});
    }

    // 在仅由 facts 构成的子图上，从 src 沿边走到 dst，得到节点序列与边序列。
    unordered_set<Proof *> remaining = facts;
    vector<string> nodes = {src};
    vector<Proof *> edges;
    string cur = src;
    while (cur != dst)
    {
        bool advanced = false;
        auto it = adj.find(cur);
        if (it != adj.end())
        {
            for (const Edge &e : it->second)
            {
                if (remaining.erase(e.via))
                {
                    edges.push_back(e.via);
                    nodes.push_back(e.to);
                    cur = e.to;
                    advanced = true;
                    break;
                }
            }
        }
        if (!advanced)
        {
            // 理论上不应发生（facts 应构成一条 src-dst 路径）；保险起见放弃。
            return std::make_pair(false, std::vector<SubProof>{});
        }
    }

    const size_t k = edges.size();
    if (k < 2)
    {
        // 只有一条边，原 AR 不该有 >阈值 依赖；无可简化。
        return std::make_pair(false, std::vector<SubProof>{});
    }

    // 二元化：逐步合并，生成完整子证明图。
    const Problem &problem = _solver->problem();
    auto make_seg_proof = [&](const string &line_a, const string &line_b) -> Proof *
    {
        const auto &pa = rep.at(line_a);
        const auto &pb = rep.at(line_b);
        vector<string> args = {pa.first, pa.second, pb.first, pb.second};
        unique_ptr<Statement> stmt = problem.create_statement(pred_name, args);
        return _solver->insert_statement(stmt->normalize());
    };

    std::vector<SubProof> subgraph;
    // 路径若用到诱导 cong 边，先把它们的支撑子图按拓扑序放在最前。
    {
        unordered_set<Proof *> seen;
        for (Proof *e : edges)
        {
            auto it = induced_support.find(e);
            if (it != induced_support.end() && seen.insert(e).second)
            {
                for (const SubProof &sp : it->second)
                    subgraph.push_back(sp);
            }
        }
    }
    Proof *chain = nullptr; // 已累积的 pred(nodes[0], nodes[i])
    for (size_t i = 1; i < k; ++i)
    {
        vector<Proof *> deps;
        if (i == 1)
        {
            deps = {edges[0], edges[1]};
        }
        else
        {
            deps = {chain, edges[i]};
        }

        Proof *conclusion;
        if (i + 1 == k)
        {
            conclusion = _target; // 最后一步结论即 target
        }
        else
        {
            conclusion = make_seg_proof(nodes[0], nodes[i + 1]);
        }

        subgraph.push_back({conclusion, deps});
        chain = conclusion;
    }

    return std::make_pair(true, subgraph);
}

// Tier 2：对「一对差对象相等」谓词（eqangle/eqratio）做链式简化。
// 节点 = 差对象(一对有序线/段)，事实 = 一条 eqangle/eqratio（连接两个差对象，含其反向）。
// 纯等价关系，复用 EquivClasses（label 恒 0）。
template <typename PredT, typename SegT>
std::pair<bool, std::vector<ARSimplifier::SubProof>>
ARSimplifier::simplify_pair(const std::string &pred_name)
{
    using SubProof = ARSimplifier::SubProof;

    const PredT *goal = dynamic_cast<const PredT *>(_target->statement().get());
    if (goal == nullptr)
    {
        return std::make_pair(false, std::vector<SubProof>{});
    }

    // 差对象键：两段键用 '|' 连接（有序）。
    auto obj_key = [](const SegT &a, const SegT &b)
    { return seg_key(a) + "|" + seg_key(b); };

    // 差对象 -> 其 4 个点名（构造 8 点中间语句用）。
    unordered_map<string, array<string, 4>> rep;
    auto remember = [&](const SegT &a, const SegT &b)
    {
        string k = obj_key(a, b);
        if (rep.find(k) == rep.end())
        {
            rep[k] = {a.left().name(), a.right().name(), b.left().name(), b.right().name()};
        }
    };

    const array<SegT, 4> gq = pred_quad(*goal);
    const string src = obj_key(gq[0], gq[1]);
    const string dst = obj_key(gq[2], gq[3]);
    if (src == dst)
    {
        return std::make_pair(false, std::vector<SubProof>{});
    }

    const vector<Proof *> orig_deps = _target->get_dependencies();
    unordered_set<Proof *> closure = dependency_closure(orig_deps);

    EquivClasses eq;
    struct Edge
    {
        string to;
        Proof *via;
    };
    unordered_map<string, vector<Edge>> adj;

    auto link = [&](const SegT &a, const SegT &b, const SegT &c, const SegT &d, Proof *via)
    {
        remember(a, b);
        remember(c, d);
        const string u = obj_key(a, b);
        const string v = obj_key(c, d);
        if (u == v)
        {
            return;
        }
        eq.connect(u, v, via);
        adj[u].push_back({v, via});
        adj[v].push_back({u, via});
    };

    for (Proof *pf : closure)
    {
        if (pf == _target || !pf->is_proved())
        {
            continue;
        }
        const PredT *p = dynamic_cast<const PredT *>(pf->statement().get());
        if (p == nullptr)
        {
            continue;
        }
        const array<SegT, 4> q = pred_quad(*p);
        // 正向：(q0,q1) ≡ (q2,q3)
        link(q[0], q[1], q[2], q[3], pf);
        // 反向：(q1,q0) ≡ (q3,q2)（差对象取负，等式仍成立）
        link(q[1], q[0], q[3], q[2], pf);
    }
    remember(gq[0], gq[1]);
    remember(gq[2], gq[3]);

    if (!eq.has(src) || !eq.has(dst) || !eq.same_class(src, dst))
    {
        return std::make_pair(false, std::vector<SubProof>{});
    }

    unordered_set<Proof *> facts = eq.path_facts(src, dst);
    if (facts.empty())
    {
        return std::make_pair(false, std::vector<SubProof>{});
    }

    unordered_set<Proof *> remaining = facts;
    vector<string> nodes = {src};
    vector<Proof *> edges;
    string cur = src;
    while (cur != dst)
    {
        bool advanced = false;
        auto it = adj.find(cur);
        if (it != adj.end())
        {
            for (const Edge &e : it->second)
            {
                if (remaining.erase(e.via))
                {
                    edges.push_back(e.via);
                    nodes.push_back(e.to);
                    cur = e.to;
                    advanced = true;
                    break;
                }
            }
        }
        if (!advanced)
        {
            return std::make_pair(false, std::vector<SubProof>{});
        }
    }

    const size_t k = edges.size();
    if (k < 2)
    {
        return std::make_pair(false, std::vector<SubProof>{});
    }

    const Problem &problem = _solver->problem();
    auto make_pair_proof = [&](const string &obj_a, const string &obj_b) -> Proof *
    {
        const auto &ra = rep.at(obj_a);
        const auto &rb = rep.at(obj_b);
        vector<string> args = {ra[0], ra[1], ra[2], ra[3], rb[0], rb[1], rb[2], rb[3]};
        unique_ptr<Statement> stmt = problem.create_statement(pred_name, args);
        return _solver->insert_statement(stmt->normalize());
    };

    std::vector<SubProof> subgraph;
    Proof *chain = nullptr; // 已累积的 pred(nodes[0], nodes[i])
    for (size_t i = 1; i < k; ++i)
    {
        vector<Proof *> deps = (i == 1) ? vector<Proof *>{edges[0], edges[1]}
                                        : vector<Proof *>{chain, edges[i]};

        Proof *conclusion;
        if (i + 1 == k)
        {
            conclusion = _target; // 最后一步结论即 target
        }
        else
        {
            conclusion = make_pair_proof(nodes[0], nodes[i + 1]);
        }

        subgraph.push_back({conclusion, deps});
        chain = conclusion;
    }

    return std::make_pair(true, subgraph);
}

// ===========================================================================
// run_new(): 轻量级「线/段变量消元」简化。
//
// 思路（比 run() 的等价类链法直接得多）：从前往后逐条扫描前提，遇到一条
// para（slope 通道）或 cong（dist 通道）就地消元：
//   - 它断言两段等价 X≡Y。选一个「幸存者」S、一个「被消者」V：
//       * 若恰有一段在当前 goal 里，则留 goal 里那段作 S；
//       * 否则消去「较大端点更大」的那段（保留较小的）。
//   - 把其余所有前提里的 V 全部换成 S，每条被改写的前提产出一步 DD：
//       P' <- { P, 本条para/cong }；
//     若 goal 里含 V，则同样改写 goal 并记一条 bridge，goal 逐步变简。
//   - 该 para/cong 用完即弃。
// 全部 para/cong 消完后，goal 要么已成平凡式，要么由剩下的非 para/cong 前提
// （被保留的「传递链」）一步证出。bridge 反序回接即得 target。
//
// soundness：para/cong 在 AR 线性系统里就是「变量=变量」的平凡方程，代入消元
// 保持解空间。只要把引用了被消变量的前提全部改写即可——本函数正是对所有其它
// 前提做替换。（此处只处理约定的 7 类谓词；不可分解的前提保持原样留作残余。）
namespace
{
    // 一段分量：两端点名（保序，用于重建语句 args）。
    struct Comp
    {
        string a, b;
    };

    // 把一条语句拆成 (谓词名, 分量序列, 可选 rational 后缀)。
    // 仅支持 run_new 关心的谓词；其余返回 name 为空。
    struct Decomp
    {
        string name;
        vector<Comp> comps;
        bool has_rhs = false;
        Rational rhs;
    };

    Decomp decompose(const Statement *s)
    {
        Decomp d;
        d.name = s->name();
        if (const Para *p = dynamic_cast<const Para *>(s))
        {
            d.comps = {{p->left().left().name(), p->left().right().name()},
                       {p->right().left().name(), p->right().right().name()}};
        }
        else if (const Perp *p = dynamic_cast<const Perp *>(s))
        {
            d.comps = {{p->left().left().name(), p->left().right().name()},
                       {p->right().left().name(), p->right().right().name()}};
        }
        else if (const EqAngle *p = dynamic_cast<const EqAngle *>(s))
        {
            d.comps = {{p->s1().left().name(), p->s1().right().name()},
                       {p->s2().left().name(), p->s2().right().name()},
                       {p->s3().left().name(), p->s3().right().name()},
                       {p->s4().left().name(), p->s4().right().name()}};
        }
        else if (const AConst *p = dynamic_cast<const AConst *>(s))
        {
            d.comps = {{p->s1().left().name(), p->s1().right().name()},
                       {p->s2().left().name(), p->s2().right().name()}};
            d.has_rhs = true;
            d.rhs = p->rhs();
        }
        else if (const Cong *p = dynamic_cast<const Cong *>(s))
        {
            d.comps = {{p->left().left().name(), p->left().right().name()},
                       {p->right().left().name(), p->right().right().name()}};
        }
        else if (const EqRatio *p = dynamic_cast<const EqRatio *>(s))
        {
            d.comps = {{p->left_up().left().name(), p->left_up().right().name()},
                       {p->left_down().left().name(), p->left_down().right().name()},
                       {p->right_up().left().name(), p->right_up().right().name()},
                       {p->right_down().left().name(), p->right_down().right().name()}};
        }
        else if (const RConst *p = dynamic_cast<const RConst *>(s))
        {
            d.comps = {{p->left().left().name(), p->left().right().name()},
                       {p->right().left().name(), p->right().right().name()}};
            d.has_rhs = true;
            d.rhs = p->ratio();
        }
        else
        {
            d.name.clear();
        }
        return d;
    }

    // 由分量序列 + 可选 rhs 重建语句 args（create_statement 用点名字符串）。
    vector<string> rebuild_args(const Decomp &d)
    {
        vector<string> args;
        args.reserve(d.comps.size() * 2 + 1);
        for (const Comp &c : d.comps)
        {
            args.push_back(c.a);
            args.push_back(c.b);
        }
        if (d.has_rhs)
        {
            args.push_back(std::to_string(d.rhs.numerator()) + "/" +
                           std::to_string(d.rhs.denominator()));
        }
        return args;
    }

    // 无序段键：两端点名归一后拼接。
    string comp_key(const Comp &c)
    {
        return (c.a < c.b) ? c.a + "," + c.b : c.b + "," + c.a;
    }
}

std::pair<bool, std::vector<ARSimplifier::SubProof>> ARSimplifier::run_new()
{
    using SubProof = ARSimplifier::SubProof;
    const auto FAIL = std::make_pair(false, std::vector<SubProof>{});

    const Statement *tgt = _target->statement().get();

    // ---- 通道判定：target 决定用 para 还是 cong 做消元 ----
    bool slope_channel;
    if (dynamic_cast<const Para *>(tgt) || dynamic_cast<const Perp *>(tgt) ||
        dynamic_cast<const EqAngle *>(tgt) || dynamic_cast<const AConst *>(tgt))
    {
        slope_channel = true;
    }
    else if (dynamic_cast<const Cong *>(tgt) || dynamic_cast<const EqRatio *>(tgt) ||
             dynamic_cast<const RConst *>(tgt))
    {
        slope_channel = false;
    }
    else
    {
        return FAIL;
    }
    const string elim_pred = slope_channel ? "para" : "cong";

    const Problem &problem = _solver->problem();
    auto make_proof = [&](const Decomp &d) -> Proof *
    {
        unique_ptr<Statement> stmt = problem.create_statement(d.name, rebuild_args(d));
        return _solver->insert_statement(stmt->normalize());
    };

    // 在 decomp 里把所有 key==vkey 的分量替换成 surv；返回是否有改动。
    auto rewrite = [&](const Decomp &d, const string &vkey, const Comp &surv,
                       bool &changed) -> Decomp
    {
        Decomp nd = d;
        changed = false;
        for (Comp &c : nd.comps)
        {
            if (comp_key(c) == vkey)
            {
                c = surv;
                changed = true;
            }
        }
        return nd;
    };
    auto contains_key = [&](const Decomp &d, const string &k) -> bool
    {
        for (const Comp &c : d.comps)
        {
            if (comp_key(c) == k)
                return true;
        }
        return false;
    };
    // c1 是否为「更该被消去」的那段（较大端点更大；端点相同则次端点更大）。
    auto elim_first = [&](const Comp &c1, const Comp &c2) -> bool
    {
        auto hi = [&](const Comp &c)
        {
            Point pa = problem.find_point(c.a), pb = problem.find_point(c.b);
            return (pb > pa) ? pb : pa;
        };
        auto lo = [&](const Comp &c)
        {
            Point pa = problem.find_point(c.a), pb = problem.find_point(c.b);
            return (pb > pa) ? pa : pb;
        };
        Point h1 = hi(c1), h2 = hi(c2);
        if (!(h1 == h2))
            return h1 > h2;
        return lo(c1) > lo(c2);
    };

    // ---- 前提槽位：随消元过程原地更新 ----
    struct Slot
    {
        Proof *proof;
        Decomp dec;
        bool dead = false; // 已被消去/变平凡，退出参与
    };
    vector<Slot> slots;
    for (Proof *pf : _target->get_dependencies())
    {
        slots.push_back({pf, decompose(pf->statement().get()), false});
    }

    Decomp goal = decompose(tgt);
    struct Bridge
    {
        Decomp post; // 更简的新 goal（旧 goal <- {新 goal, via}）
        Proof *via;
    };
    vector<Bridge> bridges;

    std::vector<SubProof> subgraph;
    int consumed = 0;

    // ---- 主循环：逐条消元 ----
    for (size_t i = 0; i < slots.size(); ++i)
    {
        if (slots[i].dead || slots[i].dec.name != elim_pred)
            continue;
        if (slots[i].dec.comps.size() != 2)
        {
            slots[i].dead = true;
            continue;
        }
        const Comp c1 = slots[i].dec.comps[0];
        const Comp c2 = slots[i].dec.comps[1];
        const string k1 = comp_key(c1), k2 = comp_key(c2);
        if (k1 == k2)
        {
            slots[i].dead = true; // 退化 para/cong
            continue;
        }

        // 选被消者 victim / 幸存者 surv。
        const bool in1 = contains_key(goal, k1);
        const bool in2 = contains_key(goal, k2);
        Comp victim, surv;
        if (in1 && !in2)
        {
            surv = c1;
            victim = c2;
        }
        else if (in2 && !in1)
        {
            surv = c2;
            victim = c1;
        }
        else if (elim_first(c1, c2))
        {
            victim = c1;
            surv = c2;
        }
        else
        {
            victim = c2;
            surv = c1;
        }
        const string vkey = comp_key(victim);
        Proof *const via = slots[i].proof;

        // 代入其它所有前提。
        for (size_t j = 0; j < slots.size(); ++j)
        {
            if (j == i || slots[j].dead || slots[j].dec.name.empty())
                continue;
            bool ch = false;
            Decomp nd = rewrite(slots[j].dec, vkey, surv, ch);
            if (!ch)
                continue;
            Proof *np = make_proof(nd);
            if (np == _target)
            {
                // 该前提被改写后恰好等于 target：这一步就是 target 的证明。
                // （此时 goal 必未被改写过——victim 不在 goal 里，故无 bridge 冲突。）
                subgraph.push_back({_target, {slots[j].proof, via}});
                return std::make_pair(true, subgraph);
            }
            if (np->statement()->trivial())
            {
                slots[j].dead = true; // 改写后成平凡式，丢弃
                continue;
            }
            subgraph.push_back({np, {slots[j].proof, via}});
            slots[j].proof = np;
            slots[j].dec = nd;
        }

        // 代入 goal。
        bool gch = false;
        Decomp ng = rewrite(goal, vkey, surv, gch);
        if (gch)
        {
            bridges.push_back({ng, via});
            goal = ng;
        }

        slots[i].dead = true;
        ++consumed;
    }

    if (consumed == 0)
    {
        return FAIL; // 没有可消的 para/cong
    }

    // ---- 残余前提（已无 para/cong）：证最终 goal 用 ----
    vector<Proof *> residual;
    for (const Slot &s : slots)
    {
        if (!s.dead)
            residual.push_back(s.proof);
    }

    // 第 t 个 goal 版本对应的 Proof*：版本 0 = target，版本 t = bridges[t-1].post。
    auto gproof = [&](size_t t) -> Proof *
    {
        return (t == 0) ? _target : make_proof(bridges[t - 1].post);
    };

    if (bridges.empty())
    {
        // goal 未变：直接由残余前提证 target（保留的传递链）。
        vector<Proof *> rdeps;
        for (Proof *p : residual)
            if (p != _target)
                rdeps.push_back(p);
        if (rdeps.empty())
            return FAIL;
        subgraph.push_back({_target, rdeps});
        return std::make_pair(true, subgraph);
    }

    // (a) 证最终 goal Gk：平凡则自动成立；若它恰好就是某条残余前提（goal 与该
    //     前提被同一组替换改写成了同一式子，dedup 后是同一 Proof*），则已被证明，
    //     无需再证——否则会产生 X <- [X] 的自环；其余情况由残余前提一步证出。
    Proof *gk = gproof(bridges.size());
    const bool gk_available =
        gk->statement()->trivial() ||
        std::find(residual.begin(), residual.end(), gk) != residual.end();
    if (gk != _target && !gk_available)
    {
        vector<Proof *> rdeps;
        for (Proof *p : residual)
            if (p != gk)
                rdeps.push_back(p);
        if (rdeps.empty())
            return FAIL;
        subgraph.push_back({gk, rdeps});
    }

    // (b) bridge 反序回接：G_{t} <- { G_{t+1}, via }，最后一条结论即 target。
    for (size_t t = bridges.size(); t-- > 0;)
    {
        Proof *pre = gproof(t);
        Proof *post = gproof(t + 1);
        if (pre == post)
            continue;
        subgraph.push_back({pre, {post, bridges[t].via}});
    }

    return std::make_pair(true, subgraph);
}
