#include "type/point.hpp"
#include "type/slope.hpp"
#include "matcher.hpp"
#include "theorem.hpp"
#include "problem.hpp"
#include "solver/ddar.hpp"
#include <iostream>
#include <map>
#include <vector>
#include <chrono>
#include <tuple>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

using StatementTokens = vector<string>;
using DepGraph = vector<tuple<StatementTokens, vector<StatementTokens>, string>>;

// Application 信息结构体
struct ApplicationInfo
{
    vector<string> premises;        // 前提
    vector<string> conclusions;     // 结论
    vector<size_t> proved_premises; // 已证明的前提索引
};

std::string join(const StatementTokens &tokens)
{
    std::ostringstream oss;
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        if (i != 0)
            oss << " "; // 添加空格分隔符
        oss << tokens[i];
    }
    return oss.str();
}

// ===========================================================================
// simplify_ar 前后依赖图分析。
//   - detect_cycles: 在依赖图中查找循环论证（有向环），返回所有找到的环。
//     每个节点 = 语句字符串，边 node -> dep 表示 node 依赖 dep。正常证明图应为
//     DAG；若出现环，说明存在循环论证（A 依赖 B 且 B（传递）依赖 A）。
//   - dep_count_distribution: 统计每个节点的直接依赖数目分布，用于对比简化前后。
// ===========================================================================

// 从 DepGraph 构建邻接表：node 字符串 -> 其依赖 node 字符串列表。
static std::unordered_map<std::string, std::vector<std::string>>
build_adjacency(const DepGraph &graph)
{
    std::unordered_map<std::string, std::vector<std::string>> adj;
    for (const auto &entry : graph)
    {
        const std::string node = join(std::get<0>(entry));
        std::vector<std::string> deps;
        deps.reserve(std::get<1>(entry).size());
        for (const auto &dep : std::get<1>(entry))
        {
            deps.push_back(join(dep));
        }
        adj[node] = std::move(deps);
    }
    return adj;
}

// 用三色 DFS 查找有向环。发现回边（指向当前递归栈中节点）即提取环节。
// 返回去重后的环列表（每个环以其构成节点集合去重，避免同一环被多次报告）。
static std::vector<std::vector<std::string>>
detect_cycles(const std::unordered_map<std::string, std::vector<std::string>> &adj)
{
    enum Color
    {
        WHITE,
        GRAY,
        BLACK
    };
    std::unordered_map<std::string, int> color;
    std::vector<std::string> path; // 当前递归路径（栈）
    std::vector<std::vector<std::string>> cycles;
    std::unordered_set<std::string> reported; // 已报告环的规范化签名，去重

    std::function<void(const std::string &)> dfs = [&](const std::string &u)
    {
        color[u] = GRAY;
        path.push_back(u);

        auto it = adj.find(u);
        if (it != adj.end())
        {
            for (const std::string &v : it->second)
            {
                // 依赖节点可能不是图中的键（如初始前提），视为叶子跳过。
                if (adj.find(v) == adj.end())
                {
                    continue;
                }
                int c = color.count(v) ? color[v] : WHITE;
                if (c == WHITE)
                {
                    dfs(v);
                }
                else if (c == GRAY)
                {
                    // 找到回边 u -> v：环为 path 中从 v 开始到 u 结束的段。
                    auto start = std::find(path.begin(), path.end(), v);
                    std::vector<std::string> cycle(start, path.end());

                    // 规范化签名去重（按最小节点旋转 + 内容拼接）。
                    std::vector<std::string> sig = cycle;
                    auto minit = std::min_element(sig.begin(), sig.end());
                    std::rotate(sig.begin(), minit, sig.end());
                    std::string key;
                    for (const auto &s : sig)
                        key += s + " || ";
                    if (reported.insert(key).second)
                    {
                        cycles.push_back(std::move(cycle));
                    }
                }
            }
        }

        path.pop_back();
        color[u] = BLACK;
    };

    for (const auto &kv : adj)
    {
        if (!color.count(kv.first) || color[kv.first] == WHITE)
        {
            dfs(kv.first);
        }
    }
    return cycles;
}

// 完整打印依赖图：每个结论节点，及其推导规则、直接依赖列表。
static void print_dep_graph(const DepGraph &graph, const std::string &label)
{
    cout << "----- [dep_graph dump: " << label << "] (" << graph.size() << " nodes) -----" << endl;
    for (const auto &entry : graph)
    {
        const std::string node = join(std::get<0>(entry));
        const auto &deps = std::get<1>(entry);
        const std::string &reason = std::get<2>(entry);
        cout << node << "   [reason: " << reason << "]" << endl;
        for (const auto &dep : deps)
        {
            cout << "    <- " << join(dep) << endl;
        }
    }
    cout << "----------------------------------------------------------" << endl;
}

// 检测同一结论（谓词字符串）在图中出现多次、且依赖集合不同的情况。
// DepGraph 是 vector，不去重同名结论；若同一结论被证明了两次且依赖链不同，
// 说明存在冗余或互相矛盾的证明分支（例如 simplify_ar 改写后残留旧节点）。
// 返回值：每组重复结论对应其在 graph 中的下标列表（同组内依赖不完全相同）。
static std::vector<std::vector<size_t>>
detect_duplicate_nodes(const DepGraph &graph)
{
    std::unordered_map<std::string, std::vector<size_t>> occurrences;
    for (size_t i = 0; i < graph.size(); ++i)
    {
        occurrences[join(std::get<0>(graph[i]))].push_back(i);
    }

    auto dep_signature = [&](size_t idx)
    {
        std::vector<std::string> deps;
        for (const auto &dep : std::get<1>(graph[idx]))
            deps.push_back(join(dep));
        std::sort(deps.begin(), deps.end());
        return deps;
    };

    std::vector<std::vector<size_t>> duplicates;
    for (const auto &kv : occurrences)
    {
        if (kv.second.size() < 2)
            continue;

        const auto first_sig = dep_signature(kv.second[0]);
        bool differs = false;
        for (size_t k = 1; k < kv.second.size(); ++k)
        {
            if (dep_signature(kv.second[k]) != first_sig)
            {
                differs = true;
                break;
            }
        }
        if (differs)
            duplicates.push_back(kv.second);
    }
    return duplicates;
}

// 打印一个依赖图的分析结果：循环论证 + 依赖数目分布 + 重复结论（不同依赖）检测。
static void analyze_dep_graph(const DepGraph &graph, const std::string &label)
{
    auto adj = build_adjacency(graph);

    // node 字符串 -> 该结论的证明理由（reason），用于在环里标注每个节点由哪
    // 个阶段/规则得到（如 "AR Simplified" 来自 run_new 改写，"AR with Slope"
    // 是原生 AR）。这样看环就能立刻判断是哪个环节引入的回边。
    std::unordered_map<std::string, std::string> reason_of;
    for (const auto &entry : graph)
    {
        reason_of[join(std::get<0>(entry))] = std::get<2>(entry);
    }

    cout << "===== [dep_graph analysis: " << label << "] =====" << endl;
    cout << "nodes (proved statements): " << graph.size() << endl;

    // ---- 循环论证检测 ----
    auto cycles = detect_cycles(adj);
    if (cycles.empty())
    {
        cout << "cycles: none (graph is acyclic)" << endl;
    }
    else
    {
        cout << "cycles: " << cycles.size() << " circular-reasoning cycle(s) found!" << endl;
        for (size_t i = 0; i < cycles.size(); ++i)
        {
            cout << "  cycle #" << i << " (length " << cycles[i].size() << "):" << endl;
            for (const auto &node : cycles[i])
            {
                auto rit = reason_of.find(node);
                const std::string reason = (rit != reason_of.end()) ? rit->second : "?";
                cout << "    -> " << node << "   [reason: " << reason << "]" << endl;
                // 同时打印该节点的完整直接依赖，便于看清回边从哪条依赖闭合。
                auto ait = adj.find(node);
                if (ait != adj.end())
                {
                    for (const auto &dep : ait->second)
                    {
                        cout << "         dep: " << dep << endl;
                    }
                }
            }
            // 闭合回到起点，直观展示环。
            cout << "    -> " << cycles[i].front() << " (back to start)" << endl;
        }
    }

    // ---- 重复结论检测（同一谓词被不同依赖证明两次）----
    auto duplicates = detect_duplicate_nodes(graph);
    if (duplicates.empty())
    {
        cout << "duplicate conclusions: none" << endl;
    }
    else
    {
        cout << "duplicate conclusions: " << duplicates.size()
             << " statement(s) proved multiple times with different dependencies!" << endl;
        for (const auto &group : duplicates)
        {
            const std::string stmt = join(std::get<0>(graph[group[0]]));
            cout << "  statement: " << stmt << " (" << group.size() << " occurrences)" << endl;
            for (size_t idx : group)
            {
                const std::string &reason = std::get<2>(graph[idx]);
                cout << "    occurrence [reason: " << reason << "]" << endl;
                for (const auto &dep : std::get<1>(graph[idx]))
                {
                    cout << "      <- " << join(dep) << endl;
                }
            }
        }
    }

    // ---- 依赖数目分布 ----
    std::map<size_t, size_t> hist; // 依赖数 -> 节点个数
    size_t total_deps = 0;
    size_t max_deps = 0;
    for (const auto &entry : graph)
    {
        size_t n = std::get<1>(entry).size();
        ++hist[n];
        total_deps += n;
        max_deps = std::max(max_deps, n);
    }
    double mean = graph.empty() ? 0.0 : static_cast<double>(total_deps) / graph.size();
    cout << "dependency-count distribution (num_deps: num_nodes):" << endl;
    for (const auto &kv : hist)
    {
        cout << "  " << kv.first << ": " << kv.second << endl;
    }
    cout << "  [total_deps=" << total_deps << " max_deps=" << max_deps
         << " mean=" << mean << "]" << endl;

    // ---- 机器可读汇总（JSONL，一行一个 run）----
    // 以固定前缀 DEPGRAPH_JSONL 打头，分析脚本可用该前缀从混杂的 stdout/txt
    // 里精确过滤出这些行；其余人类可读输出不受影响。dep_hist 的键为「前提数」，
    // 值为「拥有该前提数的结论节点个数」，即前提（依赖）数分布。
    cout << "DEPGRAPH_JSONL {"
         << "\"label\":\"" << label << "\""
         << ",\"nodes\":" << graph.size()
         << ",\"cycles\":" << cycles.size()
         << ",\"duplicate_conclusions\":" << duplicates.size()
         << ",\"total_deps\":" << total_deps
         << ",\"max_deps\":" << max_deps
         << ",\"mean\":" << mean
         << ",\"dep_hist\":{";
    bool first = true;
    for (const auto &kv : hist)
    {
        if (!first)
            cout << ",";
        cout << "\"" << kv.first << "\":" << kv.second;
        first = false;
    }
    cout << "}}" << endl;

    cout << "=================================================" << endl;
}

// 将定理字符串列表转换为 CustomRule 列表
// 每个定理字符串格式: "rule_name|premise1_type arg1 arg2...,premise2_type arg1 arg2...|conclusion1_type arg1 arg2...,conclusion2_type arg1 arg2..."
vector<CustomRule> parse_custom_theorems(const vector<string> &theorem_strings)
{
    vector<CustomRule> rules;

    for (const auto &thm_str : theorem_strings)
    {
        CustomRule rule;

        // 按 | 分割: rule_name | premises | conclusions [| guards]
        size_t first_pipe = thm_str.find('|');
        size_t second_pipe = thm_str.find('|', first_pipe + 1);
        size_t third_pipe = thm_str.find('|', second_pipe + 1);

        if (first_pipe == string::npos || second_pipe == string::npos)
            continue;

        rule.name = thm_str.substr(0, first_pipe);
        rule.rule = rule.name;

        string premises_str = thm_str.substr(first_pipe + 1, second_pipe - first_pipe - 1);
        string conclusions_str, guards_str;
        if (third_pipe != string::npos) {
            conclusions_str = thm_str.substr(second_pipe + 1, third_pipe - second_pipe - 1);
            guards_str = thm_str.substr(third_pipe + 1);
        } else {
            conclusions_str = thm_str.substr(second_pipe + 1);
        }

        // 解析前提（用逗号分隔）
        stringstream premises_ss(premises_str);
        string premise;
        while (getline(premises_ss, premise, ','))
        {
            // 去除首尾空格
            premise.erase(0, premise.find_first_not_of(" \t\n\r"));
            premise.erase(premise.find_last_not_of(" \t\n\r") + 1);

            if (premise.empty())
                continue;

            // 解析单个谓词：type arg1 arg2 ...
            stringstream pred_ss(premise);
            vector<string> tokens;
            string token;
            while (pred_ss >> token)
                tokens.push_back(token);

            if (tokens.size() >= 2)
            {
                Stmt stmt;
                stmt.first = tokens[0];
                stmt.second = vector<string>(tokens.begin() + 1, tokens.end());
                rule.premises.push_back(stmt);
            }
        }

        // 解析结论（用逗号分隔）
        stringstream conclusions_ss(conclusions_str);
        string conclusion;
        while (getline(conclusions_ss, conclusion, ','))
        {
            // 去除首尾空格
            conclusion.erase(0, conclusion.find_first_not_of(" \t\n\r"));
            conclusion.erase(conclusion.find_last_not_of(" \t\n\r") + 1);

            if (conclusion.empty())
                continue;

            // 解析单个谓词：type arg1 arg2 ...
            stringstream pred_ss(conclusion);
            vector<string> tokens;
            string token;
            while (pred_ss >> token)
                tokens.push_back(token);

            if (tokens.size() >= 2)
            {
                Stmt stmt;
                stmt.first = tokens[0];
                stmt.second = vector<string>(tokens.begin() + 1, tokens.end());
                rule.conclusions.push_back(stmt);
            }
        }

        // 解析守卫谓词（可选第四段）：格式同前提，逗号分隔
        if (!guards_str.empty()) {
            stringstream guards_ss(guards_str);
            string guard;
            while (getline(guards_ss, guard, ',')) {
                guard.erase(0, guard.find_first_not_of(" \t\n\r"));
                guard.erase(guard.find_last_not_of(" \t\n\r") + 1);
                if (guard.empty()) continue;
                stringstream pred_ss(guard);
                vector<string> tokens;
                string token;
                while (pred_ss >> token)
                    tokens.push_back(token);
                if (tokens.size() >= 2) {
                    Stmt stmt;
                    stmt.first = tokens[0];
                    stmt.second = vector<string>(tokens.begin() + 1, tokens.end());
                    rule.guards.push_back(stmt);
                }
            }
        }

        if (!rule.premises.empty() && !rule.conclusions.empty())
            rules.push_back(rule);
    }

    return rules;
}

extern "C"
{
    pair<bool, DepGraph> run_ddar(
        string name,
        vector<tuple<string, double, double>> points,
        vector<pair<string, vector<string>>> premises,
        vector<pair<string, vector<string>>> goals,
        int max_level,
        py::dict config)
    {
        // Convert Python dict to C++ map
        map<string, bool> config_map;
        for (auto item : config)
        {
            config_map[item.first.cast<string>()] = item.second.cast<bool>();
        }

        Problem problem;
        problem.load_from_data(name, points, premises, goals);

        DDARSolver solver(&problem, config_map);
        solver.run(max_level);

        // 简化前后依赖图分析（循环论证检测 + 依赖数分布对比），
        // 仅在 config 里显式打开 "analyze_ar_simplify" 时执行，避免刷屏。
        // const bool do_analyze = true;
        //     config_map.count("analyze_ar_simplify") && config_map.at("analyze_ar_simplify");
        // DepGraph dep_graph_before;
        // if (do_analyze)
        // {
        //     dep_graph_before = solver.dependency_graph();
        // }

        // 求解结束后，对前提过多的 AR 推导尝试简化
        solver.simplify_ar();

        DepGraph dep_graph = solver.dependency_graph();

        // if (do_analyze)
        // {
        //     print_dep_graph(dep_graph_before, "before simplify_ar");
        //     print_dep_graph(dep_graph, "after simplify_ar");
        //     analyze_dep_graph(dep_graph_before, "before simplify_ar");
        //     analyze_dep_graph(dep_graph, "after simplify_ar");
        // }

        return make_pair(solver.is_solved(), dep_graph);
    }

    pair<bool, DepGraph> run_ddar_with_custom_theorems(
        string name,
        vector<tuple<string, double, double>> points,
        vector<pair<string, vector<string>>> premises,
        vector<pair<string, vector<string>>> goals,
        vector<string> custom_theorem_strings,
        int max_level,
        py::dict config)
    {
        // Convert Python dict to C++ map
        map<string, bool> config_map;
        for (auto item : config)
        {
            config_map[item.first.cast<string>()] = item.second.cast<bool>();
        }

        Problem problem;
        problem.load_from_data(name, points, premises, goals);

        DDARSolver solver(&problem, config_map);

        // 添加自定义定理
        if (!custom_theorem_strings.empty())
        {
            vector<CustomRule> custom_rules = parse_custom_theorems(custom_theorem_strings);
            solver.add_custom_theorems(custom_rules);
        }

        solver.run(max_level);

        // 简化前后依赖图分析（循环论证检测 + 依赖数分布对比），
        // 仅在 config 里显式打开 "analyze_ar_simplify" 时执行，避免刷屏。
        // const bool do_analyze =
        //     config_map.count("analyze_ar_simplify") && config_map.at("analyze_ar_simplify");
        // DepGraph dep_graph_before;
        // if (do_analyze)
        // {
        //     dep_graph_before = solver.dependency_graph();
        // }

        // 求解结束后，对前提过多的 AR 推导尝试简化
        solver.simplify_ar();

        DepGraph dep_graph = solver.dependency_graph();

        // if (do_analyze)
        // {
        //     print_dep_graph(dep_graph_before, "before simplify_ar");
        // print_dep_graph(dep_graph, "after simplify_ar");
        //     analyze_dep_graph(dep_graph_before, "before simplify_ar");
        //     analyze_dep_graph(dep_graph, "after simplify_ar");
        // }

        return make_pair(solver.is_solved(), dep_graph);
    }

    vector<string> get_possible_goals(
        string name,
        vector<tuple<string, double, double>> points,
        vector<pair<string, vector<string>>> premises,
        py::dict config)
    {
        // Convert Python dict to C++ map
        map<string, bool> config_map;
        for (auto item : config)
        {
            config_map[item.first.cast<string>()] = item.second.cast<bool>();
        }

        Problem problem;
        problem.load_from_data(name, points, premises, {});

        DDARSolver solver(&problem, config_map);
        solver.run(500);

        vector<string> possible_goals;

        for (const auto &app : solver.applications())
        {
            if (app.state() != ApplicationState::PENDING)
            {
                continue;
            }
            for (const auto &c : app.conclusions())
            {
                if (c->statement()->name() == "secant" || c->is_proved())
                {
                    continue;
                }
                possible_goals.push_back(c->statement()->to_string());
            }
        }
        std::sort(possible_goals.begin(), possible_goals.end());
        auto it = std::unique(possible_goals.begin(), possible_goals.end());
        possible_goals.erase(it, possible_goals.end());
        return possible_goals;
    }

    pair<vector<int>, vector<ApplicationInfo>> check_goals(
        string name,
        vector<tuple<string, double, double>> points,
        vector<pair<string, vector<string>>> premises,
        vector<pair<string, vector<string>>> goals,
        int max_level,
        py::dict config,
        bool need_applications = false)
    {
        // Convert Python dict to C++ map
        map<string, bool> config_map;
        for (auto item : config)
        {
            config_map[item.first.cast<string>()] = item.second.cast<bool>();
        }

        Problem problem;
        problem.load_from_data(name, points, premises, goals);
        DDARSolver solver(&problem, config_map);
        solver.run(max_level);

        vector<int> solved_indices;
        const auto &solver_goals = solver.goals();
        for (size_t i = 0; i < solver_goals.size(); ++i)
        {
            if (solver_goals[i]->is_proved())
            {
                solved_indices.push_back(static_cast<int>(i));
            }
        }

        vector<ApplicationInfo> pending_apps;
        if (need_applications)
        {
            for (const auto &app : solver.applications())
            {
                if (app.state() != ApplicationState::PENDING)
                {
                    continue;
                }

                ApplicationInfo info;
                vector<size_t> proved_indices;

                // 收集前提和已证明的前提索引
                for (size_t i = 0; i < app.hypotheses().size(); ++i)
                {
                    const auto *hyp = app.hypotheses()[i];
                    info.premises.push_back(hyp->statement()->to_string());
                    if (hyp->is_proved())
                    {
                        proved_indices.push_back(i);
                    }
                }

                // 收集结论
                for (const auto *conc : app.conclusions())
                {
                    info.conclusions.push_back(conc->statement()->to_string());
                }

                info.proved_premises = proved_indices;
                pending_apps.push_back(info);
            }
        }

        return make_pair(solved_indices, pending_apps);
    }
}

PYBIND11_MODULE(DDAR, m)
{
    py::class_<ApplicationInfo>(m, "ApplicationInfo")
        .def(py::init<>())
        .def_readwrite("premises", &ApplicationInfo::premises)
        .def_readwrite("conclusions", &ApplicationInfo::conclusions)
        .def_readwrite("proved_premises", &ApplicationInfo::proved_premises);

    m.def("run_ddar", &run_ddar, "Run DDAR with given problem",
          py::arg("name"),
          py::arg("points"),
          py::arg("premises"),
          py::arg("goals"),
          py::arg("max_level"),
          py::arg("config"));

    m.def("run_ddar_with_custom_theorems", &run_ddar_with_custom_theorems,
          "Run DDAR with custom theorems",
          py::arg("name"),
          py::arg("points"),
          py::arg("premises"),
          py::arg("goals"),
          py::arg("custom_theorem_strings"),
          py::arg("max_level"),
          py::arg("config"));

    m.def("get_possible_goals", &get_possible_goals,
          "Get all possible goals for a given problem",
          py::arg("name"),
          py::arg("points"),
          py::arg("premises"),
          py::arg("config"));

    m.def("check_goals", &check_goals,
          "Check which goals are solved and return their indices",
          py::arg("name"),
          py::arg("points"),
          py::arg("premises"),
          py::arg("goals"),
          py::arg("max_level"),
          py::arg("config"),
          py::arg("need_applications") = false);
}
