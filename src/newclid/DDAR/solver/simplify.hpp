#pragma once

#include <vector>
#include <string>
#include <utility>

class DDARSolver;
class Proof;

// AR 简化器
//
// 针对一条由 AR (Algebraic Reasoning) 推导得到、且前提数量较多的语句，
// 尝试用若干步简单的 DD (Deductive Database) 二元推理替代原本的单步 AR 推导。
//
// 直接在 solver 的证明体系内工作：所有语句以 Proof* 标识，等价谓词
// (如 coll a b c / coll a c b) 经由 solver 的 _statement_proofs 归一到同一
// Proof 对象。
class ARSimplifier
{
public:
    // 子证明图的一个条目：conclusion 由 deps 经「一步」DD 推理推出。
    // 简化器保证 deps 仅指向 D1-Dn（target 原依赖）的传递依赖闭包内的语句，
    // 或本子图中更靠前的中间语句 —— 从而绝不引入指向 target/其后代的回边。
    struct SubProof
    {
        Proof *conclusion;
        std::vector<Proof *> deps;
    };

    // solver: 当前求解器（用于查找/构造中间语句对应的 Proof）
    // target: 待简化的已证明语句（其 AR 推导前提过多）
    ARSimplifier(DDARSolver *solver, Proof *target);

    // 尝试用若干步 DD 推理重新证明 target。
    //
    // 返回 {success, subgraph}:
    //   - success == true:  subgraph 为重证 target 的完整子证明图，按拓扑序排列
    //                       （靠前为新引入的中间语句，最后一条的 conclusion 即 target）。
    //                       每条 conclusion 由 deps 一步推出。
    //   - success == false: 简化失败，subgraph 无意义，调用方应保持原 AR 推导不变。
    std::pair<bool, std::vector<SubProof>> run();

    std::pair<bool, std::vector<SubProof>> run_new();

private:
    // 方向侧（para/perp）的链式简化：在带偏移的方向等价类上工作，
    // 每个二元步骤按累计偏移(0->para, 90->perp)选择结论谓词。
    std::pair<bool, std::vector<SubProof>> simplify_direction();

    // 对一类「传递性等价关系」谓词（cong）做链式简化的通用实现。
    //   PredT: 谓词类型 (Cong)
    //   SegT:  线段类型 (Dist)
    //   pred_name: 构造中间语句用的谓词名 ("cong")
    template <typename PredT, typename SegT>
    std::pair<bool, std::vector<SubProof>> simplify_transitive(const std::string &pred_name);

    // Tier 2：对「一对差对象相等」谓词（eqangle/eqratio）做链式简化。
    //   PredT: 谓词类型 (EqAngle / EqRatio)
    //   SegT:  线/线段类型 (Slope / Dist)
    //   pred_name: 构造中间语句用的谓词名 ("eqangle" / "eqratio")
    template <typename PredT, typename SegT>
    std::pair<bool, std::vector<SubProof>> simplify_pair(const std::string &pred_name);

    DDARSolver *_solver;
    Proof *_target;
};
