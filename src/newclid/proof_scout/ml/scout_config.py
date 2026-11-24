# scout_config.py
import os

# ================= 路径配置 =================
SOURCE_DIR = "/c23474/home/duzhengtong/Discovery-GenesisGeo"

RAW_POOL_FILE = os.path.join(
    SOURCE_DIR, "benchmarks/configuration_clauses6_samples1M_problems.txt"
)  # 原始的大题库
STATE_FILE = os.path.join(
    SOURCE_DIR, "datasets/proof_scout/pipeline_state.csv"
)  # 核心：状态管理表
MODEL_DIR = os.path.join(SOURCE_DIR, "datasets/proof_scout/checkpoints")  # 模型存放目录
MODEL_PATH = os.path.join(MODEL_DIR, "geo_value_model.pth")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")

# ================= 初始基准数据 (防止遗忘) =================
# 你的 12000 条初始数据文件
BASELINE_DATA_FILE = os.path.join(
    SOURCE_DIR, "datasets/proof_scout/configuration_clauses6_samples5M_problems_20k.txt"
)
# 对应的已解出 ID 列表
BASELINE_SOLVED_IDS = os.path.join(SOURCE_DIR, "datasets/proof_scout/solved_ids.txt")

# ================= 求解器与存储配置 =================
# Legacy bash evaluation entrypoints were removed. Configure this explicitly if
# proof_scout is wired to a project-specific solver script.
SOLVER_SCRIPT = ""
TEMP_INPUT_FILE = os.path.join(SOURCE_DIR, "benchmarks/tmp_problems.txt")
SOLVER_OUTPUT_JSON = os.path.join(
    SOURCE_DIR, "datasets/success_proofs/tmp_problems.txt.jsonl"
)

# [新增] 永久保存所有成功证明的主文件路径
MASTER_PROOFS_FILE = os.path.join(
    SOURCE_DIR, "datasets", "success_proofs", "all_proof_traces.jsonl"
)

# [新增] 用于记录每一轮 Batch 的详细统计指标
METRICS_FILE = os.path.join(SOURCE_DIR, "datasets/proof_scout/pipeline_metrics.csv")

# ================= 流程参数 =================
BATCH_SIZE = 1000  # 环节1: 每次处理 500 题
EXPLORATION_RATE = 0.1  # 环节2: 10% 的随机探索概率
FILTER_THRESHOLD = 0.4  # 环节2: 初始筛选阈值

RETROSPECT_INTERVAL = 5  # 环节5: 每处理 5 个 Batch，回捞一次
RETROSPECT_THRESHOLD = 0.5  # 环节5: 回捞时使用稍高的阈值（更严谨）

DEVICE = "cuda"  # 或 "cpu"
