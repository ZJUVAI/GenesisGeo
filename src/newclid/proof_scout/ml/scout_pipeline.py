# pipeline.py
import pandas as pd
import os
import random
import newclid.proof_scout.ml.scout_config as config
import newclid.proof_scout.ml.model_utils as model_utils
import torch
import subprocess
import json
import csv
from datetime import datetime


class PipelineManager:
    def __init__(self):
        self.state_file = config.STATE_FILE
        self.model = None
        self.preprocessor = None

        # 初始化或加载状态
        if os.path.exists(self.state_file):
            print("加载现有管线状态...")
            self.df = pd.read_csv(self.state_file)
        else:
            print("初始化管线：从题库加载数据...")
            self._init_from_raw_pool()

        self._load_model_if_exists()

    def _init_from_raw_pool(self):
        """
        将原始txt转换为DataFrame管理，状态初始化。
        解析格式：
        Line 1: ID
        Line 2: Content
        ...
        """
        data = []

        print(f"正在从 {config.RAW_POOL_FILE} 读取数据...")

        with open(config.RAW_POOL_FILE, "r", encoding="utf-8") as f:
            # 使用 iter(f) 创建迭代器，方便一次取两行
            lines = iter(f)
            while True:
                try:
                    # 1. 读取 ID 行
                    uid_line = next(lines).strip()
                    # 2. 读取 Content 行
                    # 注意：如果文件只有 ID 没有 Content，这里会抛出 StopIteration，被 except 捕获
                    text_line = next(lines).strip()

                    data.append({"uid": uid_line, "text": text_line})

                except StopIteration:
                    # 文件读取完毕
                    break

        self.df = pd.DataFrame(data)

        # 状态列定义初始化
        self.df["status"] = (
            "pending"  # pending (待处理), skipped (跳过), solved (解出), failed (未解出)
        )
        self.df["batch_id"] = -1  # 属于哪个批次
        self.df["pred_score"] = 0.0  # 模型打分
        self.df["label"] = -1  # 真实结果: 1(解出), 0(未解出), -1(未知)
        self.df["is_explored"] = False  # 是否是随机探索选中的

        print(f"初始化完成，共加载 {len(self.df)} 条题目。")
        self.save_state()

    def _load_model_if_exists(self):
        if os.path.exists(config.MODEL_PATH) and os.path.exists(config.VOCAB_PATH):
            print("加载最新模型...")
            prep = model_utils.GeoPreprocessor()
            prep.load_vocab(config.VOCAB_PATH)
            model = model_utils.GeometryValueModel(len(prep.token2idx)).to(
                config.DEVICE
            )
            model.load_state_dict(torch.load(config.MODEL_PATH))
            model.eval()
            self.model = model
            self.preprocessor = prep
        else:
            print("警告：未找到已训练模型，将在第一轮数据收集后初始化。")

    def save_state(self):
        self.df.to_csv(self.state_file, index=False)

    # --- 核心环节 1 & 2: 批次处理 ---
    def process_next_batch(self, batch_id):
        print(f"\n>>> [Batch {batch_id}] 开始处理...")

        # 1. 选取 Pending 题目
        mask_pending = self.df["status"] == "pending"
        # 取前 BATCH_SIZE 个
        target_indices = self.df[mask_pending].head(config.BATCH_SIZE).index

        input_count = len(target_indices)  # [统计] 处理多少数据
        if input_count == 0:
            print("题库已空，所有题目处理完毕。")
            return False

        current_texts = self.df.loc[target_indices, "text"].tolist()

        # 2. 如果有模型，进行推理筛选
        to_solve_indices = []
        if self.model:
            scores = model_utils.predict_batch(
                self.model, self.preprocessor, current_texts
            )
            self.df.loc[target_indices, "pred_score"] = scores

            for idx, score in zip(target_indices, scores):
                # 策略：分数 > 阈值 OR 随机探索
                is_explore = random.random() < config.EXPLORATION_RATE

                if score >= config.FILTER_THRESHOLD or is_explore:
                    to_solve_indices.append(idx)
                    self.df.at[idx, "is_explored"] = is_explore
                else:
                    self.df.at[idx, "status"] = "skipped"
        else:
            # 冷启动阶段：如果没有模型，全部求解（或者随机选一部分）来积累初始数据
            print("冷启动模式：全部提交求解以积累数据...")
            to_solve_indices = target_indices.tolist()

        # 3. 提交求解
        sent_count = len(to_solve_indices)  # [统计] 筛选剩余多少
        self.df.loc[target_indices, "batch_id"] = batch_id
        success_count = self._batch_solve(to_solve_indices)

        self.save_state()

        # [新增] 记录日志
        self._log_metrics(
            batch_id=batch_id,
            stage="regular_filter",
            input_count=input_count,
            sent_solver_count=sent_count,
            success_count=success_count,
        )

        return True

    # 辅助方法：将新产生的证明数据追加到大文件中 (JSONL 格式)
    def _save_proofs_to_master(self, new_data):
        """
        将数据追加写入 JSONL 文件。
        new_data: list，通常是 [[id, proof...], [id, proof...]]
        """
        if not new_data:
            return

        master_file = config.MASTER_PROOFS_FILE
        # 确保目录存在
        os.makedirs(os.path.dirname(master_file), exist_ok=True)

        try:
            # 使用 'a' (append) 模式打开，直接在文件末尾追加
            # 这种方式内存占用极低，且速度极快
            with open(master_file, "a", encoding="utf-8") as f:
                for item in new_data:
                    # json.dumps 将对象转为字符串
                    # ensure_ascii=False 保证中文正常显示
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + "\n")

            print(f"已归档: 追加 {len(new_data)} 条证明到主文件 (JSONL格式)")

        except Exception as e:
            print(f"错误：写入主文件失败 - {str(e)}")

    # [修改] 包含归档逻辑的 _batch_solve
    def _batch_solve(self, indices):
        if len(indices) == 0:
            return

        print(f"正在准备求解 {len(indices)} 道题目...")

        # --- 步骤 1: 写入临时文件 ---
        with open(config.TEMP_INPUT_FILE, "w", encoding="utf-8") as f:
            for idx in indices:
                uid = self.df.at[idx, "uid"]
                text = self.df.at[idx, "text"]
                f.write(f"{uid}\n")
                f.write(f"{text}\n")

        # --- 步骤 2: 调用 Shell 脚本 ---
        if not config.SOLVER_SCRIPT:
            raise RuntimeError(
                "No proof_scout solver script is configured. "
                "Legacy bash evaluation scripts were removed."
            )
        cmd = ["bash", config.SOLVER_SCRIPT]
        print(f"执行脚本: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"警告：求解脚本返回错误 (Code: {e.returncode})")

        # --- 步骤 3: 解析结果、归档并更新状态 ---
        solved_ids_set = set()

        # [修改] 读取 JSONL 格式的临时文件
        if os.path.exists(config.SOLVER_OUTPUT_JSON):
            result_data = []  # 用于存储解析出来的所有条目
            try:
                with open(config.SOLVER_OUTPUT_JSON, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:  # 跳过空行
                            try:
                                # 逐行解析 JSON
                                item = json.loads(line)
                                result_data.append(item)
                            except json.JSONDecodeError:
                                print(f"警告：跳过无法解析的行: {line[:50]}...")

                # [归档] 将读取到的数据列表传给归档函数
                # 注意：result_data 是一个 list，_save_proofs_to_master 会遍历它并追加写入主文件
                if len(result_data) > 0:
                    self._save_proofs_to_master(result_data)

                    # [提取 ID]
                    for item in result_data:
                        # 假设每一行数据结构依然是 list，且第一个元素是 ID
                        if isinstance(item, list) and len(item) > 0:
                            solved_ids_set.add(item[0])

                print(f"本批次成功解出: {len(solved_ids_set)}")

            except Exception as e:
                print(f"错误：读取输出文件失败 - {str(e)}")
        else:
            print(f"警告：未找到输出文件 {config.SOLVER_OUTPUT_JSON}。")

        # --- 步骤 4: 更新 DataFrame 状态 ---
        success_count = 0
        for idx in indices:
            uid = str(self.df.at[idx, "uid"]).strip()
            is_success = uid in solved_ids_set

            self.df.at[idx, "status"] = "solved" if is_success else "failed"
            self.df.at[idx, "label"] = 1 if is_success else 0
            if is_success:
                success_count += 1

        # 清理临时文件（可选）
        if os.path.exists(config.TEMP_INPUT_FILE):
            os.remove(config.TEMP_INPUT_FILE)
        if os.path.exists(config.SOLVER_OUTPUT_JSON):
            os.remove(config.SOLVER_OUTPUT_JSON)
        return success_count

    # [新增] 辅助函数：加载初始的 12000 条基准数据
    def _load_baseline_data(self):
        print("正在加载基准数据集 (Baseline Data)...")
        texts = []
        labels = []

        # 1. 加载基准的 solved IDs
        baseline_solved = set()
        if os.path.exists(config.BASELINE_SOLVED_IDS):
            with open(config.BASELINE_SOLVED_IDS, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        baseline_solved.add(line.strip())

        # 2. 加载基准题目 (ID + Content 格式)
        if os.path.exists(config.BASELINE_DATA_FILE):
            with open(config.BASELINE_DATA_FILE, "r", encoding="utf-8") as f:
                lines = iter(f)
                while True:
                    try:
                        uid = next(lines).strip()
                        if not uid:
                            continue
                        text = next(lines).strip()

                        # 只有当 text 不为空时才加入
                        if text:
                            texts.append(text)
                            # 如果 ID 在 solved 列表中，label=1，否则=0
                            labels.append(1.0 if uid in baseline_solved else 0.0)

                    except StopIteration:
                        break

        print(f"基准数据加载完毕: {len(texts)} 条样本")
        return texts, labels

    # [修改] 回训函数：合并基准数据 + 新数据
    def retrain_filter(self):
        print("\n>>> [Retrain] 准备更新筛选器...")

        # --- 1. 获取管线中新产生的数据 ---
        # 选取所有已有明确结果 (solved/failed) 的数据
        mask = self.df["label"] != -1
        new_df = self.df[mask]

        new_texts = new_df["text"].tolist()
        new_labels = new_df["label"].tolist()

        print(f"当前管线新数据: {len(new_texts)} 条")

        # --- 2. 获取基准数据 ---
        base_texts, base_labels = self._load_baseline_data()

        # --- 3. 数据合并 ---
        # 将两者合并
        all_texts = base_texts + new_texts
        all_labels = base_labels + new_labels

        total_count = len(all_texts)
        if total_count < 100:  # 安全检查
            print("可用训练数据过少，跳过本次训练。")
            return

        print(
            f"合并后训练集总量: {total_count} 条 (基准 {len(base_texts)} + 新增 {len(new_texts)})"
        )

        # --- 4. 调用训练 ---
        # 注意：建议增加 epoch，因为数据量变大了，或者保持较小的 epoch 进行微调
        # 这里我们调用 model_utils 里的 train_model，它会保存模型到 MODEL_PATH
        self.model, self.preprocessor = model_utils.train_model(
            all_texts, all_labels, config.VOCAB_PATH, config.MODEL_PATH
        )
        print("模型更新完毕，已重新加载。")

    # --- 核心环节 5: 回捞 (Retrospective Review) ---
    def retrospective_review(self, batch_ids_to_review):
        batch_range_str = f"{min(batch_ids_to_review)}-{max(batch_ids_to_review)}"
        print(f"\n>>> [Review] 回捞检查批次: {batch_range_str}")

        # 找出这些批次中被 'skipped' 的题目
        mask = (self.df["batch_id"].isin(batch_ids_to_review)) & (
            self.df["status"] == "skipped"
        )
        target_indices = self.df[mask].index

        input_count = len(target_indices)  # [统计] 这一次回捞扫描了多少旧题目
        if input_count == 0:
            return

        print(f"正在重新评估 {len(target_indices)} 个历史遗留题目...")
        texts = self.df.loc[target_indices, "text"].tolist()
        new_scores = model_utils.predict_batch(self.model, self.preprocessor, texts)

        rescued_indices = []
        for idx, score in zip(target_indices, new_scores):
            self.df.at[idx, "pred_score"] = score  # 更新分数

            # 使用更严格的阈值回捞
            if score >= config.RETROSPECT_THRESHOLD:
                rescued_indices.append(idx)

        rescued_count = len(rescued_indices)  # [统计] 回捞时候剩多少(捞回来多少)
        print(f"捞回 {rescued_count} 个题目，送入求解器...")
        success_count = self._batch_solve(rescued_indices)  # 再次调用求解
        self.save_state()
        # [新增] 记录日志
        self._log_metrics(
            batch_id=batch_range_str,
            stage="retrospective",
            input_count=input_count,
            sent_solver_count=rescued_count,
            success_count=success_count,
        )

    # [新增] 统计日志记录函数
    def _log_metrics(
        self, batch_id, stage, input_count, sent_solver_count, success_count
    ):
        """
        记录核心指标到 CSV 文件
        :param batch_id: 批次号 (可以是单个数字，也可以是字符串如 '0-4')
        :param stage: 阶段名称 ('regular_filter', 'retrospective')
        :param input_count: 本阶段起始的题目总数 (处理多少数据)
        :param sent_solver_count: 筛选后提交给 Solver 的数量 (筛选剩余多少)
        :param success_count: 求解成功的数量
        """
        file_exists = os.path.exists(config.METRICS_FILE)

        with open(config.METRICS_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 如果是新文件，先写表头
            if not file_exists:
                writer.writerow(
                    [
                        "timestamp",
                        "batch_id",
                        "stage",
                        "total_input",
                        "sent_to_solver",
                        "solved_success",
                        "filter_pass_rate",
                        "solve_success_rate",
                    ]
                )

            # 计算一些比率
            pass_rate = (sent_solver_count / input_count) if input_count > 0 else 0
            solve_rate = (
                (success_count / sent_solver_count) if sent_solver_count > 0 else 0
            )

            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    batch_id,
                    stage,
                    input_count,
                    sent_solver_count,
                    success_count,
                    f"{pass_rate:.2%}",
                    f"{solve_rate:.2%}",
                ]
            )

        print(
            f"统计已记录: Batch {batch_id} [{stage}] - 提交 {sent_solver_count}/{input_count}, 解出 {success_count}"
        )
