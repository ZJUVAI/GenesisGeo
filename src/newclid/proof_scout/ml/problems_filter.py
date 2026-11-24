import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm  # 进度条库，如果没装建议 pip install tqdm

# ================= 配置区 (请根据实际情况修改) =================
# 1. 待验证的原始题目集

# FILE_NAME = "configuration_clauses6_samples1M_problems.txt"

FILE_NAME = "raw_problems_0.txt"

INPUT_FILE = (
    "/c23474/home/duzhengtong/Discovery-GenesisGeo/datasets/train_filter/raw_problems/"
    + FILE_NAME
)

# 2. 筛选后输出的文件路径
OUTPUT_FILE = (
    "/c23474/home/duzhengtong/Discovery-GenesisGeo/datasets/train_filter/filtered_problems/"
    + FILE_NAME.replace(".txt", "_filtered.txt")
)

# 3. 模型和词表所在的目录
MODEL_DIR = (
    "/c23474/home/duzhengtong/Discovery-GenesisGeo/datasets/train_filter/checkpoints"
)
MODEL_NAME = "best_geo_model.pth"  # 或者是 geo_value_model.pth
VOCAB_NAME = "vocab.json"

# 4. 评测标准：置信度阈值 (0.0 ~ 1.0)
# 大于此值的题目会被保留。建议从 0.3 或 0.5 开始尝试。

THRESHOLD = 0.5

# 5. 批处理大小 (越大越快，显存不够就调小)
BATCH_SIZE = 128
# ==========================================================


# ---------------------------------------------------------
# 1. 必要类定义 (必须与训练时完全一致)
# ---------------------------------------------------------
class GeoPreprocessor:
    def __init__(self):
        self.special_tokens = {"<PAD>": 0, "<UNK>": 1, "<SEP>": 2}
        self.token2idx = self.special_tokens.copy()

    def canonicalize_and_tokenize(self, text):
        text = (
            text.replace("?", " ? ")
            .replace("=", " = ")
            .replace(",", " , ")
            .replace(";", " ; ")
        )
        tokens = text.split()
        mapping = {}
        new_tokens = []
        p_counter = 0
        for t in tokens:
            if len(t) == 1 and t.isalpha():
                if t not in mapping:
                    mapping[t] = f"P{p_counter}"
                    p_counter += 1
                new_tokens.append(mapping[t])
            else:
                new_tokens.append(t)
        return new_tokens

    def encode(self, text, max_len=128):
        tokens = self.canonicalize_and_tokenize(text)
        token_ids = []
        for t in tokens:
            if t == "?":
                token_ids.append(self.token2idx["<SEP>"])
            else:
                token_ids.append(self.token2idx.get(t, self.token2idx["<UNK>"]))
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        else:
            token_ids += [self.token2idx["<PAD>"]] * (max_len - len(token_ids))
        return token_ids


class GeometryValueModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x):
        emb = self.embedding(x)
        output, _ = self.lstm(emb)
        attn_weights = F.softmax(self.attention(output), dim=1)
        context_vector = torch.sum(output * attn_weights, dim=1)
        logits = self.classifier(context_vector)
        return logits


# 这是一个专门用于推理的 Dataset，它会额外返回原始文本，方便写入文件
class InferenceDataset(Dataset):
    def __init__(self, file_path, preprocessor):
        self.data = []
        self.preprocessor = preprocessor

        # 加载数据
        with open(file_path, "r", encoding="utf-8") as f:
            lines = iter(f)
            while True:
                try:
                    uid = next(lines).strip()
                    if not uid:
                        continue
                    content = next(lines).strip()
                    self.data.append({"uid": uid, "text": content})
                except StopIteration:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        ids = self.preprocessor.encode(item["text"])
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "uid": item["uid"],
            "raw_text": item["text"],  # 返回原始内容以便写入文件
        }


# ---------------------------------------------------------
# 2. 核心推理逻辑
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # A. 加载资源
    vocab_path = os.path.join(MODEL_DIR, VOCAB_NAME)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)

    if not os.path.exists(vocab_path) or not os.path.exists(model_path):
        print("错误：找不到模型或词表文件，请检查 MODEL_DIR 配置。")
        return

    print("加载词表...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        token2idx = json.load(f)

    preprocessor = GeoPreprocessor()
    preprocessor.token2idx = token2idx  # 恢复词表

    print("加载模型...")
    model = GeometryValueModel(vocab_size=len(token2idx))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # B. 准备数据
    print(f"读取输入文件: {INPUT_FILE}")
    dataset = InferenceDataset(INPUT_FILE, preprocessor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    total_count = len(dataset)
    kept_count = 0

    # C. 执行推理与筛选
    print(f"开始筛选 (阈值: {THRESHOLD})...")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        with torch.no_grad():
            # 使用 tqdm 显示进度条
            for batch in tqdm(loader, desc="Processing"):
                ids = batch["ids"].to(device)
                uids = batch["uid"]
                raw_texts = batch["raw_text"]

                # 模型推理
                logits = model(ids)
                probs = torch.sigmoid(logits).squeeze(1)  # [batch_size]

                # 批量判断
                for i in range(len(probs)):
                    prob = probs[i].item()

                    if prob >= THRESHOLD:
                        # 写入文件
                        f_out.write(f"{uids[i]}\n")
                        f_out.write(f"{raw_texts[i]}\n")
                        kept_count += 1

    # D. 输出统计报告
    discarded_count = total_count - kept_count
    keep_ratio = (kept_count / total_count) * 100 if total_count > 0 else 0

    print("\n" + "=" * 30)
    print("筛选完成 - 统计报告")
    print("=" * 30)
    print(f"原始题目总数 : {total_count}")
    print(f"保留题目数   : {kept_count} (可解预测)")
    print(f"丢弃题目数   : {discarded_count} (不可解预测)")
    print(f"保留率       : {keep_ratio:.2f}%")
    print(f"理论计算节省 : {100 - keep_ratio:.2f}%")
    print(f"结果已保存至 : {OUTPUT_FILE}")
    print("=" * 30)


if __name__ == "__main__":
    main()
