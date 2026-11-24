import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os
import json  # 用于保存词表

# ================= 配置区 =================
DATA_PATH = "/c23474/home/duzhengtong/Discovery-GenesisGeo/datasets/train_filter/configuration_clauses6_samples5M_problems_20k.txt"
LABEL_PATH = (
    "/c23474/home/duzhengtong/Discovery-GenesisGeo/datasets/train_filter/solved_ids.txt"
)
SAVE_DIR = (
    "/c23474/home/duzhengtong/Discovery-GenesisGeo/datasets/train_filter/checkpoints"
)
MODEL_NAME = "geo_value_model.pth"
VOCAB_NAME = "vocab.json"
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
# =========================================

# 确保保存目录存在
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------------------------------------------
# 1. 类定义 (保持不变，为了完整性还是贴在这里)
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

    def build_vocab(self, texts, min_freq=1):
        counter = Counter()
        for text in texts:
            tokens = self.canonicalize_and_tokenize(text)
            counter.update(tokens)
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.token2idx:
                self.token2idx[token] = len(self.token2idx)

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


class GeoDataset(Dataset):
    def __init__(self, samples, preprocessor):
        self.samples = samples
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        ids = self.preprocessor.encode(item["text"])
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(item["label"], dtype=torch.float),
        }


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


# ---------------------------------------------------------
# 2. 数据加载函数
# ---------------------------------------------------------
def load_solved_ids(label_file_path):
    solved_set = set()
    if not os.path.exists(label_file_path):
        return solved_set
    with open(label_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                solved_set.add(line.strip())
    return solved_set


def load_data_with_external_labels(data_file_path, solved_set):
    data_samples = []
    if not os.path.exists(data_file_path):
        print(f"Error: Data file {data_file_path} not found.")
        return []

    with open(data_file_path, "r", encoding="utf-8") as f:
        lines = iter(f)
        while True:
            try:
                uid_line = next(lines).strip()
                if not uid_line:
                    continue
                content_line = next(lines).strip()
                label = 1.0 if uid_line in solved_set else 0.0
                data_samples.append(
                    {"uid": uid_line, "text": content_line, "label": label}
                )
            except StopIteration:
                break
    return data_samples


# ---------------------------------------------------------
# 3. 主程序 (Execution)
# ---------------------------------------------------------
if __name__ == "__main__":
    # A. 加载数据
    print("正在加载数据...")
    solved_ids = load_solved_ids(LABEL_PATH)
    raw_data = load_data_with_external_labels(DATA_PATH, solved_ids)
    print(f"数据加载完成，共 {len(raw_data)} 条样本。")

    # B. 初始化与构建词表 (Critical Step!)
    preprocessor = GeoPreprocessor()
    preprocessor.build_vocab([d["text"] for d in raw_data])
    print(f"词表构建完成，大小: {len(preprocessor.token2idx)}")

    # >>> 保存词表 (非常重要，否则以后没法用) <<<
    with open(os.path.join(SAVE_DIR, VOCAB_NAME), "w", encoding="utf-8") as f:
        json.dump(preprocessor.token2idx, f)
    print(f"词表已保存至 {os.path.join(SAVE_DIR, VOCAB_NAME)}")

    # C. 准备 DataLoader
    dataset = GeoDataset(raw_data, preprocessor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # D. 计算 Pos Weight (处理不平衡)
    pos_count = sum(1 for d in raw_data if d["label"] == 1.0)
    neg_count = len(raw_data) - pos_count
    pos_weight = torch.tensor([neg_count / (pos_count + 1e-5)])  # 加个极小值防除零
    print(
        f"正样本: {pos_count}, 负样本: {neg_count}, Pos Weight: {pos_weight.item():.2f}"
    )

    # E. 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    model = GeometryValueModel(vocab_size=len(preprocessor.token2idx))
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # F. 训练循环
    print("\n开始训练...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            ids = batch["ids"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # G. 保存模型
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, MODEL_NAME))
    print(f"\n模型已保存至 {os.path.join(SAVE_DIR, MODEL_NAME)}")
