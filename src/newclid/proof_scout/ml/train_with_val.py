import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import os
import json
import numpy as np
from collections import Counter

# ---------------------------------------------------------
# 假设前面的 GeoPreprocessor, GeoDataset, GeometryValueModel 类定义
# 和 load_data... 函数保持不变，这里不再重复粘贴
# ---------------------------------------------------------

# ================= 配置区 =================
DATA_PATH = "/c23474/home/duzhengtong/Discovery-GenesisGeo/datasets/proof_scout/configuration_clauses6_samples5M_problems_20k.txt"
LABEL_PATH = (
    "/c23474/home/duzhengtong/Discovery-GenesisGeo/datasets/proof_scout/solved_ids.txt"
)
SAVE_DIR = (
    "/c23474/home/duzhengtong/Discovery-GenesisGeo/datasets/proof_scout/checkpoints"
)
MODEL_NAME = "geo_value_model.pth"  # 改名：保存最好的模型
VOCAB_NAME = "vocab.json"
BATCH_SIZE = 64
EPOCHS = 30  # 增加 Epoch
LEARNING_RATE = 0.001
VAL_RATIO = 0.2  # 20% 的数据用来做验证
# =========================================


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


def evaluate(model, loader, device):
    """
    在验证集上评估模型，计算 Loss, Precision, Recall, AUC
    """
    model.eval()  # 切换到评估模式
    all_targets = []
    all_probs = []
    total_loss = 0
    criterion = (
        nn.BCEWithLogitsLoss()
    )  # 验证时通常不需要 pos_weight，或者保持一致也可以

    with torch.no_grad():  # 不计算梯度，节省显存
        for batch in loader:
            ids = batch["ids"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            outputs = model(ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 记录预测概率和真实标签
            probs = torch.sigmoid(outputs).cpu().numpy()
            targets = labels.cpu().numpy()

            all_probs.extend(probs)
            all_targets.extend(targets)

    # 计算指标
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # 默认阈值 0.5，但在你的场景下，你可能更关心阈值较低时的 Recall
    preds = (all_probs > 0.5).astype(int)

    precision = precision_score(all_targets, preds, zero_division=0)
    recall = recall_score(all_targets, preds, zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5  # 如果验证集全是负样本可能会报错

    avg_loss = total_loss / len(loader)
    return avg_loss, precision, recall, auc


# ---------------------------------------------------------
# 主程序
# ---------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载所有数据
    print("正在加载数据...")
    solved_ids = load_solved_ids(LABEL_PATH)
    all_data = load_data_with_external_labels(DATA_PATH, solved_ids)

    # 2. 构建词表
    preprocessor = GeoPreprocessor()
    preprocessor.build_vocab([d["text"] for d in all_data])
    # 保存词表
    with open(os.path.join(SAVE_DIR, VOCAB_NAME), "w", encoding="utf-8") as f:
        json.dump(preprocessor.token2idx, f)

    # 3. 划分 训练集 vs 验证集 (Stratified Split)
    # stratify=labels 保证两边正负样本比例一致
    labels = [d["label"] for d in all_data]
    train_data, val_data = train_test_split(
        all_data, test_size=VAL_RATIO, stratify=labels, random_state=42
    )

    print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")

    # 4. DataLoader
    train_dataset = GeoDataset(train_data, preprocessor)
    val_dataset = GeoDataset(val_data, preprocessor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )  # 验证集不需要 shuffle

    # 5. 计算 pos_weight (只根据训练集计算)
    pos_count = sum(1 for d in train_data if d["label"] == 1.0)
    neg_count = len(train_data) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(device)
    print(
        f"Train Pos: {pos_count}, Train Neg: {neg_count}, Pos Weight: {pos_weight.item():.2f}"
    )

    # 6. 初始化模型
    model = GeometryValueModel(vocab_size=len(preprocessor.token2idx)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. 训练循环
    best_recall = 0.0  # 我们以 Recall 作为保存模型的标准

    print("\n开始训练...")
    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0
        for batch in train_loader:
            ids = batch["ids"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        val_loss, val_prec, val_recall, val_auc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"  Val Recall: {val_recall:.4f} | Val Prec: {val_prec:.4f} | Val AUC: {val_auc:.4f}"
        )

        # --- Model Checkpoint ---
        # 策略：如果 Recall 提升了，就保存模型。
        # 或者你可以改为 if val_loss < best_loss
        if val_recall >= best_recall:
            best_recall = val_recall
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, MODEL_NAME))
            print(f"  >> 模型已保存 (Best Recall: {best_recall:.4f})")
