# model_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import newclid.proof_scout.ml.scout_config as config
from collections import Counter


# --- 基础组件 (Preprocessor & Model) ---
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

    def build_vocab(self, texts, min_freq=1):
        counter = Counter()
        for text in texts:
            tokens = self.canonicalize_and_tokenize(text)
            counter.update(tokens)
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.token2idx:
                self.token2idx[token] = len(self.token2idx)

    def save_vocab(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2idx, f)

    def load_vocab(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.token2idx = json.load(f)


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


# --- 训练与推理接口 ---


def train_model(texts, labels, vocab_path, model_save_path):
    """全量训练函数"""
    print(f"开始训练，样本数: {len(texts)}")
    preprocessor = GeoPreprocessor()
    preprocessor.build_vocab(
        texts
    )  # 注意：这里会重新构建词表，实际中建议增量更新或固定大词表
    preprocessor.save_vocab(vocab_path)

    # Dataset
    class SimpleDataset(Dataset):
        def __init__(self, txts, lbls, prep):
            self.t = txts
            self.l = lbls
            self.p = prep

        def __len__(self):
            return len(self.t)

        def __getitem__(self, idx):
            return {
                "ids": torch.tensor(self.p.encode(self.t[idx]), dtype=torch.long),
                "label": torch.tensor(self.l[idx], dtype=torch.float),
            }

    dataset = SimpleDataset(texts, labels, preprocessor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = GeometryValueModel(len(preprocessor.token2idx)).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()  # 简单起见，暂不加 pos_weight，实际建议加

    model.train()
    for epoch in range(10):  # 快速迭代建议 5-10 epoch
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch["ids"].to(config.DEVICE))
            loss = criterion(out, batch["label"].to(config.DEVICE).unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), model_save_path)
    return model, preprocessor


def predict_batch(model, preprocessor, texts):
    """批量推理函数"""
    model.eval()
    scores = []
    # 简单处理，实际应使用DataLoader进行Batch处理
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            batch_texts = texts[i : i + 128]
            ids_list = [preprocessor.encode(t) for t in batch_texts]
            tensor_in = torch.tensor(ids_list, dtype=torch.long).to(config.DEVICE)
            logits = model(tensor_in)
            probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
            scores.extend(probs)
    return scores
