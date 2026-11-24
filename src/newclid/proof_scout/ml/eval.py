import torch
import json
import os
from newclid.proof_scout.ml.data_processor import GeoPreprocessor, GeometryValueModel
# 这里需要导入你之前定义的类：GeoPreprocessor, GeometryValueModel
# 或者把那些类单独放在一个 model_def.py 文件里然后 import 进来


def load_predictor(model_dir="checkpoints"):
    # 1. 加载词表
    vocab_path = os.path.join(model_dir, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        token2idx = json.load(f)

    # 2. 初始化 Preprocessor 并恢复词表
    preprocessor = GeoPreprocessor()
    preprocessor.token2idx = token2idx  # 关键！必须覆盖

    # 3. 初始化模型结构
    model = GeometryValueModel(vocab_size=len(token2idx))

    # 4. 加载权重
    model_path = os.path.join(model_dir, "geo_value_model.pth")
    # map_location='cpu' 保证即使没有 GPU 也能跑
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()  # 切换到评估模式 (关闭 Dropout 等)

    return model, preprocessor


def predict_solvability(text_input, model, preprocessor, threshold=0.5):
    # 1. 编码
    input_ids = preprocessor.encode(text_input)
    tensor_in = torch.tensor([input_ids], dtype=torch.long)  # [1, seq_len]

    # 2. 推理
    with torch.no_grad():
        logits = model(tensor_in)
        prob = torch.sigmoid(logits).item()

    return prob, prob >= threshold


# --- 使用示例 ---
if __name__ == "__main__":
    # 加载一次模型 (开销较大，不要放在循环里)
    model, preprocessor = load_predictor(
        "/c23474/home/duzhengtong/Discovery-GenesisGeo/datasets/train_filter/checkpoints"
    )

    # 模拟一个新的题目
    new_problem = "a b c = r_triangle a b c; d = on_dia d c a ? cong a b c d"

    probability, is_solvable = predict_solvability(
        new_problem, model, preprocessor, threshold=0.3
    )

    print(f"题目: {new_problem}")
    print(f"可解概率: {probability:.4f}")
    print(f"判决结果: {'保留 (送去计算)' if is_solvable else '丢弃 (跳过)'}")
