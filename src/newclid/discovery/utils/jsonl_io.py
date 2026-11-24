import json


def read_json(file_path):
    """读取标准 JSON 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, file_path, indent=4):
    """写入标准 JSON 文件，默认美化层级并支持中文"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_jsonl(file_path):
    """读取 JSONL 文件，返回对象列表"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def write_jsonl(data, file_path, append=False):
    """
    写入 JSONL 文件
    :param data: 列表对象
    :param file_path: 输出路径
    :param append: 是否为追加模式
    """
    mode = "a" if append else "w"
    with open(file_path, mode, encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --- 测试用例 ---
if __name__ == "__main__":
    # 测试数据
    my_data = [{"id": 1, "text": "你好"}, {"id": 2, "text": "世界"}]

    # 写入并读取 JSONL
    write_jsonl(my_data, "test.jsonl")
    content = read_jsonl("test.jsonl")

    # 写入并读取 JSON
    write_json(my_data, "test.json")
    content_json = read_json("test.json")
