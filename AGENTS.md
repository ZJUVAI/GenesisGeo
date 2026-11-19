# GenesisGeo 仓库约定

## 入口

- 实验编排：`scripts/train_pipeline.py`
- 数据生成：`src/newclid/generation/pipeline.py`
- 训练实现：`scripts/train_lm.sh`
- 模型评测：`scripts/evaluation.py`

## 求解与数据格式

- `src/newclid/api.py` 提供求解接口，`proof.py` 管理证明状态，`agent/` 包含符号推理、LM 和 VLM agent。
- `generation/` 中的 `sampler.py` 采样构造，`point_naming.py` 管理点名，`filter.py` 筛选目标，`worker.py` 处理单题，`writer.py` 写出数据和图像。
- `fl_problem` 格式为 `构造 ? 目标`。构造子句用分号分隔，每句为 `点名 = 构造类型 参数`，同一句的多个构造用逗号分隔。求解器把构造转换为谓词，如 `coll`、`perp`、`cyclic`、`eqangle`、`eqratio`。
- `llm_input_renamed` 使用 `<problem> 点名 : 谓词 参数 [id] ; … ? 目标 </problem>`。
- `llm_output_renamed` 使用 `<proof> 结论 [id] 规则 [依赖id] ; … </proof>`。

## 文件职责

- `runs/`：实验运行产物；`results/`：独立评测输出。

## 训练约定

- 统一通过 `scripts/train_pipeline.py` 编排生成和训练；训练底层使用 Swift SFT。
- 基模对比的固定配置：500K 共享 JSONL、`aux_only=1`、`prune=true`、
  `non_aux_keep_probability=0.3`、`n_clauses=10`、`base_seed=42`、`img=0`。
- 训练为 full SFT、1 epoch、序列长度 2048、4 卡、per-device batch 16、学习率
  `1e-4`、bfloat16、packing、padding-free、Flash Attention、Liger、ZeRO-1。
- 最终 checkpoint 必须保留并确认可被 vLLM 加载；中间 checkpoint 可在实验完成后清理。
- 不修改公共数据集或 `datasets/seed_cache.jsonl`。数据比例以 `profile.json` 实测值为准。

典型训练命令：

```bash
.venv/bin/python scripts/train_pipeline.py \
  --config runs/<experiment>/pipeline_config.json \
  --run-dir runs/<experiment>
```

## 评估约定

- 评估入口是 `scripts/evaluation.py`，编排入口是 `scripts/train_pipeline.py`。
- 评估题单使用 `benchmarks/imo-dev.txt`；当前题单为 20 题。历史 CSV 可能是旧的
  30 题，不能与当前题单混合统计。
- 标准参数：`decoding_size=32`、`beam_size=512`、`search_depth=4`、
  `search_version=hybrid`、`timeout=7200`、`ray_num_cpus=50`、`think=false`。
- Qwen 使用 `evaluation.template=qwen3`；SmolLM2 使用训练一致的
  `evaluation.template=llama` 和对应的 `server.chat_template` 文件。
- 启动评估服务后必须检查 `/v1/models` 返回的 served checkpoint 与目标 checkpoint
  一致，不能只检查端口健康。
- 每次评估必须保留完整 CSV、日志和 summary；重复评估应使用独立 `repeat_N` 目录。
- 当前采样使用 `temperature=1.0` 且未固定 seed，结果必须报告多次评估统计，不以最高分
  作为结论。

典型评估命令：

```bash
.venv/bin/python scripts/train_pipeline.py \
  --config runs/<experiment>/pipeline_config.json \
  --run-dir runs/<experiment> \
  --skip-generation --skip-training \
  --set training.enabled=false \
  --set evaluation.enabled=true \
  --set evaluation.problems_paths='["benchmarks/imo-dev.txt"]' \
  --set evaluation.template=qwen3 \
  --set evaluation.repeats=1 \
  --set evaluation.start_repeat=1 \
  --set server.chat_template=runs/<experiment>/chat_template.jinja
```

SmolLM2 评估时将上例中的 `evaluation.template` 改为 `llama`，并使用其训练导出的
`chat_template.jinja`；不要把 Llama `[INST]` 模板替换成 ChatML 模板。

## 运行长时间任务时
  - 不要轮询进程状态
  - 不要使用 ps/top/sleep 循环检查
  - 使用前台阻塞等待，或 wait PID
  - 任务完成后再汇报结果
