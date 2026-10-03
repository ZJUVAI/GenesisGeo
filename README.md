<div align="center">

# GenesisGeo

**Neuro-Symbolic Geometry Theorem Proving at Olympiad Level**

[![Paper](https://img.shields.io/badge/arXiv-2509.21896-b31b1b.svg)](https://arxiv.org/abs/2509.21896) [![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE) [![Model](https://img.shields.io/badge/🤗_Model-GenesisGeo--2B-blue.svg)](https://huggingface.co/ZJUVAI/GenesisGeo-2B)

</div>

## Overview

GenesisGeo is a neuro-symbolic system that proves geometry theorems by combining a symbolic deduction engine (DDAR) with a vision-language model. It uses rendered diagrams and formal problem statements to propose auxiliary constructions, then resumes symbolic deduction to check whether they prove the goal.

**Highlights:**

- Synthetic data generation pipeline producing **1 million** multimodal geometry problems with diagrams and machine-checkable proof traces
- Enhanced DDAR engine with **~20x** speedup over the AlphaGeometry engine
- Neuro-symbolic prover fine-tuned from **Qwen3-VL-2B**

## Results

| Variant | IMO-30 | IMO-95 | HAGeo-409 |
|---------|-------:|-------:|----------:|
| Text | 28/30 | 59/95 | 270/409 |
| Vision + Text | **29/30** | **63/95** | **278/409** |

Paper-reported results with a 32 × 512 × 4 search budget and a
60-minute time limit per problem.

## Setup

```bash
git clone https://github.com/ZJUVAI/GenesisGeo.git
cd GenesisGeo
uv venv
source .venv/bin/activate
uv sync --extra full
```

## Data Generation

Generate synthetic geometry problems with proof traces and diagrams:

```bash
python src/newclid/generation/pipeline.py \
  --n_clauses 10 \
  --n_samples 1000000 \
  --n_threads 20 \
  --aux_only 2 \
  --img 3 \
  --seed_cache
```

### General Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_clauses` | `15` | Max number of construction clauses per problem |
| `--n_samples` | `10000` | Total number of problems to generate |
| `--n_threads` | `10` | Number of parallel Ray workers |
| `--timeout` | `3600` | Per-task timeout in seconds |
| `--max_level` | `500` | Maximum DDAR search depth |
| `--base_seed` | `42` | Base random seed for generation |
| `--log_level` | `info` | Logging level (`debug`, `info`, `warning`, `error`) |
| `--construction_config` | `None` | Path to JSON config for construction sets and sampler steps |
| `--seed_cache` | `off` | Enable seed cache to skip seeds without real auxiliary points |

### Auxiliary Point Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--add_auxiliary` / `--no-add_auxiliary` | `enabled` | Whether to add auxiliary points during generation |
| `--max_auxiliary_points` | `2` | Maximum auxiliary points per problem |
| `--aux_only` | `0` | Data filter: `0` = all, `1` = include non-aux at 0.1 prob, `2` = aux-only |

### Output Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dir` | `./datasets` | Output directory |
| `--img` | `0` | Image mode: `0` = none, `1` = annotated only, `2` = unannotated only, `3` = both |
| `--direct_png` / `--no-direct_png` | `enabled` | Save PNG directly or keep the legacy `svg -> png` pipeline |
| `--img_pixels` | `512` | Output image width in pixels |
| `--prune` / `--no-prune` | `enabled` | Prune clauses to keep only the deepest clause chain |
| `--remove_coords` | `off` | Remove coordinate information from output |
| `--clear` | `off` | Clear old dataset files before generation |

## Training

### Text SFT

```bash
bash scripts/train_lm.sh
```

### VLM SFT

```bash
bash scripts/train_vlm.sh
```

> **Note:** Update dataset paths, checkpoint directories, output paths, and `CUDA_VISIBLE_DEVICES` in the scripts before running.

## Evaluation

Start a vLLM OpenAI-compatible server for the checkpoint first:

```bash
python scripts/launch_vllm_server.py \
  --model_name /path/to/checkpoint \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu_ids 0,1,2,3
```

### Qwen3-VL (Text)

```bash
python scripts/evaluation.py \
  --agent qwen3_vl_text \
  --problems_path benchmarks/dev_imo.txt \
  --vllm_base_url http://127.0.0.1:8000 \
  --decoding_size 32 \
  --beam_size 512 \
  --search_depth 4 \
  --ray_num_cpus 40 \
  --timeout 3600
```

The `qwen3_text` agent supports `--think true` to start with `<think>` and extract the generated `<aux>` block instead of forcing the next point name. The Qwen3-VL agents start directly from `<aux>` for Qwen3-VL-Instruct checkpoints.

### Qwen3-VL (Vision + Text)

```bash
python scripts/evaluation.py \
  --agent qwen3_vl \
  --problems_path benchmarks/imo_95.txt \
  --vllm_base_url http://127.0.0.1:8000 \
  --decoding_size 32 \
  --beam_size 512 \
  --search_depth 4 \
  --ray_num_cpus 40 \
  --timeout 3600
```

### Benchmarks

| File | Description |
|------|-------------|
| `benchmarks/imo_30.txt` | IMO-30 (30 problems) |
| `benchmarks/imo_95.txt` | IMO-95 (95 problems) |
| `benchmarks/hageo_409.txt` | HAGeo-409 (409 problems) |
| `benchmarks/jgex_231.txt` | JGEX-AG-231 (231 problems) |
| `benchmarks/dev_imo.txt` | Small IMO development subset |
| `benchmarks/dev_jgex.txt` | Small JGEX development subset |
| `benchmarks/examples.txt` | Mixed example and debugging problems |
| `benchmarks/larger_imo_eval.txt` | Extended IMO-style evaluation set |
| `benchmarks/testing_minimal_rules.txt` | Minimal regression checks for individual rules |

## Project Structure

```
GenesisGeo/
├── src/newclid/                    # Main source code
│   ├── __main__.py                 # CLI entry point
│   ├── api.py                      # GeometricSolver interface
│   ├── proof.py                    # Proof state management
│   ├── agent/                      # Reasoning agents
│   │   ├── ddarn.py                # DDAR symbolic engine
│   │   ├── base.py                 # Shared neural-guided search logic
│   │   └── vllm.py                 # vLLM text and vision-language agents
│   ├── evaluation/                 # vLLM evaluation runtime and trace helpers
│   │   ├── search_runtime.py       # Search DSL, DDAR tasks, beam utilities
│   │   └── search_trace.py         # Evaluation trace recording
│   ├── generation/                 # Data generation pipeline
│   │   ├── pipeline.py             # ProblemPipeline orchestrator
│   │   ├── sampler.py              # Geometry construction sampling
│   │   ├── worker.py               # Per-problem processing
│   │   ├── writer.py               # Data writing & image rendering
│   │   ├── filter.py               # Goal filtering
│   │   ├── point_naming.py         # Point naming management
│   │   ├── constructions.py        # Construction type constants
│   │   ├── statistics.py           # Generation statistics
│   │   └── auxiliary/              # Auxiliary point discovery
│   ├── DDAR/                       # C++ symbolic engine
│   ├── dependencies/               # Dependency graph management
│   ├── formulations/               # Problem representations
│   ├── numerical/                  # Numerical geometry
│   ├── algebraic_reasoning/        # Algebraic reasoning
│   ├── predicates/                 # Geometry predicates
│   └── proof_scout/                # Theorem discovery and rule reduction
├── scripts/                        # Training, evaluation, and data utilities
│   ├── train_lm.sh                 # Text SFT training
│   ├── train_vlm.sh                # VLM SFT training
│   ├── launch_vllm_server.py       # Managed vLLM gateway launcher
│   └── evaluation.py               # vLLM evaluation CLI
├── tests/                          # Test suite
├── benchmarks/                     # Benchmark problem sets
└── docs/                           # Documentation
```

## Acknowledgements

- [AlphaGeometry](https://github.com/google-deepmind/alphageometry) — the original neuro-symbolic geometry prover
- [Newclid](https://arxiv.org/abs/2411.11938) — the DDAR symbolic engine
- [Qwen](https://github.com/QwenLM) — base language models
- [ms-swift](https://github.com/modelscope/ms-swift) — training framework

## Citation

```bibtex
@article{zhu2025genesisgeo,
  title={GenesisGeo: Technical Report},
  author={Zhu, Minfeng and Wang, Zi and Ji, Sizhe and Du, Zhengtong and Tai, Shengqiang and Ke, Junming and Deng, Xiao and Yin, Zanlang and Huang, Xiuqi and Wang, Heyu and Chen, Wei},
  journal={arXiv preprint arXiv:2509.21896},
  year={2025}
}
```
