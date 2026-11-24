"""命令行入口。

解析 CLI 参数（--config <path>），调用 pipeline.run_pipeline(config_path)。
可通过 python -m newclid.discovery 运行。
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    """CLI 入口。

    Parameters
    ----------
    argv : list[str] | None
        命令行参数列表，None 时使用 sys.argv[1:]。
    """
    parser = argparse.ArgumentParser(
        description="Discovery Pipeline: 从合成数据中提取并规约几何推理规则",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON 配置文件路径",
    )
    args = parser.parse_args(argv)

    from newclid.discovery.pipeline import run_pipeline

    run_pipeline(args.config)


if __name__ == "__main__":
    main()
