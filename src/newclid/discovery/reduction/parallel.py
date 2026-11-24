"""Ray 并行工具：bounded in-flight 调度，避免线程闲置。

模式：一次最多提交 inflight 个任务，用 ray.wait(num_returns=1) 等到任一完成就取回
结果并立刻补一个新任务，直到全部 drain 完。相比"一次性提交全部"能控内存，相比
"分批 barrier"能避免 worker 空转（完成一个补一个）。
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator


def run_bounded(
    remote_fn: Any,
    task_args: Iterable[tuple],
    inflight: int = 32,
) -> Iterator[Any]:
    """并发执行 remote_fn(*args)，以完成顺序 yield 结果。

    Parameters
    ----------
    remote_fn : ray remote function
        形如 `fn.remote(*args)` 可调用。
    task_args : Iterable[tuple]
        每个元素是一次调用的位置参数元组。
    inflight : int
        在飞任务上限（≈ worker 数）。

    Yields
    ------
    每个任务的返回值（完成即产出，顺序非提交序）。
    """
    import ray

    it = iter(task_args)
    pending: list = []

    # 先填满在飞窗口
    for _ in range(inflight):
        try:
            args = next(it)
        except StopIteration:
            break
        pending.append(remote_fn.remote(*args))

    # 完成一个 -> 取回 -> 补一个
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        yield ray.get(done[0])
        try:
            args = next(it)
            pending.append(remote_fn.remote(*args))
        except StopIteration:
            pass


def ensure_ray(n_workers: int) -> None:
    """按需初始化 Ray（已初始化则跳过）。"""
    import ray

    if not ray.is_initialized():
        ray.init(num_cpus=n_workers, ignore_reinit_error=True, log_to_driver=False)
