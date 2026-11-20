from __future__ import annotations

from importlib.machinery import PathFinder
import sys


_QWEN2_MODULE = "transformers.models.qwen2.tokenization_qwen2"


def _patch_loaded_module(module) -> None:
    tokenizer = getattr(module, "Qwen2Tokenizer", None)
    if tokenizer is not None and not hasattr(tokenizer, "all_special_tokens_extended"):
        tokenizer.all_special_tokens_extended = property(
            lambda self: list(self.all_special_tokens)
        )


def patch_qwen2_tokenizer() -> None:
    try:
        from transformers.models.qwen2 import tokenization_qwen2
    except Exception:
        return
    _patch_loaded_module(tokenization_qwen2)


class _Qwen2Loader:
    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _patch_loaded_module(module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _Qwen2Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _QWEN2_MODULE:
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _Qwen2Loader(spec.loader)
        return spec


_finder = _Qwen2Finder()


def install_qwen2_tokenizer_compat() -> None:
    # The startup .pth also runs in geometry-only processes.
    module = sys.modules.get(_QWEN2_MODULE)
    if module is not None:
        _patch_loaded_module(module)
    if _finder not in sys.meta_path:
        sys.meta_path.insert(sys.meta_path.index(PathFinder), _finder)


install_qwen2_tokenizer_compat()
