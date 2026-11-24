"""
Proof Scout ML Module - Machine learning components (requires torch, etc.)

This module provides ML-based functionality for proof scout:
- ScoutConfig: Configuration management
- PipelineManager: ML pipeline orchestration
- model_utils: Model utilities
- data_processor: Data processing utilities
- train_with_val: Training with validation
- eval: Evaluation utilities
- problems_filter: Problem filtering
"""


# Lazy imports to avoid loading torch when not needed
def __getattr__(name):
    if name == "ScoutConfig":
        from newclid.proof_scout.ml.scout_config import ScoutConfig

        return ScoutConfig
    elif name == "PipelineManager":
        from newclid.proof_scout.ml.scout_pipeline import PipelineManager

        return PipelineManager
    elif name == "parse_llm_input":
        from newclid.proof_scout.ml.data_processor import parse_llm_input

        return parse_llm_input
    elif name == "parse_llm_output":
        from newclid.proof_scout.ml.data_processor import parse_llm_output

        return parse_llm_output
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ScoutConfig",
    "PipelineManager",
    "parse_llm_input",
    "parse_llm_output",
]
