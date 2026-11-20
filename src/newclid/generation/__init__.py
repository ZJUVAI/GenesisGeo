"""
Generation module for geometry problem generation.

This module provides tools for:
- Sampling geometric problems and constructions (ProblemSampler)
- Processing geometry problems (ProblemWorker)
- Batch generation pipeline (ProblemPipeline)
- Filtering goals (GoalFilter)
- Generating statistics reports (Statistics, get_first_predicate)

Public API:
    ProblemSampler: Sample geometric constructions
    ProblemWorker: Process single problems with proof validation
    ProblemPipeline: Batch generation with parallel processing
    GoalFilter: Filter valid geometry goals
    Statistics: Statistics collection and reporting
    get_first_predicate: Extract first predicate from problem string
"""

from newclid.generation.sampler import ProblemSampler
from newclid.generation.filter import GoalFilter
from newclid.generation.statistics import Statistics, get_first_predicate
from newclid.generation.worker import ProblemWorker

__all__ = [
    # Core public API
    "ProblemSampler",
    "ProblemWorker",
    "ProblemPipeline",
    # Filtering and statistics
    "GoalFilter",
    "Statistics",
    "get_first_predicate",
]


def __getattr__(name):
    if name == "ProblemPipeline":
        from newclid.generation.pipeline import ProblemPipeline

        return ProblemPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
