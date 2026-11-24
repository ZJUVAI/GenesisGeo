"""
Proof Scout - Knowledge Discovery and Theorem Extraction Module

Submodules:
- core: Core graph structures and utilities (no ML dependencies)
- reduction: Rule reduction via subsumption testing
- ml: Machine learning components (requires torch, etc.)

Quick Start:
    from newclid.proof_scout import ProofGraph, GraphPruner
    from newclid.proof_scout.core import FilterAndPruneEngine
"""

from newclid.proof_scout.core import (
    ProofGraph,
    SingleProofGraph,
    GraphPruner,
    FilterAndPruneEngine,
    AuxExtractor,
)


# Lazy import for ProofGraphVisualizer to avoid visualization dependencies
def __getattr__(name):
    if name == "ProofGraphVisualizer":
        from newclid.proof_scout.core import ProofGraphVisualizer

        return ProofGraphVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ProofGraph",
    "SingleProofGraph",
    "GraphPruner",
    "ProofGraphVisualizer",
    "FilterAndPruneEngine",
    "AuxExtractor",
]
