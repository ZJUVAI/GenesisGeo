"""
Proof Scout Core Module - Core graph structures and utilities (no ML dependencies)

This module provides the fundamental building blocks for proof graph analysis:
- ProofGraph: Main proof graph data structure
- SingleProofGraph: Single-problem graph wrapper
- GraphPruner: Iterative graph pruning
- ProofGraphVisualizer: Graph visualization (lazy loaded)
- FilterAndPruneEngine: Filter and prune pipeline
- AuxExtractor: Auxiliary point extraction
- SubgraphExtractor: Subgraph extraction and pruning
"""

from newclid.proof_scout.core.proof_graph import ProofGraph, GSpanMiner, MergedGraph
from newclid.proof_scout.core.single_proof_graph import SingleProofGraph
from newclid.proof_scout.core.graph_pruner import GraphPruner
from newclid.proof_scout.core.filter_and_prune_engine import FilterAndPruneEngine
from newclid.proof_scout.core.aux_extractor import AuxExtractor
from newclid.proof_scout.core.subgraph_extractor import GraphPruner as SubgraphExtractor


# Lazy import for ProofGraphVisualizer to avoid pydot dependency at import time
def __getattr__(name):
    if name == "ProofGraphVisualizer":
        from newclid.proof_scout.core.proof_graph_visualizer import ProofGraphVisualizer

        return ProofGraphVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ProofGraph",
    "SingleProofGraph",
    "GraphPruner",
    "ProofGraphVisualizer",
    "FilterAndPruneEngine",
    "AuxExtractor",
    "SubgraphExtractor",
    "GSpanMiner",
    "MergedGraph",
]
