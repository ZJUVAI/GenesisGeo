"""
ClauseDAG-based problem sampler for geometric constructions.

This module provides a DAG-based approach to managing geometric clauses,
with ClauseDAG as the core data structure for tracking dependencies
and enabling efficient pruning.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Union

import numpy

from newclid.formulations.definition import DefinitionJGEX
from newclid.formulations.clause import Clause, translate_sentence
from newclid.statement import Statement
from newclid.configs import default_defs_path
from newclid.dependencies.symbols import Point
from newclid.dependencies.dependency_graph import DependencyGraph
from newclid.algebraic_reasoning.algebraic_manipulator import AlgebraicManipulator
from newclid.numerical.sketch import sketch
from newclid.numerical.geometries import LineNum, ObjNum, PointNum, reduce
from newclid.numerical.distances import (
    PointTooCloseError,
    PointTooFarError,
    check_too_far_numerical,
    check_too_close_numerical,
)

from newclid.generation.point_naming import PointNaming
from newclid.generation.constructions import resolve_construction_config
from newclid.generation.auxiliary import (
    add_potential_points as _add_potential_points_cpp,
)


def add_potential_points(point_names, coords, max_points, seed=0):
    """Wrapper that converts C++ output format to Python-compatible format."""
    result = _add_potential_points_cpp(point_names, coords, max_points, seed)
    converted = []
    for coord, construction_strings in result:
        constructions = []
        for constr_str in construction_strings:
            parts = constr_str.split()
            if len(parts) >= 1:
                ctype = parts[0]
                args = parts[1:] if len(parts) > 1 else []
                constructions.append((ctype, args))
        converted.append((coord, constructions))
    return converted


MAX_TRY = 10


def basics_sentence_count(construction_def) -> int:
    """
    Count how many predicate sentences a construction's `basics` clauses bind
    at once (e.g. `on_line` binds one `coll`, `centroid` binds eight sentences
    across coll/cong). This is a purely structural property of the definition,
    independent of its name.
    """
    return sum(len(clause.sentences) for clause in construction_def.basics)


def basics_complexity_weights(
    construction_defs: dict, bias: float
) -> dict[str, float]:
    """
    Multiplicative weight factor per construction: 1 / max(1, n_sentences) ** bias.

    Constructions whose basics reduce to a single elementary predicate (or none,
    e.g. `free`/`triangle`) keep factor 1.0; constructions that bind several
    predicate sentences at once are progressively down-weighted. `bias=0`
    disables this entirely (factor 1.0 for everything).
    """
    if bias == 0.0:
        return {}
    factors = {}
    for name, cdef in construction_defs.items():
        n = basics_sentence_count(cdef)
        factors[name] = 1.0 / max(1, n) ** bias
    return factors


def argument_roles(construction_def) -> dict[str, int] | None:
    """
    Coarse structural role of each of a construction's own args, derived from
    its `basics`: for an 8-token eqangle/eqratio sentence (4 segments), each
    arg's role is its position parity (0 = segment-start, 1 = segment-end).

    Returns None for constructions without such a sentence (most of them) --
    those are left untouched by role-biased argument merging.
    """
    for clause in construction_def.basics:
        for sentence in clause.sentences:
            if sentence[0] in ("eqangle", "eqratio") and len(sentence) == 9:
                names = sentence[1:]
                return {
                    name: pos % 2
                    for pos, name in enumerate(names)
                    if name in construction_def.args
                }
    return None


def required_distinct_pairs(construction_def) -> set[frozenset[str]]:
    """
    Derive "these argument names may never share a point" from a construction def.

    The construction's `require` clause is the authoritative statement of this:

      - ``diff p q``            -> p and q must be different points (else the
                                   segment has zero length)
      - ``ncoll a b c ...``     -> the listed points are pairwise distinct
                                   (non-collinear implies distinct)
      - ``npara/nperp/perp/cong a b c d`` -> both segments must be non-degenerate
        (a zero-length segment is parallel/perpendicular/congruent to nothing
        meaningful)

    Only names that are *not* constrained this way may be merged onto the same
    point, so e.g. ``on_tline``'s ``diff b c`` permits ``a == b`` while
    ``eqratio``'s ``diff a b, diff c d, diff e f`` leaves the rest free.
    """
    pairs: set[frozenset[str]] = set()
    for sentence in construction_def.require.sentences:
        if not sentence:
            continue
        pred, args = sentence[0], tuple(sentence[1:])
        if pred == "diff" and len(args) >= 2:
            pairs.add(frozenset(args[:2]))
        elif pred == "ncoll":
            for i in range(len(args)):
                for j in range(i + 1, len(args)):
                    pairs.add(frozenset((args[i], args[j])))
        elif pred in ("npara", "nperp", "perp", "cong") and len(args) >= 4:
            pairs.add(frozenset(args[:2]))
            pairs.add(frozenset(args[2:4]))
    return pairs


class EquivalencePairs:
    """
    Manages structural equivalence relationships between geometric points.

    Each pair (a, b) indicates that a and b are structurally equivalent:
    a construction performed on one side can be mirrored to the other.
    A point belongs to at most one pair. Points not in any pair implicitly
    map to themselves.
    """

    def __init__(self):
        self._pairs: list[tuple[str, str]] = []
        self._partner: dict[str, str] = {}

    def add_pair(self, a: str, b: str) -> None:
        """Add a new equivalent pair (a, b)."""
        self._pairs.append((a, b))
        self._partner[a] = b
        self._partner[b] = a

    def get_partner(self, point: str) -> str | None:
        """Return the partner of a point, or None if the point has no partner."""
        return self._partner.get(point)

    def is_in_pair(self, point: str) -> bool:
        """Check whether a point belongs to any equivalent pair."""
        return point in self._partner

    def get_pairs(self) -> list[tuple[str, str]]:
        """Return a copy of all equivalent pairs."""
        return list(self._pairs)

    def get_all_paired_points(self) -> set[str]:
        """Return the set of all points that belong to at least one pair."""
        return set(self._partner.keys())

    def __len__(self) -> int:
        return len(self._pairs)


class EquivalenceTriples:
    """
    Manages structural equivalence groups of three points (cyclic symmetry).

    A triple (a, b, c) means a -> b -> c -> a is a symmetry of the
    configuration: a construction performed at a can be rotated to b and c.
    This is the three-fold analogue of `EquivalencePairs`, used for triangle
    centred constructions such as the three angle bisectors, altitudes or
    side midpoints of one triangle.

    A point belongs to at most one triple. Points not in any triple implicitly
    map to themselves.
    """

    def __init__(self):
        self._triples: list[tuple[str, str, str]] = []
        self._triple_of: dict[str, tuple[str, str, str]] = {}
        self._position: dict[str, int] = {}

    def add_triple(self, a: str, b: str, c: str) -> None:
        """Add a new cyclic triple (a, b, c)."""
        triple = (a, b, c)
        self._triples.append(triple)
        for i, p in enumerate(triple):
            self._triple_of[p] = triple
            self._position[p] = i

    def is_in_triple(self, point: str) -> bool:
        """Check whether a point belongs to any triple."""
        return point in self._triple_of

    def get_triple(self, point: str) -> tuple[str, str, str] | None:
        """Return the triple a point belongs to, or None."""
        return self._triple_of.get(point)

    def rotate(self, point: str, k: int = 1) -> str:
        """Image of `point` under k applications of a -> b -> c -> a."""
        triple = self._triple_of.get(point)
        if triple is None:
            return point
        return triple[(self._position[point] + k) % 3]

    def get_triples(self) -> list[tuple[str, str, str]]:
        """Return a copy of all triples."""
        return list(self._triples)

    def get_all_tripled_points(self) -> set[str]:
        """Return the set of all points that belong to at least one triple."""
        return set(self._triple_of.keys())

    def __len__(self) -> int:
        return len(self._triples)


class ClauseNode:
    """
    A node in the ClauseDAG representing a single clause.

    Tracks the clause's points, construction dependencies, and relationships
    to other nodes in the DAG.
    """

    def __init__(self, clause: Clause):
        """
        Initialize a ClauseNode.

        Args:
            clause: The Clause object this node represents.
        """
        self.clause = clause
        self.points = self._extract_point_names()
        self.construction_deps = self._extract_construction_deps()
        self.rely_on_points = (
            set().union(*self.construction_deps) if self.construction_deps else set()
        )
        self.children: list[ClauseNode] = []
        self.parents: list[ClauseNode] = []
        self.depth: int = 0

    def _extract_point_names(self) -> set[str]:
        """Extract point names (without coordinates) from clause points."""
        return {p.split("@")[0] for p in self.clause.points}

    def _extract_construction_deps(self) -> list[set[str]]:
        """
        Extract dependency points for each construction in the clause.

        Returns:
            List of sets, where each set contains the dependency points
            for the corresponding construction sentence.
        """
        deps = []
        for sentence in self.clause.sentences:
            if not sentence:
                deps.append(set())
                continue
            # Extract point names from sentence arguments (skip construction
            # name).  Angle values such as "45o" begin with a digit and are
            # already excluded above; additionally dropping every name ending
            # in "o" would orphan the point literally named "o", letting prune
            # delete the clause that defines it.
            sentence_points = {s for s in sentence[1:] if s and s[0].isalpha()}
            # Dependency points are those not defined by this clause
            dep_points = sentence_points - self.points
            deps.append(dep_points)
        return deps


class ClauseDAG:
    """
    Directed Acyclic Graph for managing geometric clauses.

    Tracks clause dependencies, enables topological sorting, and provides
    pruning functionality to keep only the deepest construction chains.
    """

    def __init__(self, seed: int = 0):
        """Initialize an empty ClauseDAG."""
        self.seed = seed
        self.nodes: list[ClauseNode] = []
        self.point_to_node: dict[str, ClauseNode] = {}
        self.point_coords: dict[str, tuple[float, float]] = {}

    def add_clause(self, clause: Clause) -> ClauseNode:
        """
        Add a clause to the DAG.

        Establishes parent-child relationships based on point dependencies
        and computes the node's depth in the DAG.

        Args:
            clause: The Clause to add.

        Returns:
            The created ClauseNode.
        """
        node = ClauseNode(clause)

        # Build parent-child relationships
        parent_nodes = set()
        for dep_point in node.rely_on_points:
            if dep_point in self.point_to_node:
                parent_nodes.add(self.point_to_node[dep_point])

        for parent in parent_nodes:
            parent.children.append(node)
            node.parents.append(parent)

        # Compute depth (max parent depth + 1)
        node.depth = max((p.depth for p in node.parents), default=-1) + 1

        # Extract coordinates and update point mappings
        for p in clause.points:
            point_name = p.split("@")[0]
            if "@" in p:
                parts = p.split("@")
                try:
                    if len(parts) >= 2 and "_" in parts[1]:
                        # Format: name@x_y
                        x_str, y_str = parts[1].split("_")
                        self.point_coords[point_name] = (float(x_str), float(y_str))
                    elif len(parts) >= 3:
                        # Format: name@x@y
                        self.point_coords[point_name] = (
                            float(parts[1]),
                            float(parts[2]),
                        )
                except (ValueError, IndexError):
                    pass
            self.point_to_node[point_name] = node

        self.nodes.append(node)
        return node

    def prune(self, topk: int = 1) -> None:
        """
        Prune the DAG to keep only the deepest k layers of leaf nodes and their ancestors.

        This removes branches that don't contribute to the deepest construction
        chains, simplifying the problem while preserving the most complex paths.

        Args:
            topk: Number of deepest layers to keep (default 1, i.e., only the deepest layer).
        """
        if not self.nodes:
            return

        # Find maximum depth
        max_depth = max(node.depth for node in self.nodes)
        min_keep_depth = max(0, max_depth - topk + 1)

        # Find leaf nodes in the top k layers (depth >= min_keep_depth)
        target_leaves = [
            n for n in self.nodes if n.depth >= min_keep_depth and not n.children
        ]

        if not target_leaves:
            # If no leaves in target range, pick random nodes from target depth range
            target_nodes = [n for n in self.nodes if n.depth >= min_keep_depth]
            if target_nodes:
                target_leaves = [random.choice(target_nodes)]
            else:
                return

        # Collect nodes to keep (target leaves + all ancestors)
        keep_nodes: set[ClauseNode] = set()

        def collect_ancestors(node: ClauseNode) -> None:
            if node in keep_nodes:
                return
            keep_nodes.add(node)
            for parent in node.parents:
                collect_ancestors(parent)

        for leaf in target_leaves:
            collect_ancestors(leaf)

        # Find points to remove
        removed_points: set[str] = set()
        for node in self.nodes:
            if node not in keep_nodes:
                removed_points.update(node.points)

        # Update data structures
        self.nodes = [n for n in self.nodes if n in keep_nodes]
        for point in removed_points:
            self.point_to_node.pop(point, None)
            self.point_coords.pop(point, None)

        # Clean up children references
        for node in self.nodes:
            node.children = [c for c in node.children if c in keep_nodes]

    def add_auxiliary_points(
        self, max_points: int = 2, point_naming: PointNaming = None
    ) -> None:
        """
        Add potential auxiliary points to the DAG.

        Finds meaningful geometric points (intersections, midpoints, reflections, feet)
        and adds them as new clauses to the DAG.

        Args:
            max_points: Maximum number of auxiliary points to add.
            point_naming: PointNaming instance for generating new point names.
                         If None, creates a new instance.
        """
        if not self.nodes or not self.point_coords:
            return

        if point_naming is None:
            point_naming = PointNaming()
            # Pre-define existing points
            for p in sorted(self.point_to_node.keys()):
                point_naming.define_points([p])

        # Use the DAG's coordinate data directly
        point_names = list(self.point_coords.keys())
        coords = self.point_coords.copy()

        # Call the new add_potential_points function
        potential_points = add_potential_points(
            point_names, coords, max_points, seed=self.seed
        )

        # Directly create Clause and add to DAG
        for coord, constructions in potential_points:
            new_name = point_naming.prefetch_points(1)[0]
            point_naming.define_points([new_name])

            # Build Clause
            point_with_coord = f"{new_name}@{coord[0]}_{coord[1]}"
            sentences = tuple(
                tuple([ctype, new_name] + args) for ctype, args in constructions
            )
            clause = Clause((point_with_coord,), sentences)
            self.add_clause(clause)

    def get_all_points(self) -> set[str]:
        """Get all point names in the DAG."""
        return set(self.point_to_node.keys())

    def get_max_depth(self) -> int:
        """Get the maximum depth of any node in the DAG."""
        if not self.nodes:
            return -1
        return max(node.depth for node in self.nodes)

    def _topological_sort(self) -> list[ClauseNode]:
        """
        Perform topological sort on the DAG.

        Nodes are sorted so that dependencies come first, with ties broken
        by depth (shallower first), then lexicographic order of point names,
        then number of dependencies (fewer first).

        Returns:
            List of ClauseNodes in topological order.
        """
        result = []
        defined_points: set[str] = set()
        remaining = set(self.nodes)

        while remaining:
            # Find nodes whose dependencies are satisfied
            ready = [n for n in remaining if n.rely_on_points <= defined_points]
            if not ready:
                # Circular dependency (shouldn't happen)
                break

            # Sort by: 1) depth (shallower first), 2) min point name, 3) number of dependencies
            ready.sort(key=lambda n: (n.depth, min(n.points), len(n.rely_on_points)))

            # Take the first ready node
            node = ready[0]
            result.append(node)
            defined_points.update(node.points)
            remaining.remove(node)

        return result

    def to_problem(self, with_coords: bool = False, rename: bool = True) -> str:
        """
        Convert the DAG to a problem string.

        Performs topological sort, optionally renames points to standard names (a, b, c, ...),
        and formats the output.

        Args:
            with_coords: Whether to include coordinate information.
            rename: Whether to rename points using PointNaming convention.

        Returns:
            Problem string with clauses separated by semicolons.
        """
        if not self.nodes:
            return ""

        # Topological sort
        sorted_nodes = self._topological_sort()

        # Build point renaming map
        old_to_new: dict[str, str] = {}
        if rename:
            point_naming = PointNaming()
            for node in sorted_nodes:
                for p in sorted(node.points):
                    if p not in old_to_new:
                        new_names = point_naming.prefetch_points(1)
                        point_naming.define_points(new_names)
                        old_to_new[p] = new_names[0]
        else:
            # Identity mapping (no renaming)
            for node in sorted_nodes:
                for p in node.points:
                    old_to_new[p] = p

        # Generate output
        clause_strs = []
        for node in sorted_nodes:
            renamed_clause = node.clause.renamed(old_to_new)
            if with_coords:
                clause_strs.append(str(renamed_clause))
            else:
                # Remove coordinates from points
                points_str = " ".join(old_to_new[p] for p in sorted(node.points))
                sentences_str = ", ".join(" ".join(s) for s in renamed_clause.sentences)
                clause_strs.append(f"{points_str} = {sentences_str}")

        return "; ".join(clause_strs)


class ProblemSampler:
    """
    Sampler for geometric problems using ClauseDAG.

    Samples geometric constructions by combining basic shapes, intersections,
    and other construction types, managing them through a DAG structure.

    Supports weighted construction sampling and construction-type
    compatibility boosts to increase the probability of geometrically
    meaningful configurations without over-constraining the search space.
    """

    # Down-weight constructions that bind several predicate sentences at once
    # (e.g. `centroid`, `cc_tangent`) in favor of ones that bind a single
    # elementary predicate (e.g. `on_line`, `midpoint`) -- see
    # basics_complexity_weights(). Baked in as a class default rather than
    # threaded through every ProblemSampler(...) call site / pipeline config.
    DEFAULT_BASICS_COMPLEXITY_BIAS = 2.5

    # When a construction's own predicate exposes segment-start/segment-end
    # argument roles (e.g. eqratio's a,c,e,g vs b,d,f,h), bias arg-merging
    # toward pairing same-role arguments -- mirrors r27-style "apex" figures
    # where several segment-starts coincide at one point. Baked in as a class
    # default for the same reason as DEFAULT_BASICS_COMPLEXITY_BIAS above.
    DEFAULT_ROLE_BIAS = 0.85

    # Default strength of argument reuse (see arg_repetition below) when no
    # external construction_config sets construction_arg_repetition. Without
    # this, DEFAULT_ROLE_BIAS above is dead code on the production pipeline,
    # which runs with construction_config=None.
    DEFAULT_ARG_REPETITION = 0.35

    def __init__(
        self,
        seed: int = None,
        defs: dict = None,
        construction_config: dict | None = None,
        basics_complexity_bias: float | None = None,
    ):
        """
        Initialize the problem sampler.

        Args:
            seed: Random seed for reproducibility.
            defs: Definition dictionary. If None, loads from default path.
            construction_config: Optional external config defining the
                construction sets and the three sampler steps.
            basics_complexity_bias: Exponent for down-weighting constructions
                whose `basics` bind several predicate sentences at once
                (factor = 1 / max(1, n_sentences) ** bias). None (default)
                uses DEFAULT_BASICS_COMPLEXITY_BIAS; pass 0.0 explicitly to
                disable and get the historical unweighted behavior.
        """
        self.construction_defs = defs or DefinitionJGEX.to_dict(
            DefinitionJGEX.parse_txt_file(default_defs_path())
        )
        resolved_profile = resolve_construction_config(
            construction_config=construction_config,
            available_constructions=set(self.construction_defs.keys()),
        )
        self.rng = numpy.random.default_rng(seed)
        random.seed(seed)
        self.seed = seed if seed is not None else 0
        self.point_naming: PointNaming = None
        self.symbols_graph = None
        self.dep_graph: DependencyGraph = None
        self.dag: ClauseDAG = None
        self.construction_config = construction_config
        self.step1_pool = resolved_profile["step1_pool"]
        self.step2_pool = resolved_profile["step2_pool"]
        self.step3_pool = resolved_profile["step3_pool"]
        self._weights = dict(resolved_profile.get("weights", {}))
        if basics_complexity_bias is None:
            basics_complexity_bias = self.DEFAULT_BASICS_COMPLEXITY_BIAS
        complexity_factors = basics_complexity_weights(
            self.construction_defs, basics_complexity_bias
        )
        for name, factor in complexity_factors.items():
            self._weights[name] = self._weights.get(name, 1.0) * factor
        self._compatibility = resolved_profile.get("compatibility", {})
        # Strength of argument reuse: > 0 lets a construction map several of its
        # arguments onto one point (as far as `require` permits), which is how
        # figures like `eqratio f a b b c d e e` become samplable at all.
        # 0.0 = the historical behaviour, every argument a distinct point.
        # resolve_construction_config always fills in "arg_repetition" (0.0
        # when construction_config doesn't set construction_arg_repetition),
        # so an explicit external config is distinguished by inspecting the
        # raw config dict directly rather than resolved_profile.
        if (
            isinstance(construction_config, dict)
            and "construction_arg_repetition" in construction_config
        ):
            self.arg_repetition = float(resolved_profile["arg_repetition"])
        else:
            self.arg_repetition = self.DEFAULT_ARG_REPETITION
        self.role_bias = self.DEFAULT_ROLE_BIAS

    def generate(
        self,
        length: int = 0,
        add_auxiliary: bool = True,
        max_auxiliary_points: int = 2,
        prune: bool = True,
        prune_topk: int = 1,
        with_coords: bool = True,
        rename: bool = True,
        return_timings: bool = False,
        max_points: int = 0,
        symmetry_enabled: bool = True,
        triple_symmetry_enabled: bool = True,
        arg_repetition: float | None = None,
    ) -> Union[str, tuple[str, dict]]:
        """
        Generate geometric clauses.

        Args:
            length: Number of clause sets to generate.
            add_auxiliary: Whether to add auxiliary points using enhancement.
            max_auxiliary_points: Maximum number of auxiliary points to add.
            prune: Whether to prune clauses to preserve only the deepest chains.
            prune_topk: Number of deepest layers to keep when pruning (default 1).
            with_coords: Whether to include coordinate information in output.
            rename: Whether to rename points to standard names (a, b, c, ...).
            return_timings: Whether to return detailed timing information.
            max_points: Target maximum point count, used to compute dynamic
                probabilities for symmetry modes. A heuristic default is used
                when set to 0.
            symmetry_enabled: Whether structural-equivalence modes (create,
                expand) are active. When False the sampler behaves identically
                to the original version.
            triple_symmetry_enabled: Whether the three-fold cyclic modes
                (create_triple, expand_triple) are active. They rotate a
                construction over the three vertices of a triangle and record
                the pairwise intersections of the rotated lines. Enabled by
                default; pass False for the pre-triple behaviour.

        Returns:
            A string of generated clauses separated by semicolons.
            If return_timings=True, returns (problem_string, timings_dict).
        """
        timings = {}

        if arg_repetition is not None:
            self.arg_repetition = float(arg_repetition)

        self.point_naming = PointNaming()
        self.dep_graph = DependencyGraph(AlgebraicManipulator())
        self.symbols_graph = self.dep_graph.symbols_graph
        self.dag = ClauseDAG(seed=self.seed)
        self.equivalent_pairs = EquivalencePairs()
        self.equivalent_triples = EquivalenceTriples()
        self._triple_construction_cache: list[str] | None = None

        if max_points <= 0:
            max_points = max(length * 3 + 10, 20)

        # Step 1: Sample initial clauses
        t0 = time.time()
        for clause_set in range(length):
            if clause_set == 0:
                # First clause: always a BASIC construction (no symmetry)
                new_clause_strs = self._sample_clauses(
                    construction_type_candidates=self.step1_pool,
                    num_construction_types=1,
                    num_clauses=1,
                )
            else:
                new_clause_strs = self._sample_one_step(
                    max_points=max_points,
                    symmetry_enabled=symmetry_enabled,
                    triple_enabled=triple_symmetry_enabled,
                )

            # Parse clause strings and add to DAG
            for clause_str in new_clause_strs:
                clause = self._parse_clause_str(clause_str)
                self.dag.add_clause(clause)
        timings["sampling"] = time.time() - t0

        # Step 2: Prune to keep deepest chains
        t0 = time.time()
        if prune:
            self.dag.prune(topk=prune_topk)
        timings["prune"] = time.time() - t0

        # Step 3: Add auxiliary points
        t0 = time.time()
        if add_auxiliary:
            self.dag.add_auxiliary_points(
                max_points=max_auxiliary_points, point_naming=self.point_naming
            )
        timings["add_auxiliary"] = time.time() - t0

        # Step 4: Convert to problem string
        t0 = time.time()
        result = self.dag.to_problem(with_coords=with_coords, rename=rename)
        timings["to_problem"] = time.time() - t0

        if return_timings:
            result = result, timings
        return result

    # ------------------------------------------------------------------
    # Symmetry-aware sampling helpers
    # ------------------------------------------------------------------

    def _sample_one_step(
        self,
        max_points: int,
        symmetry_enabled: bool,
        triple_enabled: bool = False,
    ) -> list[str]:
        """
        Sample one clause set using either normal / create / expand mode.

        After the initial BASIC clause, each subsequent step chooses a
        sampling mode dynamically.  When symmetry is disabled this always
        falls back to the original normal behaviour.
        """
        if not symmetry_enabled:
            return self._sample_clause_normal()

        mode = self._choose_mode(max_points, triple_enabled)

        if mode == "normal":
            return self._sample_clause_normal()
        elif mode == "create":
            return self._create_equivalent_pair()
        elif mode == "expand":
            return self._expand_equivalent_pair()
        elif mode == "create_triple":
            return self._create_equivalent_triple()
        else:  # expand_triple
            return self._expand_equivalent_triple()

    def _choose_mode(self, max_points: int, triple_enabled: bool = False) -> str:
        """
        Choose a sampling mode based on the current graph state.

        Probabilities are computed dynamically:
            r = N / max_points   (progress ratio)
            P_create ∼ base_create * (1-r) / (1 + num_pairs)
            P_expand ∼ base_expand * min(1, num_pairs / k)
            P_create_triple ∼ base_create_triple * (1-r) / (1 + num_triples)
            P_expand_triple ∼ base_expand_triple * min(1, num_triples / k_t)
            P_normal = 1 - the rest

        The triple modes mirror the pairwise ones but rotate over three
        vertices.  Once a triangle exists, expanding it is much more likely
        than seeding a new one, so a chosen triangle tends to be reused.
        """
        N = len(self.point_naming.defined_points)
        r = N / max(max_points, 1)
        num_pairs = len(self.equivalent_pairs)
        num_triples = len(self.equivalent_triples) if triple_enabled else 0

        base_create = 0.2
        base_expand = 0.3
        k = 5.0

        # Create probability: peaks early, decays as graph fills and pairs accumulate
        shortage = 1.0 / (1.0 + num_pairs)
        if num_pairs == 0:
            shortage *= 2.5  # boost when no pair exists yet
        p_create = base_create * (1.0 - r) * shortage

        # Expand probability: only available when pairs exist
        if num_pairs > 0:
            p_expand = base_expand * min(1.0, num_pairs / k)
        else:
            p_expand = 0.0

        p_create_triple = 0.0
        p_expand_triple = 0.0
        if triple_enabled:
            base_create_triple = 0.12
            base_expand_triple = 0.5
            if num_triples > 0:
                p_expand_triple = base_expand_triple * min(1.0, num_triples / 2.0)
                p_create_triple = (
                    base_create_triple * (1.0 - r) / (1.0 + num_triples)
                )
            else:
                p_create_triple = base_create_triple * (1.0 - r) * 2.0

        p_create = max(0.0, min(1.0, p_create))
        p_expand = max(0.0, min(1.0, p_expand))
        p_create_triple = max(0.0, min(1.0, p_create_triple))
        p_expand_triple = max(0.0, min(1.0, p_expand_triple))
        p_normal = max(
            0.0,
            1.0 - p_create - p_expand - p_create_triple - p_expand_triple,
        )

        # Normalise so the probabilities sum to 1
        total = p_normal + p_create + p_expand + p_create_triple + p_expand_triple
        if total == 0.0:
            return "normal"
        p_normal /= total
        p_create /= total
        p_create_triple /= total

        r_val = random.random()
        if r_val < p_normal:
            return "normal"
        r_val -= p_normal
        if r_val < p_create:
            return "create"
        r_val -= p_create
        if r_val < p_expand:
            return "expand"
        r_val -= p_expand
        if r_val < p_create_triple:
            return "create_triple"
        return "expand_triple"

    # 抽 step2_pool（两个构造相交定义一个点，产出两条句子的clause）而不是
    # step3_pool（单构造定义一个点）的概率。原来是0.5；调大这个值会让更多
    # 单点clause具备"两句"形状，从而能进入反向重构的候选池
    # （`_split_defining_clause` 要求clause恰好两条句子才会拆分）。
    STEP2_POOL_PROB = 0.8

    def _sample_clause_normal(self) -> list[str]:
        """Original normal sampling: randomly pick step2 or step3 pool."""
        if random.random() < self.STEP2_POOL_PROB:
            return self._sample_clauses(
                construction_type_candidates=self.step2_pool,
                num_construction_types=2,
                num_clauses=2,
                num_points=1,
            )
        else:
            return self._sample_clauses(
                construction_type_candidates=self.step3_pool,
                num_construction_types=1,
                num_clauses=1,
            )

    def _create_equivalent_pair(self) -> list[str]:
        """
        Create a new equivalent pair from two unused points.

        Does NOT generate any new clause.  Returns an empty list so the
        main loop skips the DAG-add step for this iteration.
        """
        paired = self.equivalent_pairs.get_all_paired_points()
        available = [
            p for p in self.point_naming.defined_points if p not in paired
        ]
        if len(available) < 2:
            return self._sample_clause_normal()

        a, b = random.sample(available, 2)
        self.equivalent_pairs.add_pair(a, b)
        return []

    def _expand_equivalent_pair(self) -> list[str]:
        """
        Expand an existing equivalent pair by mirroring a construction.

        1. Pick a random pair (a, b).
        2. Sample a construction using only "one side" of each pair as input.
        3. Build the mirrored construction by replacing every paired input
           point with its partner.
        4. If the mirrored construction uses the same input-point set the
           construction is treated as non-symmetric (submitted once, no new
           pair).
        5. If the mirrored construction fails validation, fall back to
           keeping only the original (equivalent to a normal construction).
        """
        pairs = self.equivalent_pairs.get_pairs()
        if not pairs:
            return self._sample_clause_normal()

        # Pick a random pair whose *a* side we will sample from
        a, b = random.choice(pairs)

        # Build the set of point names to exclude from candidate selection.
        # For the chosen pair we exclude b (so we only sample from a's side).
        # For every other pair we randomly exclude one side.
        excluded: set[str] = {b}
        for p1, p2 in pairs:
            if (p1 == a and p2 == b) or (p1 == b and p2 == a):
                continue
            excluded.add(random.choice([p1, p2]))

        # Pick pool (same logic as normal mode)
        if random.random() < self.STEP2_POOL_PROB:
            pool = self.step2_pool
            num_construction_types = 2
            num_clauses = 2
            num_points = 1
        else:
            pool = self.step3_pool
            num_construction_types = 1
            num_clauses = 1
            num_points = None

        # ---- build original (commits to point_naming / symbols_graph) ----
        result = self._sample_clauses_return_info(
            construction_type_candidates=pool,
            num_construction_types=num_construction_types,
            num_clauses=num_clauses,
            num_points=num_points,
            exclude_points=excluded,
        )
        if result is None:
            return self._sample_clause_normal()

        original_clause_strs, selected_types, original_mappings_saved, num_points_used = result

        # ---- build mirrored mapping ----
        mirrored_mappings: list[dict[str, str]] = []
        for mapping in original_mappings_saved:
            mir: dict[str, str] = {}
            for arg, val in mapping.items():
                partner = self.equivalent_pairs.get_partner(val)
                mir[arg] = partner if partner is not None else val
            mirrored_mappings.append(mir)

        # ---- check whether mirrored is the same construction ----
        original_inputs: set[str] = set()
        for m in original_mappings_saved:
            original_inputs.update(m.values())
        mirrored_inputs: set[str] = set()
        for m in mirrored_mappings:
            mirrored_inputs.update(m.values())
        if original_inputs == mirrored_inputs:
            # Same construction — keep only the original (already committed)
            return original_clause_strs

        # ---- build mirrored clauses ----
        try:
            mirrored_clause_strs = self._build_clauses_from_mappings(
                selected_types=selected_types,
                args_mappings=mirrored_mappings,
                num_clauses=num_clauses,
                num_points=num_points_used,
            )
        except Exception:
            logging.debug("Mirrored construction failed; keeping original only.")
            return original_clause_strs

        # ---- both sides succeeded: record new equivalent pairs ----
        orig_new_points = self._extract_new_point_names(original_clause_strs)
        mir_new_points = self._extract_new_point_names(mirrored_clause_strs)
        for op, mp in zip(orig_new_points, mir_new_points):
            self.equivalent_pairs.add_pair(op, mp)

        return original_clause_strs + mirrored_clause_strs

    # ------------------------------------------------------------------
    # Three-fold (triangle) symmetry helpers
    # ------------------------------------------------------------------

    def _create_equivalent_triple(self) -> list[str]:
        """
        Create a new cyclic triple from three unused points.

        Does NOT generate any new clause.  Returns an empty list so the
        main loop skips the DAG-add step for this iteration, mirroring
        `_create_equivalent_pair`.
        """
        used = self.equivalent_pairs.get_all_paired_points() | (
            self.equivalent_triples.get_all_tripled_points()
        )
        available = [p for p in self.point_naming.defined_points if p not in used]
        if len(available) < 3:
            return self._sample_clause_normal()

        a, b, c = random.sample(available, 3)
        self.equivalent_triples.add_triple(a, b, c)
        return []

    def _expand_equivalent_triple(self) -> list[str]:
        """
        Expand an existing triple by rotating a construction over its vertices.

        1. Pick a random triple (p, q, r).
        2. Sample a construction that consumes two or three points and produces
           a single new point, then map its arguments onto the triangle.
        3. Rotate that mapping twice (p -> q -> r -> p) to obtain three
           constructions, build them, and record the new points as a triple.
        4. When each rotation yields a line anchored at exactly one triangle
           vertex, add the pairwise intersections of those three lines as
           further new points.

        Constructions that cannot be rotated (degenerate results, e.g. three
        copies of the same circumcenter) stop early and keep whatever was
        built, like `_expand_equivalent_pair` does when mirroring fails.
        """
        triples = self.equivalent_triples.get_triples()
        if not triples:
            return self._sample_clause_normal()

        triangle = random.choice(triples)
        construction = self._pick_triple_construction()
        if construction is None:
            return self._sample_clause_normal()

        construction_def = self.construction_defs[construction]
        base_points = list(triangle[: len(construction_def.args)])

        # ---- build the three rotated constructions ----
        clause_strs: list[str] = []
        built: list[tuple[dict[str, str], list[str]]] = []
        for step in range(3):
            args_mapping = {
                arg: self.equivalent_triples.rotate(base_points[i], step)
                for i, arg in enumerate(construction_def.args)
            }
            built_clause = self._build_explicit_clause(
                construction, args_mapping, len(construction_def.points)
            )
            if built_clause is None:
                break
            clause_str, new_points, full_mapping = built_clause
            clause_strs.append(clause_str)
            built.append((full_mapping, new_points))

        if not built:
            return self._sample_clause_normal()

        # ---- record the rotated new points as a new triple ----
        if len(built) == 3 and all(len(pts) == 1 for _, pts in built):
            self.equivalent_triples.add_triple(*[pts[0] for _, pts in built])

            # ---- pairwise intersections of the three rotated lines ----
            clause_strs.extend(self._build_triple_intersections(construction, built))

        return clause_strs

    def _pick_triple_construction(self) -> str | None:
        """
        Pick a construction suitable for cyclic rotation over a triangle.

        Restricted to constructions that consume exactly two or three points
        and produce a single new point, so that every argument can be mapped
        onto a triangle vertex.  The admissible set is cached per sampler.
        """
        if self._triple_construction_cache is None:
            candidates: list[str] = []
            for name in dict.fromkeys(self.step2_pool + self.step3_pool):
                construction_def = self.construction_defs.get(name)
                if construction_def is None:
                    continue
                if len(construction_def.points) != 1:
                    continue
                if len(construction_def.args) not in (2, 3):
                    continue
                candidates.append(name)
            self._triple_construction_cache = candidates

        if not self._triple_construction_cache:
            return None
        return self._weighted_choice(self._triple_construction_cache)

    def _build_explicit_clause(
        self,
        construction: str,
        args_mapping: dict[str, str],
        num_points: int,
    ) -> tuple[str, list[str], dict[str, str]] | None:
        """
        Build a single clause from an explicit argument mapping.

        The counterpart of `_sample_clauses` for symmetry-driven steps, where
        the arguments are dictated by the rotation rather than sampled.
        Returns (clause_str, new_point_names, full_mapping), or None when the
        construction cannot be instantiated on those arguments.
        """
        construction_def = self.construction_defs[construction]
        if not self._validate_construction_requirements(construction_def, args_mapping):
            return None

        numerics = self._extract_numerics(construction_def, args_mapping)
        construction_points = self._extract_construction_points(
            construction_def, args_mapping
        )

        try:
            new_points = self.point_naming.prefetch_points(num_points)
            self._draw_diagram(new_points, numerics, construction_points)
            self.point_naming.define_points(new_points)
        except Exception as e:
            logging.debug(f"Explicit clause construction failed: {e}")
            return None

        full_mapping = dict(args_mapping)
        full_mapping.update(dict(zip(construction_def.points, new_points)))
        self._validate_construction_basics(construction_def, full_mapping)

        point_strs = self._format_points_with_coords(new_points)
        text = self._format_construction_text(construction_def, full_mapping)
        return " ".join(point_strs) + " = " + text, new_points, full_mapping

    def _axis_vertex_for_mapping(
        self, construction: str, full_mapping: dict[str, str]
    ) -> str | None:
        """
        Find the line a construction draws through exactly one of its points.

        Returns the name of that point (the line's anchor), or None when the
        construction produces no such line -- e.g. it yields a point (midpoint)
        or a line through no single named argument (perpendicular bisector).

        Requiring *exactly* one anchor keeps us from picking up the base side of
        an altitude, which passes through two of the triangle's vertices. Only
        the construction's input arguments are considered, so the freshly
        created point -- which trivially lies on the line -- is not an anchor.
        """
        construction_def = self.construction_defs[construction]
        numerics = self._extract_numerics(construction_def, full_mapping)

        named: dict[str, PointNum] = {}
        for arg in construction_def.args:
            name = full_mapping.get(arg)
            if name is None or name in named:
                continue
            node = self.symbols_graph.name2node.get(name)
            if node is not None and hasattr(node, "num"):
                named[name] = node.num
        if not named:
            return None

        for numeric in numerics:
            args: list = []
            usable = True
            for token in numeric[1:]:
                if isinstance(token, str) and token and token[0].isalpha():
                    node = self.symbols_graph.name2node.get(token)
                    if node is None or not hasattr(node, "num"):
                        usable = False
                        break
                    args.append(node.num)
                else:
                    args.append(token)
            if not usable:
                continue

            try:
                objects = sketch(numeric[0], tuple(args), self.rng)
            except Exception:
                continue

            for obj in objects:
                if not isinstance(obj, LineNum):
                    continue
                on_line = [name for name, p in named.items() if obj(p) == 0]
                if len(on_line) == 1:
                    return on_line[0]

        return None

    def _build_triple_intersections(
        self,
        construction: str,
        built: list[tuple[dict[str, str], list[str]]],
    ) -> list[str]:
        """
        Add the pairwise intersections of the three rotated construction lines.

        Each rotated line is anchored at one triangle vertex and carries its
        own new point, so the intersection of two of them is expressed with
        `intersection_ll`.  Rotations whose line has no single anchor are
        skipped, as are intersections that are numerically degenerate (the
        three lines of an angle-bisector rotation all meet at the incenter,
        which the too-close check rejects).
        """
        anchors: list[tuple[str, str] | None] = []
        for full_mapping, new_points in built:
            anchor = self._axis_vertex_for_mapping(construction, full_mapping)
            anchors.append((anchor, new_points[0]) if anchor else None)

        clause_strs: list[str] = []
        for i in range(3):
            j = (i + 1) % 3
            if anchors[i] is None or anchors[j] is None:
                continue
            vertex_i, point_i = anchors[i]
            vertex_j, point_j = anchors[j]
            built_clause = self._build_explicit_clause(
                "intersection_ll",
                {"a": vertex_i, "b": point_i, "c": vertex_j, "d": point_j},
                1,
            )
            if built_clause is None:
                continue
            clause_strs.append(built_clause[0])

        return clause_strs

    # ------------------------------------------------------------------
    # Refactored clause-sampling internals
    # ------------------------------------------------------------------

    def _sample_clauses(
        self,
        construction_type_candidates: list[str],
        num_construction_types: int,
        num_clauses: int,
        num_points: int = None,
        exclude_points: set[str] | None = None,
    ) -> list[str]:
        """
        Sample clauses using specified construction type candidates.

        Args:
            construction_type_candidates: List of candidate construction types.
            num_construction_types: Number of construction types per clause.
            num_clauses: Number of clauses to generate.
            num_points: Number of new points per clause (optional).
            exclude_points: Point names to exclude from arg selection.
        """
        num_points_backup = num_points
        for _ in range(MAX_TRY):
            try:
                (
                    selected_construction_types,
                    args_mappings,
                    numeric_list,
                    construction_points,
                    num_points,
                ) = self._select_construction_types(
                    construction_type_candidates,
                    num_construction_types,
                    num_points_backup,
                    exclude_points=exclude_points,
                )

                if len(selected_construction_types) != num_construction_types:
                    continue

                try:
                    clause_strs = self._build_clause_strings(
                        selected_construction_types,
                        args_mappings,
                        numeric_list,
                        construction_points,
                        num_clauses,
                        num_points,
                    )
                except Exception as e:
                    logging.debug(f"Clause application failed: {e}")
                    continue

                return clause_strs

            except Exception as e:
                logging.debug(f"Clause generation attempt failed: {e}")
                continue
        return []

    def _sample_clauses_return_info(
        self,
        construction_type_candidates: list[str],
        num_construction_types: int,
        num_clauses: int,
        num_points: int = None,
        exclude_points: set[str] | None = None,
    ) -> tuple[list[str], list[str], list[dict[str, str]], int] | None:
        """
        Like _sample_clauses but also returns construction metadata.

        Returns:
            (clause_strs, selected_types, original_mappings, num_points)
            or None if all attempts failed.

        The returned ``original_mappings`` are snapshots taken BEFORE
        _build_clause_strings updates them with output-point names, so they
        only contain input-arg → point-name entries.
        """
        num_points_backup = num_points
        for _ in range(MAX_TRY):
            try:
                (
                    selected_types,
                    args_mappings,
                    numeric_list,
                    construction_points,
                    num_points,
                ) = self._select_construction_types(
                    construction_type_candidates,
                    num_construction_types,
                    num_points_backup,
                    exclude_points=exclude_points,
                )

                if len(selected_types) != num_construction_types:
                    continue

                # Snapshot input-arg mappings before they are mutated
                original_mappings = [dict(m) for m in args_mappings]

                try:
                    clause_strs = self._build_clause_strings(
                        selected_types,
                        args_mappings,
                        numeric_list,
                        construction_points,
                        num_clauses,
                        num_points,
                    )
                except Exception as e:
                    logging.debug(f"Clause application failed: {e}")
                    continue

                return clause_strs, selected_types, original_mappings, num_points

            except Exception as e:
                logging.debug(f"Clause generation attempt failed: {e}")
                continue
        return None

    def _build_clauses_from_mappings(
        self,
        selected_types: list[str],
        args_mappings: list[dict[str, str]],
        num_clauses: int,
        num_points: int,
    ) -> list[str]:
        """
        Build clauses from pre-determined construction types and arg mappings.

        Re-validates requirements, recomputes numerics and construction points,
        then delegates to _build_clause_strings.  Used for the mirrored side of
        an expand operation.
        """
        numeric_list: list[tuple] = []
        construction_points: set[Point] = set()

        for construction, mapping in zip(selected_types, args_mappings):
            construction_def = self.construction_defs[construction]
            if not self._validate_construction_requirements(construction_def, mapping):
                raise Exception(
                    f"Mirrored construction '{construction}' requirements not met"
                )
            numerics = self._extract_numerics(construction_def, mapping)
            numeric_list.extend(numerics)
            cp = self._extract_construction_points(construction_def, mapping)
            construction_points.update(cp)

        return self._build_clause_strings(
            selected_types,
            args_mappings,
            numeric_list,
            construction_points,
            num_clauses,
            num_points,
        )

    @staticmethod
    def _extract_new_point_names(clause_strs: list[str]) -> list[str]:
        """
        Extract the new point names from clause strings.

        Each clause_str has the form "a@x_y b@x_y = ...".  We return the
        point name (the part before '@') for every point in every clause.
        """
        names: list[str] = []
        for clause_str in clause_strs:
            parts = clause_str.split("=")
            if len(parts) != 2:
                continue
            for token in parts[0].strip().split():
                name = token.split("@")[0]
                names.append(name)
        return names

    def _parse_clause_str(self, clause_str: str) -> Clause:
        """
        Parse a clause string into a Clause object.

        Args:
            clause_str: String in format "a@x_y b@x_y = construction1 args, construction2 args"

        Returns:
            Parsed Clause object.
        """
        parts = clause_str.split("=")
        if len(parts) != 2:
            raise ValueError(f"Invalid clause string: {clause_str}")

        points_part, constructions_part = parts
        points = tuple(points_part.strip().split())

        sentences = []
        for c in constructions_part.strip().split(","):
            tokens = c.strip().split()
            if tokens:
                sentences.append(tuple(tokens))

        return Clause(points, tuple(sentences))

    def _weighted_choice(
        self,
        candidates: list[str],
        compat_boost: dict[str, float] | None = None,
    ) -> str:
        """
        Weighted random selection from construction type candidates.

        Base weight from `construction_weights` config (default 1.0),
        multiplied by any compatibility boost from the previously
        selected construction type.

        Args:
            candidates: Construction type names to choose from.
            compat_boost: Optional dict mapping type name → multiplier.
                          Applied multiplicatively on top of base weight.

        Returns:
            Selected construction type name.
        """
        weights = []
        for c in candidates:
            w = self._weights.get(c, 1.0)
            if compat_boost and c in compat_boost:
                w *= compat_boost[c]
            weights.append(max(w, 0.01))  # floor to avoid zero-weight
        total = sum(weights)
        probs = [w / total for w in weights]
        idx = self.rng.choice(len(candidates), p=probs)
        return candidates[idx]

    def _select_construction_types(
        self,
        construction_type_candidates: list[str],
        num_construction_types: int,
        num_points: int,
        exclude_points: set[str] | None = None,
    ) -> tuple[list[str], list[dict[str, str]], list[tuple], set[Point], int]:
        """
        Select construction types and map their arguments.

        Args:
            exclude_points: Point names to exclude from arg selection
                (used by symmetry expand mode to restrict to one side).

        Returns:
            A tuple of (selected_construction_types, args_mappings, numeric_list,
            construction_points, num_points).
        """
        selected_construction_types: list[str] = []
        args_mappings: list[dict[str, str]] = []
        numeric_list: list[tuple] = []
        construction_points: set[Point] = set()

        # Compatibility boost: after picking construction type X, types listed
        # in _compatibility[X] get their weights multiplied for the next pick
        # within the same clause.  Reset between clauses.
        compat_boost: dict[str, float] | None = None

        for _ in range(num_construction_types):
            # Pre-filter to structurally-compatible candidates (same num_points,
            # enough defined points for args).  Deeper validation (require
            # checks, arg mappings) is done after weighted selection.
            pref_candidates = []
            for c in construction_type_candidates:
                cdef = self.construction_defs[c]
                if num_points is None or len(cdef.points) == num_points:
                    if len(cdef.args) <= len(self.point_naming.defined_points):
                        pref_candidates.append(c)

            if not pref_candidates:
                break

            # Weighted selection with retry (discard invalid and re-weight)
            remaining = list(pref_candidates)
            chosen = None
            chosen_mapping = None
            chosen_numerics = None
            chosen_cpoints = None

            for _ in range(min(len(remaining), MAX_TRY)):
                construction = self._weighted_choice(remaining, compat_boost)
                construction_def = self.construction_defs[construction]

                if num_points is None:
                    num_points = len(construction_def.points)

                args_mapping = self._map_construction_args(
                    construction_def,
                    self.point_naming.defined_points,
                    exclude_points=exclude_points,
                )
                if not self._validate_construction_requirements(
                    construction_def, args_mapping
                ):
                    remaining.remove(construction)
                    continue

                numerics = self._extract_numerics(construction_def, args_mapping)
                new_construction_points = self._extract_construction_points(
                    construction_def, args_mapping
                )

                chosen = construction
                chosen_mapping = args_mapping
                chosen_numerics = numerics
                chosen_cpoints = new_construction_points
                break

            if chosen is None:
                # Fall back to original shuffle-based approach
                random_construction_type_candidates = construction_type_candidates.copy()
                self.rng.shuffle(random_construction_type_candidates)
                for construction in random_construction_type_candidates:
                    construction_def = self.construction_defs[construction]
                    if num_points is not None and len(construction_def.points) != num_points:
                        continue
                    if len(construction_def.args) > len(self.point_naming.defined_points):
                        continue
                    args_mapping = self._map_construction_args(
                        construction_def,
                        self.point_naming.defined_points,
                        exclude_points=exclude_points,
                    )
                    if not self._validate_construction_requirements(
                        construction_def, args_mapping
                    ):
                        continue
                    chosen = construction
                    chosen_mapping = args_mapping
                    chosen_numerics = self._extract_numerics(construction_def, args_mapping)
                    chosen_cpoints = self._extract_construction_points(
                        construction_def, args_mapping
                    )
                    break

            if chosen is None:
                break

            selected_construction_types.append(chosen)
            args_mappings.append(chosen_mapping)
            numeric_list.extend(chosen_numerics)
            construction_points.update(chosen_cpoints)

            # Set up compatibility boost for next selection within this clause
            compat_boost = self._compatibility.get(chosen)

        return (
            selected_construction_types,
            args_mappings,
            numeric_list,
            construction_points,
            num_points,
        )

    def _build_clause_strings(
        self,
        selected_construction_types: list[str],
        args_mappings: list[dict[str, str]],
        numeric_list: list[tuple],
        construction_points: set[Point],
        num_clauses: int,
        num_points: int,
    ) -> list[str]:
        """Build clause strings from selected construction types."""
        clause_strs: list[str] = []

        for i in range(num_clauses):
            try:
                new_points = self.point_naming.prefetch_points(num_points)
                # Check numerics by drawing diagram
                self._draw_diagram(new_points, numeric_list, construction_points)
                self.point_naming.define_points(new_points)

                construction_strs: list[str] = []
                for construction, mapping in zip(
                    selected_construction_types, args_mappings
                ):
                    mapping.update(
                        dict(
                            zip(self.construction_defs[construction].points, new_points)
                        )
                    )
                    construction_strs.append(
                        self._format_construction_text(
                            self.construction_defs[construction], mapping
                        )
                    )
                    self._validate_construction_basics(
                        self.construction_defs[construction], mapping
                    )

                new_point_strs = self._format_points_with_coords(new_points)
                clause_str = (
                    " ".join(new_point_strs) + " = " + ", ".join(construction_strs)
                )
                clause_strs.append(clause_str)

            except Exception as e:
                if i == 0:
                    raise e
                else:
                    logging.debug(f"Multiple clause generation attempt failed: {e}")
                    break

        return clause_strs

    def _validate_construction_requirements(self, construction_def, mapping) -> bool:
        """Validate construction requirements (premises)."""
        try:
            for premise in construction_def.require.sentences:
                if len(premise) == 0:
                    continue
                statement = Statement.from_tokens(
                    translate_sentence(mapping, premise), self.dep_graph
                )
                if statement is None or not statement.check_numerical():
                    return False
            return True
        except Exception as e:
            logging.debug(f"Requirement validation failed: {e}")
            return False

    def _validate_construction_basics(self, construction_def, mapping) -> bool:
        """Validate construction basic properties."""
        try:
            for basic_statement in construction_def.basics:
                for t in basic_statement.sentences:
                    Statement.from_tokens(
                        translate_sentence(mapping, t), self.dep_graph
                    )
            return True
        except Exception as e:
            logging.warning(f"Error processing construction basics: {e}")
            return False

    def _extract_numerics(self, construction_def, mapping) -> list:
        """Extract numerical constraints from construction definition."""
        numerics = []
        for n in construction_def.numerics:
            numerics.append(tuple(mapping[a] if a in mapping else a for a in n))
        return numerics

    def _extract_construction_points(self, construction_def, mapping) -> set[Point]:
        """Extract construction points from construction definition."""
        construction_points = set()
        for arg in construction_def.args:
            if mapping[arg] in self.symbols_graph.name2node:
                construction_points.add(self.symbols_graph.name2node[mapping[arg]])
        return construction_points

    def _format_points_with_coords(self, point_names: list[str]) -> list[str]:
        """Format point names with their coordinates."""
        points = self.symbols_graph.names2points(point_names)
        return [f"{p.name}@{p.num.x}_{p.num.y}" for p in points]

    def _draw_diagram(
        self, new_points: list[str], numerics: list, construction_points: set[Point]
    ):
        """Draw diagram and validate point positions."""

        def draw_fn() -> tuple[PointNum, ...]:
            to_be_intersected: list[ObjNum] = []
            for n in numerics:
                args: list[Union[PointNum, str]] = []
                for t in n[1:]:
                    if str.isalpha(t[0]):
                        args.append(self.symbols_graph.names2points([t])[0].num)
                    else:
                        args.append(t)
                to_be_intersected += sketch(n[0], tuple(args), self.rng)

            return reduce(
                to_be_intersected,
                [p.num for p in _existing_points],
                [p.num for p in construction_points],
                rng=self.rng,
            )

        _existing_points = list(self.symbols_graph.nodes_of_type(Point))
        _existing_points = [p for p in _existing_points if hasattr(p, "num")]
        _new_points = self.symbols_graph.names2points(new_points)
        _new_numerical_point = draw_fn()

        if len(_new_numerical_point) != len(_new_points):
            raise Exception("Point count mismatch in draw result")

        # Check point distances
        _existing_numerical_points = [p.num for p in _existing_points]
        if check_too_close_numerical(
            _new_numerical_point, _existing_numerical_points, round=-1.0
        ):
            raise PointTooCloseError()
        if check_too_far_numerical(_new_numerical_point, _existing_numerical_points):
            raise PointTooFarError()

        # Set point positions
        for p, num in zip(_new_points, _new_numerical_point):
            p.num = num

    def _sample_arg_mapping(
        self,
        args_to_map: tuple[str, ...],
        candidate_points: list[str],
        must_differ: set[frozenset[str]],
        repeat_p: float,
        roles: dict[str, int] | None = None,
        role_bias: float = 0.0,
    ) -> dict[str, str] | None:
        """
        Assign points to a construction's arguments, allowing some to coincide.

        Start from one block per argument and repeatedly try to merge two random
        blocks, each attempt happening with probability `repeat_p`; a merge that
        would put a `require`-declared must-differ pair in one block is dropped.
        Every surviving block then gets its own distinct point, so arguments in
        the same block share a point and arguments in different blocks do not.

        When `roles` is given (see `argument_roles`), each merge attempt prefers
        -- with probability `role_bias` -- a pair of same-role, role-pure blocks
        over a uniformly random pair (e.g. for eqratio, two segment-starts are
        preferred over a segment-start/segment-end pair), matching how the goal
        predicate itself groups its arguments. `roles=None` or `role_bias=0.0`
        reproduces the original uniformly-random merge exactly.

        Returns None when the assignment cannot be drawn (caller falls back to
        the all-distinct path).
        """
        n = len(args_to_map)
        blocks: list[list[str]] = [[a] for a in args_to_map]

        def block_role(block: list[str]) -> int | str | None:
            if roles is None:
                return None
            block_roles = {roles.get(a) for a in block if a in roles}
            return next(iter(block_roles)) if len(block_roles) == 1 else "mixed"

        for _ in range(n - 1):
            if len(blocks) < 2 or random.random() >= repeat_p:
                continue
            same_role_pairs = []
            if roles is not None:
                for bi in range(len(blocks)):
                    for bj in range(bi + 1, len(blocks)):
                        ri, rj = block_role(blocks[bi]), block_role(blocks[bj])
                        if ri is not None and ri == rj and ri != "mixed":
                            same_role_pairs.append((bi, bj))
            if same_role_pairs and random.random() < role_bias:
                i, j = random.choice(same_role_pairs)
            else:
                i, j = random.sample(range(len(blocks)), 2)
            merged = blocks[i] + blocks[j]
            if any(
                frozenset((x, y)) in must_differ
                for idx, x in enumerate(merged)
                for y in merged[idx + 1 :]
            ):
                continue
            blocks[min(i, j)] = merged
            blocks.pop(max(i, j))

        if len(blocks) > len(candidate_points):
            return None

        chosen = random.sample(candidate_points, len(blocks))
        mapping: dict[str, str] = {}
        for block, point in zip(blocks, chosen):
            for arg in block:
                mapping[arg] = point
        return mapping

    def _map_construction_args(
        self,
        construction_def: DefinitionJGEX,
        defined_points: list[str],
        exclude_points: set[str] | None = None,
    ) -> dict[str, str]:
        """Map construction definition args to actual point names."""
        mapping: dict[str, str] = {}

        # Calculate number of points needed
        if construction_def.declare[0] == "s_angle":
            n_needed = len(construction_def.args) - 1
            args_to_map = construction_def.args[:-1]
        else:
            n_needed = len(construction_def.args)
            args_to_map = construction_def.args

        # Filter out excluded points; fall back to full list if too few remain
        if exclude_points:
            filtered = [p for p in defined_points if p not in exclude_points]
            if len(filtered) >= n_needed:
                candidate_points = filtered
            else:
                candidate_points = defined_points
        else:
            candidate_points = defined_points

        if self.arg_repetition > 0 and len(args_to_map) == n_needed and n_needed >= 2:
            # Draw merged mappings and keep the first one the construction's own
            # `require` accepts.  Retrying here matters: the caller drops a
            # construction type from the clause's candidates as soon as a mapping
            # fails validation, so returning an invalid merge would cost the type
            # rather than just this draw.  Falling back to the all-distinct
            # mapping keeps that from ever happening.
            must_differ = required_distinct_pairs(construction_def)
            roles = argument_roles(construction_def)
            for _ in range(MAX_TRY):
                merged = self._sample_arg_mapping(
                    args_to_map,
                    candidate_points,
                    must_differ,
                    self.arg_repetition,
                    roles=roles,
                    role_bias=self.role_bias,
                )
                if merged and self._validate_construction_requirements(
                    construction_def, merged
                ):
                    mapping = merged
                    break

        if not mapping:
            points = random.sample(candidate_points, n_needed)
            mapping = dict(zip(args_to_map, points))

        # Handle special angle case
        if construction_def.declare[0] == "s_angle":
            mapping[construction_def.args[-1]] = f"{random.choice(range(15, 180, 15))}o"

        return mapping

    def _format_construction_text(self, construction_def, mapping: dict) -> str:
        """Format construction text from definition and mapping."""
        text = f"{construction_def.declare[0]} {' '.join([mapping[p] for p in construction_def.declare[1:]])}"
        return text


def main():
    """简单测试。"""
    import time
    from collections import Counter

    seed = 42
    length = 20
    print(f"seed: {seed}, length: {length}\n")

    # 1. ClauseDAG 测试
    print("=== ClauseDAG 测试 ===")
    dag = ClauseDAG()
    dag.add_clause(
        Clause(("a@0_0", "b@1_0", "c@0.5_1"), (("triangle", "a", "b", "c"),))
    )
    dag.add_clause(Clause(("d@0.5_0",), (("midpoint", "d", "a", "b"),)))
    dag.add_clause(Clause(("e@0.25_0.5",), (("midpoint", "e", "a", "c"),)))
    dag.add_clause(Clause(("f@0.375_0.25",), (("midpoint", "f", "d", "e"),)))
    print(f"节点: {len(dag.nodes)}, 深度: {dag.get_max_depth()}")
    print(f"问题: {dag.to_problem(rename=True)}\n")

    # 2. 单个问题
    print("=== 单个问题 ===")
    sampler = ProblemSampler(seed=seed)
    result = sampler.generate(
        length=length, add_auxiliary=True, prune=True, with_coords=False
    )
    print(result)
    print(f"节点: {len(sampler.dag.nodes)}, 深度: {sampler.dag.get_max_depth()}\n")

    # 3. 批量生成
    print("=== 批量生成 (50题) ===")
    n_samples = 50
    depths = []
    point_counts = []
    start = time.time()

    for i in range(n_samples):
        s = ProblemSampler(seed=seed + i)
        s.generate(length=length, add_auxiliary=True, prune=True, with_coords=False)
        if s.dag.nodes:
            depths.append(s.dag.get_max_depth())
            point_counts.append(len(s.dag.get_all_points()))

    elapsed = time.time() - start
    print(f"耗时: {elapsed:.2f}s, 速度: {n_samples / elapsed:.1f}题/秒\n")

    # 深度分布 (排序 + barchart)
    print("深度分布:")
    depth_counts = Counter(depths)
    max_count = max(depth_counts.values())
    for d in sorted(depth_counts.keys()):
        count = depth_counts[d]
        bar_len = int(count / max_count * 40)
        bar = "█" * bar_len
        print(f"  {d}: {count:3d} {bar}")

    # 点数量分布
    print("\n点数量分布:")
    point_dist = Counter(point_counts)
    max_count = max(point_dist.values())
    for p in sorted(point_dist.keys()):
        count = point_dist[p]
        bar_len = int(count / max_count * 40)
        bar = "█" * bar_len
        print(f"  {p:2d}: {count:3d} {bar}")


if __name__ == "__main__":
    main()
