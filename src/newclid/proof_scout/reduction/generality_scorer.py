#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeneralityScorer - Compute generality scores for rules.

Generality is measured by:
- Primary: Number of premises (fewer = more general)
- Secondary: Number of conclusions (more = more general)

Score tuple: (-n_premises, n_conclusions)
This ensures sorting puts more general rules first.
"""

import re
from typing import Tuple


class GeneralityScorer:
    """Compute generality scores for geometric rules."""

    @staticmethod
    def score(rule_text: str) -> Tuple[int, int]:
        """Compute generality score for a rule.

        Args:
            rule_text: Rule in format "premise1, premise2 => conclusion1, conclusion2"

        Returns:
            Tuple of (-n_premises, n_conclusions) for sorting.
            More general rules (fewer premises, more conclusions) sort first.

        Example:
            "cong a b c d => para a b c d" -> (-1, 1)
            "cong a b c d, cong e f g h => para a b c d" -> (-2, 1)
        """
        if "=>" not in rule_text:
            return (0, 0)

        premise_part, conclusion_part = rule_text.split("=>", 1)

        # Count premises
        n_premises = 0
        for condition in premise_part.split(","):
            parts = re.findall(r"\w+", condition)
            if parts:
                n_premises += 1

        # Count conclusions
        n_conclusions = 0
        for condition in conclusion_part.split(","):
            parts = re.findall(r"\w+", condition)
            if parts:
                n_conclusions += 1

        return (-n_premises, n_conclusions)


__all__ = ["GeneralityScorer"]
