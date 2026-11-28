"""Cancel polynomial factors whose nonzero domains are explicit."""

from functools import lru_cache

import sympy as sp

from .native_basis import integer_terms


def rational_row_basis(expressions, variables):
    import flint

    expressions = [sp.expand(e) for e in expressions if e != 0]
    if not expressions:
        return []
    if not variables:
        return [sp.Integer(1)]
    # Clearing each row's denominators is multiplication by a nonzero
    # rational unit. The row space and its reduced basis stay identical.
    terms = [integer_terms(expression, variables) for expression in expressions]
    monomials = sorted(
        {monomial for row in terms for monomial in row},
        key=lambda monomial: (sum(monomial), monomial),
        reverse=True,
    )
    if not monomials:
        return []
    columns = {monomial: i for i, monomial in enumerate(monomials)}
    matrix = flint.fmpq_mat(len(terms), len(columns))
    for i, row in enumerate(terms):
        for monomial, coefficient in row.items():
            matrix[i, columns[monomial]] = coefficient
    reduced, rank = matrix.rref()
    monomial_expressions = [
        sp.Mul(
            *(variable**power for variable, power in zip(variables, monomial) if power)
        )
        for monomial in monomials
    ]
    return [
        sp.Add(
            *(
                sp.Rational(int(c.p), int(c.q)) * monomial_expressions[j]
                for j in range(len(monomials))
                if (c := reduced[i, j]) != 0
            )
        )
        for i in range(rank)
    ]


def cancel_guarded_factors(equations, variables, nonzero, substitutions):
    """Simplify equations on the explicitly guarded nondegenerate branch.

    From a*b=0 and a!=0 derive b=0. Every cancelled irreducible factor
    comes from a declared nonzero fact, never from a positive equation or
    a numerical source alone.
    """
    from .native_basis import NativeBasis
    from .exact_geometry import eliminate_linear_variables, substitute_coordinates

    equations, variables, substitutions = (
        list(equations),
        list(variables),
        list(substitutions),
    )
    native = NativeBasis([], variables)
    factor = lru_cache(maxsize=None)(native.factor_for_zero_tests)

    def nonzero_factors():
        result = set()
        for obligation in nonzero:
            expression = (
                sum(substitute_coordinates(c, substitutions) ** 2 for c in obligation)
                if isinstance(obligation, tuple)
                else substitute_coordinates(obligation, substitutions)
            )
            valid, factors = factor(expression)
            if not valid:
                return None
            result.update(f for f, _ in factors)
        return result

    known = None
    already_reduced = False
    for _ in range(2 * len(variables) + 4):
        before = tuple(equations), tuple(variables)
        if known is None:
            known = nonzero_factors()
            if known is None:
                return equations, variables, substitutions, None
        changed = False
        updated = []
        for equation in equations:
            _, factors = factor(equation)
            cancelled = [(f, k) for f, k in factors if f in known]
            if cancelled:
                remainder = sp.expand(
                    sp.prod(f**k for f, k in factors if f not in known)
                )
                changed = True
                updated.append(remainder)
            else:
                updated.append(equation)
        # The previous row basis is already canonical if it needed no
        # coordinate elimination. With no new cancellation it is unchanged.
        if already_reduced and not changed:
            break
        equations = rational_row_basis(updated, variables)
        equations, variables, extra = eliminate_linear_variables(equations, variables)
        substitutions.extend(extra)
        already_reduced = not extra
        if extra:
            known = None
        after = tuple(equations), tuple(variables)
        if after == before:
            break
        if after[1] != before[1]:
            native = NativeBasis([], variables)
            factor = lru_cache(maxsize=None)(native.factor_for_zero_tests)
    return (
        equations,
        variables,
        substitutions,
        known if known is not None else nonzero_factors(),
    )
