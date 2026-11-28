"""Exact QQ ideal operations using FLINT's primitive integer polynomials.

Clearing rational denominators and taking primitive parts multiply by nonzero
rational units, preserving QQ ideal membership and all zero/nonzero tests used
by this certificate kernel. No finite fields or floating-point approximations.
"""

from functools import lru_cache
from math import lcm

import sympy as sp


@lru_cache(maxsize=16)
def _variable_indexes(variables):
    return {variable: i for i, variable in enumerate(variables)}


def rational_terms(expression, variables):
    """Return exact coefficients of a polynomial, without a dense expansion."""
    variables = tuple(variables)
    indexes = _variable_indexes(variables)
    terms = {}
    for term in sp.Add.make_args(expression):
        coefficient, factors = term.as_coeff_mul(rational=True)
        powers = [0] * len(variables)
        for factor in factors:
            base, exponent = factor.as_base_exp()
            index = indexes.get(base)
            if index is None or not exponent.is_Integer or exponent < 0:
                return dict(sp.Poly(expression, *variables, domain=sp.QQ).terms())
            powers[index] += int(exponent)
        monomial = tuple(powers)
        terms[monomial] = terms.get(monomial, 0) + coefficient
    return terms


def integer_terms(expression, variables):
    """Clear rational denominators with a sparse path for expanded monomials."""
    terms = rational_terms(expression, variables)
    denominator = lcm(*(int(coefficient.q) for coefficient in terms.values()))
    if denominator == 1:
        return {
            monomial: int(coefficient)
            for monomial, coefficient in terms.items()
            if coefficient
        }
    return {
        monomial: int(coefficient * denominator)
        for monomial, coefficient in terms.items()
        if coefficient
    }


def _symbolic_expression(polynomial, variables):
    return sp.Add(
        *(
            sp.Integer(int(coefficient))
            * sp.Mul(
                *(
                    variable ** int(power)
                    for variable, power in zip(variables, monomial)
                )
            )
            for monomial, coefficient in polynomial.to_dict().items()
        )
    )


@lru_cache(maxsize=2048)
def _exact_factors(variables, expression):
    import flint

    context = flint.fmpz_mpoly_ctx.get(tuple(map(str, variables)), "degrevlex")
    polynomial = context.from_dict(integer_terms(expression, variables))
    _, factors = polynomial.factor()
    return int(bool(polynomial)), tuple(
        (_symbolic_expression(factor, variables), int(power))
        for factor, power in factors
    )


class NativeBasisLimit(RuntimeError):
    pass


class NativeBasis:
    def __init__(self, expressions, variables, limits=(1000, 50000, 4096)):
        import flint

        flint.ctx.threads = 1
        self.variables = tuple(variables) or (sp.Dummy("cas"),)
        self.context = flint.fmpz_mpoly_ctx.get(
            tuple(str(variable) for variable in self.variables), "degrevlex"
        )
        self.cache = {}
        inputs = [
            self.convert(expression) for expression in expressions if expression != 0
        ]
        self.original = flint.fmpz_mpoly_vec(inputs, self.context)
        if inputs:
            result, complete = self.original.buchberger_naive(limits=limits)
            if not complete:
                raise NativeBasisLimit("native Gröbner resource limit")
            self.basis = result.autoreduction(groebner=True)
            if not self.basis.is_groebner(self.original):
                raise RuntimeError("native Gröbner verification failed")
        else:
            self.basis = flint.fmpz_mpoly_vec([], self.context)
        self.empty = not tuple(self.basis)

    def convert(self, expression):
        from flint import fmpq_mpoly

        if isinstance(expression, fmpq_mpoly):
            # Multiplying by a nonzero rational unit preserves ideal membership.
            coefficients = expression.to_dict()
            denominator = lcm(
                *(int(coefficient.denominator) for coefficient in coefficients.values())
            )
            return self.context.from_dict(
                {
                    monomial: int(coefficient * denominator)
                    for monomial, coefficient in coefficients.items()
                }
            )
        expression = sp.sympify(expression)
        if expression not in self.cache:
            self.cache[expression] = self.context.from_dict(
                integer_terms(expression, self.variables)
            )
        return self.cache[expression]

    def remainder(self, polynomial):
        return (
            polynomial
            if self.empty
            else polynomial.reduction_primitive_part(self.basis)
        )

    def expression(self, polynomial):
        return _symbolic_expression(polynomial, self.variables)

    def factor_for_zero_tests(self, expression):
        """Exact irreducible factors over QQ, ignoring a nonzero rational unit.

        The first return value indicates zero/nonzero, not the original scale.
        Factors have FLINT's canonical primitive integer normalization, so they
        can be compared across calls in the same coordinate context.
        """
        nonzero, factors = _exact_factors(self.variables, sp.sympify(expression))
        return nonzero, list(factors)
