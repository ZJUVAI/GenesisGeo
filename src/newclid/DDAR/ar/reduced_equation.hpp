#ifndef REDUCED_EQUATION_HPP
#define REDUCED_EQUATION_HPP

#include <vector>
#include <optional>
#include "ar/equation.hpp"

class LinearSystem;
class Proof;

class ReducedEquation final
{
private:
    Equation _original_equation;
    LinearSystem *_system;
    Equation _remainder;
    std::optional<size_t> _last_reduced_rule_version;

public:
    explicit ReducedEquation(Equation &equation, LinearSystem *system);

    const Equation &original_equation() const { return _original_equation; }

    const LinearSystem *linear_system() const { return _system; }

    const Equation &remainder() const { return _remainder; }

    bool is_solved() const;

    void reduce();

    std::vector<Proof *> statement_dependencies() const;
};

#endif // REDUCED_EQUATION_HPP
