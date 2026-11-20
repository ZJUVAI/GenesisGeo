#include "ar/reduced_equation.hpp"
#include "ar/linear_system.hpp"
#include <set>

using namespace std;

class Proof;

ReducedEquation::ReducedEquation(Equation equation, LinearSystem *system)
    : _original_equation(std::move(equation)), _system(system), _remainder(_original_equation)
{
    _remainder.make_monic();
}

void ReducedEquation::reduce()
{
    if (_system == nullptr || _last_reduced_rule_version == _system->rule_version())
    {
        return;
    }
    if (auto reduced = _system->normal_form_if_changed(_remainder))
    {
        _remainder = std::move(*reduced);
    }
    _last_reduced_rule_version = _system->rule_version();
}

bool ReducedEquation::is_solved() const
{
    return _remainder.empty();
}

vector<Proof *> ReducedEquation::statement_dependencies() const
{
    set<Proof *> uniq;
    for (size_t idx : _remainder.dependency_indices())
    {
        if (idx >= _system->size())
        {
            continue;
        }
        Proof *p = _system->pair_at(idx).second;
        if (p != nullptr)
        {
            uniq.insert(p);
        }
    }
    return vector<Proof *>(uniq.begin(), uniq.end());
}
