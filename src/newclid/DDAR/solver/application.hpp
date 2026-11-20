#ifndef APPLICATION_HPP
#define APPLICATION_HPP
#include "theorem.hpp"
#include "predicate/statement.hpp"

class Proof;
class DDARSolver;

enum class ApplicationState : uint8_t
{
    PENDING,
    PROVED,
    DISCARDED,
};

class Application
{
public:
    Application(DDARSolver *solver, Theorem &&theorem);

    void advance_proof(long long depth, bool check_conclusions_with_ar);

    ApplicationState state() const { return _state; }

    const std::vector<Proof *> &hypotheses() const { return _hypotheses; }

    const std::vector<Proof *> &conclusions() const { return _conclusions; }

    const Point &max_point() const { return _max_point; }

    const Theorem &theorem() const { return _theorem; }

private:
    Theorem _theorem;
    ApplicationState _state{ApplicationState::PENDING};
    std::vector<Proof *> _hypotheses;
    std::vector<Proof *> _conclusions;
    Point _max_point;
};

std::ostream &operator<<(std::ostream &out, const ApplicationState &state);

#endif // APPLICATION_HPP