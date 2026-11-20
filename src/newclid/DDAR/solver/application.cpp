#include "solver/application.hpp"
#include "solver/ddar.hpp"
#include "solver/proof.hpp"

using namespace std;

Application::Application(DDARSolver *solver, Theorem &&theorem) : _theorem(move(theorem)),
                                                                  _max_point(_theorem.max_point())
{
    for (const auto &stmt : _theorem.hypotheses())
    {
        _hypotheses.push_back(solver->insert_statement(stmt));
    }
    for (const auto &stmt : _theorem.conclusions())
    {
        _conclusions.push_back(solver->insert_statement(stmt));
    }
}

void Application::advance_proof(long long depth, bool check_conclusions_with_ar)
{
    if (_state != ApplicationState::PENDING)
    {
        return;
    }

    bool hypotheses_proved = true;

    for (auto *pf : _hypotheses)
    {
        pf->ar(depth);
        if (!pf->is_proved() || pf->depth() >= depth)
        {
            hypotheses_proved = false;
        }
    }

    if (hypotheses_proved)
    {
        _state = ApplicationState::PROVED;
        return;
    }

    bool conclusion_proved = true;

    for (auto *pf : _conclusions)
    {
        if (check_conclusions_with_ar)
        {
            pf->ar(depth);
        }
        conclusion_proved &= pf->is_proved();
    }

    if (conclusion_proved)
    {
        _state = ApplicationState::DISCARDED;
        return;
    }
}

ostream &operator<<(ostream &out, const ApplicationState &state)
{
    switch (state)
    {
    case ApplicationState::PENDING:
        return out << "pending";
    case ApplicationState::PROVED:
        return out << "proved";
    case ApplicationState::DISCARDED:
        return out << "discarded";
    }
    return out;
}