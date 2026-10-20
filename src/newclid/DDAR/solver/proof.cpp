#include "solver/proof.hpp"
#include "predicate/statement.hpp"
#include "solver/ddar.hpp"
#include "type/dist.hpp"
#include "type/slope.hpp"
#include "type/distlog.hpp"
#include <set>
#include <vector>
#include <optional>
#include <cassert>
#include <sstream>
#include <iomanip>

using namespace std;

Proof::Proof(DDARSolver *solver, std::unique_ptr<Statement> &&p)
    : _solver(solver),
      _statement(move(p)),
      _dep(nullptr)
{
}

void Proof::ensure_equations_initialized()
{
    if (_equations_initialized)
    {
        return;
    }
    _eqn_dist = _solver->insert_equation(_statement, "dist");
    _eqn_slope = _solver->insert_equation(_statement, "slope");
    _eqn_distlog = _solver->insert_equation(_statement, "distlog");
    _equations_initialized = true;
}

void Proof::prove_by_assumption()
{
    if (is_proved())
    {
        return;
    }
    set_proved(ProofState::PROVED_BY_ASSUMPTION);
    _depth = 0;
}

void Proof::initial()
{
    if (_statement->numerical_only())
    {
        if (_statement->check_numerically())
        {
            set_proved(ProofState::PROVED_NUMERICALLY);
            _depth = 0;
            return;
        }
        std::string err_message = "尝试添加错误的仅数值检验命题:";
        err_message += _statement->to_string();
        err_message += "  points:";
        for (const auto &pt : _statement->points())
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(15);
            oss << " (" << pt.name() << ", " << pt.num().x() << ", " << pt.num().y() << ")";
            err_message += oss.str();
        }
        throw runtime_error(err_message);
    }
    if (_statement->trivial())
    {
        set_proved(ProofState::PROVED_TRIVIAL);
        _depth = 0;
        return;
    }
}

void Proof::ar(long long depth)
{
    if (_state != ProofState::NOT_PROVED)
    {
        return;
    }

    if (_statement->name() == "secant")
    {
        return;
    }

    ensure_equations_initialized();
    for (const auto &req : _eqn_dist)
    {
        req->reduce();
        if (req->is_solved())
        {
            _dep = req;
            set_proved(ProofState::PROVED_AR_DIST);
            _depth = depth;
            return;
        }
    }
    for (const auto &req : _eqn_slope)
    {
        req->reduce();
        if (req->is_solved())
        {
            _dep = req;
            set_proved(ProofState::PROVED_AR_SLOPE);
            _depth = depth;
            return;
        }
    }
    for (const auto &req : _eqn_distlog)
    {
        req->reduce();
        if (req->is_solved())
        {
            _dep = req;
            set_proved(ProofState::PROVED_AR_DISTLOG);
            _depth = depth;
            return;
        }
    }
    return;
}

void Proof::set_theorem(size_t index, long long depth)
{
    _theoremId = index;
    set_proved(ProofState::PROVED_BY_THEOREM);
    _depth = depth;
}

const unique_ptr<Statement> &Proof::statement() const
{
    return _statement;
}

void Proof::print_equations() const
{
    const_cast<Proof *>(this)->ensure_equations_initialized();
    cout << "Proof Equations for statement: " << _statement->to_string() << endl;
    for (const auto &eq : _eqn_dist)
    {
        cout << "Original Equation: " << eq->original_equation() << endl;
        cout << "Reduced Equation: " << eq->remainder() << endl;
    }
    for (const auto &eq : _eqn_distlog)
    {
        cout << "Original Equation: " << eq->original_equation() << endl;
        cout << "Reduced Equation: " << eq->remainder() << endl;
    }
    for (const auto &eq : _eqn_slope)
    {
        cout << "Original Equation: " << eq->original_equation() << endl;
        cout << "Reduced Equation: " << eq->remainder() << endl;
    }
}

vector<Proof *> Proof::get_dependencies() const
{
    switch (_state)
    {
    case ProofState::PROVED_BY_ASSUMPTION:
    case ProofState::PROVED_NUMERICALLY:
    case ProofState::PROVED_TRIVIAL:
    case ProofState::NOT_PROVED:
        return {};
    case ProofState::PROVED_BY_THEOREM:
        return _solver->applications()[_theoremId].hypotheses();
    case ProofState::PROVED_BY_DOUBLEPOINT:
    case ProofState::PROVED_AR_SIMPLIFIED:
        return _deps;
    case ProofState::PROVED_AR_DIST:
    case ProofState::PROVED_AR_DISTLOG:
    case ProofState::PROVED_AR_SLOPE:
        return _dep->statement_dependencies();
    }
    return {};
}

string Proof::reason() const
{
    switch (_state)
    {
    case ProofState::NOT_PROVED:
        return "Not Proved";
    case ProofState::PROVED_BY_ASSUMPTION:
        return "Premise";
    case ProofState::PROVED_NUMERICALLY:
        return "Numerical Check";
    case ProofState::PROVED_TRIVIAL:
        return "Trivial";
    case ProofState::PROVED_BY_THEOREM:
        return _solver->applications()[_theoremId].theorem().rule();
    case ProofState::PROVED_AR_DIST:
        return "AR with Dist";
    case ProofState::PROVED_AR_DISTLOG:
        return "AR with Distlog";
    case ProofState::PROVED_AR_SLOPE:
        return "AR with Slope";
    case ProofState::PROVED_BY_DOUBLEPOINT:
        return "Transfer";
    case ProofState::PROVED_AR_SIMPLIFIED:
        return "AR Simplified";
    default:
        return "Unknown";
    }
}

void Proof::set_proved(Proof *doublepoint, Proof *original)
{
    if (_state != ProofState::NOT_PROVED)
    {
        throw runtime_error("Proof already proved");
    }
    ensure_equations_initialized();
    _deps.emplace_back(doublepoint);
    _deps.emplace_back(original);
    _state = ProofState::PROVED_BY_DOUBLEPOINT;
    _solver->push_established_statement(this);
    assert(_statement->check_numerically());

    for (const auto req : _eqn_dist)
    {
        req->reduce();
    }
    for (const auto req : _eqn_distlog)
    {
        req->reduce();
    }
    for (const auto req : _eqn_slope)
    {
        req->reduce();
    }
    _solver->add_established_equations(this);
}

void Proof::update(std::vector<Proof *> deps)
{
    _deps = std::move(deps);
    _state = ProofState::PROVED_AR_SIMPLIFIED;
}

void Proof::set_proved(ProofState state)
{
    if (state == ProofState::NOT_PROVED)
    {
        return;
    }

    if (_state != ProofState::NOT_PROVED)
    {
        throw runtime_error("Proof already proved");
    }

    ensure_equations_initialized();

    _state = state;
    _solver->push_established_statement(this);

    assert(_statement->check_numerically());

    for (const auto req : _eqn_dist)
    {
        req->reduce();
    }
    for (const auto req : _eqn_distlog)
    {
        req->reduce();
    }
    for (const auto req : _eqn_slope)
    {
        req->reduce();
    }

    _solver->add_established_equations(this);
}