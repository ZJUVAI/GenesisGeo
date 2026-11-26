// c++ -O3 -Wall -shared -std=c++14 `python3 -m pybind11 --includes` matchinC.cpp -o matchinC`python3-config --extension-suffix` -fPIC
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <string>
#include <set>
#include <iostream>
#include <unordered_map>

namespace py = pybind11;

using Point = std::pair<double, double>;

double angle(const Point &a, const Point &b)
{
    return std::atan2(a.second - b.second, a.first - b.first);
}

double distance(const Point &a, const Point &b)
{
    return std::sqrt(std::pow(b.first - a.first, 2) + std::pow(b.second - a.second, 2));
}

double distance2(const Point &a, const Point &b)
{
    return std::pow(b.first - a.first, 2) + std::pow(b.second - a.second, 2);
}

double modpi(double x)
{
    double result = std::fmod(x, M_PI);
    if (result < 0)
    {
        result += M_PI;
    }
    return result;
}

double dot(const Point &a, const Point &b)
{
    return a.first * b.first + a.second * b.second;
}


#define ATOM 1e-9
#define REL_TOL 0.001

bool close_enough(double a, double b)
{
    return fabs(a - b) < 4 * ATOM || fabs(a - b) / std::max(fabs(a), fabs(b)) < REL_TOL;
}

bool close_enough(Point a, Point b)
{
    return close_enough(a.first, b.first) && close_enough(a.second, b.second);
}

double angle4p(const Point &a, const Point &b, const Point &c, const Point &d)
{
    double angle1 = angle(a, b);
    double angle2 = angle(c, d);
    double result = modpi(angle2 - angle1);
    if (close_enough(result, M_PI))
    {
        result = 0.0;
    }
    return result;
}

double ratio(const Point &a, const Point &b, const Point &c, const Point &d)
{
    double dis1 = distance(a, b);
    double dis2 = distance(c, d);
    double result = dis1 / dis2;
    return result;
}

bool nearly_zero(double a)
{
    return fabs(a) < 2 * ATOM;
}

int sign(double x)
{
    if(nearly_zero(x))
    {
        return 0;
    }
    if (x> 0)
    {
        return 1;
    }
    return -1;
}

bool same(const Point &a, const Point &b)
{
    return close_enough(a.first, b.first) && close_enough(a.second, b.second);
}

bool has_duplicates(const std::vector<Point> &vec)
{
    std::set<Point> unique_elements(vec.begin(), vec.end());
    return unique_elements.size() != vec.size();
}

Point calculate_center(const Point &a, const Point &b, const Point &c)
{
    double x1 = a.first, y1 = a.second;
    double x2 = b.first, y2 = b.second;
    double x3 = c.first, y3 = c.second;
    double d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
    double dx = (x1 * x1 * (y2 - y3) + x2 * x2 * (y3 - y1) + x3 * x3 * (y1 - y2)) - (y1 - y2) * (y2 - y3) * (y3 - y1);
    double dy = (y1 * y1 * (x3 - x2) + y2 * y2 * (x1 - x3) + y3 * y3 * (x2 - x1)) + (x1 - x2) * (x2 - x3) * (x3 - x1);

    double x = dx / d;
    double y = dy / d;
    if (nearly_zero(x))
    {
        x = 0;
    }
    if (nearly_zero(y))
    {
        y = 0;
    }
    return std::make_pair(x, y);
}

double clock(const Point &a, const Point &b, const Point &c)
{
    double abx = b.first - a.first;
    double aby = b.second - a.second;
    double acx = c.first - a.first;
    double acy = c.second - a.second;
    return abx * acy - aby * acx;
}

bool sameclock(const Point &a,const Point &b,const Point &c,const Point &d,const Point &e,const Point &f)
{

    double clock1 = clock(a, b, c);
    double clock2 = clock(d, e, f);
    if (clock1 * clock2 > ATOM)
    {
        return true;
    }
    return false;
}

bool check_numerical(std::string type, const std::vector<Point> &points)
{

    if (type == "PythagoreanPremises")
    {
        if (points.size() != 3)
            return false;
        if (same(points[0], points[1]) || same(points[0], points[2]))
            return false;
        Point ba = std::make_pair(points[0].first - points[1].first, points[0].second - points[1].second);
        Point ca = std::make_pair(points[0].first - points[2].first, points[0].second - points[2].second);
        double dot_res = fabs(dot(ba, ca));
        return nearly_zero(dot_res);
    }
    else if (type == "cong")
    {
        if (points.size() % 2 != 0)
            return false;
        double length = -1;
        for (size_t i = 0; i < points.size() - 1; i = i + 2)
        {
            if (close_enough(points[i], points[i + 1]))
            {
                return false;
            }
            double current_length = distance(points[i], points[i + 1]);
            if (length >= 0 && !close_enough(length, current_length))
            {
                return false;
            }
            length = current_length;
        }
        return true;
    }
    else if (type == "coll")
    {
        if (points.size() <= 2)
            return false;
        if (has_duplicates(points))
            return false;
        if (close_enough(points[0].first, points[1].first))
        {
            for (size_t i = 2; i < points.size(); ++i)
            {
                if (!close_enough(points[0].first, points[i].first))
                {
                    return false;
                }
            }
            return true;
        }
        else
        {
            double slope = (points[1].second - points[0].second) / (points[1].first - points[0].first);
            double intercept = points[0].second - slope * points[0].first;
            for (size_t i = 2; i < points.size(); ++i)
            {
                if (!close_enough(points[i].second, slope * points[i].first + intercept))
                {
                    return false;
                }
            }
        }
        return true;
    }
    else if (type == "ncoll")
    {
        if (points.size() <= 2)
            return false;
        if (has_duplicates(points))
            return false;
        if (close_enough(points[0].first, points[1].first))
        {
            for (size_t i = 2; i < points.size(); ++i)
            {
                if (!close_enough(points[0].first, points[i].first))
                {
                    return true;
                }
            }
            return false;
        }
        else
        {
            double slope = (points[1].second - points[0].second) / (points[1].first - points[0].first);
            double intercept = points[0].second - slope * points[0].first;
            for (size_t i = 2; i < points.size(); ++i)
            {
                if (!close_enough(points[i].second, slope * points[i].first + intercept))
                {
                    return true;
                }
            }
        }
        return false;
    }
    else if (type == "para")
    {
        if (points.size() < 4 || points.size() % 2 != 0 || close_enough(points[0], points[1]))
            return false;
        double reference = modpi(angle(points[0], points[1]));
        for (size_t i = 2; i < points.size(); i += 2)
        {
            if (close_enough(points[i], points[i + 1]))
                return false;
            const double direction = modpi(angle(points[i], points[i + 1]));
            if (!close_enough(reference, direction)
                && !close_enough(std::remainder(direction - reference, M_PI), 0.0))
                return false;
            reference = direction;
        }
        return true;
    }
    else if (type == "npara")
    {
        if (points.size() != 4)
            return false;
        if (close_enough(points[0], points[1]) || close_enough(points[2], points[3]))
        {
            return false;
        }
        double angle1 = angle(points[0], points[1]);
        double angle2 = angle(points[2], points[3]);
        if (!close_enough(angle1, angle2))
        {
            return true;
        }
        return false;
    }
    else if (type == "cyclic")
    {
        if (points.size() <= 3)
            return false;
        if (has_duplicates(points))
            return false;
        Point center = calculate_center(points[0], points[1], points[2]);
        double radius2 = -1;
        for (size_t i = 0; i < points.size(); ++i)
        {
            double current_radius2 = distance2(center, points[i]);
            if (radius2 >= 0 && !close_enough(current_radius2, radius2))
            {
                return false;
            }
            radius2 = current_radius2;
        }
        return true;
    }
    else if (type == "circle")
    {
        if (points.size() <= 2)
            return false;
        if (has_duplicates(points))
            return false;
        double radius2 = -1;
        for (size_t i = 1; i < points.size(); ++i)
        {
            double current_radius2 = distance2(points[0], points[i]);
            if (radius2 >= 0 && !close_enough(current_radius2, radius2))
            {
                return false;
            }
            radius2 = current_radius2;
        }
        return true;
    }
    else if (type == "perp")
    {
        if (points.size() != 4)
            return false;
        if (close_enough(points[0], points[1]) || close_enough(points[2], points[3]))
        {
            return false;
        }
        Point ba = std::make_pair(points[0].first - points[1].first, points[0].second - points[1].second);
        Point dc = std::make_pair(points[2].first - points[3].first, points[2].second - points[3].second);
        double dot_res = fabs(dot(ba, dc));
        return nearly_zero(dot_res);
    }
    else if (type == "eqangle")
    {
        if (points.size() != 8)
        {
            return false;
        }
        double angle1 = angle4p(points[0], points[1], points[2], points[3]);
        double angle2 = angle4p(points[4], points[5], points[6], points[7]);
        return close_enough(angle1, angle2);
    }
    else if (type == "eqratio")
    {
        if (points.size() != 8)
        {
            return false;
        }
        double ratio1 = ratio(points[0], points[1], points[2], points[3]);
        double ratio2 = ratio(points[4], points[5], points[6], points[7]);
        return close_enough(ratio1, ratio2);
    }
    else if(type == "sameclock")
    {
        if(points.size()!=6)
        {
            return false;
        }
        return sameclock(points[0], points[1], points[2], points[3],points[4], points[5]);
    }
    else if(type == "sameside")
    {
        if(points.size()!=6)
        {
            return false;
        }
        Point a = points[0];
        Point b = points[1];
        Point c = points[2];
        Point d = points[3];
        Point e = points[4];
        Point f = points[5];
        Point ab = std::make_pair(b.first-a.first,b.second-a.second);
        Point ac = std::make_pair(c.first-a.first,c.second-a.second);
        Point de = std::make_pair(e.first-d.first,e.second-d.second);
        Point df = std::make_pair(f.first-d.first,f.second-d.second);
        return sign(dot(ab,ac)) == sign(dot(de,df));  
    }
    else if(type == "nsameside")
    {
        if(points.size()!=6)
        {
            return false;
        }
        Point a = points[0];
        Point b = points[1];
        Point c = points[2];
        Point d = points[3];
        Point e = points[4];
        Point f = points[5];
        Point ab = std::make_pair(b.first-a.first,b.second-a.second);
        Point ac = std::make_pair(c.first-a.first,c.second-a.second);
        Point de = std::make_pair(e.first-d.first,e.second-d.second);
        Point df = std::make_pair(f.first-d.first,f.second-d.second);
        return sign(dot(ab,ac)) != sign(dot(de,df)); 
    }
    else if(type == "midp")
    {
        if(points.size() != 3)
        {
            return false;
        }
        Point m = points[0];
        Point l = points[1];
        Point r = points[2];
        Point m1 = std::make_pair((l.first + r.first) / 2, (l.second + r.second) / 2);
        return close_enough(m, m1);
    }
    else if (type == "rconst")
    {
        if(points.size() != 5)
        {
            return false;
        }
        Point a = points[0];
        Point b = points[1];
        Point c = points[2];
        Point d = points[3];
        double rat = ratio(a, b, c, d);
        return close_enough(rat, points[4].first);
    }
    else if (type == "aconst")
    {
        if(points.size() != 5)
        {
            return false;
        }
        Point a = points[0];
        Point b = points[1];
        Point c = points[2];
        Point d = points[3];
        double ang = angle4p(a, b, c, d);
        return close_enough(ang, points[4].first);
    }
    else
    {
        std::cout << type << std::endl;
        return true;
    }
    return false;
}

long long parseInteger(const std::string &text)
{
    size_t used = 0;
    const auto value = std::stoll(text, &used);
    if (used != text.size())
        throw std::invalid_argument("Invalid scalar constant: " + text);
    return value;
}

double parseFraction(const std::string &input, bool angle_constant)
{
    long long numerator, denominator = 1;
    const auto pi = input.find("pi/");
    const auto slash = input.find('/');
    if (!input.empty() && input.back() == 'o')
    {
        numerator = parseInteger(input.substr(0, input.size()-1));
        denominator = 180;
    }
    else if (pi != std::string::npos)
    {
        numerator = parseInteger(input.substr(0, pi));
        denominator = parseInteger(input.substr(pi+3));
    }
    else if (slash != std::string::npos)
    {
        numerator = parseInteger(input.substr(0, slash));
        denominator = parseInteger(input.substr(slash+1));
    }
    else
        numerator = parseInteger(input);
    if (denominator == 0)
        throw std::invalid_argument("Denominator cannot be zero");
    if (angle_constant)
    {
        // std::fmod operates on the small exact integer remainder, not on a
        // potentially enormous floating-point multiple of pi.
        const auto remainder = denominator == -1 ? 0LL : numerator % denominator;
        double coefficient = static_cast<double>(remainder) / denominator;
        if (coefficient < 0) coefficient += 1;
        return coefficient * M_PI;
    }
    return static_cast<double>(numerator) / denominator;
}

namespace
{
struct CompiledPremise
{
    std::string predicate;
    std::vector<size_t> slots;
    size_t first_new_slot;
    size_t end_new_slot;
    std::vector<Point> arguments;
};

void match_premise(size_t index, size_t assigned_end,
                   std::vector<CompiledPremise> &premises,
                   const std::vector<Point> &points,
                   const std::vector<std::string> &variables,
                   std::vector<int> &binding,
                   std::vector<std::unordered_map<std::string, int>> &results)
{
    if (index == premises.size())
    {
        std::unordered_map<std::string, int> result;
        result.reserve(variables.size());
        for (size_t i = 0; i < variables.size(); ++i)
            result.emplace(variables[i], binding[i]);
        results.push_back(std::move(result));
        return;
    }

    auto &premise = premises[index];
    if (assigned_end > premise.first_new_slot)
    {
        const size_t slot = assigned_end - 1;
        for (size_t point = 0; point < points.size(); ++point)
        {
            binding[slot] = point;
            match_premise(index, slot, premises, points, variables, binding, results);
        }
        return;
    }

    for (size_t i = 0; i < premise.slots.size(); ++i)
        premise.arguments[i] = points[binding[premise.slots[i]]];
    if (check_numerical(premise.predicate, premise.arguments))
    {
        const size_t next_end = index + 1 < premises.size()
            ? premises[index + 1].end_new_slot : binding.size();
        match_premise(index + 1, next_end, premises, points, variables, binding, results);
    }
}
}

extern "C"
{
    std::vector<std::unordered_map<std::string, int>> mapping_nomal_theorem(const std::vector<std::vector<std::string>> &theorem, const std::vector<Point> &points)
    {
        std::vector<std::unordered_map<std::string, int>> results;
        std::vector<std::string> variables;
        std::unordered_map<std::string, size_t> slots;
        std::vector<CompiledPremise> premises;
        for (const auto &tokens : theorem)
        {
            if (tokens.size() < 2)
                throw py::value_error("Each premise requires a predicate and arguments");
            const bool has_constant = tokens[0] == "aconst" || tokens[0] == "rconst";
            if (has_constant && tokens.size() != 6)
                throw py::value_error("Constant predicates require four points and a scalar");
            const size_t point_end = tokens.size() - (has_constant ? 1 : 0);
            CompiledPremise premise{tokens[0], {}, variables.size(), 0, {}};
            for (size_t i = 1; i < point_end; ++i)
            {
                const auto entry = slots.emplace(tokens[i], variables.size());
                if (entry.second)
                    variables.push_back(tokens[i]);
                premise.slots.push_back(entry.first->second);
            }
            premise.end_new_slot = variables.size();
            premise.arguments.resize(tokens.size() - 1);
            if (has_constant)
                premise.arguments.back() = {parseFraction(tokens.back(), tokens[0] == "aconst"), 0};
            premises.push_back(std::move(premise));
        }
        std::vector<int> binding(variables.size());
        match_premise(0, premises.empty() ? 0 : premises.front().end_new_slot,
                      premises, points, variables, binding, results);
        return results;
    }

    std::vector<std::unordered_map<std::string, int>> mapping_eq_theorem(const std::vector<std::vector<std::string>> &theorem, const std::vector<Point> &points, const std::vector<std::tuple<int, int, int, int, int, int, int, int>> eqs)
    {
        std::vector<std::unordered_map<std::string, int>> mappings;
        std::vector<std::unordered_map<std::string, int>> candidate_mappings;
        if (theorem.empty() || theorem[0].size() != 9 ||
            (theorem[0][0] != "eqangle" && theorem[0][0] != "eqratio"))
            throw py::value_error("Equation matching requires an eight-point anchor");
        std::set<std::string> anchored(theorem[0].begin()+1, theorem[0].end());
        for (const auto &premise : theorem)
        {
            if (premise.size() < 2 || premise[0] == "aconst" || premise[0] == "rconst")
                throw py::value_error("Use ordinary matching for scalar constants or empty premises");
            for (size_t i = 1; i < premise.size(); ++i)
                if (!anchored.count(premise[i]))
                    throw py::value_error("Equation anchor does not bind every variable");
        }
        for (const auto &equation : eqs)
        {
            const int indices[] = {std::get<0>(equation),std::get<1>(equation),std::get<2>(equation),std::get<3>(equation),
                                   std::get<4>(equation),std::get<5>(equation),std::get<6>(equation),std::get<7>(equation)};
            for (int index : indices)
                if (index < 0 || static_cast<size_t>(index) >= points.size())
                    throw py::value_error("Equation point index is out of range");
        }
        for (size_t i = 0; i < eqs.size(); i++)
        {
            int a = std::get<0>(eqs[i]);
            int b = std::get<1>(eqs[i]);
            int c = std::get<2>(eqs[i]);
            int d = std::get<3>(eqs[i]);
            int e = std::get<4>(eqs[i]);
            int f = std::get<5>(eqs[i]);
            int g = std::get<6>(eqs[i]);
            int h = std::get<7>(eqs[i]);
            std::vector<int> args = {a, b, c, d, e, f, g, h};
            std::set<std::vector<int>> args_permutation;

            if (args[0] != args[4])
            {
                args_permutation.insert({args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]});
                args_permutation.insert({args[2], args[3], args[0], args[1], args[6], args[7], args[4], args[5]});
                args_permutation.insert({args[4], args[5], args[6], args[7], args[0], args[1], args[2], args[3]});
                args_permutation.insert({args[6], args[7], args[4], args[5], args[2], args[3], args[0], args[1]});
            }
            else
            {
                args_permutation.insert({args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]});
                args_permutation.insert({args[2], args[3], args[0], args[1], args[6], args[7], args[4], args[5]});
                args_permutation.insert({args[4], args[5], args[6], args[7], args[0], args[1], args[2], args[3]});
                args_permutation.insert({args[6], args[7], args[4], args[5], args[2], args[3], args[0], args[1]});
                args_permutation.insert({args[0], args[1], args[4], args[5], args[2], args[3], args[6], args[7]});
                args_permutation.insert({args[4], args[5], args[0], args[1], args[6], args[7], args[2], args[3]});
                args_permutation.insert({args[6], args[7], args[2], args[3], args[4], args[5], args[0], args[1]});
                args_permutation.insert({args[2], args[3], args[6], args[7], args[0], args[1], args[4], args[5]});
            }
            for (const auto &perm : args_permutation)
            {
                std::unordered_map<std::string, int> current_map;
                bool flag = true;
                for (size_t j = 1; j < theorem[0].size(); j++)
                {
                    std::string v = theorem[0][j];
                    int p = perm[j - 1];
                    if (current_map.find(v) == current_map.end())
                    {
                        current_map[v] = p;
                    }
                    else if (current_map[v] != p)
                    {
                        flag = false;
                        break;
                    }
                }
                if (flag)
                {
                    candidate_mappings.push_back(current_map);
                }
            }
        }
        for (size_t i = 0; i < candidate_mappings.size(); i++)
        {   
            bool flag = true;
            for (size_t j = 1; j < theorem.size(); j++)
            {
                std::string type = theorem[j][0];
                std::vector<Point> mapped_points;
                for (size_t k = 1; k < theorem[j].size(); k++)
                {
                    mapped_points.push_back(points[candidate_mappings[i][theorem[j][k]]]);
                }
                if (!check_numerical(type, mapped_points))
                {
                    flag = false;
                    break;
                }
            }
            if (flag)
            {
                mappings.push_back(candidate_mappings[i]);
            }
        }
        return mappings;
    }
}

PYBIND11_MODULE(matchinC, m)
{
    m.def("mapping_normal_theorem", &mapping_nomal_theorem, "Mapping normal theorem to points");
    m.def("mapping_eq_theorem", &mapping_eq_theorem, "Mapping equation theorem to points");
}