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

void generate_mappings(const std::vector<std::string> &vec, size_t n, std::vector<std::unordered_map<std::string, int>> &result)
{
    size_t total_combinations = 1;
    for (size_t i = 0; i < vec.size(); ++i)
    {
        total_combinations *= n;
    }

    result.resize(total_combinations, std::unordered_map<std::string, int>(vec.size()));

    for (size_t i = 0; i < total_combinations; ++i)
    {
        size_t temp = i;
        for (size_t j = 0; j < vec.size(); ++j)
        {
            result[i][vec[j]] = temp % n;
            temp /= n;
        }
    }
}

std::vector<std::vector<std::unordered_map<std::string, int>>> mapping_theorem_to_points(const std::vector<std::vector<std::string>> &theorem, const size_t &n)
{
    std::vector<std::vector<std::unordered_map<std::string, int>>> result;
    std::vector<std::string> points;
    for (const auto &premise : theorem)
    {
        std::vector<std::string> new_points;
        std::vector<std::unordered_map<std::string, int>> mapping;
        for (size_t i = 1; i < premise.size(); i++)
        {
            if (std::find(points.begin(), points.end(), premise[i]) == points.end())
            {
                new_points.push_back(premise[i]);
                points.push_back(premise[i]);
            }
        }
        if (!new_points.empty())
        {
            generate_mappings(new_points, n, mapping);
        }
        result.push_back(mapping);
    }
    return result;
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
        if (points.size() % 2 != 0)
            return false;
        double ang = -1;
        for (size_t i = 0; i < points.size() - 1; i = i + 2)
        {
            if (close_enough(points[i], points[i + 1]))
            {
                return false;
            }
            double current_ang = angle(points[i], points[i + 1]);
            if (ang >= 0 && !close_enough(ang, current_ang))
            {
                return false;
            }
            ang = current_ang;
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
    else
    {
        std::cout << type << std::endl;
        return true;
    }
    return false;
}

void check_submapping(const size_t index, const std::vector<std::vector<std::string>> &theorem, const std::vector<Point> &points, std::vector<std::vector<std::unordered_map<std::string, int>>> &mappings, std::vector<std::unordered_map<std::string, int>> &results, std::unordered_map<std::string, int> &current_mapping)
{
    if (index == theorem.size())
    {
        results.push_back(current_mapping);
        return;
    }

    auto premise = theorem[index];
    auto submappings = mappings[index];

    for (auto submapping : submappings)
    {
        auto new_mapping = current_mapping;
        for (auto &pair : submapping)
        {
            new_mapping[pair.first] = pair.second;
        }
        std::string type = premise[0];
        std::vector<Point> mapped_points;
        for (size_t i = 1; i < premise.size(); ++i)
        {
            mapped_points.push_back(points[new_mapping[premise[i]]]);
        }
        if (check_numerical(type, mapped_points))
        {
            check_submapping(index + 1, theorem, points, mappings, results, new_mapping);
        }
    }
    if (submappings.empty())
    {
        std::string type = premise[0];
        std::vector<Point> mapped_points;
        for (size_t i = 1; i < premise.size(); ++i)
        {
            mapped_points.push_back(points[current_mapping[premise[i]]]);
        }
        if (check_numerical(type, mapped_points))
        {
            check_submapping(index + 1, theorem, points, mappings, results, current_mapping);
        }
    }
}

extern "C"
{
    std::vector<std::unordered_map<std::string, int>> mapping_nomal_theorem(const std::vector<std::vector<std::string>> &theorem, const std::vector<Point> &points)
    {
        auto mappings = mapping_theorem_to_points(theorem, points.size());
        std::vector<std::unordered_map<std::string, int>> mapping;
        std::unordered_map<std::string, int> current_mapping;
        check_submapping(0, theorem, points, mappings, mapping, current_mapping);
        return mapping;
    }

    std::vector<std::unordered_map<std::string, int>> mapping_eq_theorem(const std::vector<std::vector<std::string>> &theorem, const std::vector<Point> &points, const std::vector<std::tuple<int, int, int, int, int, int, int, int>> eqs)
    {
        std::vector<std::unordered_map<std::string, int>> mappings;
        std::vector<std::unordered_map<std::string, int>> candidate_mappings;
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