// c++ -O3 -Wall -shared -std=c++14 `python3 -m pybind11 --includes` geometry.cpp -o geometry`python3-config --extension-suffix` -fPIC
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <string>
#include <set>
#include <iostream>

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

double modpi(double x)
{
    double result = std::fmod(x, M_PI);
    if (result < 0)
    {
        result += M_PI;
    }
    return result;
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

Point midpoint(const Point &a, const Point &b)
{
    return {(a.first + b.first) / 2, (a.second + b.second) / 2};
}

bool goal_filter(const std::string &name, const std::tuple<std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string> &args)
{
    // 创建两个集合 seg_1, seg_2, seg_3, seg_4
    std::set<std::string> seg_1 = {std::get<0>(args), std::get<1>(args)};
    std::set<std::string> seg_2 = {std::get<2>(args), std::get<3>(args)};
    std::set<std::string> seg_3 = {std::get<4>(args), std::get<5>(args)};
    std::set<std::string> seg_4 = {std::get<6>(args), std::get<7>(args)};

    // case: eqratio AB/CD = DC/BA
    if (seg_1 == seg_3 && seg_2 == seg_4)
    {
        return false;
    }
    if (seg_1 == seg_4 && seg_2 == seg_3)
    {
        return false;
    }

    // AB/AB = CD/EF => cong CD = EF
    if (seg_1 == seg_2 || seg_3 == seg_4)
    {
        return false;
    }

    return true;
}

std::tuple<int, int, int, int, int, int, int, int> parse_eq(std::tuple<double, int, int, int, int> angle1, std::tuple<double, int, int, int, int> angle2)
{
    int a = std::get<1>(angle1);
    int b = std::get<2>(angle1);
    int c = std::get<3>(angle1);
    int d = std::get<4>(angle1);
    int e = std::get<1>(angle2);
    int f = std::get<2>(angle2);
    int g = std::get<3>(angle2);
    int h = std::get<4>(angle2);
    int a1 = -1, b1 = -1, c1 = -1, d1 = -1, e1 = -1, f1 = -1, g1 = -1, h1 = -1;

    if (a == b || c == d || e == f || g == h)
        return std::make_tuple(-1, -1, -1, -1, -1, -1, -1, -1);
    if (a == c && e == g)
    {
        a1 = a;
        b1 = b;
        c1 = c;
        d1 = d;
        e1 = e;
        f1 = f;
        g1 = g;
        h1 = h;
    }
    else if (a == c && e == h)
    {
        a1 = a;
        b1 = b;
        c1 = c;
        d1 = d;
        e1 = e;
        f1 = f;
        g1 = h;
        h1 = g;
    }
    else if (a == c && f == g)
    {
        a1 = a;
        b1 = b;
        c1 = c;
        d1 = d;
        e1 = f;
        f1 = e;
        g1 = g;
        h1 = h;
    }
    else if (a == c && f == h)
    {
        a1 = a;
        b1 = b;
        c1 = c;
        d1 = d;
        e1 = f;
        f1 = e;
        g1 = h;
        h1 = g;
    }
    else if (a == d && e == g)
    {
        a1 = a;
        b1 = b;
        c1 = d;
        d1 = c;
        e1 = e;
        f1 = f;
        g1 = g;
        h1 = h;
    }
    else if (a == d && e == h)
    {
        a1 = a;
        b1 = b;
        c1 = d;
        d1 = c;
        e1 = e;
        f1 = f;
        g1 = h;
        h1 = g;
    }
    else if (a == d && f == g)
    {
        a1 = a;
        b1 = b;
        c1 = d;
        d1 = c;
        e1 = f;
        f1 = e;
        g1 = g;
        h1 = h;
    }
    else if (a == d && f == h)
    {
        a1 = a;
        b1 = b;
        c1 = d;
        d1 = c;
        e1 = f;
        f1 = e;
        g1 = h;
        h1 = g;
    }
    else if (b == c && e == g)
    {
        a1 = b;
        b1 = a;
        c1 = c;
        d1 = d;
        e1 = e;
        f1 = f;
        g1 = g;
        h1 = h;
    }
    else if (b == c && e == h)
    {
        a1 = b;
        b1 = a;
        c1 = c;
        d1 = d;
        e1 = e;
        f1 = f;
        g1 = h;
        h1 = g;
    }
    else if (b == c && f == g)
    {
        a1 = b;
        b1 = a;
        c1 = c;
        d1 = d;
        e1 = f;
        f1 = e;
        g1 = g;
        h1 = h;
    }
    else if (b == c && f == h)
    {
        a1 = b;
        b1 = a;
        c1 = c;
        d1 = d;
        e1 = f;
        f1 = e;
        g1 = h;
        h1 = g;
    }
    else if (b == d && e == g)
    {
        a1 = b;
        b1 = a;
        c1 = d;
        d1 = c;
        e1 = e;
        f1 = f;
        g1 = g;
        h1 = h;
    }
    else if (b == d && e == h)
    {
        a1 = b;
        b1 = a;
        c1 = d;
        d1 = c;
        e1 = e;
        f1 = f;
        g1 = h;
        h1 = g;
    }
    else if (b == d && f == g)
    {
        a1 = b;
        b1 = a;
        c1 = d;
        d1 = c;
        e1 = f;
        f1 = e;
        g1 = g;
        h1 = h;
    }
    else if (b == d && f == h)
    {
        a1 = b;
        b1 = a;
        c1 = d;
        d1 = c;
        e1 = f;
        f1 = e;
        g1 = h;
        h1 = g;
    }
    else if (a == e && c == g)
    {
        a1 = a;
        b1 = b;
        c1 = e;
        d1 = f;
        e1 = c;
        f1 = d;
        g1 = g;
        h1 = h;
    }
    else if (a == e && c == h)
    {
        a1 = a;
        b1 = b;
        c1 = e;
        d1 = f;
        e1 = c;
        f1 = d;
        g1 = h;
        h1 = g;
    }
    else if (a == e && d == g)
    {
        a1 = a;
        b1 = b;
        c1 = e;
        d1 = f;
        e1 = d;
        f1 = c;
        g1 = g;
        h1 = h;
    }
    else if (a == e && d == h)
    {
        a1 = a;
        b1 = b;
        c1 = e;
        d1 = f;
        e1 = d;
        f1 = c;
        g1 = h;
        h1 = g;
    }
    else if (a == f && c == g)
    {
        a1 = a;
        b1 = b;
        c1 = f;
        d1 = e;
        e1 = c;
        f1 = d;
        g1 = g;
        h1 = h;
    }
    else if (a == f && c == h)
    {
        a1 = a;
        b1 = b;
        c1 = f;
        d1 = e;
        e1 = c;
        f1 = d;
        g1 = h;
        h1 = g;
    }
    else if (a == f && d == g)
    {
        a1 = a;
        b1 = b;
        c1 = f;
        d1 = e;
        e1 = d;
        f1 = c;
        g1 = g;
        h1 = h;
    }
    else if (a == f && d == h)
    {
        a1 = a;
        b1 = b;
        c1 = f;
        d1 = e;
        e1 = d;
        f1 = c;
        g1 = h;
        h1 = g;
    }
    else if (b == e && c == g)
    {
        a1 = b;
        b1 = a;
        c1 = e;
        d1 = f;
        e1 = c;
        f1 = d;
        g1 = g;
        h1 = h;
    }
    else if (b == e && c == h)
    {
        a1 = b;
        b1 = a;
        c1 = e;
        d1 = f;
        e1 = c;
        f1 = d;
        g1 = h;
        h1 = g;
    }
    else if (b == e && d == g)
    {
        a1 = b;
        b1 = a;
        c1 = e;
        d1 = f;
        e1 = d;
        f1 = c;
        g1 = g;
        h1 = h;
    }
    else if (b == e && d == h)
    {
        a1 = b;
        b1 = a;
        c1 = e;
        d1 = f;
        e1 = d;
        f1 = c;
        g1 = h;
        h1 = g;
    }
    else if (b == f && c == g)
    {
        a1 = b;
        b1 = a;
        c1 = f;
        d1 = e;
        e1 = c;
        f1 = d;
        g1 = g;
        h1 = h;
    }
    else if (b == f && c == h)
    {
        a1 = b;
        b1 = a;
        c1 = f;
        d1 = e;
        e1 = c;
        f1 = d;
        g1 = h;
        h1 = g;
    }
    else if (b == f && d == g)
    {
        a1 = b;
        b1 = a;
        c1 = f;
        d1 = e;
        e1 = d;
        f1 = c;
        g1 = g;
        h1 = h;
    }
    else if (b == f && d == h)
    {
        a1 = b;
        b1 = a;
        c1 = f;
        d1 = e;
        e1 = d;
        f1 = c;
        g1 = h;
        h1 = g;
    }
    else
    {
        return std::make_tuple(-1, -1, -1, -1, -1, -1, -1, -1);
    }

    std::tuple<int, int, int, int> g1a = std::make_tuple(a1, b1, c1, d1);
    std::tuple<int, int, int, int> g1b = std::make_tuple(e1, f1, g1, h1);
    std::tuple<int, int, int, int> g2a = std::make_tuple(c1, d1, a1, b1);
    std::tuple<int, int, int, int> g2b = std::make_tuple(g1, h1, e1, f1);

    // 比较元组并组合
    std::tuple<int, int, int, int, int, int, int, int> groups1, groups2;

    // g1a 和 g1b 的比较
    if (g1a <= g1b)
    {
        groups1 = std::make_tuple(std::get<0>(g1a), std::get<1>(g1a), std::get<2>(g1a), std::get<3>(g1a), std::get<0>(g1b), std::get<1>(g1b), std::get<2>(g1b), std::get<3>(g1b));
    }
    else
    {
        groups1 = std::make_tuple(std::get<0>(g1b), std::get<1>(g1b), std::get<2>(g1b), std::get<3>(g1b), std::get<0>(g1a), std::get<1>(g1a), std::get<2>(g1a), std::get<3>(g1a));
    }

    // g2a 和 g2b 的比较
    if (g2a <= g2b)
    {
        groups2 = std::make_tuple(std::get<0>(g2a), std::get<1>(g2a), std::get<2>(g2a), std::get<3>(g2a), std::get<0>(g2b), std::get<1>(g2b), std::get<2>(g2b), std::get<3>(g2b));
    }
    else
    {
        groups2 = std::make_tuple(std::get<0>(g2b), std::get<1>(g2b), std::get<2>(g2b), std::get<3>(g2b), std::get<0>(g2a), std::get<1>(g2a), std::get<2>(g2a), std::get<3>(g2a));
    }

    return groups1 <= groups2 ? groups1 : groups2;
    // return groups1;
}

std::tuple<int, int, int, int, int, int> parse_simitri(std::tuple<int, int, int, int, int, int> &simitri)
{
    int a = std::get<0>(simitri);
    int b = std::get<1>(simitri);
    int c = std::get<2>(simitri);
    int d = std::get<3>(simitri);
    int e = std::get<4>(simitri);
    int f = std::get<5>(simitri);
    auto aa = std::make_pair(a, d);
    auto bb = std::make_pair(b, e);
    auto cc = std::make_pair(c, f);
    std::vector<std::tuple<int, int>> group1 = {aa, bb, cc};
    sort(group1.begin(), group1.end());
    auto aa1 = std::make_pair(d, a);
    auto bb1 = std::make_pair(e, b);
    auto cc1 = std::make_pair(f, c);
    std::vector<std::tuple<int, int>> group2 = {aa1, bb1, cc1};
    sort(group2.begin(), group2.end());

    std::tuple<int, int, int, int, int, int> result1 = {std::get<0>(group1[0]), std::get<0>(group1[1]), std::get<0>(group1[2]), std::get<1>(group1[0]), std::get<1>(group1[1]), std::get<1>(group1[2])};
    std::tuple<int, int, int, int, int, int> result2 = {std::get<0>(group2[0]), std::get<0>(group2[1]), std::get<0>(group2[2]), std::get<1>(group2[0]), std::get<1>(group2[1]), std::get<1>(group2[2])};

    return result1 <= result2 ? result1 : result2;
}

bool check_simtri(const std::vector<Point> &points, const std::tuple<int, int, int, int, int, int> &ids)
{
    Point a = points[std::get<0>(ids)];
    Point b = points[std::get<1>(ids)];
    Point c = points[std::get<2>(ids)];
    Point d = points[std::get<3>(ids)];
    Point e = points[std::get<4>(ids)];
    Point f = points[std::get<5>(ids)];
    double ab = distance(a, b);
    double de = distance(d, e);
    double ratio = ab / de;
    double bc = distance(b, c);
    double ef = distance(e, f);
    double bc1 = ef * ratio;
    return close_enough(bc, bc1);
}

double clock(const Point &a, const Point &b, const Point &c)
{
    double abx = b.first - a.first;
    double aby = b.second - a.second;
    double acx = c.first - a.first;
    double acy = c.second - a.second;
    return abx * acy - aby * acx;
}

bool sameclock(const std::vector<Point> &points, const std::tuple<int, int, int, int, int, int> &ids)
{
    Point a = points[std::get<0>(ids)];
    Point b = points[std::get<1>(ids)];
    Point c = points[std::get<2>(ids)];
    Point d = points[std::get<3>(ids)];
    Point e = points[std::get<4>(ids)];
    Point f = points[std::get<5>(ids)];

    double clock1 = clock(a, b, c);
    double clock2 = clock(d, e, f);
    if (clock1 * clock2 > ATOM)
    {
        return true;
    }
    return false;
}

extern "C"
{
    std::tuple<std::vector<std::tuple<double, int, int, int, int>>,
               std::vector<std::tuple<double, int, int, int, int>>>
    process_points(const std::vector<Point> &points)
    {
        std::vector<std::tuple<double, int, int, int, int>> angles;
        std::vector<std::tuple<double, int, int, int, int>> ratios;

        std::vector<std::tuple<double, int, int>> all_angle;
        std::vector<std::tuple<double, int, int>> all_dis;

        for (size_t i = 0; i < points.size(); ++i)
        {
            for (size_t j = i + 1; j < points.size(); ++j)
            {
                all_angle.push_back({angle(points[i], points[j]), i, j});
                all_dis.push_back({distance(points[i], points[j]), i, j});
            }
        }

        for (size_t i = 0; i < all_angle.size(); ++i)
        {
            for (size_t j = i + 1; j < all_angle.size(); ++j)
            {
                const auto &a = all_angle[i];
                const auto &b = all_angle[j];
                angles.push_back({modpi(std::get<0>(b) - std::get<0>(a)), std::get<1>(a), std::get<2>(a), std::get<1>(b), std::get<2>(b)});
                angles.push_back({modpi(std::get<0>(a) - std::get<0>(b)), std::get<1>(b), std::get<2>(b), std::get<1>(a), std::get<2>(a)});
            }
        }

        for (size_t i = 0; i < all_dis.size(); ++i)
        {
            for (size_t j = i + 1; j < all_dis.size(); ++j)
            {
                const auto &a = all_dis[i];
                const auto &b = all_dis[j];
                if (std::get<0>(a) > std::get<0>(b))
                {
                    ratios.push_back({std::get<0>(a) / std::get<0>(b), std::get<1>(a), std::get<2>(a), std::get<1>(b), std::get<2>(b)});
                }
                else
                {
                    ratios.push_back({std::get<0>(b) / std::get<0>(a), std::get<1>(b), std::get<2>(b), std::get<1>(a), std::get<2>(a)});
                }
            }
        }

        std::sort(angles.begin(), angles.end(), [](const auto &lhs, const auto &rhs)
                  { return std::get<0>(lhs) < std::get<0>(rhs); });

        std::sort(ratios.begin(), ratios.end(), [](const auto &lhs, const auto &rhs)
                  { return std::get<0>(lhs) < std::get<0>(rhs); });

        return {angles, ratios};
    }

    std::vector<std::tuple<int, int, int>>
    findmidp(const std::vector<Point> &points, const std::vector<std::tuple<double, int, int, int, int>> &ratios)
    {
        std::vector<std::tuple<int, int, int>> midpoints;
        for (const auto &ratio : ratios)
        {
            double r = std::get<0>(ratio);
            int i = std::get<1>(ratio);
            int j = std::get<2>(ratio);
            int k = std::get<3>(ratio);
            int l = std::get<4>(ratio);

            if (close_enough(r, 1.0))
            {
                if (i == k && close_enough(points[i], midpoint(points[j], points[l])))
                {
                    midpoints.push_back({i, j, l});
                }
                else if (i == l && close_enough(points[i], midpoint(points[j], points[k])))
                {
                    midpoints.push_back({i, j, k});
                }
                else if (j == k && close_enough(points[j], midpoint(points[i], points[l])))
                {
                    midpoints.push_back({j, i, l});
                }
                else if (j == l && close_enough(points[j], midpoint(points[i], points[k])))
                {
                    midpoints.push_back({j, i, k});
                }
            }
            else
            {
                break;
            }
        }
        return midpoints;
    }

    std::vector<std::tuple<int, int, int, int, int, int, int, int>>
    findeq(const std::vector<Point> &points, const std::vector<std::tuple<double, int, int, int, int>> &datas)
    {
        std::vector<std::tuple<int, int, int, int, int, int, int, int>> eqs;
        for (size_t i = 0; i < datas.size(); ++i)
        {
            if (close_enough(std::get<0>(datas[i]), 0) || close_enough(std::get<0>(datas[i]), M_PI))
            {
                continue;
            }
            for (size_t j = i + 1; j < datas.size(); ++j)
            {
                if (!close_enough(std::get<0>(datas[i]), std::get<0>(datas[j])))
                {
                    break;
                }
                auto parsed = parse_eq(datas[i], datas[j]);
                if (std::get<0>(parsed) == -1)
                {
                    continue;
                }
                eqs.push_back(parsed);
            }
        }
        return eqs;
    }

    std::tuple<std::vector<std::tuple<int, int, int, int, int, int>>, std::vector<std::tuple<int, int, int, int, int, int>>>
    findsimitri(const std::vector<Point> &points, std::vector<std::tuple<int, int, int, int, int, int, int, int>> &eqratios)
    {
        std::vector<std::tuple<int, int, int, int, int, int>> simitris;
        std::vector<std::tuple<int, int, int, int, int, int>> simitrirs;

        for (size_t i = 0; i < eqratios.size(); ++i)
        {
            std::tuple<int, int, int, int, int, int> tmp1 = {std::get<0>(eqratios[i]), std::get<1>(eqratios[i]), std::get<3>(eqratios[i]), std::get<4>(eqratios[i]), std::get<5>(eqratios[i]), std::get<7>(eqratios[i])};
            std::tuple<int, int, int, int, int, int> tmp2 = {std::get<0>(eqratios[i]), std::get<3>(eqratios[i]), std::get<1>(eqratios[i]), std::get<4>(eqratios[i]), std::get<5>(eqratios[i]), std::get<7>(eqratios[i])};
            if (check_simtri(points, tmp1))
            {
                if (sameclock(points, tmp1))
                {
                    simitris.push_back(parse_simitri(tmp1));
                }
                else if (sameclock(points, tmp2))
                {
                    simitrirs.push_back(parse_simitri(tmp1));
                }
            }
            if (std::get<0>(eqratios[i]) == std::get<4>(eqratios[i]))
            {
                std::tuple<int, int, int, int, int, int> tmp3 = {std::get<0>(eqratios[i]), std::get<1>(eqratios[i]), std::get<5>(eqratios[i]), std::get<4>(eqratios[i]), std::get<3>(eqratios[i]), std::get<7>(eqratios[i])};
                std::tuple<int, int, int, int, int, int> tmp4 = {std::get<0>(eqratios[i]), std::get<5>(eqratios[i]), std::get<1>(eqratios[i]), std::get<4>(eqratios[i]), std::get<3>(eqratios[i]), std::get<7>(eqratios[i])};
                if (check_simtri(points, tmp3))
                {
                    if (sameclock(points, tmp3))
                    {
                        simitris.push_back(parse_simitri(tmp3));
                    }
                    else if (sameclock(points, tmp4))
                    {
                        simitrirs.push_back(parse_simitri(tmp3));
                    }
                }
            }
        }
        return std::make_tuple(simitris, simitrirs);
    }

    std::vector<std::tuple<int, int, int, int>> findpara(const std::vector<Point> &points, std::vector<std::tuple<double, int, int, int, int>> &angles)
    {
        std::vector<std::tuple<int, int, int, int>> paras;
        for (const auto &angle : angles)
        {
            double ag = std::get<0>(angle);
            int a = std::get<1>(angle);
            int b = std::get<2>(angle);
            int c = std::get<3>(angle);
            int d = std::get<4>(angle);
            if (close_enough(ag, 0) || close_enough(ag, M_PI))
            {
                if (a > b)
                {
                    int tmp = a;
                    a = b;
                    b = tmp;
                }
                if (c > d)
                {
                    int tmp = c;
                    c = d;
                    d = tmp;
                }
                if (a == c && b == d)
                {
                    continue;
                }
                if (a < c || (a == c && b < d))
                {
                    paras.push_back({a, b, c, d});
                }
                else
                {
                    paras.push_back({c, d, a, b});
                }
            }
        }
        return paras;
    }

    std::vector<std::tuple<int, int, int, int>> findperp(const std::vector<Point> &points, std::vector<std::tuple<double, int, int, int, int>> &angles)
    {
        std::vector<std::tuple<int, int, int, int>> perps;
        for (const auto &angle : angles)
        {
            double ag = std::get<0>(angle);
            int a = std::get<1>(angle);
            int b = std::get<2>(angle);
            int c = std::get<3>(angle);
            int d = std::get<4>(angle);
            if (close_enough(ag, M_PI / 2))
            {
                if (a > b)
                {
                    int tmp = a;
                    a = b;
                    b = tmp;
                }
                if (c > d)
                {
                    int tmp = c;
                    c = d;
                    d = tmp;
                }
                if (a == c && b == d)
                {
                    continue;
                }
                if (a < c || (a == c && b < d))
                {
                    perps.push_back({a, b, c, d});
                }
                else
                {
                    perps.push_back({c, d, a, b});
                }
            }
        }
        return perps;
    }

    std::vector<std::tuple<int, int, int, int>> findcong(const std::vector<Point> &points, std::vector<std::tuple<double, int, int, int, int>> &ratios)
    {
        std::vector<std::tuple<int, int, int, int>> congs;
        for (const auto &ratio : ratios)
        {
            double r = std::get<0>(ratio);
            int a = std::get<1>(ratio);
            int b = std::get<2>(ratio);
            int c = std::get<3>(ratio);
            int d = std::get<4>(ratio);
            if (close_enough(r, 1.0))
            {
                if (a > b)
                {
                    int tmp = a;
                    a = b;
                    b = tmp;
                }
                if (c > d)
                {
                    int tmp = c;
                    c = d;
                    d = tmp;
                }
                if (a == c && b == d)
                {
                    continue;
                }
                if (a < c || (a == c && b < d))
                {
                    congs.push_back({a, b, c, d});
                }
                else
                {
                    congs.push_back({c, d, a, b});
                }
            }
        }
        return congs;
    }
}

PYBIND11_MODULE(geometry, m)
{
    m.def("process_points", &process_points, "Process points to calculate angles and ratios");
    m.def("findmidp", &findmidp, "Find midpoints based on ratios");
    m.def("findeq", &findeq, "Find equals based on angle data or ratio data");
    m.def("findsimitri", &findsimitri, "Find similiary triangles based on eqratios");
    m.def("findpara", &findpara, "Find parallel lines based on angles");
    m.def("findperp", &findperp, "Find perpendicular lines based on angles");
    m.def("findcong", &findcong, "Find congruent lines based on ratios");
}
