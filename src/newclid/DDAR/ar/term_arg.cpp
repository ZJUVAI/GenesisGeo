#include "ar/term_arg.hpp"
#include "type/slope.hpp"
#include "type/dist.hpp"
#include "type/distlog.hpp"
#include <iostream>

using namespace std;

namespace
{
    shared_ptr<const vector<Point>> sorted_points(vector<Point> points)
    {
        if (points.empty())
        {
            return {};
        }
        sort(points.begin(), points.end());
        return make_shared<const vector<Point>>(move(points));
    }
}

TermArg::TermArg(string t, vector<Point> pts)
{
    if (t == "Dist")
        _type = ArgType::Dist;
    else if (t == "DistLog")
        _type = ArgType::DistLog;
    else if (t == "Slope")
        _type = ArgType::Slope;
    else if (t == "Pi")
        _type = ArgType::Pi;
    else
    {
        throw invalid_argument("Invalid term_arg type: " + t);
    }
    _points = sorted_points(move(pts));
}

TermArg::TermArg(ArgType t, vector<Point> pts)
    : _type(t), _points(sorted_points(move(pts)))
{
}

TermArg::TermArg(Slope s) : TermArg(ArgType::Slope, s.points()) {}

TermArg::TermArg(Dist d) : TermArg(ArgType::Dist, d.points()) {}

TermArg::TermArg(DistLog d) : TermArg(ArgType::DistLog, d.points()) {}

const vector<Point> &TermArg::points() const
{
    static const vector<Point> empty;
    return _points ? *_points : empty;
}

string TermArg::to_string() const
{
    switch (_type)
    {
    case ArgType::Dist:
    case ArgType::DistLog:
    case ArgType::Slope:
    {
        string inner;
        for (size_t i = 0; i < points().size(); ++i)
        {
            if (i > 0)
                inner += "-";
            inner += points()[i].name();
        }

        if (_type == ArgType::Dist)
        {
            return "|" + inner + "|";
        }
        else if (_type == ArgType::DistLog)
        {
            return "log(|" + inner + "|)";
        }
        else
        {
            return "∠(" + inner + ")";
        }
    }
    case ArgType::Pi:
        return "π";
    default:
        return "Unknown";
    }
}

double TermArg::to_double() const
{
    switch (_type)
    {
    case ArgType::Dist:
    case ArgType::DistLog:
    {
        if (points().size() != 2)
        {
            throw runtime_error("Dist/DistLog requires exactly 2 points");
        }

        const Point &p1 = points()[0];
        const Point &p2 = points()[1];

        double dx = p2.x() - p1.x();
        double dy = p2.y() - p1.y();
        double dist = sqrt(dx * dx + dy * dy);

        if (_type == ArgType::Dist)
        {
            return dist;
        }
        else
        {
            if (dist <= 0.0)
            {
                throw runtime_error("DistLog: distance must be positive");
            }
            return log(dist);
        }
    }

    case ArgType::Slope:
    {
        if (points().size() != 2)
        {
            throw runtime_error("Slope requires exactly 2 points");
        }

        const Point &left = points()[0];
        const Point &right = points()[1];

        double dx = right.x() - left.x();
        double dy = right.y() - left.y();

        double ang = std::atan2(dy, dx);

        if (ang < 0)
        {
            ang += M_PI;
        }

        if (Numerical::close_enough(ang, M_PI))
        {
            return 0.0;
        }

        return ang;
    }

    case ArgType::Pi:
        return M_PI;

    default:
        throw std::runtime_error("TermArg type cannot be converted to double: " +
                                 std::to_string(static_cast<int>(_type)));
    }
}

bool TermArg::operator==(const TermArg &other) const
{
    return _type == other._type && points() == other.points();
}

bool TermArg::operator!=(const TermArg &other) const
{
    return !(*this == other);
}

bool TermArg::operator<(const TermArg &other) const
{
    if (_type != other._type)
    {
        return _type < other._type;
    }
    return points() < other.points();
}