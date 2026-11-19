#include "type/point.hpp"

#include <string_view>
#include <utility>

size_t Point::prefix_length(const std::string &name)
{
    size_t length = 0;
    while (length < name.size() && std::isalpha(static_cast<unsigned char>(name[length])))
    {
        ++length;
    }
    return length;
}

int Point::parse_index(const std::string &name, size_t length)
{
    return length == name.size() ? -1 : std::stoi(name.substr(length));
}

Point::Point(std::string name, double x, double y)
    : Point(std::move(name), PointNum(x, y))
{
}

Point::Point(std::string name, PointNum num)
    : _num(num), _name(std::move(name)), _prefix_length(prefix_length(_name)),
      _index(parse_index(_name, _prefix_length))
{
}

const std::string &Point::name() const & { return _name; }
std::string Point::name() const && { return _name; }

double Point::x() const { return _num.x(); }

double Point::y() const { return _num.y(); }

PointNum Point::num() const { return _num; }

std::string Point::to_string() const
{
    return _name + "(" + std::to_string(_num.x()) + "," + std::to_string(_num.y()) + ")";
}

bool Point::is_close(const Point &other) const
{
    return Numerical::close_enough(x(), other.x()) && Numerical::close_enough(y(), other.y());
}

bool Point::operator==(const Point &other) const
{
    return _name == other.name();
}

bool Point::operator!=(const Point &other) const
{
    return !(_name == other.name());
}

bool Point::operator<(const Point &other) const
{
    if (other._index != _index)
    {
        return _index < other._index;
    }
    return std::string_view(_name).substr(0, _prefix_length) <
           std::string_view(other._name).substr(0, other._prefix_length);
}

bool Point::operator>(const Point &other) const
{
    return other < *this;
}

bool Point::operator<=(const Point &other) const
{
    return !(*this > other);
}

bool Point::operator>=(const Point &other) const
{
    return !(*this < other);
}

std::ostream &operator<<(std::ostream &os, const Point &pt)
{
    os << pt.name();
    return os;
}
