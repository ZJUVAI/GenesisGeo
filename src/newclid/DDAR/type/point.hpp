#ifndef POINT_HPP
#define POINT_HPP

#include "type/pointnum.hpp"
#include "numerical.hpp"
#include <string>
#include <ostream> // 引入 std::ostream 以便声明 operator<<

// 声明类 Point
class Point
{
public:
    Point(std::string name, double x, double y); // 构造函数
    Point(std::string name, PointNum num);       // 构造函数

    const std::string &name() const &;
    std::string name() const &&;
    double x() const;
    double y() const;
    PointNum num() const;
    std::string to_string() const;

    bool is_close(const Point &other) const;

    bool operator==(const Point &other) const;
    bool operator!=(const Point &other) const;
    bool operator<(const Point &other) const;
    bool operator>(const Point &other) const;
    bool operator<=(const Point &other) const;
    bool operator>=(const Point &other) const;

    static void parseName(const std::string &name,
                          std::string &prefix,
                          int &index)
    {
        const size_t length = prefix_length(name);
        prefix = name.substr(0, length);
        index = parse_index(name, length);
    }

private:
    static size_t prefix_length(const std::string &name);
    static int parse_index(const std::string &name, size_t prefix_length);

    PointNum _num;
    std::string _name;
    size_t _prefix_length;
    int _index;
};

// 声明 operator<<
std::ostream &operator<<(std::ostream &os, const Point &pt);

#endif // POINT_HPP
