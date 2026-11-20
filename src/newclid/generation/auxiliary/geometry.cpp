#include "geometry.h"
#include "line.h"
#include "circle.h"
#include "utils.h"
#include <algorithm>
#include <cstdio>
#include <charconv>
#include <cfenv>
#include <clocale>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace auxiliary_c {

constexpr double TOLERANCE = 1e-8;

// Helper: get point from coords
inline Point get_point(const std::unordered_map<std::string, Point>& coords, const std::string& name) {
    return coords.at(name);
}

// ===================== Intersection primitives =====================

// Intersection of two lines
std::vector<Point> intersect_two_lines(const Point& a1, const Point& a2,
                                       const Point& b1, const Point& b2) {
    std::vector<Point> result;
    double dx1 = a2[0] - a1[0];
    double dy1 = a2[1] - a1[1];
    double dx2 = b2[0] - b1[0];
    double dy2 = b2[1] - b1[1];
    double det = dx1 * dy2 - dy1 * dx2;

    if (std::abs(det) < TOLERANCE) return result;

    // Python: CA = A - C, t = (CA x CD) / denom, inter = A - t * AB
    double t = ((a1[0] - b1[0]) * dy2 - (a1[1] - b1[1]) * dx2) / det;
    result.push_back({a1[0] - t * dx1, a1[1] - t * dy1});
    return result;
}

// Intersection of two circles
std::vector<Point> intersect_two_circles(const Circle& c1, const Circle& c2) {
    std::vector<Point> result;
    double d = dist(c1.center, c2.center);

    if (d > c1.radius + c2.radius + TOLERANCE ||
        d < std::abs(c1.radius - c2.radius) - TOLERANCE ||
        d < TOLERANCE) {
        return result;
    }

    double a = (c1.radius * c1.radius - c2.radius * c2.radius + d * d) / (2 * d);
    double h_squared = c1.radius * c1.radius - a * a;
    double h = std::sqrt(std::max(0.0, h_squared));

    double dx = c2.center[0] - c1.center[0];
    double dy = c2.center[1] - c1.center[1];
    double px = c1.center[0] + a * dx / d;
    double py = c1.center[1] + a * dy / d;

    if (h < TOLERANCE) {
        result.push_back({px, py});
    } else {
        double ux = -dy / d;
        double uy = dx / d;
        result.push_back({px + h * ux, py + h * uy});
        result.push_back({px - h * ux, py - h * uy});
    }
    return result;
}

// Intersection of line and circle
std::vector<Point> intersect_line_circle(const Point& A, const Point& B,
                                          const Circle& c) {
    std::vector<Point> result;
    double dx = B[0] - A[0];
    double dy = B[1] - A[1];
    double fx = A[0] - c.center[0];
    double fy = A[1] - c.center[1];

    double a_quad = dx * dx + dy * dy;
    double b_quad = 2 * (fx * dx + fy * dy);
    double c_quad = fx * fx + fy * fy - c.radius * c.radius;
    double disc = b_quad * b_quad - 4 * a_quad * c_quad;

    if (disc < -TOLERANCE) return result;

    disc = std::max(0.0, disc);
    double sqrt_disc = std::sqrt(disc);

    if (std::abs(a_quad) < TOLERANCE) {
        // Degenerate line: A == B
        double dist_sq = fx * fx + fy * fy;
        if (std::abs(dist_sq - c.radius * c.radius) < TOLERANCE) {
            result.push_back(A);
        }
    } else {
        for (int sign : {1, -1}) {
            double t = (-b_quad + sign * sqrt_disc) / (2 * a_quad);
            result.push_back({A[0] + t * dx, A[1] + t * dy});
        }
    }
    return result;
}

// ===================== check_on_line for geometric constraints =====================

bool check_on_line_constraint(
    const Point& coord,
    const std::vector<std::pair<std::string, std::vector<std::string>>>& all_lines,
    const std::unordered_map<std::string, Point>& coords,
    const std::vector<std::vector<std::string>>& current_lines = {}) {

    for (const auto& [_, line_points] : all_lines) {
        if (line_points.size() < 2) continue;

        // Skip if this line is a subset of any excluded line
        bool excluded = false;
        for (const auto& curr : current_lines) {
            std::set<std::string> curr_set(curr.begin(), curr.end());
            std::set<std::string> line_set(line_points.begin(), line_points.end());
            // Check if line_set contains all elements of curr_set
            bool contains_all = true;
            for (const auto& p : curr_set) {
                if (line_set.find(p) == line_set.end()) {
                    contains_all = false;
                    break;
                }
            }
            if (contains_all && curr_set.size() <= line_set.size()) {
                excluded = true;
                break;
            }
        }
        if (excluded) continue;

        const Point& x1 = coords.at(line_points[0]);
        const Point& x2 = coords.at(line_points[1]);
        double ldx = x2[0] - x1[0];
        double ldy = x2[1] - x1[1];
        double line_len = std::sqrt(ldx * ldx + ldy * ldy);
        if (line_len < TOLERANCE) continue;

        double vx = coord[0] - x1[0];
        double vy = coord[1] - x1[1];
        double cross = vx * ldy - vy * ldx;
        double d = std::abs(cross) / line_len;

        if (d <= TOLERANCE) return true;
    }
    return false;
}

bool check_on_circle_constraint(
    const Point& coord,
    const std::vector<std::tuple<std::string, Point, double, std::vector<std::string>>>& all_circles) {

    for (const auto& [_, center, radius, __] : all_circles) {
        double d = dist(coord, center);
        if (std::abs(d - radius) < TOLERANCE) return true;
    }
    return false;
}

// ===================== Deduplicate candidates =====================

std::vector<Point> deduplicate_candidates(const std::vector<Point>& candidates) {
    std::vector<Point> unique;
    for (const auto& pt : candidates) {
        bool dup = false;
        for (const auto& u : unique) {
            if (std::abs(pt[0] - u[0]) < TOLERANCE && std::abs(pt[1] - u[1]) < TOLERANCE) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            unique.push_back(round_coord(pt));
        }
    }
    return unique;
}

// ===================== Main intersection functions =====================

PotentialPoints intersection_between_lines(
    const std::vector<std::string>& point_names,
    const std::unordered_map<std::string, Point>& coords,
    const std::vector<std::pair<std::string, std::vector<std::string>>>* lines_data,
    const std::unordered_map<std::string, Point>* rounded_coords) {

    PotentialPoints result;
    if (!lines_data || lines_data->size() < 2) return result;

    // Map: rounded intersection point -> set of lines passing through it
    std::unordered_map<std::string, std::set<std::vector<std::string>>> point_to_lines;
    std::unordered_map<std::string, Point> point_coords_map;

    for (size_t i = 0; i < lines_data->size(); ++i) {
        for (size_t j = i + 1; j < lines_data->size(); ++j) {
            const auto& l1 = (*lines_data)[i];
            const auto& l2 = (*lines_data)[j];

            const Point& A = coords.at(l1.second[0]);
            const Point& B = coords.at(l1.second[1]);
            const Point& C = coords.at(l2.second[0]);
            const Point& D = coords.at(l2.second[1]);

            auto intersections = intersect_two_lines(A, B, C, D);

            for (const auto& inter : intersections) {
                Point rounded = round_coord(inter);

                // Remove points that coincide with intersection
                std::vector<std::string> cleaned_l1, cleaned_l2;
                for (const auto& name : l1.second) {
                    if ((*rounded_coords).at(name) != rounded) {
                        cleaned_l1.push_back(name);
                    }
                }
                for (const auto& name : l2.second) {
                    if ((*rounded_coords).at(name) != rounded) {
                        cleaned_l2.push_back(name);
                    }
                }

                if (cleaned_l1.size() <= 1 && cleaned_l2.size() <= 1) continue;

                std::vector<std::string> use_l1 = (cleaned_l1.size() > 1) ? cleaned_l1 : l1.second;
                std::vector<std::string> use_l2 = (cleaned_l2.size() > 1) ? cleaned_l2 : l2.second;

                std::sort(use_l1.begin(), use_l1.end());
                std::sort(use_l2.begin(), use_l2.end());

                std::ostringstream oss1;
                oss1 << std::fixed << std::setprecision(9) << rounded[0] << "," << rounded[1];
                std::string key = oss1.str();
                point_coords_map[key] = rounded;
                point_to_lines[key].insert(use_l1);
                point_to_lines[key].insert(use_l2);
            }
        }
    }

    // Only keep points where at least 3 lines meet
    for (const auto& [key, line_set] : point_to_lines) {
        if (line_set.size() < 3) continue;

        std::vector<std::vector<std::string>> line_list(line_set.begin(), line_set.end());
        std::sort(line_list.begin(), line_list.end(),
                  [](const auto& a, const auto& b) { return a[0] < b[0]; });

        const auto& line1 = line_list[0];
        const auto& line2 = line_list[1];

        // Format construction strings: "on_line point1 point2"
        std::vector<std::string> constructions = {
            "on_line " + line1[0] + " " + line1[1],
            "on_line " + line2[0] + " " + line2[1]
        };
        result.push_back({point_coords_map[key], constructions});
    }

    return result;
}

// For already rounded coordinates below one million, scaling back to
// the nine-digit integer grid is exact after rounding to the nearest integer.
static char* write_rounded_coordinate(char* out, char* end, double value) {
    constexpr long long scale = 1000000000;
    static_assert(ROUND_DECIMALS == 9);
    if (value < 0) *out++ = '-';
    const auto scaled = std::llround(std::abs(value) * scale);
    out = std::to_chars(out, end, scaled / scale).ptr;
    *out++ = '.';
    auto fraction = scaled % scale;
    for (int i = ROUND_DECIMALS - 1; i >= 0; --i) {
        out[i] = '0' + fraction % 10;
        fraction /= 10;
    }
    return out + ROUND_DECIMALS;
}

PotentialPoints intersection_between_circles(
    const std::vector<std::string>& point_names,
    const std::unordered_map<std::string, Point>& coords,
    const std::vector<std::tuple<std::string, Point, double, std::vector<std::string>>>* circles_data,
    const std::unordered_map<std::string, Point>* rounded_coords) {

    PotentialPoints result;
    if (!circles_data || circles_data->size() < 2) return result;

    using Membership = std::vector<std::string>;
    std::set<Membership> memberships;
    struct CircleMembers {
        const Membership* original;
        std::vector<std::pair<Point, const Membership*>> without_intersection;
    };
    std::vector<CircleMembers> circle_members;
    circle_members.reserve(circles_data->size());
    for (const auto& circle : *circles_data) {
        auto names = std::get<3>(circle);
        std::sort(names.begin(), names.end());
        const auto* original = &*memberships.insert(std::move(names)).first;
        CircleMembers members{original, {}};
        for (const auto& name : *original) {
            const Point& coord = rounded_coords->at(name);
            const auto duplicate = std::find_if(
                members.without_intersection.begin(), members.without_intersection.end(),
                [&](const auto& entry) { return entry.first == coord; });
            if (duplicate != members.without_intersection.end()) continue;
            Membership remaining;
            for (const auto& point : *original) {
                if (rounded_coords->at(point) != coord) remaining.push_back(point);
            }
            const auto* usable = remaining.size() > 2
                ? &*memberships.insert(std::move(remaining)).first
                : nullptr;
            members.without_intersection.emplace_back(coord, usable);
        }
        circle_members.push_back(std::move(members));
    }

    auto retained_members = [](const CircleMembers& circle, const Point& intersection) {
        const auto entry = std::find_if(
            circle.without_intersection.begin(), circle.without_intersection.end(),
            [&](const auto& member) { return member.first == intersection; });
        if (entry != circle.without_intersection.end()) return entry->second;
        return circle.original->size() > 2 ? circle.original : nullptr;
    };

    struct Intersection {
        Point coord;
        std::array<const Membership*, 2> first{};
        std::vector<const Membership*> additional;

        void add(const Membership* members) {
            for (auto& entry : first) {
                if (entry == members) return;
                if (!entry) {
                    entry = members;
                    return;
                }
            }
            if (std::find(additional.begin(), additional.end(), members) == additional.end()) {
                additional.push_back(members);
            }
        }
    };
    std::unordered_map<std::string, Intersection> intersections_by_key;
    const bool use_fixed_key = std::fegetround() == FE_TONEAREST &&
        std::string_view(std::localeconv()->decimal_point) == ".";

    for (size_t i = 0; i < circles_data->size(); ++i) {
        for (size_t j = i + 1; j < circles_data->size(); ++j) {
            const auto& [name1, center1, radius1, points1] = (*circles_data)[i];
            const auto& [name2, center2, radius2, points2] = (*circles_data)[j];

            Circle c1 = {center1, radius1};
            Circle c2 = {center2, radius2};

            for (const auto& pt : deduplicate_candidates(intersect_two_circles(c1, c2))) {
                const auto* left = retained_members(circle_members[i], pt);
                const auto* right = retained_members(circle_members[j], pt);
                if (!left && !right) continue;

                std::string point_key;
                // Keep the old key bytes, since hash iteration affects sampling order.
                if (use_fixed_key && std::abs(pt[0]) < 1e6 && std::abs(pt[1]) < 1e6) {
                    char key[40];
                    auto* end = write_rounded_coordinate(key, key + sizeof(key), pt[0]);
                    *end++ = ',';
                    end = write_rounded_coordinate(end, key + sizeof(key), pt[1]);
                    point_key.assign(key, end);
                } else {
                    // A finite double needs at most 309 integral digits in fixed format.
                    char key[2 * (309 + ROUND_DECIMALS + 2) + 1];
                    const auto size = std::snprintf(
                        key, sizeof(key), "%.*f,%.*f", ROUND_DECIMALS, pt[0], ROUND_DECIMALS, pt[1]);
                    if (size >= 0 && static_cast<size_t>(size) < sizeof(key)) {
                        point_key.assign(key, size);
                    } else {
                        std::ostringstream stream;
                        stream << std::fixed << std::setprecision(ROUND_DECIMALS) << pt[0] << "," << pt[1];
                        point_key = stream.str();
                    }
                }
                auto& intersection = intersections_by_key[std::move(point_key)];
                intersection.coord = pt;
                intersection.add(left ? left : circle_members[i].original);
                intersection.add(right ? right : circle_members[j].original);
            }
        }
    }

    // Only keep points where at least 3 circles meet
    for (const auto& entry : intersections_by_key) {
        const auto& [coord, first, additional] = entry.second;
        if (additional.empty()) continue;

        std::vector<const Membership*> circle_list(first.begin(), first.end());
        circle_list.insert(circle_list.end(), additional.begin(), additional.end());
        std::sort(circle_list.begin(), circle_list.end(),
                  [](const auto* a, const auto* b) { return *a < *b; });
        std::sort(circle_list.begin(), circle_list.end(),
                  [](const auto* a, const auto* b) { return (*a)[0] < (*b)[0]; });

        const auto& c1 = *circle_list[0];
        const auto& c2 = *circle_list[1];

        std::vector<std::string> constructions = {
            "on_circum " + c1[0] + " " + c1[1] + " " + c1[2],
            "on_circum " + c2[0] + " " + c2[1] + " " + c2[2]
        };
        result.push_back({coord, constructions});
    }

    return result;
}

PotentialPoints intersection_between_line_and_circle(
    const std::vector<std::string>& point_names,
    const std::unordered_map<std::string, Point>& coords,
    const std::vector<std::pair<std::string, std::vector<std::string>>>* lines_data,
    const std::vector<std::tuple<std::string, Point, double, std::vector<std::string>>>* circles_data,
    const std::unordered_map<std::string, Point>* rounded_coords) {

    PotentialPoints result;
    if (!lines_data || !circles_data) return result;

    // Map: rounded point -> set of (type, sorted_points) objects
    using ObjKey = std::pair<std::string, std::vector<std::string>>;
    std::unordered_map<std::string, std::set<ObjKey>> point_to_objects;
    std::unordered_map<std::string, Point> point_coords_map;

    for (const auto& [line_name, line_points] : *lines_data) {
        if (line_points.size() < 2) continue;

        const Point& A = coords.at(line_points[0]);
        const Point& B = coords.at(line_points[1]);

        for (const auto& [circle_name, center, radius, circle_points] : *circles_data) {
            Circle c = {center, radius};
            auto intersections = intersect_line_circle(A, B, c);

            // Deduplicate candidates
            auto unique = deduplicate_candidates(intersections);

            for (const auto& pt : unique) {
                // Remove points that coincide with intersection
                std::vector<std::string> cleaned_line, cleaned_circle;
                for (const auto& name : line_points) {
                    if ((*rounded_coords).at(name) != pt) {
                        cleaned_line.push_back(name);
                    }
                }
                for (const auto& name : circle_points) {
                    if ((*rounded_coords).at(name) != pt) {
                        cleaned_circle.push_back(name);
                    }
                }

                if (cleaned_line.size() <= 2 && cleaned_circle.size() <= 3) continue;

                std::vector<std::string> use_line = (cleaned_line.size() >= 2) ? cleaned_line : line_points;
                std::vector<std::string> use_circle = (cleaned_circle.size() >= 3) ? cleaned_circle : circle_points;

                std::sort(use_line.begin(), use_line.end());
                std::sort(use_circle.begin(), use_circle.end());

                std::ostringstream oss3;
                oss3 << std::fixed << std::setprecision(9) << pt[0] << "," << pt[1];
                std::string key = oss3.str();
                point_coords_map[key] = pt;
                point_to_objects[key].insert({"line", use_line});
                point_to_objects[key].insert({"circle", use_circle});
            }
        }
    }

    // Only keep points where at least 3 objects meet (at least 1 line and 1 circle)
    for (const auto& [key, obj_set] : point_to_objects) {
        int line_count = 0, circle_count = 0;
        for (const auto& [type, _] : obj_set) {
            if (type == "line") line_count++;
            else circle_count++;
        }

        if ((line_count + circle_count) < 3) continue;
        if (line_count <= 0 || circle_count <= 0) continue;

        // Find smallest line and circle
        std::vector<std::vector<std::string>> line_objs, circle_objs;
        for (const auto& [type, pts] : obj_set) {
            if (type == "line") line_objs.push_back(pts);
            else circle_objs.push_back(pts);
        }

        std::sort(line_objs.begin(), line_objs.end(),
                  [](const auto& a, const auto& b) { return a[0] < b[0]; });
        std::sort(circle_objs.begin(), circle_objs.end(),
                  [](const auto& a, const auto& b) { return a[0] < b[0]; });

        const auto& smallest_line = line_objs[0];
        const auto& smallest_circle = circle_objs[0];

        std::vector<std::string> constructions = {
            "on_circum " + smallest_circle[0] + " " + smallest_circle[1] + " " + smallest_circle[2],
            "on_line " + smallest_line[0] + " " + smallest_line[1]
        };
        result.push_back({point_coords_map[key], constructions});
    }

    return result;
}

PotentialPoints midpoint(
    const std::vector<std::string>& point_names,
    const std::unordered_map<std::string, Point>& coords,
    const std::vector<std::pair<std::string, std::vector<std::string>>>* lines_data,
    const std::vector<std::tuple<std::string, Point, double, std::vector<std::string>>>* circles_data) {

    PotentialPoints result;
    int n = point_names.size();

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const Point& a = get_point(coords, point_names[i]);
            const Point& b = get_point(coords, point_names[j]);
            Point mid = {(a[0] + b[0]) / 2, (a[1] + b[1]) / 2};

            // Require midpoint to be on a non-trivial line or circle
            // (excluding the line formed by the two points themselves)
            bool on_line = false, on_circle = false;
            if (lines_data) {
                on_line = check_on_line_constraint(mid, *lines_data, coords,
                    {{point_names[i], point_names[j]}});
            }
            if (!on_line && circles_data) {
                on_circle = check_on_circle_constraint(mid, *circles_data);
            }

            if (on_line || on_circle) {
                std::vector<std::string> constructions = {
                    "midpoint " + point_names[i] + " " + point_names[j]
                };
                result.push_back({mid, constructions});
            }
        }
    }

    return result;
}

PotentialPoints reflection(
    const std::vector<std::string>& point_names,
    const std::unordered_map<std::string, Point>& coords,
    const std::vector<std::pair<std::string, std::vector<std::string>>>* lines_data,
    const std::vector<std::tuple<std::string, Point, double, std::vector<std::string>>>* circles_data) {

    PotentialPoints result;
    int n = point_names.size();

    // Reflection over points (mirror)
    for (int i = 0; i < n; ++i) {
        const std::string& center_name = point_names[i];
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            const std::string& pt_name = point_names[j];
            const Point& center = coords.at(center_name);
            const Point& pt = coords.at(pt_name);
            Point refl = {2 * center[0] - pt[0], 2 * center[1] - pt[1]};

            bool on_line = false, on_circle = false;
            if (lines_data) {
                on_line = check_on_line_constraint(refl, *lines_data, coords,
                    {{center_name, pt_name}});
            }
            if (!on_line && circles_data) {
                on_circle = check_on_circle_constraint(refl, *circles_data);
            }

            if (on_line || on_circle) {
                std::vector<std::string> constructions = {
                    "mirror " + pt_name + " " + center_name
                };
                result.push_back({refl, constructions});
            }
        }
    }

    // Reflection over lines (reflect)
    if (!lines_data || lines_data->empty()) return result;

    for (const auto& [line_name, line_points] : *lines_data) {
        if (line_points.size() < 2) continue;

        std::vector<std::string> line_sorted = line_points;
        std::sort(line_sorted.begin(), line_sorted.end());
        std::set<std::string> line_set(line_sorted.begin(), line_sorted.end());

        const Point& a = get_point(coords, line_sorted[0]);
        const Point& b = get_point(coords, line_sorted[1]);
        double dx = b[0] - a[0];
        double dy = b[1] - a[1];
        double len_sq = dx * dx + dy * dy;

        if (std::abs(len_sq) < TOLERANCE) continue;

        for (int i = 0; i < n; ++i) {
            const std::string& pt_name = point_names[i];
            // Skip points on the line
            if (line_set.count(pt_name)) continue;

            const Point& p = coords.at(pt_name);
            double t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq;
            Point foot_pt = {a[0] + t * dx, a[1] + t * dy};
            Point refl = {2 * foot_pt[0] - p[0], 2 * foot_pt[1] - p[1]};

            bool on_line = false, on_circle = false;
            if (lines_data) {
                on_line = check_on_line_constraint(refl, *lines_data, coords);
            }
            if (!on_line && circles_data) {
                on_circle = check_on_circle_constraint(refl, *circles_data);
            }

            if (on_line || on_circle) {
                std::vector<std::string> constructions = {
                    "reflect " + pt_name + " " + line_sorted[0] + " " + line_sorted[1]
                };
                result.push_back({refl, constructions});
            }
        }
    }

    return result;
}

PotentialPoints foot(
    const std::vector<std::string>& point_names,
    const std::unordered_map<std::string, Point>& coords,
    const std::vector<std::pair<std::string, std::vector<std::string>>>* lines_data) {

    PotentialPoints result;
    if (!lines_data || lines_data->empty()) return result;

    int n = point_names.size();

    for (const auto& [line_name, line_points] : *lines_data) {
        if (line_points.size() < 2) continue;

        std::vector<std::string> line_sorted = line_points;
        std::sort(line_sorted.begin(), line_sorted.end());
        std::set<std::string> line_set(line_sorted.begin(), line_sorted.end());

        const Point& a = get_point(coords, line_sorted[0]);
        const Point& b = get_point(coords, line_sorted[1]);
        double dx = b[0] - a[0];
        double dy = b[1] - a[1];
        double len_sq = dx * dx + dy * dy;

        if (std::abs(len_sq) < TOLERANCE) continue;

        for (int i = 0; i < n; ++i) {
            const std::string& pt_name = point_names[i];
            // Skip points on the line
            if (line_set.count(pt_name)) continue;

            const Point& p = coords.at(pt_name);
            double t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq;
            Point foot_pt = {a[0] + t * dx, a[1] + t * dy};

            // Require foot to be on a non-trivial line
            if (check_on_line_constraint(foot_pt, *lines_data, coords,
                {{pt_name}, line_sorted})) {
                std::vector<std::string> constructions = {
                    "foot " + pt_name + " " + line_sorted[0] + " " + line_sorted[1]
                };
                result.push_back({foot_pt, constructions});
            }
        }
    }

    return result;
}

PotentialPoints add_potential_points(
    const std::vector<std::string>& point_names,
    const std::unordered_map<std::string, Point>& coords,
    int max_points,
    uint32_t seed) {

    PotentialPoints result;
    std::vector<Point> existing_coords;
    for (const auto& name : point_names) {
        existing_coords.push_back(coords.at(name));
    }

    // Step 1: Randomly select construction types (matching Python behavior)
    std::mt19937 gen(seed);

    std::vector<int> all_types = {0, 1, 2, 3, 4, 5};
    std::shuffle(all_types.begin(), all_types.end(), gen);

    int max_type_count = std::max(1, std::min(max_points, (int)all_types.size()));
    std::uniform_int_distribution<> count_dist(1, max_type_count);
    int type_count = count_dist(gen);
    std::vector<int> selected_types(all_types.begin(), all_types.begin() + type_count);

    // Step 2: Precompute lines and circles if needed
    std::vector<std::pair<std::string, std::vector<std::string>>> lines_data;
    std::vector<std::tuple<std::string, Point, double, std::vector<std::string>>> circles_data;
    std::unordered_map<std::string, Point> rounded_coords;

    bool needs_lines = false, needs_circles = false, needs_rounded = false;
    for (int t : selected_types) {
        if (t == 0 || t == 2 || t == 3 || t == 4 || t == 5) needs_lines = true;
        if (t == 1 || t == 2 || t == 3 || t == 4) needs_circles = true;
        if (t == 0 || t == 1 || t == 2) needs_rounded = true;
    }

    if (needs_lines) lines_data = lines(point_names, coords);
    if (needs_circles) circles_data = circles(point_names, coords);
    if (needs_rounded) {
        for (const auto& name : point_names) {
            rounded_coords[name] = round_coord(coords.at(name));
        }
    }

    // Step 3: Compute potential points for each selected type
    std::map<int, PotentialPoints> type_to_points;
    for (int t : selected_types) {
        if (t == 0) {
            type_to_points[t] = intersection_between_lines(point_names, coords, &lines_data, &rounded_coords);
        } else if (t == 1) {
            type_to_points[t] = intersection_between_circles(point_names, coords, &circles_data, &rounded_coords);
        } else if (t == 2) {
            type_to_points[t] = intersection_between_line_and_circle(point_names, coords, &lines_data, &circles_data, &rounded_coords);
        } else if (t == 3) {
            type_to_points[t] = midpoint(point_names, coords, &lines_data, &circles_data);
        } else if (t == 4) {
            type_to_points[t] = reflection(point_names, coords, &lines_data, &circles_data);
        } else if (t == 5) {
            type_to_points[t] = foot(point_names, coords, &lines_data);
        }
    }

    // Return empty if no potential points found
    bool all_empty = true;
    for (const auto& [_, pts] : type_to_points) {
        if (!pts.empty()) { all_empty = false; break; }
    }
    if (all_empty) return result;

    // Step 4: Randomly select points with distance filtering
    std::uniform_int_distribution<> type_selector(0, selected_types.size() - 1);
    int max_trials = max_points * 20;
    int trials = 0;

    while ((int)result.size() < max_points && trials < max_trials) {
        trials++;
        int t_idx = type_selector(gen);
        int t = selected_types[t_idx];
        auto it = type_to_points.find(t);
        if (it == type_to_points.end() || it->second.empty()) continue;

        auto& candidates = it->second;
        std::uniform_int_distribution<> item_selector(0, (int)candidates.size() - 1);
        int item_idx = item_selector(gen);

        auto [coord, constructions] = candidates[item_idx];

        // Distance filtering
        if (is_point_too_close({coord}, existing_coords) ||
            is_point_too_far({coord}, existing_coords)) {
            // Remove unsuitable point from candidates
            candidates.erase(candidates.begin() + item_idx);
            continue;
        }

        // Add the point
        existing_coords.push_back(coord);
        result.push_back({coord, constructions});

        // Remove used point from candidates
        candidates.erase(candidates.begin() + item_idx);
    }

    return result;
}

} // namespace auxiliary_c
