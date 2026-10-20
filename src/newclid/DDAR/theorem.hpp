#ifndef THEOREM_HPP
#define THEOREM_HPP

#include "predicate/circumcenter.hpp"
#include "predicate/coll.hpp"
#include "predicate/cyclic.hpp"
#include "predicate/midpoint.hpp"
#include "predicate/ncoll.hpp"
#include "predicate/orthocenter.hpp"
#include "predicate/pappus.hpp"
#include "predicate/similar_triangles.hpp"
#include "predicate/congruent_triangles.hpp"
#include "predicate/statement.hpp"
#include "predicate/eqpoint.hpp"
#include "predicate/thales.hpp"
#include "predicate/secant.hpp"
#include "type/angle.hpp"
#include <vector>

class Theorem
{
public:
    Theorem(const Theorem &other) = delete;
    Theorem(Theorem &&other) noexcept = default;
    Theorem() = default;
    Theorem(std::string name, std::string rule) : _name(name), _rule(rule) {}

    const std::vector<std::unique_ptr<Statement>> &hypotheses() const { return _hypotheses; }
    const std::vector<std::unique_ptr<Statement>> &conclusions() const { return _conclusions; }
    const std::string &name() const { return _name; }
    const std::string &rule() const { return _rule; }

    bool check_hypotheses_numerically() const;
    bool check_conclusions_numerically() const;

    bool check_numerically() const;

    void print() const;

    Theorem &add_hypothesis(std::unique_ptr<Statement> stmt);
    Theorem &add_conclusion(std::unique_ptr<Statement> stmt);

    Theorem converse(const std::string &name, const std::string &rule) const;
    Theorem normalize() const;
    Theorem clone() const;

    Point max_point() const;

    // r03
    static Theorem cyclic_properties(const Cyclic &p);

    // r04
    static Theorem cyclic_of_equal_angles(const Cyclic &p);

    // r07
    static Theorem thales_eqratio_of_para_with_common_point(const Coll &left, const Coll &right);

    // r11
    static Theorem triangle_bisector_of_eqratio(const Point &point, const Angle &angle);

    // r12
    static Theorem triangle_bisector_of_equal_angles(const Point &point, const Angle &angle);

    // r19
    static Theorem hypotenuse_is_diameter(const Midp &p, const Point &pt);

    // r27
    static Theorem thales_para_of_eqratio_with_common_point(const Coll &left, const Coll &right);

    // r28
    static Theorem coll_of_para(const Coll &c);

    // ratio segment-to-whole (collinear): AB/BC = DE/EF → AB/AC = DE/DF
    static Theorem ratio_segment_to_whole(const Coll &left, const Coll &right);

    // r82
    static Theorem para_of_coll(const Coll &c);

    // r34, r35
    static Theorem similar_triangles_of_aa(const SimilarTriangles &p);

    // r41
    static Theorem thales_para_of_eqratio(const Thales &p);

    // r42
    static Theorem thales_eqratio_of_para(const Thales &p);

    // r43
    static Theorem orthocenter(const OrthoCenter &p);

    // r44
    static Theorem pappus(const Pappus &p);

    // r46
    static Theorem incenter(const Point &point, const Angle &angle);

    // r49
    static Theorem cong_of_circumcenter_of_cyclic(const CircumCenter &p, const Point &pt);

    // r50
    static Theorem center_of_cyclic_of_cong_of_cong(const Cyclic &p, const Point &pt);

    // r51
    static Theorem midpoint_ratio_dist(const Midp &p);

    // r52, r53
    static Theorem similar_triangles_properties(const SimilarTriangles &p);

    // r54
    static Theorem midpoint_of_coll_cong(const Midp &p);

    // r56
    static Theorem coll_cong_of_midpoint(const Midp &p);

    // TODO: r57

    // TODO: r58

    // TODO: r59

    // r60, r61
    static Theorem similar_triangles_of_sss(const SimilarTriangles &p);

    // r62, r63
    static Theorem similar_triangles_of_sas(const SimilarTriangles &p);

    // r72
    static Theorem cong_of_circumcenter(const CircumCenter &p);

    // r73
    static Theorem circumcenter_of_cong(const CircumCenter &p);

    // r101, r102
    static Theorem congruent_triangles_of_cong(const CongruentTriangles &p);

    // r103, r104
    static Theorem congruent_triangles_properties(const CongruentTriangles &p);

    // Congruent triangles from SSS (Side-Side-Side)
    static Theorem congruent_triangles_of_sss(const CongruentTriangles &p);

    // Congruent triangles from SAS (Side-Angle-Side)
    static Theorem congruent_triangles_of_sas(const CongruentTriangles &p);

    // r105
    static Theorem eqratio_of_coll(const Coll &left, const Coll &right);

    // r106
    static Theorem definition_of_secant(const Secant &sec);

    // r107
    static Theorem eqpoints_of_same_intersections(const Point &p, const Point &q, const Point &a, const Point &b, const Point &c, const Point &d);

    // r108
    static Theorem cong_of_eqpoints(const EqPoint &eq, const Point &p);

    // r109
    static Theorem eqpoints_of_same_ratio_on_line(const Point &p1, const Point &p2, const Point &a, const Point &c);

    // r110
    static Theorem cong_of_eqangle(const Point &vertex, const Point &a, const Point &b);

    // r111
    static Theorem eqangle_of_cong(const Point &vertex, const Point &a, const Point &b);

private:
    std::string _name;
    std::string _rule;
    std::vector<std::unique_ptr<Statement>> _hypotheses;
    std::vector<std::unique_ptr<Statement>> _conclusions;
};

#endif // THEOREM_HPP