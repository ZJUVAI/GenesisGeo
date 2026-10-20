#include "theorem.hpp"
#include "predicate/circumcenter.hpp"
#include "predicate/coll.hpp"
#include "predicate/cong.hpp"
#include "predicate/cyclic.hpp"
#include "predicate/const_line.hpp"
#include "predicate/eqangle.hpp"
#include "predicate/eqratio.hpp"
#include "predicate/eqpoint.hpp"
#include "predicate/midpoint.hpp"
#include "predicate/ncoll.hpp"
#include "predicate/npara.hpp"
#include "predicate/orthocenter.hpp"
#include "predicate/pappus.hpp"
#include "predicate/para.hpp"
#include "predicate/perp.hpp"
#include "predicate/rconst.hpp"
#include "predicate/sameclock.hpp"
#include "predicate/sameside.hpp"
#include "predicate/similar_triangles.hpp"
#include "predicate/congruent_triangles.hpp"
#include "predicate/statement.hpp"
#include "predicate/thales.hpp"
#include "type/angle.hpp"
#include "type/dist.hpp"
#include "type/rational.hpp"
#include <vector>

using namespace std;

bool Theorem::check_hypotheses_numerically() const
{
    for (const auto &stmt : _hypotheses)
    {
        if (!stmt->check_numerically())
        {
            return false;
        }
    }
    return true;
}

bool Theorem::check_conclusions_numerically() const
{
    for (const auto &stmt : _conclusions)
    {
        if (!stmt->check_numerically())
        {
            return false;
        }
    }
    return true;
}

bool Theorem::check_numerically() const
{
    return check_hypotheses_numerically() && check_conclusions_numerically();
}

void Theorem::print() const
{
    cout << _name << endl;
    cout << "Hypotheses: " << endl;
    for (const auto &stmt : _hypotheses)
    {
        cout << *stmt << endl;
    }
    cout << "Conclusions: " << endl;
    for (const auto &stmt : _conclusions)
    {
        cout << *stmt << endl;
    }
}

Theorem &Theorem::add_hypothesis(unique_ptr<Statement> stmt)
{
    _hypotheses.emplace_back(move(stmt));
    return *this;
}

Theorem &Theorem::add_conclusion(unique_ptr<Statement> stmt)
{
    _conclusions.emplace_back(move(stmt));
    return *this;
}

Theorem Theorem::converse(const string &name, const string &rule) const
{
    Theorem theorem(name, rule);
    for (const auto &stmt : _hypotheses)
    {
        theorem.add_conclusion(stmt->clone());
    }
    for (const auto &stmt : _conclusions)
    {
        theorem.add_hypothesis(stmt->clone());
    }
    return theorem;
}

Theorem Theorem::normalize() const
{
    Theorem thm(_name, _rule);
    for (const auto &stmt : _hypotheses)
    {
        thm.add_hypothesis(stmt->normalize());
    }
    for (const auto &stmt : _conclusions)
    {
        thm.add_conclusion(stmt->normalize());
    }
    return thm;
}

Point Theorem::max_point() const
{
    Point max_pt = _hypotheses[0]->points()[0];
    for (const auto &stmt : _hypotheses)
    {
        for (const auto &pt : stmt->points())
        {
            if (pt > max_pt)
            {
                max_pt = pt;
            }
        }
    }

    for (const auto &stmt : _conclusions)
    {
        for (const auto &pt : stmt->points())
        {
            if (pt > max_pt)
            {
                max_pt = pt;
            }
        }
    }

    return max_pt;
}

Theorem Theorem::cyclic_properties(const Cyclic &p)
{
    Theorem theorem("Arc determines internal angles", "r03");
    theorem.add_hypothesis(p.clone());
    theorem.add_conclusion(p.eqangles_cad_cbd().clone());
    theorem.add_conclusion(p.eqangles_bad_bcd().clone());
    theorem.add_conclusion(p.eqangles_abd_acd().clone());
    return theorem;
}

Theorem Theorem::cyclic_of_equal_angles(const Cyclic &p)
{
    Theorem theorem("Congruent angles are in a circle", "r04");
    theorem.add_conclusion(p.clone());
    theorem.add_hypothesis(make_unique<NColl>(NColl(p.a(), p.c(), p.d())));
    theorem.add_hypothesis(p.eqangles_cad_cbd().clone());
    return theorem;
}

Theorem Theorem::thales_eqratio_of_para_with_common_point(const Coll &left, const Coll &right)
{
    Theorem theorem("Thales theorem I", "r07");
    theorem.add_hypothesis(make_unique<Para>(Para(Slope(left.b(), right.b()), Slope(left.c(), right.c()))));
    theorem.add_hypothesis(left.clone());
    theorem.add_hypothesis(right.clone());
    theorem.add_hypothesis(make_unique<NColl>(NColl(left.b(), left.c(), right.b())));
    theorem.add_conclusion(make_unique<EqRatio>(EqRatio(Dist(left.a(), left.b()), Dist(left.a(), left.c()), Dist(right.a(), right.b()), Dist(right.a(), right.c()))));
    theorem.add_conclusion(make_unique<EqRatio>(EqRatio(Dist(left.a(), left.b()), Dist(left.a(), left.c()), Dist(left.b(), right.b()), Dist(left.c(), right.c()))));
    return theorem;
}

Theorem Theorem::triangle_bisector_of_eqratio(const Point &point, const Angle &angle)
{
    Theorem theorem("Bisector theorem I", "r11");
    theorem.add_hypothesis(make_unique<NColl>(NColl(angle.left(), angle.vertex(), angle.right())));
    theorem.add_hypothesis(make_unique<Coll>(Coll(angle.left(), point, angle.right())));
    theorem.add_hypothesis(make_unique<EqRatio>(EqRatio(Dist(point, angle.left()),
                                                        Dist(point, angle.right()),
                                                        Dist(angle.vertex(), angle.left()),
                                                        Dist(angle.vertex(), angle.right()))));
    theorem.add_conclusion(make_unique<EqAngle>(EqAngle(Angle(angle.left(), angle.vertex(), point), Angle(point, angle.vertex(), angle.right()))));
    return theorem;
}

Theorem Theorem::triangle_bisector_of_equal_angles(const Point &point, const Angle &angle)
{
    Theorem theorem("Bisector theorem II", "r12");
    theorem.add_hypothesis(make_unique<EqAngle>(EqAngle(Angle(angle.left(), angle.vertex(), point),
                                                        Angle(point, angle.vertex(), angle.right()))));
    theorem.add_hypothesis(make_unique<NColl>(NColl(angle.left(), angle.vertex(), angle.right())));
    theorem.add_hypothesis(make_unique<Coll>(Coll(angle.left(), point, angle.right())));
    theorem.add_conclusion(make_unique<EqRatio>(EqRatio(Dist(point, angle.left()),
                                                        Dist(point, angle.right()),
                                                        Dist(angle.vertex(), angle.left()),
                                                        Dist(angle.vertex(), angle.right()))));
    return theorem;
}

Theorem Theorem::hypotenuse_is_diameter(const Midp &p, const Point &pt)
{
    Theorem theorem("Hypotenuse is diameter", "r19");
    theorem.add_hypothesis(make_unique<Perp>(Perp(Slope(p.left(), pt), Slope(p.right(), pt))));
    theorem.add_hypothesis(p.clone());
    theorem.add_conclusion(make_unique<Cong>(Cong(Dist(p.left(), p.middle()), Dist(pt, p.middle()))));
    theorem.add_conclusion(make_unique<Cong>(Cong(Dist(p.right(), p.middle()), Dist(pt, p.middle()))));
    return theorem;
}

Theorem Theorem::thales_para_of_eqratio_with_common_point(const Coll &left, const Coll &right)
{
    Theorem theorem("Thales theorem II", "r27");
    theorem.add_hypothesis(make_unique<EqRatio>(EqRatio(Dist(left.a(), left.b()), Dist(left.c(), left.b()), Dist(right.a(), right.b()), Dist(right.c(), right.b()))));
    theorem.add_hypothesis(left.clone());
    theorem.add_hypothesis(right.clone());
    theorem.add_hypothesis(make_unique<NColl>(NColl(left.b(), left.c(), right.b())));
    theorem.add_hypothesis(make_unique<SameSide>(SameSide(left.a(), left.b(), left.c(), right.a(), right.b(), right.c())));
    theorem.add_conclusion(make_unique<Para>(Para(Slope(left.b(), right.b()), Slope(left.c(), right.c()))));
    return theorem;
}

Theorem Theorem::coll_of_para(const Coll &c)
{
    Theorem theorem("Overlapping parallels", "r28");
    theorem.add_hypothesis(make_unique<Para>(Para(Slope(c.a(), c.b()), Slope(c.b(), c.c()))));
    theorem.add_conclusion(c.clone());
    return theorem;
}

Theorem Theorem::para_of_coll(const Coll &c)
{
    Theorem theorem("Overlapping parallels", "r82");
    theorem.add_hypothesis(c.clone());
    theorem.add_conclusion(make_unique<Para>(Para(Slope(c.a(), c.b()), Slope(c.b(), c.c()))));
    theorem.add_conclusion(make_unique<Para>(Para(Slope(c.a(), c.b()), Slope(c.a(), c.c()))));
    return theorem;
}

Theorem Theorem::similar_triangles_of_aa(const SimilarTriangles &p)
{
    Theorem theorem("AA Similarity of triangles", p.sameclock() ? "r34" : "r35");
    theorem.add_hypothesis(p.eqangle_abc().clone());
    theorem.add_hypothesis(p.eqangle_bca().clone());
    theorem.add_hypothesis(p.to_sameclock().clone());
    theorem.add_conclusion(p.clone());
    return theorem;
}

Theorem Theorem::thales_para_of_eqratio(const Thales &p)
{
    Theorem theorem("Thales theorem III", "r41");
    theorem.add_hypothesis(p.left().clone());
    theorem.add_hypothesis(p.right().clone());
    theorem.add_hypothesis(p.para_bc().clone());
    theorem.add_hypothesis(p.left().eqratio_ab_ac(p.right()).clone());
    theorem.add_hypothesis(make_unique<SameSide>(SameSide(p.left(), p.right())));
    theorem.add_conclusion(p.para_ab().clone());
    return theorem;
}

Theorem Theorem::thales_eqratio_of_para(const Thales &p)
{
    Theorem theorem("Thales Theorem IV", "r42");
    theorem.add_hypothesis(p.left().clone());
    theorem.add_hypothesis(p.right().clone());
    theorem.add_hypothesis(p.para_ab().clone());
    theorem.add_hypothesis(p.para_bc().clone());
    theorem.add_hypothesis(make_unique<NColl>(NColl(p.left().a(), p.right().a(), p.left().b())));
    theorem.add_conclusion(p.left().eqratio_ab_bc(p.right()).clone());
    theorem.add_conclusion(p.left().eqratio_ab_ac(p.right()).clone());
    return theorem;
}

Theorem Theorem::orthocenter(const OrthoCenter &p)
{
    Theorem theorem("Orthocenter theorem", "r43");
    theorem.add_hypothesis(p.perp_a().clone());
    theorem.add_hypothesis(p.perp_b().clone());
    theorem.add_conclusion(p.perp_c().clone());
    return theorem;
}

Theorem Theorem::pappus(const Pappus &p)
{
    Theorem theorem("Pappus theorem", "r44");
    theorem.add_hypothesis(p.coll_ab().clone());
    theorem.add_hypothesis(p.coll_ba().clone());
    theorem.add_hypothesis(p.coll_bc().clone());
    theorem.add_hypothesis(p.coll_cb().clone());
    theorem.add_hypothesis(p.coll_ca().clone());
    theorem.add_hypothesis(p.coll_ac().clone());
    theorem.add_hypothesis(p.left().clone());
    theorem.add_hypothesis(p.right().clone());
    theorem.add_conclusion(p.middle().clone());
    return theorem;
}

Theorem Theorem::incenter(const Point &point, const Angle &angle)
{
    Theorem theorem("Incenter theorem", "r46");
    theorem.add_hypothesis(make_unique<EqAngle>(EqAngle(Angle(angle.vertex(), angle.left(), point),
                                                        Angle(point, angle.left(), angle.right()))));
    theorem.add_hypothesis(make_unique<EqAngle>(EqAngle(Angle(angle.left(), angle.right(), point),
                                                        Angle(point, angle.right(), angle.vertex()))));
    theorem.add_hypothesis(make_unique<NColl>(NColl(angle.left(), angle.vertex(), angle.right())));
    theorem.add_conclusion(make_unique<EqAngle>(EqAngle(Angle(angle.left(), angle.vertex(), point),
                                                        Angle(point, angle.vertex(), angle.right()))));
    return theorem;
}

Theorem Theorem::cong_of_circumcenter_of_cyclic(const CircumCenter &p, const Point &pt)
{
    Theorem theorem("Recognize center of cyclic", "r49");
    theorem.add_hypothesis(p.clone());
    theorem.add_hypothesis(Cyclic(pt, p.a(), p.b(), p.c()).clone());
    theorem.add_conclusion(Cong(Dist(p.center(), p.a()), Dist(p.center(), pt)).clone());
    return theorem;
}

Theorem Theorem::center_of_cyclic_of_cong_of_cong(const Cyclic &p, const Point &pt)
{
    Theorem theorem("Recognize center of cyclic", "r50");
    theorem.add_hypothesis(p.clone());
    theorem.add_hypothesis(Cong(Dist(pt, p.a()), Dist(pt, p.b())).clone());
    theorem.add_hypothesis(Cong(Dist(pt, p.c()), Dist(pt, p.d())).clone());
    theorem.add_hypothesis(NPara(Slope(p.a(), p.b()),
                                 Slope(p.c(), p.d()))
                               .clone());
    theorem.add_conclusion(Cong(Dist(pt, p.a()), Dist(pt, p.c())).clone());
    return theorem;
}

Theorem Theorem::midpoint_ratio_dist(const Midp &p)
{
    Theorem theorem("Midp splits in two", "r51");
    theorem.add_hypothesis(p.clone());
    theorem.add_conclusion(RConst(Dist(p.left(), p.middle()),
                                  Dist(p.left(), p.right()),
                                  Rational(0.5))
                               .clone());
    theorem.add_conclusion(RConst(Dist(p.right(), p.middle()),
                                  Dist(p.left(), p.right()),
                                  Rational(0.5))
                               .clone());
    return theorem;
}

Theorem Theorem::similar_triangles_properties(const SimilarTriangles &p)
{
    Theorem theorem("Properties of similar triangles", p.sameclock() ? "r52" : "r53");
    theorem.add_hypothesis(p.to_sameclock().clone());
    theorem.add_hypothesis(p.clone());
    theorem.add_conclusion(p.eqratio_abbc().clone());
    theorem.add_conclusion(p.eqratio_abac().clone());
    theorem.add_conclusion(p.eqratio_bcac().clone());
    theorem.add_conclusion(p.eqangle_abc().clone());
    theorem.add_conclusion(p.eqangle_bca().clone());
    return theorem;
}

Theorem Theorem::midpoint_of_coll_cong(const Midp &p)
{
    Theorem theorem("Definition of midpoint", "r54");
    theorem.add_hypothesis(p.to_coll().clone());
    theorem.add_hypothesis(p.to_cong().clone());
    theorem.add_conclusion(p.clone());
    return theorem;
}

Theorem Theorem::coll_cong_of_midpoint(const Midp &p)
{
    Theorem theorem("Properties of midpoint (coll, cong)", "r56");
    theorem.add_hypothesis(p.clone());
    theorem.add_conclusion(p.to_coll().clone());
    theorem.add_conclusion(p.to_cong().clone());
    return theorem;
}

Theorem Theorem::similar_triangles_of_sss(const SimilarTriangles &p)
{
    Theorem theorem("SSS Similarity of triangles", p.sameclock() ? "r60" : "r61");
    theorem.add_hypothesis(p.eqratio_abbc().clone());
    theorem.add_hypothesis(p.eqratio_abac().clone());
    theorem.add_hypothesis(p.to_sameclock().clone());
    theorem.add_conclusion(p.clone());
    return theorem;
}

Theorem Theorem::similar_triangles_of_sas(const SimilarTriangles &p)
{
    Theorem theorem("SAS Similarity of triangles", p.sameclock() ? "r62" : "r63");
    theorem.add_hypothesis(p.eqratio_abbc().clone());
    theorem.add_hypothesis(p.eqangle_abc().clone());
    theorem.add_hypothesis(p.to_sameclock().clone());
    theorem.add_conclusion(p.clone());
    return theorem;
}

Theorem Theorem::cong_of_circumcenter(const CircumCenter &p)
{
    Theorem theorem("Congruence of circumcenter", "r72");
    theorem.add_hypothesis(p.cong_ab().clone());
    theorem.add_hypothesis(p.cong_ac().clone());
    theorem.add_conclusion(p.clone());
    return theorem;
}

Theorem Theorem::circumcenter_of_cong(const CircumCenter &p)
{
    Theorem theorem("Circumcenter of congruent triangles", "r73");
    theorem.add_hypothesis(p.clone());
    theorem.add_conclusion(p.cong_ab().clone());
    theorem.add_conclusion(p.cong_ac().clone());
    theorem.add_conclusion(p.cong_bc().clone());
    return theorem;
}

Theorem Theorem::congruent_triangles_of_cong(const CongruentTriangles &p)
{
    Theorem theorem("Congruent triangles from similarity triangles", p.sameclock() ? "r101" : "r102");
    theorem.add_hypothesis(make_unique<SimilarTriangles>(SimilarTriangles(p.left(), p.right(), p.sameclock())));
    theorem.add_hypothesis(p.cong_ab().clone());
    theorem.add_conclusion(p.clone());
    return theorem;
}

Theorem Theorem::congruent_triangles_properties(const CongruentTriangles &p)
{
    Theorem theorem("Properties of congruent triangles", p.sameclock() ? "r103" : "r104");
    theorem.add_hypothesis(p.clone());
    theorem.add_conclusion(p.cong_ab().clone());
    theorem.add_conclusion(p.cong_ac().clone());
    theorem.add_conclusion(p.cong_bc().clone());
    theorem.add_conclusion(p.eqangle_abc().clone());
    theorem.add_conclusion(p.eqangle_bca().clone());
    theorem.add_conclusion(p.eqangle_cab().clone());
    return theorem;
}

Theorem Theorem::congruent_triangles_of_sss(const CongruentTriangles &p)
{
    Theorem theorem("SSS Congruence of triangles", p.sameclock() ? "r110" : "r111");
    theorem.add_hypothesis(p.cong_ab().clone());
    theorem.add_hypothesis(p.cong_ac().clone());
    theorem.add_hypothesis(p.cong_bc().clone());
    theorem.add_hypothesis(p.to_sameclock().clone());
    theorem.add_conclusion(p.clone());
    return theorem;
}

Theorem Theorem::congruent_triangles_of_sas(const CongruentTriangles &p)
{
    Theorem theorem("SAS Congruence of triangles", p.sameclock() ? "r112" : "r113");
    theorem.add_hypothesis(p.cong_ab().clone());
    theorem.add_hypothesis(p.cong_bc().clone());
    theorem.add_hypothesis(p.eqangle_abc().clone());
    theorem.add_hypothesis(p.to_sameclock().clone());
    theorem.add_conclusion(p.clone());
    return theorem;
}

Theorem Theorem::eqratio_of_coll(const Coll &left, const Coll &right)
{
    Theorem theorem("Ratio of collinear points", "r105");
    theorem.add_hypothesis(make_unique<EqRatio>(EqRatio(Dist(left.a(), left.b()), Dist(left.a(), left.c()), Dist(right.a(), right.b()), Dist(right.a(), right.c()))));
    theorem.add_hypothesis(left.clone());
    theorem.add_hypothesis(right.clone());
    theorem.add_conclusion(make_unique<EqRatio>(EqRatio(Dist(left.a(), left.b()), Dist(left.b(), left.c()), Dist(right.a(), right.b()), Dist(right.b(), right.c()))));
    theorem.add_conclusion(make_unique<EqRatio>(EqRatio(Dist(left.a(), left.c()), Dist(left.b(), left.c()), Dist(right.a(), right.c()), Dist(right.b(), right.c()))));
    return theorem;
}

Theorem Theorem::definition_of_secant(const Secant &sec)
{
    Theorem theorem("Definition of secant", "r106");
    theorem.add_hypothesis(sec.coll_pab().clone());
    theorem.add_hypothesis(sec.cong_ab().clone());
    theorem.add_conclusion(sec.clone());
    return theorem;
}

Theorem Theorem::eqpoints_of_same_intersections(const Point &p, const Point &q, const Point &a, const Point &b, const Point &c, const Point &d)
{
    Theorem theorem("Two distinct lines determine a unique point", "r107");
    if (p == a || q == a)
    {
        theorem.add_hypothesis(ConstLine(b, p, q).clone());
    }
    else
    {
        theorem.add_hypothesis(Coll(p, a, b).clone());
        theorem.add_hypothesis(Coll(q, a, b).clone());
    }
    if (p == c || q == c)
    {
        theorem.add_hypothesis(ConstLine(d, p, q).clone());
    }
    else
    {
        theorem.add_hypothesis(Coll(p, c, d).clone());
        theorem.add_hypothesis(Coll(q, c, d).clone());
    }
    theorem.add_hypothesis(NColl(a, b, d).clone());
    theorem.add_conclusion(EqPoint(p, q).clone());
    return theorem;
}

Theorem Theorem::cong_of_eqpoints(const EqPoint &eq, const Point &p)
{
    Theorem theorem("Distances to overlapped point are equal", "r108");
    theorem.add_hypothesis(eq.clone());
    theorem.add_conclusion(make_unique<Cong>(Cong(Dist(p, eq.a()), Dist(p, eq.b()))));
    theorem.add_conclusion(make_unique<ConstLine>(ConstLine(p, eq.a(), eq.b())));
    return theorem;
}

Theorem Theorem::eqpoints_of_same_ratio_on_line(const Point &p1, const Point &p2, const Point &a, const Point &c)
{
    Theorem theorem("Same ratio on same line implies equal points", "r109");
    theorem.add_hypothesis(make_unique<Coll>(Coll(a, p1, c)));
    theorem.add_hypothesis(make_unique<Coll>(Coll(a, p2, c)));
    theorem.add_hypothesis(make_unique<Cong>(Cong(Dist(a, p1), Dist(a, p2))));
    theorem.add_hypothesis(make_unique<Cong>(Cong(Dist(c, p1), Dist(c, p2))));
    theorem.add_conclusion(make_unique<EqPoint>(EqPoint(p1, p2)));
    return theorem;
}

Theorem Theorem::cong_of_eqangle(const Point &vertex, const Point &a, const Point &b)
{
    Theorem theorem("equal angles imply equal opposite sides", "r110");
    theorem.add_hypothesis(make_unique<EqAngle>(EqAngle(Angle(vertex, a, b), Angle(a, b, vertex))));
    theorem.add_conclusion(make_unique<Cong>(Cong(Dist(vertex, a), Dist(vertex, b))));
    return theorem;
}

Theorem Theorem::eqangle_of_cong(const Point &vertex, const Point &a, const Point &b)
{
    Theorem theorem("equal sides imply equal opposite angles", "r111");
    theorem.add_hypothesis(make_unique<Cong>(Cong(Dist(vertex, a), Dist(vertex, b))));
    theorem.add_conclusion(make_unique<EqAngle>(EqAngle(Angle(vertex, a, b), Angle(a, b, vertex))));
    return theorem;
}

Theorem Theorem::ratio_segment_to_whole(const Coll &left, const Coll &right)
{
    // coll A B C, coll D E F, eqratio A B B C D E E F
    // => eqratio A B A C D E D F
    Theorem theorem("Ratio segment to whole (collinear)", "r29");
    theorem.add_hypothesis(left.clone());
    theorem.add_hypothesis(right.clone());
    theorem.add_hypothesis(make_unique<EqRatio>(EqRatio(
        Dist(left.a(), left.b()), Dist(left.b(), left.c()),
        Dist(right.a(), right.b()), Dist(right.b(), right.c()))));
    theorem.add_conclusion(make_unique<EqRatio>(EqRatio(
        Dist(left.a(), left.b()), Dist(left.a(), left.c()),
        Dist(right.a(), right.b()), Dist(right.a(), right.c()))));
    return theorem;
}

Theorem Theorem::clone() const
{
    Theorem thm(_name, _rule);

    thm._hypotheses.clear();
    transform(_hypotheses.begin(), _hypotheses.end(), back_inserter(thm._hypotheses),
              [](const unique_ptr<Statement> &stmt)
              { return stmt->clone(); });

    thm._conclusions.clear();
    transform(_conclusions.begin(), _conclusions.end(), back_inserter(thm._conclusions),
              [](const unique_ptr<Statement> &stmt)
              { return stmt->clone(); });

    return thm;
}