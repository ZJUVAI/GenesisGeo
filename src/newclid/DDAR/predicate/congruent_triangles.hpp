#ifndef CONGRUENT_TRIANGLES_HPP
#define CONGRUENT_TRIANGLES_HPP

#include "predicate/statement.hpp"
#include "type/triangle.hpp"
#include "predicate/similar_triangles.hpp"
#include "predicate/cong.hpp"

class CongruentTriangles : public SimilarTriangles
{
public:
    CongruentTriangles(const Triangle &t1, const Triangle &t2, bool sameclock);

    CongruentTriangles(const std::vector<statement_arg> &args);

    std::string name() const override;

    std::unique_ptr<Statement> clone() const override;

    bool check_nondegen() const override;

    bool check_equations() const override;

    std::unique_ptr<Statement> replace(Point p, Point q) const override;

    std::unique_ptr<Statement> normalize() const override;

    Cong cong_ab() const;

    Cong cong_bc() const;

    Cong cong_ac() const;

    std::ostream &print(std::ostream &os) const override;

    std::vector<CongruentTriangles> cyclic_rotations() const;

    bool numerical_only() const { return false; }
};

#endif // CONGRUENT_TRIANGLES_HPP
