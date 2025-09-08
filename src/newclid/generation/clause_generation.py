import random
import string
import numpy
import time
from newclid.formulations.definition import DefinitionJGEX
from newclid.formulations.clause import translate_sentence
from newclid.statement import Statement
from newclid.configs import default_defs_path
from newclid.dependencies.symbols import Point
from newclid.dependencies.dependency_graph import DependencyGraph
from newclid.predicates import NAME_TO_PREDICATE
from newclid.proof import ConstructionError
from newclid.algebraic_reasoning.algebraic_manipulator import AlgebraicManipulator
from newclid.numerical.sketch import sketch
from newclid.numerical.geometries import (
    ObjNum,
    PointNum,
    reduce,
)
from newclid.numerical.distances import (
    PointTooCloseError,
    PointTooFarError,
    check_too_far_numerical,
    check_too_close_numerical,
)


BASIC = [
    'segment',
    'triangle',
    'triangle12',
    'r_triangle',
    'iso_triangle',
    # 'iso_triangle0', 
    'ieq_triangle',
    'risos',
    'quadrangle',
    'rectangle',
    'isquare',
    'trapezoid',
    'r_trapezoid',
    'iso_trapezoid',
    'eq_quadrangle',
    'eqdia_quadrangle',
    'pentagon',
]

BASIC_FREE = [
    'free',
]

INTERSECT = [
    'angle_bisector', # => bisect => LineNum
    'angle_mirror', # => amirror => LineNum
    'eqdistance', # => circle => CircleNum     
    'on_line', # => line => LineNum
    'on_aline', # => aline => LineNum
    # 'on_aline0', # => aline => LineNum
    'on_bline', # => bline => LineNum
    'on_pline', # => pline => LineNum
    # 'on_pline0', # => pline => LineNum
    'on_tline', # => tline => LineNum
    'on_dia', # => dia => CircleNum
    'on_circle', # => circle => CircleNum
    'eqangle3', # => eqangle3 => CircleNum
    'on_circum', # => cyclic =>  CircleNum, 
    'eqratio', # => eqratio => CircleNum
    'eqratio6',  # => eqratio6 => LineNum / CircleNum
    'lc_tangent', # => tline => LineNum  # should be here
    # TODO: double check. do we need this?
    # 'rconst', # => rconst => CircleNum
    # 'rconst2', # => rconst2 => LineNum / CircleNum
    # 'aconst', # => aconst => LineNum !一般在goal中，可以不放
    's_angle', # => s_angle => LineNum 
    # 'lconst', # => lconst => CircleNum
]

OTHER = [
    'circle',
    'circumcenter',
    'eq_triangle',
    'eqangle2',
    'foot',
    'incenter',
    'incenter2',
    'excenter',
    'excenter2',
    'centroid',
    'ninepoints',
    'intersection_cc',
    'intersection_lc',
    'intersection_ll',
    'intersection_lp',
    'intersection_lt',
    'intersection_pp',
    'intersection_tt',
    'midpoint',
    'mirror',
    'nsquare',
    'orthocenter',
    'parallelogram',
    'psquare',
    'reflect',
    'shift',
    'square',
    '2l1c',
    'e5128',
    '3peq',
    'trisect',
    'trisegment',
    'cc_tangent',
    'tangent',
    # 'iso_triangle_vertex',
    # 'iso_triangle_vertex_angle',
]


class PointGenerator:
    def __init__(self, max_points=260):
        """Point generator, creates unique point names"""
        self.max_points = max_points
        self.defined_points = []

    def get_point_name(self, va_idx):
        """Generate a point name using letters and numbers"""
        letter_part = string.ascii_lowercase[va_idx % 26]
        number_part = va_idx // 26
        return f"{letter_part}{number_part - 1}" if number_part else letter_part  # a, b, ..., z, a0, b0, ...

    def prefetch_points(self, n):
        res = []
        for i in range(n):
            if len(self.defined_points) >= self.max_points:
                raise ValueError("All point names exhausted.")
            point_name = self.get_point_name(len(self.defined_points) + i)
            res.append(point_name)
        return res
        
    def define_points(self, points):
        for p in points:
            if len(self.defined_points) >= self.max_points:
                raise ValueError("All point names exhausted.")
            self.defined_points.append(p)
          
                
class CompoundClauseGen:
    def __init__(self, seed = None, defs=None):
        """Initialize the compound clause generator"""
        self.defs = defs or DefinitionJGEX.to_dict(
            DefinitionJGEX.parse_txt_file(default_defs_path())
            )
        self.rng = numpy.random.default_rng(seed)
        random.seed(seed)
        self.point_generator = None
        self.symbols_graph = None
        self.dep_graph = None

    def generate(self, length = 0):
        self.point_generator = PointGenerator()
        self.dep_graph = DependencyGraph(AlgebraicManipulator())
        self.symbols_graph = self.dep_graph.symbols_graph

        max_basic_clause = int(0.15 * length)
        res = []
        for clause_set in range(length):
            # step 1: add clause with basic 
            if len(res) == 0: 
                new_clause = self.get_clause_with_n_constructions(BASIC, 1)
            # step 2: add clause with basic (free) 
            elif clause_set < max_basic_clause:
                new_clause = self.get_clause_with_n_constructions(BASIC_FREE, 1)
            # step 3: add cluase with single constructions or two constructions
            else:
                if random.random() < 0.5:
                    new_clause = self.get_clause_with_n_constructions(INTERSECT, 2)
                else:
                    new_clause = self.get_clause_with_n_constructions(OTHER+INTERSECT, 1)
            if new_clause:
                res.append(new_clause)
        return "; ".join(res)

    def get_clause_with_n_constructions(self, construction_candidates, n: int):
        try_count = 0
        while try_count < 10:
            try_count += 1
            # samples constructions
            constructions = []
            numerics = []
            if n == 1:
                new_points, construction, numeric = self.choose_construction(construction_candidates)
                constructions.append(construction)
                numerics += numeric
            else:
                # multiple n_constructions shares the same new points. Only support one new point
                new_points = self.point_generator.prefetch_points(1)
                for _ in range(n):
                    _, construction, numeric = self.choose_construction(construction_candidates, new_points)
                    constructions.append(construction)
                    numerics += numeric
            # check numerics by drawing diagram
            try:
                self.draw_diagram(new_points, numerics)
            except Exception as e:
                # print(f"Exception type: {type(e).__name__}, message: {e}")
                # import traceback
                # traceback.print_exc()
                continue

            self.point_generator.define_points(new_points)
            # output clause str. add xy for new points to accelerate building
            new_points_str=[]
            for p in new_points:
                p_num = self.symbols_graph.names2points([p])[0]
                new_points_str.append(f'{p}@{p_num.num.x}_{p_num.num.y}')
            return ' '.join(new_points_str) + " = " + ', '.join(constructions)
    
    def draw_diagram(self, new_points, numerics,):
        def draw_fn() -> tuple[PointNum, ...]:
            to_be_intersected: list[ObjNum] = []
            for n in numerics:
                args: list[Union[PointNum, str]] = []
                for t in n[1:]:
                    if str.isalpha(t[0]): # a1 => a                      
                        args.append(self.symbols_graph.names2points([t])[0].num)
                    else:
                        args.append(t)
                to_be_intersected += sketch(n[0], tuple(args), self.rng)

            return reduce(
                to_be_intersected, [p.num for p in _existing_points], rng=self.rng
            )

        # some points are created in previous draw, but not pass the check. we should replace this points
        _existing_points = list(self.symbols_graph.nodes_of_type(Point))
        _existing_points = [p for p in _existing_points if hasattr(p, "num")]
        _new_points = self.symbols_graph.names2points(new_points)
        _new_numerical_point = draw_fn()
        
        # check draw result
        if len(_new_numerical_point) != len(_new_points):
            raise Exception("why no error ??? TO FIX!!!")

        # check point distance
        _existing_numerical_points = [p.num for p in _existing_points]
        if check_too_close_numerical(_new_numerical_point, _existing_numerical_points):
            raise PointTooCloseError()
        if check_too_far_numerical(_new_numerical_point, _existing_numerical_points):
            raise PointTooFarError()

        # set point position
        for p, num in zip(_new_points, _new_numerical_point):
            p.num = num 
    def choose_construction(self, construction_candidates, new_points = None):
        while True:
            construction = random.choice(construction_candidates)
            construction_def = self.defs[construction]

            # create new point if new_points is None
            if not new_points:
                new_points = self.point_generator.prefetch_points(len(construction_def.points))

            # check number of points
            if len(construction_def.points) != len(new_points):
                continue
            if len(construction_def.args) > len(self.point_generator.defined_points):
                continue

            # create mapping
            mapping = self.map_points(construction_def, self.point_generator.defined_points, new_points)

            # check construction requirements
            try:
                for premise in construction_def.require.sentences:
                    if len(premise) == 0:
                        continue
                    statement = Statement.from_tokens(translate_sentence(mapping, premise), self.dep_graph)
                    if statement is None or not statement.check_numerical():
                        raise ConstructionError("Requirement check_numerical failed. " + str(construction))
            except Exception as e:
                continue
            try:
                for bs in construction_def.basics:
                    for t in bs.sentences:
                        statement = Statement.from_tokens(translate_sentence(mapping, t), self.dep_graph)
            except Exception as e:
                continue  

            # output numerics for draw check
            numerics = []
            for n in construction_def.numerics:
                numerics.append(tuple(mapping[a] if a in mapping else a for a in n))

            return new_points, self.construction_text(construction_def, mapping), numerics
        
    
    def map_points(self, construction_def, defined_points, new_points):
        # mapping point to new points
        mapping = dict(zip(construction_def.points, new_points))
        # mapping args to predefined_points
        if construction_def.declare[0] in ['s_angle']:
            points = random.sample(defined_points, len(construction_def.args) - 1)
            for i, point in enumerate(points):
                mapping[construction_def.args[i]] = point
            if construction_def.declare[0] == 's_angle':
                mapping[construction_def.args[-1]] = f'{random.choice(range(15, 180, 15))}o'
        else:
            points = random.sample(defined_points, len(construction_def.args))
            for i, point in enumerate(points):
                mapping[construction_def.args[i]] = point
        return mapping

    def construction_text(self, construction_def, mapping):
        text = f"{construction_def.declare[0]} {' '.join([mapping[p] for p in construction_def.declare[1:]])}"
        return text


if __name__ == "__main__":
    cc_gen = CompoundClauseGen(42)
    clause_text = cc_gen.generate(50)
    clause_text = cc_gen.generate(50)
    clause_text = cc_gen.generate(50)
    print(clause_text)
    for i in range(20):
        s_time = time.time()
        cc_gen = CompoundClauseGen(i)
        clause_text = cc_gen.generate(50)
        print(f'{time.time() - s_time:.2f}s')
