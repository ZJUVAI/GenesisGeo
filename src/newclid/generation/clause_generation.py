import random
import string
from newclid.formulations.definition import DefinitionJGEX
from newclid.configs import default_defs_path
from newclid.agent.ddarn import DDARN
from newclid.api import GeometricSolverBuilder
import time

BASIC = [
    'ieq_triangle',
    'isquare',
    'trapezoid',
    'triangle',
    'triangle12',
    'risos',
    'segment',
    'quadrangle',
    'r_trapezoid',
    'r_triangle',
    'rectangle',
    'pentagon',
    'iso_triangle',
    'free',
    'eq_quadrangle',
    'iso_trapezoid',
    'eqdia_quadrangle',
    'iso_triangle0'
]

SINGLE = [
    'incenter2',
    'excenter2',
    'centroid',
    'ninepoints',
    'square',
    '2l1c',
    'e5128',
    '3peq',
    'trisect',
    'trisegment',
    'cc_tangent',
    'tangent',
    'eqratio',
    'eqratio6',
    'rconst',
    'rconst2',
    'aconst',
    's_angle',
    'lconst',
    'iso_triangle_vertex',
    'iso_triangle_vertex_angle',
    'orthocenter',
    'parallelogram',
    'psquare',
    'reflect',
    'shift',
    'foot',
    'incenter',
    'excenter',
    'intersection_cc',
    'intersection_lc',
    'intersection_ll',
    'intersection_lp',
    'intersection_lt',
    'intersection_pp',
    'intersection_tt',
    'lc_tangent',
    'midpoint',
    'mirror',
    'nsquare',
    'circumcenter',
    'eq_triangle',
    'eqangle2',
]

COMMA = [
    'angle_bisector',
    'angle_mirror',
    'circle',
    'eqdistance',
    'on_aline',
    'on_bline',
    'on_circle',
    'on_line',
    'on_pline',
    'on_tline',
    'on_dia',
    'eqangle3',
    'on_circum',
    'on_pline0',
    'on_aline0',
]

def get_var_name(va_idx):
    """Generate a point name using letters and numbers"""
    letter_part = string.ascii_lowercase[va_idx % 26]
    number_part = va_idx // 26
    return f"{letter_part}{number_part - 1}" if number_part else letter_part  # a, b, ..., z, a0, b0, ...

class PointGenerator:
    def __init__(self, max_points=260, shuffle = False, seed = None):
        """Point generator, creates unique point names"""
        self.max_points = max_points
        self.point_counter = 0
        self.var_idx = list(range(self.max_points))
        self.defined_points = []
        if not seed:
            seed = random.randint(0, 1000000)
        if shuffle:
            random.seed(seed)
            random.shuffle(self.var_idx)

    def generate_point(self):
        """Generate the next point name"""
        if self.point_counter >= self.max_points:
            raise ValueError("All point names exhausted.")
        point_name = get_var_name(self.var_idx[self.point_counter])
        self.point_counter += 1
        return point_name
    
    def generate_unique_point(self):
        """Generate a point name that has not been defined yet"""
        while True:
            point = self.generate_point()
            if point not in self.defined_points:
                self.defined_points.append(point)
                return point

class ClauseGenerator:
    def __init__(self, defs, clause_relations, is_single_point, point_generator):
        """Initialize the geometric clause generator"""
        self.defs = defs
        self.is_single_point = is_single_point
        self.clause_relations = clause_relations
        self.point_generator = point_generator
    
    def map_points(self, clause, defined_points, new_point = None):
        defined_points = defined_points[:]
        clause_def = self.defs[clause]
        if new_point and new_point in defined_points:
            defined_points.remove(new_point)
        if len(clause_def.args) > len(defined_points):
            return None
        mapping = {}
        if new_point:
            assert len(clause_def.points) == 1
            mapping[clause_def.points[0]] = new_point
        else:
            for point in clause_def.points:
                new_point = self.point_generator.generate_unique_point()
                mapping[point] = new_point
        points = random.sample(defined_points, len(clause_def.args))
        for i,point in enumerate(points):
            mapping[clause_def.args[i]] = point
        return mapping

    def choose_random_clause(self, new_point = None):
        """Randomly choose a relation based on the number of defined points"""
        while True:
            clause_relation = random.choice(self.clause_relations)
            if new_point and len(self.defs[clause_relation].points) != 1:
                continue
            mapping = self.map_points(clause_relation, self.point_generator.defined_points, new_point)
            if new_point and mapping:
                return self.generate_clause_text(clause_relation, mapping)
            elif mapping:
                clause = self.generate_clause_text(clause_relation, mapping)
                new_points = [mapping[point] for point in self.defs[clause_relation].points]
                return ' '.join(new_points) + " = " + clause

    def generate_clause_text(self, clause_relation, mapping):
        """Generate the standard clause text for a given relation"""
        clause_def = self.defs[clause_relation]
        clause_txt = f"{clause_relation} {' '.join([mapping[p] for p in clause_def.declare[1:]])}"
        if clause_relation == 's_angle':
            clause_txt += f' {random.choice(range(15, 180, 15))}o'
        return clause_txt

    def generate_clauses(self, n, current_text):
        """Generate a specified number of clauses"""
        start_time = time.time()
        while True:
            if time.time() - start_time > 60:
                return None
            clauses = []
            origin_pt_gen_pc = self.point_generator.point_counter
            origin_pt_gen_dfp = self.point_generator.defined_points[:]
            if self.is_single_point:
                new_point = self.point_generator.generate_unique_point()
                for _ in range(n):
                    clause = self.choose_random_clause(new_point)
                    clauses.append(clause)
                new_clause = f"{new_point} = "  + ', '.join(clauses)
            else:
                new_clause = self.choose_random_clause()
            new_text = current_text + new_clause
            solver_builder = GeometricSolverBuilder(seed=998244353)
            solver_builder.with_deductive_agent(DDARN())
            solver_builder.load_problem_from_txt(new_text.strip())
            try:
                solver = solver_builder.build(max_attempts=100)
            except Exception as e:
                self.point_generator.point_counter = origin_pt_gen_pc
                self.point_generator.defined_points = origin_pt_gen_dfp
                continue
            return new_clause
                
class CompoundClauseGen:
    def __init__(self, max_sets, seed = None, shuffle=False):
        """Initialize the compound clause generator"""
        definitions = DefinitionJGEX.to_dict(DefinitionJGEX.parse_txt_file(default_defs_path()))
        self.max_basic_clause = int(0.3 * max_sets)
        self.sets = 0
        point_generator = PointGenerator(seed=seed, shuffle=shuffle)
        self.point_generator = point_generator
        self.basic_cg = ClauseGenerator(definitions, BASIC, False, point_generator)
        self.single_cg = ClauseGenerator(definitions, SINGLE, False, point_generator)
        self.comma_cg = ClauseGenerator(definitions, COMMA, True, point_generator)

    def generate_clauses(self, length = 0, clause_text = None):
        """Generate compound clauses"""
        if not clause_text:
            clause_text = ''
        for clause_set in range(self.sets, self.sets + length):
            if clause_text:
                clause_text += '; '
            if clause_set < self.max_basic_clause:
                new_clause = self.basic_cg.generate_clauses(1, clause_text)
                if not new_clause:
                    break
                clause_text += new_clause
            else:
                prob = random.random()
                if prob < 0.5:
                    new_clause = self.comma_cg.generate_clauses(2, clause_text)
                else:
                    new_clause = self.single_cg.generate_clauses(1, clause_text)
                if not new_clause:
                    break
                clause_text += new_clause
            self.sets += 1
        return clause_text.strip().rstrip(';')

if __name__ == "__main__":
    for _ in range(5):
        cc_gen = CompoundClauseGen(10, 42)
        clause_text = cc_gen.generate_clauses(5)
        print(clause_text)
        clause_text = cc_gen.generate_clauses(5, clause_text)
        print(clause_text)