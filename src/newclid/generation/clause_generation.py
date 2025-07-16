import copy
import random
import string
from newclid.formulations.definition import DefinitionJGEX
from newclid.configs import default_defs_path

INTERSECT = [
    'angle_bisector',
    'angle_mirror',
    'eqdistance',
    'lc_tangent',
    'on_aline',
    'on_bline',
    'on_circle',
    'on_line',
    'on_pline',
    'on_tline',
    'on_dia',
    's_angle',
    # 'on_opline',
    'eqangle3',
]

# constructions with corresponding sketch functions
# TODO: might not need this since these two constructions are not in the new defs.txt
NO_SKETCH = [
    'on_aline2',
    'cc_tangent0'
]

def get_ordering_index(first_list, second_list):
    """
    Returns the indices to reorder the second list to match the order of the first list.
    """
    # Create a mapping from element to its index in the first list
    index_map = {element: index for index, element in enumerate(first_list)}

    # Create a list of tuples (index_in_first_list, index_in_second_list)
    ordering = sorted((index_map[element], index) for index, element in enumerate(second_list))

    # Extract and return just the indices from the second list in the correct order
    return [index for _, index in ordering]

def get_wrapped_points(all_points, start, num_points):
    """
    Gets points from the list starting from 'start', wrapping around if necessary,
    to collect 'num_points' items.

    :param all_points: List of all available points.
    :param start: Starting index to pick points.
    :param num_points: Number of points to pick.
    :return: List of points as per specified conditions.
    """
    list_len = len(all_points)  # Length of your points list
    # Use list comprehension and modulo to wrap around
    wrapped_points = [all_points[(start + i) % list_len] for i in range(num_points)]
    return wrapped_points


def get_apha_geo_solver_var(va_idx):
    letter_part = string.ascii_lowercase[va_idx % 26]
    number_part = va_idx // 26

    # Prepare the point name
    if number_part == 0:
        # For the first cycle (A-Z), we don't add a number part
        point_name = letter_part
    else:
        # For subsequent cycles, add the number part (reduced by 1 to start from 0)
        point_name = f"{letter_part}{number_part - 1}"

    return point_name


class ClauseGenerator:
    def __init__(self, defs, clause_relations, is_comma_sep, seed, shuffle_var_names=False):
        self.defs = defs
        self.defined_points = []
        self.is_comma_sep = is_comma_sep
        self.clause_relations = clause_relations
        # To limit to a few concepts uncomment the following line
        # self.clause_relations = ['triangle', 'parallelogram',]
        self.point_counter = 0  # Start from 0
        self.max_points = 26 * 10  # 26 letters, 10 cycles (0 to 9, inclusive)
        self.var_idxs = list(range(self.max_points))
        if shuffle_var_names:
            random.seed(seed)
            random.shuffle(self.var_idxs)
        self.alpha_geo_solv_var_2_used_var_ = {}

    def get_varname_2_alpha_geo_var_map(self):
        return copy.deepcopy(self.alpha_geo_solv_var_2_used_var_)

    def get_pt_ctr_def_pts(self):
        return self.point_counter, self.defined_points

    def set_pt_ctr_def_pts(self, point_counter, defined_points):
        self.point_counter = point_counter
        self.defined_points = defined_points

    def generate_point(self):
        """
        Generate the next point in sequence: A, B, ..., Z, A0, B0, ..., Z9.
        After Z9, raise an error.
        """
        if self.point_counter >= self.max_points:
            # If we've exhausted all possible names, raise an error
            raise ValueError("All point names have been exhausted.")

        # Calculate the letter and number parts of the name
        selected_var_idx = self.var_idxs[self.point_counter]
        point_name = get_apha_geo_solver_var(selected_var_idx)
        self.alpha_geo_solv_var_2_used_var_[get_apha_geo_solver_var(self.point_counter)] = point_name

        # Increment the counter for the next call
        self.point_counter += 1

        return point_name

    def generate_new_point(self):
        while True:
            point = self.generate_point()
            if point not in self.defined_points:
                return point

    def choose_random_n_defined_points(self, n):
        """
        Choose n random defined points
        """
        return random.sample(self.defined_points, n)

    def get_text_clause(self, clause_relation, arg_vars, result_vars, all_will_be_defined_pts, result_vars_in_eq):
        """
        Make a canonical clause for a given relation
        """
        pos_new_pts_idx = get_ordering_index(self.defs[clause_relation].declare[1:],
                                            self.defs[clause_relation].points + self.defs[clause_relation].args)
        all_inp_pts = result_vars + arg_vars
        all_inp_pts_reordered = [all_inp_pts[pos_new_pts_idx[i]] for i, _ in enumerate(all_inp_pts)]

        if result_vars_in_eq:
            clause_txt = f'{" ".join(all_will_be_defined_pts)} = {clause_relation} {" ".join(all_inp_pts_reordered)}'
        else:
            clause_txt = f'{clause_relation} {" ".join(all_inp_pts_reordered)}'

        # handle special cases
        if clause_relation in ['s_angle', ]:
            clause_txt += f' {random.choice(range(0, 180, 15))}o'
        return clause_txt

    def generate_clauses(self, n):
        """
        Generate n random clauses with all points defined
        """
        clauses = []
        if self.is_comma_sep:
            sub_clause_relation = []
            sub_clause_defines_points = []
            sub_clause_needs_defined_points = []
            for i in range(n):
                # choose a random definition key as the first clause
                clause_relation, defines_points, needs_defined_points = self.choose_suitable_clause()
                sub_clause_relation.append(clause_relation)
                # this_clause_must_define = max((0, defines_points - max([0, ] + sub_clause_defines_points)))
                # sub_clause_defines_points.append(random.choice(range(defines_points, defines_points + 1)))
                sub_clause_defines_points.append(defines_points)
                sub_clause_needs_defined_points.append(needs_defined_points)

            assert max(sub_clause_defines_points) == 1
            # defines_points = random.randint(max(sub_clause_defines_points), sum(sub_clause_defines_points))
            defines_points = 1
            all_will_be_defined_pts = self.get_points_that_this_clause_defines(defines_points)

            pts_defined_sp_far = 0
            for i in range(n):
                pts_this_clause_defines = get_wrapped_points(all_will_be_defined_pts, pts_defined_sp_far,
                                                        sub_clause_defines_points[i])
                pts_defined_sp_far += sub_clause_defines_points[i]
                chosen_defined_pts = random.sample(self.defined_points, sub_clause_needs_defined_points[i])
                clause = self.get_text_clause(sub_clause_relation[i], chosen_defined_pts, pts_this_clause_defines,
                                            all_will_be_defined_pts, result_vars_in_eq=(i == 0))
                clauses.append(clause)

            self.add_newly_defined_pts(all_will_be_defined_pts)
            return ', '.join(clauses)
        else:
            for i in range(n):
                # choose a random definition key as the first clause
                clause_relation, defines_points, needs_defined_points = self.choose_suitable_clause()

                chosen_defined_pts = random.sample(self.defined_points, needs_defined_points)
                # Generate names of points that are needed for the clause
                will_be_defined_pts = self.get_points_that_this_clause_defines(defines_points)

                clause = self.get_text_clause(clause_relation, chosen_defined_pts, will_be_defined_pts,
                                            all_will_be_defined_pts=will_be_defined_pts, result_vars_in_eq=True)
                clauses.append(clause)
                self.add_newly_defined_pts(will_be_defined_pts)

            return '; '.join(clauses)

    def get_points_that_this_clause_defines(self, defines_points):
        will_be_defined_pts = []
        while defines_points > 0:
            will_be_defined_pts.append(self.generate_new_point())
            defines_points -= 1
        return will_be_defined_pts

    def add_newly_defined_pts(self, defined_pts):
        self.defined_points += defined_pts

    def choose_suitable_clause(self):
        suitable_clause = False
        while not suitable_clause:
            clause_relation = random.choice(self.clause_relations)
            needs_defined_points = len(self.defs[clause_relation].args)
            defines_points = len(self.defs[clause_relation].points)
            # handle special cases
            if clause_relation in ['s_angle', ]:
                needs_defined_points -= 1
            if needs_defined_points <= len(self.defined_points):
                suitable_clause = True
        return clause_relation, defines_points, needs_defined_points

    def reset(self):
        self.defined_points = []
        self.point_counter = 0


class CompoundClauseGen:
    def __init__(self, max_comma_sep_clause, max_single_clause, max_sets, seed, shuffle_var_names=False):
        self.max_comma_sep_clause = max_comma_sep_clause
        self.max_single_clause = max_single_clause
        self.max_sets = max_sets
        definitions = DefinitionJGEX.to_dict(DefinitionJGEX.parse_txt_file(default_defs_path()))
        self.cg_comma_sep = ClauseGenerator(definitions, INTERSECT, is_comma_sep=True, seed=seed,
                                            shuffle_var_names=shuffle_var_names)
        constructions_excluding_sketch = [key for key in definitions.keys() if key not in NO_SKETCH]
        self.cg_single_clause = ClauseGenerator(definitions, constructions_excluding_sketch, is_comma_sep=False, seed=seed,
                            shuffle_var_names=shuffle_var_names)
        # self.cg_single_clause = ClauseGenerator(definitions, list(definitions.keys()), is_comma_sep=False, seed=seed,
        #                                         shuffle_var_names=shuffle_var_names)

    def get_varname_2_alpha_geo_var_map(self):
        return self.cg_single_clause.get_varname_2_alpha_geo_var_map()

    def reset(self):
        self.cg_comma_sep.reset()
        self.cg_single_clause.reset()

    def generate_clauses(self):
        clause_text = ''
        for clause_set in range(self.max_sets):
            if clause_text == '' or random.random() < 0.5:  # 50% chance of comma separated clauses
                if clause_text != '':
                    clause_text += '; '
                clause_text += self.cg_single_clause.generate_clauses(
                    random.choice(range(1, self.max_single_clause + 1)))
                point_counter, defined_points = self.cg_single_clause.get_pt_ctr_def_pts()

            if point_counter < 3:
                continue

            if random.random() < 0.5:  # 50% chance of comma separated clauses
                self.cg_comma_sep.set_pt_ctr_def_pts(point_counter, defined_points)
                if not clause_text.endswith('; '):
                    clause_text += '; '

                # clause_text += self.cg_comma_sep.generate_clauses(
                    # random.choice(range(1, self.max_comma_sep_clause + 1)))
                clause_text += self.cg_comma_sep.generate_clauses(2) # generate cluase with two constructions
                self.cg_single_clause.set_pt_ctr_def_pts(*self.cg_comma_sep.get_pt_ctr_def_pts())
        self.reset()
        return clause_text


if __name__ == "__main__":
    random.seed(3)

    cc_gen = CompoundClauseGen(2, 3, 2, 42)

    for j in range(5):
        clause_text = cc_gen.generate_clauses()
        print(clause_text)
