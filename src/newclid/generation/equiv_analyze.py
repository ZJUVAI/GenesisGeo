import csv
import json
import sys
import re
import os
from itertools import permutations
from collections import defaultdict


class GeometryEquivalenceAnalyzer:
    """
    A class to analyze and determine equivalence between geometric figures.
    """

    def __init__(self):
        # Define all known geometric constructs
        self.definitions = [
            "angle_bisector", "angle_mirror", "circle", "circumcenter", "eq_quadrangle", "iso_trapezoid"
            "eq_trapezoid", "eq_triangle", "eqangle2", "eqdia_quadrangle", "eqdistance",
            "foot", "free", "incenter", "incenter2", "excenter", "excenter2",
            "centroid", "ninepoints", "intersection_cc", "intersection_lc",
            "intersection_ll", "intersection_lp", "intersection_lt", "intersection_pp",
            "intersection_tt", "iso_triangle", "iso_triangle0", "lc_tangent", "midpoint", "mirror",
            "nsquare", "on_aline", "on_aline2", "on_aline0", "on_bline", "on_circle", "on_line",
            "on_pline", "on_pline0", "on_tline", "orthocenter", "parallelogram", "pentagon",
            "psquare", "quadrangle", "r_trapezoid", "r_triangle", "rectangle", "reflect",
            "risos", "s_angle", "segment", "shift", "square", "isquare", "trapezoid",
            "triangle", "triangle12", "2l1c", "e5128", "3peq", "trisect", "trisegment",
            "on_dia", "ieq_triangle", "on_opline", "cc_tangent0", "cc_tangent",
            "eqangle3", "tangent", "on_circum", "eqangle", "eqratio", "eqratio6", "perp", "para", "cong",
            "cyclic", "coll", "midp", "aconst", "rconst", "rconst2", "lconst", "sameside", "acompute",
            "simtrir", "simtri", "iso_triangle_vertex", "iso_triangle_vertex_angle"
        ]
        self.construct_to_id = {name: idx + 1 for idx, name in enumerate(self.definitions)}

        self.CONSTRUCT_EQUIVALENCES = [
            # 角度平分线相关
            ('angle_bisector', 'x a b c', lambda args: [
                args,  # 原始顺序: x a b c
                self.swap(args, 1, 3)  # a<->c交换: x c b a
            ]),
            
            # 三点排列组合（circle, circumcenter, incenter, excenter等）
            (['circle', 'circumcenter', 'incenter', 'excenter'], 'x a b c', lambda args: [
                [args[0]] + list(p) for p in permutations(args[1:4])  # a,b,c的所有排列(3! = 6种)
            ]),

            (['eq_quadrangle', 'trapezoid', 'iso_trapezoid', 'eqdia_quadrangle'], 'a b c d', lambda args: [
                args,  # a b c d
                [args[2], args[3], args[0], args[1]],
                [args[3], args[2], args[1], args[0]],
                [args[1], args[0], args[3], args[2]]
            ]),
            
            # 等边三角形
            (['eq_triangle', 'iso_triangle', 'midpoint', 'midp'], 'x b c', lambda args: [
                args,  # x b c
                self.swap(args, 1, 2) 
            ]),
            
            # 角度相等
            ('eqangle2', 'x a b c', lambda args: [
                args,  # x a b c
                self.swap(args, 1, 3) 
            ]),
            
            # 三等分线段
            ('trisegment', 'x y a b', lambda args: [
                args,  # x y a b
                [args[1], args[0], args[3], args[2]]
            ]),
            
            # 三角形中心（轮换等价）
            (['incenter2', 'excenter2', 'centroid', 'ninepoints'], 'x y z i a b c', lambda args: [
                [args[p[0]], args[p[1]], args[p[2]], args[3],  # x,y,z,i
                args[p[0]+4], args[p[1]+4], args[p[2]+4]]     # 对应的a,b,c
                for p in permutations([0, 1, 2])  # 生成所有排列 (0,1,2), (0,2,1), 等
            ]),
            
            (['ieq_triangle', 'triangle'], 'a b c', lambda args: [
                list(p) for p in permutations(args)  # 所有排列(3! = 6种)
            ]),
            
            ('r_triangle', 'a b c', lambda args: [
                args,  # a b c
                [args[0], args[2], args[1]]  # a c b
            ]),
            
            # 梯形相关
            ('r_trapezoid', 'a b c d', lambda args: [
                args,  # a b c d
                [args[3], args[2], args[1], args[0]]
            ]),
        
            
            # 距离和线段相关
            (['eqdistance', 'foot'], 'x a b c', lambda args: [
                args,  # x a b c
                self.swap(args, 2, 3)
            ]),
            
            ('segment', 'a b', lambda args: [
                args,  # a b
                self.swap(args, 0, 1)
            ]),
            
            # 共线相关
            ('coll', '*points', lambda args: [
                list(p) for p in permutations(args)  # 所有点的排列(n!种)
            ]),
            
            # 四边形相关
            ('parallelogram', 'a b c x', lambda args: [
                args,  # a b c x
                [args[2], args[1], args[0], args[3]]  # c b a x
            ]),
            
            ('square', 'a b x y', lambda args: [
                args,  # a b x y
                [args[1], args[0], args[3], args[2]]  # b a y x
            ]),
            
            (['rectangle', 'isquare', 'quadrangle'], 'a b c d', lambda args: [
                [args[i], args[(i+1)%4], args[(i+2)%4], args[(i+3)%4]] for i in range(4)  # 循环排列
            ]),
            
            ('pentagon', 'a b c d e', lambda args: [
                [args[i], args[(i+1)%5], args[(i+2)%5], args[(i+3)%5], args[(i+4)%5]] for i in range(5)  # 循环排列
            ]),
            
            # 其他构造类型
            ('cyclic', 'a b c d', lambda args: [
                list(p) for p in permutations(args)  # 所有排列(4! = 24种)
            ]),
            
            (['perp', 'cong'], 'a b c d', lambda args: [
                perp
                for perp in self.generate_angle_equivalences(args[0:4])
            ]),
            
            (['eqangle','eqratio'], 'a b c d e f g h', lambda args: [
                                                            angle1 + angle2
                                                            for angle1 in self.generate_angle_equivalences(args[0:4])
                                                            for angle2 in self.generate_angle_equivalences(args[4:8])
                                                        ] + [
                                                            angle2 + angle1
                                                            for angle1 in self.generate_angle_equivalences(args[0:4])
                                                            for angle2 in self.generate_angle_equivalences(args[4:8])
                                                        ]),
            

            # 切线相关
            ('on_tline', 'x a b c', lambda args: [
                args,  # x a b c
                [args[0], args[1], args[3], args[2]]  # x a c b
            ]),
            
            ('tangent', 'x y a o b', lambda args: [
                args,  # x y a o b
                [args[1], args[0]] + args[2:]  # y x a o b
            ]),
            
            # 固定角度
            ('aconst', 'x a b c angle', lambda args: [
                args  # 只保留原始顺序，不生成任何置换
            ]),
            
            # 特殊角度
            ('s_angle', 'x a b angle', lambda args: [
                args,  # 原始顺序: x a b angle
                [args[2], args[1], args[0], args[3]]  # 交换a和b: x b a angle
            ]),

            (['simtrir', 'simtri'], 'a b c d e f', lambda args: [
                [args[p[0]], args[p[1]], args[p[2]],  # x,y,z,i
                args[p[0]+3], args[p[1]+3], args[p[2]+3]]     # 对应的a,b,c
                for p in permutations([0, 1, 2]) 
            ]),

        ]

        self.EQUIVALENT_COUNTS = {}
        for item in self.CONSTRUCT_EQUIVALENCES:
            construct_names = item[0] if isinstance(item[0], list) else [item[0]]
            
            sample_args = item[1].split() if item[1] != '*points' else ['a', 'b', 'c']
            
            equiv_count = len(item[2](sample_args))
            
            for name in construct_names:
                self.EQUIVALENT_COUNTS[name] = equiv_count


        escaped_defs = [re.escape(d) for d in self.definitions]
        pattern = r'\b(' + '|'.join(escaped_defs) + r')\b'
        self.construct_pattern = re.compile(pattern)
        self.target_pattern = re.compile(pattern)
        self.geometry_map = {}

    def swap(self, args, *swap_groups):
        new_args = args.copy()
        for i in range(0, len(swap_groups), 2):
            a, b = swap_groups[i], swap_groups[i+1]
            new_args[a], new_args[b] = new_args[b], new_args[a]
        return new_args

    def generate_angle_equivalences(self, angle):
        """Generate all equivalent representations of an angle [a,b,c,d]"""
        if len(angle) != 4:
            return [angle]

        a, b, c, d = angle
        return [
            [a, b, c, d],  # Original
            [b, a, c, d],  # Reversed first line
            [a, b, d, c],  # Reversed second line
            [b, a, d, c],  # Both lines reversed
            [c, d, a, b],  # Swapped lines
            [d, c, a, b],  # Swapped and first line reversed
            [c, d, b, a],  # Swapped and second line reversed
            [d, c, b, a],  # Swapped and both lines reversed
        ]
    

    def are_same_figure(self, figure1, figure2):
        """
        Determine whether two strings represent the same figure.
        """
        # Step 1: Parse the figure descriptions
        components1 = self.parse_figure(figure1)
        components2 = self.parse_figure(figure2)

        # Step 2: Check if they have the same number of components
        if len(components1) != len(components2):
            return False

        # Step 3: Try to find a mapping between points in figure1 and figure2
        point_mapping = self.find_point_mapping(components1, components2)

        # Step 4: If no valid mapping is found, they are not the same figure
        if point_mapping is None:
            return False

        # Step 5: Check if all components match under the point mapping
        return self.check_components_match(components1, components2, point_mapping)

    def parse_figure(self, figure_str):
        """Parse a figure string into its component parts."""
        # First split by question mark to separate definitions from queries
        if '?' in figure_str:
            definitions_part, queries_part = figure_str.split('?', 1)
        else:
            definitions_part, queries_part = figure_str, ""

        # Process definitions (before the question mark)
        parsed_components = []
        if definitions_part:
            # Split definitions by semicolon
            definition_parts = [part.strip() for part in definitions_part.split(';') if part.strip()]

            for part in definition_parts:
                if '=' in part:
                    left, right = part.split('=', 1)
                    points = left.strip().split()

                    # Split the right side by comma to handle multiple constraints
                    constraints = [c.strip() for c in right.split(',')]

                    for constraint in constraints:
                        construct_parts = constraint.split()
                        construct_type = construct_parts[0]
                        construct_args = construct_parts[1:]

                        parsed_components.append({
                            'type': 'definition',
                            'points': points,
                            'construct': construct_type,
                            'args': construct_args
                        })

        # Process queries (after the question mark)
        if queries_part:
            # Split queries by semicolon
            query_parts = [part.strip() for part in queries_part.split(';') if part.strip()]

            for query_part in query_parts:
                query_parts = query_part.split()
                query_type = query_parts[0]
                query_args = query_parts[1:]

                parsed_components.append({
                    'type': 'query',
                    'query': query_type,
                    'args': query_args
                })

        return parsed_components

    def extract_points(self, components):
        """Extract all unique points mentioned in the components."""
        points = set()
        for component in components:
            if component['type'] == 'definition':
                points.update(component['points'])
                points.update(component['args'])

        return points

    def find_unique_constructs(self, components1, components2):
        """Find construct types that appear exactly once in each figure."""
        constructs1 = [c.get('construct') for c in components1 if 'construct' in c]
        constructs2 = [c.get('construct') for c in components2 if 'construct' in c]

        # Find constructs that appear exactly once in both
        unique_constructs = []
        for construct in set(constructs1).intersection(set(constructs2)):
            if constructs1.count(construct) == 1 and constructs2.count(construct) == 1:
                unique_constructs.append(construct)

        # Sort by the number of points they define (fewer is better for starting)
        unique_constructs.sort(key=lambda c: self.EQUIVALENT_COUNTS.get(c, 1))
        return unique_constructs


    def update_mapping_for_construct(self, comp1, comp2, current_mapping):
        """
        Update the point mapping based on a pair of components with the same construct type.
        Takes into account the equivalence rules for different construct types using CONSTRUCT_EQUIVALENCES.
        """
        construct_type = comp1.get('construct')

        # Find the equivalence rule for this construct type
        equiv_rule = None
        param_format = None
        for rule_type, params, rule_func in self.CONSTRUCT_EQUIVALENCES:
            rule_types = [rule_type] if not isinstance(rule_type, list) else rule_type
            if construct_type in rule_types:
                equiv_rule = rule_func
                param_format = params
                break

        # If no equivalence rule is found, use direct mapping
        if equiv_rule is None:
            return self._direct_mapping(comp1, comp2, current_mapping)

        # Special handling for variable number of points
        if param_format == '*points':
            points1 = comp1.get('points', []) + comp1.get('args', [])
            points2 = comp2.get('points', []) + comp2.get('args', [])

            if len(points1) != len(points2):
                return None

            # Try each possible permutation from the equivalence rules
            for perm in equiv_rule(points1):
                # Try to create a mapping using this permutation
                new_mapping = current_mapping.copy()
                valid = True

                for i, p1 in enumerate(points1):
                    if i < len(perm):
                        p2_idx = points1.index(perm[i])
                        p2 = points2[p2_idx] if p2_idx < len(points2) else None

                        if p2 is None or (p1 in new_mapping and new_mapping[p1] != p2):
                            valid = False
                            break
                        new_mapping[p1] = p2

                if valid:
                    return new_mapping

            return None

        # Normal case: extract points based on parameter format
        param_list = param_format.split()

        # Gather the points from comp1 and comp2
        points1 = []
        points2 = []

        # First try to use 'args' if available
        if 'args' in comp1 and 'args' in comp2:
            points1 = comp1.get('args', [])
            points2 = comp2.get('args', [])
        # Otherwise use 'points'
        elif 'points' in comp1 and 'points' in comp2:
            points1 = comp1.get('points', [])
            points2 = comp2.get('points', [])
        # Or combine both
        else:
            points1 = comp1.get('points', []) + comp1.get('args', [])
            points2 = comp2.get('points', []) + comp2.get('args', [])

        # Check that we have the right number of points
        if len(points1) != len(param_list) or len(points2) != len(param_list):
            return None

        # Generate all possible equivalent orderings
        equivalent_orderings = equiv_rule(points1)

        # Try each ordering to find a valid mapping
        for ordering in equivalent_orderings:
            new_mapping = current_mapping.copy()
            valid = True

            for i, p1 in enumerate(ordering):
                if i < len(points2):
                    p2 = points2[i]
                    if p1 in new_mapping and new_mapping[p1] != p2:
                        valid = False
                        break
                    new_mapping[p1] = p2

            if valid:
                return new_mapping

        return None

    def _direct_mapping(self, comp1, comp2, current_mapping):
        """
        Helper method for direct mapping when no specific equivalence rule exists.
        """
        new_mapping = current_mapping.copy()

        # Map points based on their positions
        if 'points' in comp1 and 'points' in comp2:
            if len(comp1['points']) != len(comp2['points']):
                return None

            for i, p1 in enumerate(comp1['points']):
                p2 = comp2['points'][i]
                if p1 in new_mapping and new_mapping[p1] != p2:
                    return None
                new_mapping[p1] = p2

        if 'args' in comp1 and 'args' in comp2:
            if len(comp1['args']) != len(comp2['args']):
                return None

            for i, p1 in enumerate(comp1['args']):
                p2 = comp2['args'][i]
                if p1 in new_mapping and new_mapping[p1] != p2:
                    return None
                new_mapping[p1] = p2

        return new_mapping

    def find_point_mapping(self, components1, components2):
        """
        Find a valid mapping between points in figure1 and figure2.
        """
        # Extract all unique points from both figures
        points1 = self.extract_points(components1)
        points2 = self.extract_points(components2)

        # If they have different numbers of points, they can't be the same figure
        if len(points1) != len(points2):
            return None

        # Start with unique constructs (those that appear only once)
        unique_constructs = self.find_unique_constructs(components1, components2)

        # Initialize the point mapping
        point_mapping = {}

        # Use the unique constructs to establish initial point mappings
        for construct_type in unique_constructs:
            # Find the components with this construct type
            comp1 = next((c for c in components1 if c.get('construct') == construct_type), None)
            comp2 = next((c for c in components2 if c.get('construct') == construct_type), None)

            if comp1 and comp2:
                # Update the point mapping based on the equivalence rules for this construct
                updated_mapping = self.update_mapping_for_construct(comp1, comp2, point_mapping)

                # If we couldn't find a consistent mapping, this might not be the same figure
                if updated_mapping is None:
                    return None

                point_mapping = updated_mapping

        # If we haven't mapped all points, try to infer the remaining mappings
        if len(point_mapping) < len(points1):
            # Use the remaining constructs to fill in the mapping
            point_mapping = self.complete_mapping(components1, components2, point_mapping, points1, points2)

        # Check if the mapping is complete and consistent
        if len(point_mapping) == len(points1) and self.is_mapping_consistent(components1, components2, point_mapping):
            return point_mapping

        return None

    def complete_mapping(self, components1, components2, partial_mapping, points1, points2):
        """
        Try to complete the partial mapping using the remaining components.
        """
        # Create a copy of the partial mapping
        mapping = partial_mapping.copy()

        # Identify unmapped points
        unmapped_points1 = points1 - set(mapping.keys())
        unmapped_points2 = points2 - set(mapping.values())

        # If no more points to map, return the current mapping
        if not unmapped_points1:
            return mapping

        # Try all possible assignments for the first unmapped point
        p1 = next(iter(unmapped_points1))

        for p2 in unmapped_points2:
            test_mapping = mapping.copy()
            test_mapping[p1] = p2

            # Check if this mapping is consistent with all components
            if self.is_mapping_consistent(components1, components2, test_mapping):
                # Recursively try to complete the mapping
                remaining_unmapped1 = unmapped_points1 - {p1}
                remaining_unmapped2 = unmapped_points2 - {p2}

                if not remaining_unmapped1:
                    return test_mapping

                complete_test_mapping = test_mapping.copy()
                # Pair remaining unmapped points in order (simplified approach)
                for r1, r2 in zip(remaining_unmapped1, remaining_unmapped2):
                    complete_test_mapping[r1] = r2

                if self.is_mapping_consistent(components1, components2, complete_test_mapping):
                    return complete_test_mapping

        # If no valid complete mapping is found
        return mapping


    def is_args_equivalent(self, construct_type, args1, args2):
        """
        Check if two sets of arguments are equivalent based on the construct type and CONSTRUCT_EQUIVALENCES.
        """
        if len(args1) != len(args2):
            return False

        if construct_type in ['aconst', 's_angle', 'lconst', 'rconst']:
            # 比较前面的点参数
            if not self._compare_point_args(args1[:-1], args2[:-1], construct_type):
                return False

            # 比较角度值，必须完全一致
            return args1[-1] == args2[-1]

        # Find the equivalence rule for this construct type
        equiv_rule = None
        for rule_type, params, rule_func in self.CONSTRUCT_EQUIVALENCES:
            rule_types = [rule_type] if not isinstance(rule_type, list) else rule_type
            if construct_type in rule_types:
                equiv_rule = rule_func
                break

        # If no equivalence rule is found, use direct comparison
        if equiv_rule is None:
            return args1 == args2

        # Generate all equivalent forms of args1
        equiv_forms = equiv_rule(args1)

        # Check if args2 matches any equivalent form
        return args2 in equiv_forms

    def _compare_point_args(self, points1, points2, construct_type):
        equiv_rule = None
        for rule_type, params, rule_func in self.CONSTRUCT_EQUIVALENCES:
            rule_types = [rule_type] if not isinstance(rule_type, list) else rule_type
            if construct_type in rule_types:
                equiv_rule = rule_func
                break

        if equiv_rule is None:
            return points1 == points2

        dummy_args = points1 + ['0']
        equiv_forms = [x[:-1] for x in equiv_rule(dummy_args)]

        return points2 in equiv_forms

    def is_mapping_consistent(self, components1, components2, mapping):
        """
        Check if the given mapping is consistent with all components.
        """
        # Apply the mapping to the first figure and check if it matches the second figure
        transformed_components = []

        for comp1 in components1:
            if comp1['type'] == 'definition':
                transformed_points = [mapping.get(p, p) for p in comp1['points']]
                transformed_args = [mapping.get(p, p) for p in comp1['args']]

                transformed_comp = {
                    'type': 'definition',
                    'points': transformed_points,
                    'construct': comp1['construct'],
                    'args': transformed_args
                }
                transformed_components.append(transformed_comp)

            elif comp1['type'] == 'query':
                transformed_args = [mapping.get(p, p) for p in comp1['args']]

                transformed_comp = {
                    'type': 'query',
                    'query': comp1['query'],
                    'args': transformed_args
                }
                transformed_components.append(transformed_comp)

        # Check if each transformed component matches a component in figure2
        for trans_comp in transformed_components:
            found_match = False

            for comp2 in components2:
                if trans_comp['type'] != comp2['type']:
                    continue

                if trans_comp['type'] == 'definition':
                    if (trans_comp['construct'] == comp2['construct'] and
                            set(trans_comp['points']) == set(comp2['points']) and
                            self.is_args_equivalent(trans_comp['construct'], trans_comp['args'], comp2['args'])):
                        found_match = True
                        break

                elif trans_comp['type'] == 'query':
                    if (trans_comp['query'] == comp2['query'] and
                            self.is_args_equivalent(trans_comp['query'], trans_comp['args'], comp2['args'])):
                        found_match = True
                        break

            if not found_match:
                return False

        return True
    
    def check_components_match(self, components1, components2, point_mapping):
        """
        Check if all components in figure1 match components in figure2 under the given point mapping.
        This improved version ensures all components must match exactly, including ensuring
        the exact same number of components in both figures.
        """
        # First, check if the number of components is the same
        if len(components1) != len(components2):
            return False

        # Apply the mapping to components1
        mapped_components = []
        for comp in components1:
            mapped_comp = self.map_component(comp, point_mapping)
            mapped_components.append(mapped_comp)

        # Check if each mapped component matches a component in figure2
        for mapped_comp in mapped_components:
            found_match = False
            for comp2 in components2:
                if mapped_comp['type'] != comp2['type']:
                    continue

                if mapped_comp['type'] == 'definition':
                    if (mapped_comp['construct'] == comp2['construct'] and
                            sorted(mapped_comp['points']) == sorted(comp2['points']) and
                            self.is_args_equivalent(mapped_comp['construct'], mapped_comp['args'], comp2['args'])):
                        found_match = True
                        break
                elif mapped_comp['type'] == 'query':
                    if (mapped_comp['query'] == comp2['query'] and
                            self.is_args_equivalent(mapped_comp['query'], mapped_comp['args'], comp2['args'])):
                        found_match = True
                        break

            if not found_match:
                return False

        # Also check in reverse - every component in figure2 should match a component in mapped_components
        for comp2 in components2:
            found_match = False
            for mapped_comp in mapped_components:
                if mapped_comp['type'] != comp2['type']:
                    continue

                if mapped_comp['type'] == 'definition':
                    if (mapped_comp['construct'] == comp2['construct'] and
                            sorted(mapped_comp['points']) == sorted(comp2['points']) and
                            self.is_args_equivalent(mapped_comp['construct'], mapped_comp['args'], comp2['args'])):
                        found_match = True
                        break
                elif mapped_comp['type'] == 'query':
                    if (mapped_comp['query'] == comp2['query'] and
                            self.is_args_equivalent(mapped_comp['query'], mapped_comp['args'], comp2['args'])):
                        found_match = True
                        break

            if not found_match:
                return False

        return True

    def map_component(self, component, point_mapping):
        """Map the points in a component according to the given point mapping."""
        mapped_component = component.copy()

        if component['type'] == 'definition':
            mapped_component['points'] = [point_mapping.get(p, p) for p in component['points']]
            mapped_component['args'] = [point_mapping.get(p, p) for p in component['args']]

        elif component['type'] == 'query':
            mapped_component['args'] = [point_mapping.get(p, p) for p in component['args']]

        return mapped_component

    def process_geometry_block(self, data_num, content):
        """Process a single geometry block from the input file."""
        if '?' not in content:
            print(f"⚠️ Data {data_num} missing question mark, skipping")
            return

        before_q, after_q = content.split('?', 1)
        # Extract constructs using regex
        constructs = re.findall(self.construct_pattern, before_q)
        construct_ids = sorted(self.construct_to_id[c] for c in constructs if c in self.construct_to_id)

        # Extract target construct
        target_match = re.search(self.target_pattern, '?' + after_q)
        if not target_match:
            print(f"⚠️ Data {data_num} missing target construct, skipping")
            return

        target_keyword = target_match.group(1)
        if target_keyword not in self.construct_to_id:
            print(f"⚠️ Data {data_num}'s target construct {target_keyword} not in definitions")
            return

        target_id = self.construct_to_id[target_keyword]

        # Store the structure and geometry
        key = (tuple(construct_ids), target_id)
        if key not in self.structure_map:
            self.structure_map[key] = []
        self.structure_map[key].append(data_num)
        self.geometry_map[data_num] = content
    
    def get_dataset_path(self, filename):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, "dataset", filename)

    def analyze_input_file(self, input_file):
        """Read and analyze geometry figures from the input CSV file."""
        self.structure_map = defaultdict(list)
        self.geometry_map = {}

        try:
            input_path = self.get_dataset_path(input_file)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(script_dir, "fl_problem.txt")

            with open(input_path, newline='', encoding='utf-8') as infile, \
                open(output_path, 'w', encoding='utf-8') as outfile:
                for i, row in enumerate(infile):
                    data = json.loads(row)
                    fl_problem = data['fl_problem']
                    outfile.write(fl_problem + '\n')
                    self.process_geometry_block(i, fl_problem)
            return True
        except Exception as e:
            print(f"Error reading input file: {e}")
            return False
        
        
    def remove_duplicates_based_on_output(self, input_filename, output_filename, new_filename):
        """Remove duplicate entries from infile based on analysis results."""
        input_path = self.get_dataset_path(input_filename)
        output_path = self.get_dataset_path(output_filename)
        new_path = self.get_dataset_path(new_filename)

        rows = []
        with open(input_path, 'r', newline='', encoding='utf-8') as infile:
            # reader = csv.DictReader(infile)
            # fieldnames = reader.fieldnames
            # for row in reader:
            #     rows.append(row)
            for row in infile:
                data = json.loads(row)
                rows.append(data)

        # Identify IDs to remove
        to_remove = set()
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    dup_id = int(parts[1].strip())
                    to_remove.add(dup_id)

        # Filter remaining rows
        remaining_rows = []
        for new_id, row in enumerate([r for i, r in enumerate(rows) if i not in to_remove]):
            new_row = row.copy()
            new_row['id'] = str(new_id) 
            remaining_rows.append(new_row)

        # Write cleaned CSV (overwrite original)
        # with open(input_path, 'w', newline='', encoding='utf-8') as infile:
        #     writer = csv.DictWriter(infile, fieldnames=fieldnames)
        #     writer.writeheader()
        #     writer.writerows(remaining_rows)
            
        with open(new_path, 'w', encoding='utf-8') as outfile:
            for row in remaining_rows:
                json.dump(row, outfile, ensure_ascii=False)
                outfile.write('\n')

    def find_equivalent_figures(self):
        """Find all pairs of equivalent figures."""

        equivalent_pairs = []
        matched_numbers = set()

        # For each structure group, check pairs within the group
        for structure_key, data_nums in self.structure_map.items():
            if len(data_nums) > 1:
                sorted_nums = sorted(data_nums)
                for i in range(len(sorted_nums)):
                    a = sorted_nums[i]
                    if a in matched_numbers:
                        continue
                    for j in range(i + 1, len(sorted_nums)):
                        b = sorted_nums[j]
                        if b in matched_numbers:
                            continue
                        fig1 = self.geometry_map[a]
                        fig2 = self.geometry_map[b]
                        if self.are_same_figure(fig1, fig2):
                            equivalent_pairs.append((a, b))
                            matched_numbers.add(b)

        return equivalent_pairs

    def write_results_to_file(self, output_file, equivalent_pairs):
        """Write the results to the output file."""
        output_path = self.get_dataset_path(output_file)
        try:
            with open(output_path, "w", encoding="utf-8") as out_file:
                for a, b in equivalent_pairs:
                    out_file.write(f"{a}, {b}\n")
            return True
        except Exception as e:
            print(f"Error writing output file: {e}")
            return False
        

    def run_analysis(self, input_file="input.jsonl", output_file="output.txt", new_file="new.jsonl"):
        """Run the complete analysis process."""
        print(f"Analyzing dataset/{input_file}...")
        if not self.analyze_input_file(input_file):
            return False

        print("Identifying equivalent figures...")
        equivalent_pairs = self.find_equivalent_figures()

        print(f"Saving results to {output_file}...")
        if not self.write_results_to_file(output_file, equivalent_pairs):
            return False

        print("Removing duplicates from original file...")
        self.remove_duplicates_based_on_output(input_file, output_file, new_file)

        print(f"Analysis completed. Found {len(equivalent_pairs)} equivalent pairs.")
        return True
    

if __name__ == "__main__":
    input_filename = "geometry_clauses15_samples10K_aux.jsonl" 
    output_filename = f"analysis_{input_filename.split('.')[0]}_output.txt"

    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
    if len(sys.argv) > 2:
        output_filename = sys.argv[2]

    analyzer = GeometryEquivalenceAnalyzer()
    analyzer.run_analysis(input_filename, output_filename, new_file="new_" + input_filename)
