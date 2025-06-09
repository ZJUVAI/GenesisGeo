import unittest
from translate import translate_constrained_to_constructive

class TestTranslateConstrainedToConstructive(unittest.TestCase):
    def test_perp(self):
        # 测试点在c,d
        self.assertEqual(
            translate_constrained_to_constructive('C', 'perp', ['A', 'B', 'C', 'D']),
            ('on_tline', ['C', 'D', 'A', 'B'])
        )
        # 测试点是b
        self.assertEqual(
            translate_constrained_to_constructive('B', 'perp', ['A', 'B', 'C', 'D']),
            ('on_tline', ['B', 'A', 'C', 'D'])
        )
        # 测试点是d
        self.assertEqual(
            translate_constrained_to_constructive('D', 'perp', ['A', 'B', 'C', 'D']),
            ('on_tline', ['D', 'C', 'A', 'B'])
        )
        # 测试on_dia分支
        self.assertEqual(
            translate_constrained_to_constructive('A', 'perp', ['A', 'B', 'A', 'D']),
            ('on_dia', ['A', 'D', 'B'])
        )
        # 测试on_dia分支
        self.assertEqual(
            translate_constrained_to_constructive('D', 'perp', ['A', 'D', 'D', 'B']),
            ('on_dia', ['D', 'B', 'A'])
        )

    def test_para(self):
        # 点在c,d
        self.assertEqual(
            translate_constrained_to_constructive('C', 'para', ['A', 'B', 'C', 'D']),
            ('on_pline', ['C', 'D', 'A', 'B'])
        )
        # 点是b
        self.assertEqual(
            translate_constrained_to_constructive('B', 'para', ['A', 'B', 'C', 'D']),
            ('on_pline', ['B', 'A', 'C', 'D'])
        )

    def test_cong(self):
        # 点在c,d
        self.assertEqual(
            translate_constrained_to_constructive('C', 'cong', ['A', 'B', 'C', 'D']),
            ('eqdistance', ['C', 'D', 'A', 'B'])
        )
        # 点是b
        self.assertEqual(
            translate_constrained_to_constructive('B', 'cong', ['A', 'B', 'C', 'D']),
            ('eqdistance', ['B', 'A', 'C', 'D'])
        )
        # 点是d
        self.assertEqual(
            translate_constrained_to_constructive('D', 'cong', ['A', 'B', 'C', 'D']),
            ('eqdistance', ['D', 'C', 'A', 'B'])
        )
        # on_bline分支
        self.assertEqual(
            translate_constrained_to_constructive('A', 'cong', ['A', 'B', 'A', 'D']),
            ('on_bline', ['A', 'D', 'B'])
        )
        # eqdistance分支
        self.assertEqual(
            translate_constrained_to_constructive('E', 'cong', ['A', 'B', 'C', 'D']),
            ('eqdistance', ['A', 'B', 'C', 'D'])
        )

    def test_coll(self):
        # 点是b
        self.assertEqual(
            translate_constrained_to_constructive('B', 'coll', ['A', 'B', 'C']),
            ('on_line', ['B', 'A', 'C'])
        )
        # 点是c
        self.assertEqual(
            translate_constrained_to_constructive('C', 'coll', ['A', 'B', 'C']),
            ('on_line', ['C', 'A', 'B'])
        )
        # 普通情况
        self.assertEqual(
            translate_constrained_to_constructive('A', 'coll', ['A', 'B', 'C']),
            ('on_line', ['A', 'B', 'C'])
        )

    def test_eqangle(self):
        # 点在d,e,f
        self.assertEqual(
            translate_constrained_to_constructive('D', 'eqangle', ['A', 'B', 'C', 'D', 'E', 'F']),
            ('on_aline', ['D', 'E', 'F', 'A', 'B', 'C'])
        )
        # 其它分支
        self.assertEqual(
            translate_constrained_to_constructive('B', 'eqangle', ['A', 'B', 'C', 'D', 'E', 'F']),
            ('eqangle3', ['B', 'A', 'C', 'E', 'D', 'F'])
        )

    def test_cyclic(self):
        self.assertEqual(
            translate_constrained_to_constructive('A', 'cyclic', ['A', 'B', 'C', 'D']),
            ('on_circum', ['A', 'B', 'C', 'D'])
        )
        self.assertEqual(
            translate_constrained_to_constructive('B', 'cyclic', ['A', 'B', 'C', 'D']),
            ('on_circum', ['B', 'A', 'C', 'D'])
        )

    def test_default(self):
        self.assertEqual(
            translate_constrained_to_constructive('A', 'other', ['A', 'B']),
            ('other', ['A', 'B'])
        )

if __name__ == '__main__':
    unittest.main()