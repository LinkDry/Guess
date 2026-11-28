"""Regression checks for the two additional cubic support stages."""
from collections import Counter
from itertools import combinations_with_replacement
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from general_4_9_model import NUMBER_OF_VARIABLES, VARIABLES, basis_blocks
from level3_4_9_model import cubic_basis


class CubicSupportTests(unittest.TestCase):
    def test_new_orbits_and_tail_subgroups(self):
        previous = set(cubic_basis('support4'))
        lower = tuple(word for degree in range(3)
                      for word in combinations_with_replacement(range(NUMBER_OF_VARIABLES), degree))
        expected = {
            5: (13440, [1, 1, 1, 1, 2], (1452, 1342, 1007, 1035, 1007, 1035, 722, 722)),
            6: (3360, [1, 1, 1, 1, 1, 1], (3341, 3209, 2327, 2365, 2327, 2365, 1627, 1623)),
        }
        for support, (count, vertex_degrees, block_sizes) in expected.items():
            words = cubic_basis(f'support{support}')
            new = set(words) - previous
            self.assertEqual(len(new), count)
            for word in new:
                degree = Counter(v for index in word for v in VARIABLES[index][1:])
                self.assertEqual(sorted(degree.values()), vertex_degrees)
            blocks = basis_blocks(lower + tuple(words), True, fixed_vertices=support)
            self.assertEqual(tuple(len(basis) for _, basis in blocks), block_sizes)
            previous.update(new)
