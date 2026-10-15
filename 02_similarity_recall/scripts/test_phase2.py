import unittest
from pathlib import Path
import sys
import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_experiment import exact_similarity, top_neighbors


class Phase2Tests(unittest.TestCase):
    def test_exact_cosine_and_self_exclusion(self):
        matrix = sparse.csr_matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float32)
        sim, meta = exact_similarity(matrix)
        self.assertEqual(sim.diagonal().sum(), 0)
        np.testing.assert_allclose(sim[0, 1], 0.5)
        np.testing.assert_allclose(sim[0, 2], 0.5)
        self.assertEqual(meta["nnz"], 6)

    def test_neighbor_tie_uses_item_id(self):
        row = sparse.csr_matrix(([0.5, 0.5], ([0, 0], [3, 1])), shape=(1, 5))
        ids, values = top_neighbors(row, 2)
        np.testing.assert_array_equal(ids, [1, 3])
        np.testing.assert_allclose(values, [0.5, 0.5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
