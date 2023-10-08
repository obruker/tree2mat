from unittest import TestCase

import numpy as np

from transformers.tree_transform import tree2matrix, TreeNode


class TreeTransformTest(TestCase):
    def test_that_1_on_1_matrix_is_returned_for_a_single_node_tree(self):
        val = 42
        actual = tree2matrix(root=TreeNode(val), level_sizes=[1])
        np.testing.assert_array_equal(actual, np.array([val]))
