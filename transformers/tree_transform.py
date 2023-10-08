from typing import List

import numpy as np


class TreeNode:
    value: float
    children: List

    def __init__(self, value, children=None):
        self.value = value
        self.children = children or list()


def tree2matrix(root: TreeNode, level_sizes: List[int]) -> np.array:
    """
    :param root: the root of the tree
    :param level_sizes: a list of the sizes of each level (len of this list equals to h-1)
    :return: numpy array
    """
    matrix = np.zeros(level_sizes)
    matrix[0] = root.value
    return matrix
