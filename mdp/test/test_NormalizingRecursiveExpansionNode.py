"""
Tests for the NormalizingRecursiveExpansionNode.
"""

from mdp.nodes import (RecursiveExpansionNode,
                       NormalizingRecursiveExpansionNode)
from mdp.nodes.recursive_expansion_nodes import recfs
from mdp.test._tools import *
from mdp import numx as np
import py.test


def test_NormalizingRecursiveExpansionNode():
    """Essentially testing the domain transformation."""
    degree = 10
    episodes = 5
    num_obs = 500
    num_vars = 4

    for func_name in recfs:
        x = np.zeros((0, num_vars))
        expn = NormalizingRecursiveExpansionNode(degree, recf=func_name,
                                                 check=True, with0=True)
        for i in range(episodes):
            chunk = (np.random.rand(num_obs, num_vars)-0.5)*1000
            expn.train(chunk)
            x = np.concatenate((x, chunk), axis=0)
        expn.stop_training()
        expn.execute(x)
