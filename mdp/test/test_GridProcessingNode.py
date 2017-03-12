
from ._tools import *
from mdp.nodes import GridProcessingNode


def test_grid_processing_node():
    node = GridProcessingNode(grid_lims=[(-8, -10), (8, 10)], n_grid_pts=[11, 11])
    inp = mdp.numx.asarray([[3.2, 4], [-6.4, -4.]])
    for output_type in ['graphx', 'graphindx']:
        node.output_type = output_type
        out = node(inp)
        inp1 = node.inverse(out)
        assert_array_almost_equal(inp, inp1, decimal=10)
