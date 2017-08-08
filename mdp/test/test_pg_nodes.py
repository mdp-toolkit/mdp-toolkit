
from _tools import *
import os

requires_pyqtgraph = skip_on_condition("not mdp.config.has_pyqtgraph","This test requires PyQtGraph")

@requires_pyqtgraph
def test_pg_nodes():
    if os.environ.has_key('DISPLAY')  and (os.environ['DISPLAY'] != ''):
        for dispnode in [mdp.nodes.PG2DNode(), mdp.nodes.PGCurveNode(), mdp.nodes.PGImageNode(img_shapes=(10,10))]:
            x = mdp.numx_rand.randn(1 ,100)
            y = dispnode(x)
            assert dispnode._viewer.is_alive()
            dispnode.stop_rendering()
            assert not dispnode._viewer.is_alive()
            assert_array_equal(x,y)

