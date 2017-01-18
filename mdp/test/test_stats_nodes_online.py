
from mdp.nodes import MovingAvgNode, MovingTimeDiffNode
from ._tools import *


def test_moving_avg_node():
    node = MovingAvgNode()
    x = mdp.numx_rand.randn(1000, 5) + mdp.numx_rand.uniform(-3,3,5)
    node.train(x)
    assert_array_almost_equal(node.avg[0], x.mean(axis=0))

def test_movingtimediffnode():
    node = MovingTimeDiffNode()
    x = mdp.numx_rand.randn(10,10)
    out=[]
    for i in xrange(x.shape[0]):
        node.train(x[i:i+1])
        out.append(node.execute(x[i:i+1]))
    assert_array_equal(mdp.numx.asarray(out).squeeze()[1:], x[1:]-x[:-1])

