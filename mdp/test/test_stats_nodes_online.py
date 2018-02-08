from builtins import range
from mdp.nodes import OnlineCenteringNode, OnlineTimeDiffNode
from ._tools import *


def test_online_centering_node():
    node = OnlineCenteringNode()
    x = mdp.numx_rand.randn(1000, 5) + mdp.numx_rand.uniform(-3,3,5)
    node.train(x)
    assert_array_almost_equal(node.get_average()[0], x.mean(axis=0))

def test_online_centering_node_ema():
    n = 4
    node = OnlineCenteringNode(avg_n=n)
    alpha = 2./(n + 1.)
    x = mdp.numx_rand.randint(0, 10, (n, 4)).astype('float')
    weights = [alpha * mdp.numx.power(1-alpha, t) for t in range(n-1)] + [mdp.numx.power(1-alpha, n-1)]
    ema = (mdp.numx.asarray(weights)[:, None] * x[::-1]).sum(axis=0)
    node.train(x)
    assert_array_almost_equal(node.get_average()[0], ema)

def test_online_time_diff_node_sample():
    node = OnlineTimeDiffNode()
    x = mdp.numx_rand.randn(10,10)
    out=[]
    for i in range(x.shape[0]):
        node.train(x[i:i+1])
        out.append(node.execute(x[i:i+1]))
    assert_array_equal(mdp.numx.asarray(out).squeeze()[1:], x[1:]-x[:-1])

def test_online_time_diff_node_block():
    node = OnlineTimeDiffNode()
    x = mdp.numx_rand.randn(10,10)
    x_0_5 = x[0:6]
    x_5_9 = x[5:]
    node.train(x_0_5)
    out = node.execute(x_5_9)
    assert_array_equal(out, (x[1:]-x[:-1])[-5:])


