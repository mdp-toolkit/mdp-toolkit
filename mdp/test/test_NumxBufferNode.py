
import mdp
from mdp.nodes import NumxBufferNode
from _tools import assert_array_equal

def test_NumxBufferNode():
    x = mdp.numx.random.randn(1000,100)

    node = NumxBufferNode(buffer_size=500)
    for i in xrange(1000):
        node.train(x[i:i+1])
    out = node.execute(x[-1:])
    assert_array_equal(out, x[-500:])

    for i in xrange(10):
        node.train(x[i*100:(i+1)*100])
        out = node.execute(x[i*100:(i+1)*100])
    assert_array_equal(out, x[-500:])

    node = NumxBufferNode(buffer_size=10)
    for i in xrange(10):
        node.train(x[i:i + 1])
        out = node(x[i:i + 1])
    assert_array_equal(out[0], x[0])


