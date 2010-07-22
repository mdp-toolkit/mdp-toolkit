from _tools import *
from mdp import numx

def testNoiseNode():
    def bogus_noise(mean, size=None):
        return numx.ones(size)*mean

    node = mdp.nodes.NoiseNode(bogus_noise, (1.,))
    out = node.execute(numx.zeros((100,10),'d'))
    assert_array_equal(out, numx.ones((100,10),'d'))
    node = mdp.nodes.NoiseNode(bogus_noise, (1.,), 'multiplicative')
    out = node.execute(numx.zeros((100,10),'d'))
    assert_array_equal(out, numx.zeros((100,10),'d'))

def testNormalNoiseNode():
    node = mdp.nodes.NormalNoiseNode(noise_args=(2.1, 0.001))
    x = numx.array([range(100), range(100)])
    node.execute(x)

def testNoiseNodePickling():
    node = mdp.nodes.NoiseNode()
    node.copy()
    dummy = node.save(None)
