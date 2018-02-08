from builtins import range
from mdp.nodes import PCANode, WhiteningNode, PolynomialExpansionNode, MCANode
from ._tools import *
import time


def test_mcanode_v1():
    line_x = numx.zeros((1000, 2), "d")
    line_y = numx.zeros((1000, 2), "d")
    line_x[:, 0] = numx.linspace(-1, 1, num=1000, endpoint=1)
    line_y[:, 1] = numx.linspace(-0.2, 0.2, num=1000, endpoint=1)
    mat = numx.concatenate((line_x, line_y))
    utils.rotate(mat, uniform() * 2 * numx.pi)
    mat += uniform(2)
    mat -= mat.mean(axis=0)
    mca = MCANode()
    for i in range(5):
        mca.train(mat)

    bpca = PCANode()
    bpca.train(mat)
    bpca.stop_training()

    v = mca.get_projmatrix()
    bv = bpca.get_projmatrix()[:, ::-1]

    dcosines = numx.zeros(v.shape[1])
    for dim in range(v.shape[1]):
        dcosines[dim] = numx.fabs(numx.dot(v[:, dim], bv[:, dim].T)) / (
            numx.linalg.norm(v[:, dim]) * numx.linalg.norm(bv[:, dim]))
    assert_almost_equal(numx.ones(v.shape[1]), dcosines)


def test_mcanode_v2():
    iterval = 30
    t = numx.linspace(0, 4 * numx.pi, 500)
    x = numx.zeros([t.shape[0], 2])
    x[:, 0] = numx.real(numx.sin(t) + numx.power(numx.cos(11 * t), 2))
    x[:, 1] = numx.cos(11 * t)
    expnode = PolynomialExpansionNode(2)
    input_data = expnode(x)
    input_data = input_data - input_data.mean(axis=0)
    wtnnode = WhiteningNode()
    input_data = wtnnode(input_data)
    input_data = mdp.utils.timediff(input_data)

    ##Setup node/trainer
    output_dim = 4
    node = MCANode(output_dim=output_dim, eps=0.05)

    bpcanode = PCANode()
    bpcanode(input_data)
    # bv = bpcanode.v / numx.linalg.norm(bpcanode.v, axis=0)
    bv = bpcanode.v / mdp.numx.sum(bpcanode.v ** 2, axis=0) ** 0.5
    bv = bv[:, ::-1][:, :output_dim]

    _tcnt = time.time()

    v = []

    for i in range(iterval * input_data.shape[0]):
        node.train(input_data[i % input_data.shape[0]:i % input_data.shape[0] + 1])
        if (node.get_current_train_iteration() % 100 == 0):
            v.append(node.v)

    dcosines = numx.zeros([len(v), output_dim])
    for i in range(len(v)):
        for dim in range(output_dim):
            dcosines[i, dim] = numx.fabs(numx.dot(v[i][:, dim], bv[:, dim].T)) / (
                numx.linalg.norm(v[i][:, dim]) * numx.linalg.norm(bv[:, dim]))

    print('\nTotal Time for {} iterations: {}'.format(iterval, time.time() - _tcnt))
    assert_almost_equal(numx.ones(output_dim), dcosines[-1], decimal=2)


def test_mcanode_numx_rng():
    x = mdp.numx_rand.randn(100, 5)

    numx_rng = mdp.numx_rand.RandomState(seed=10)
    node1 = MCANode(numx_rng=numx_rng)
    node1.train(x)
    init_v1 = node1.init_eigen_vectors
    v1 = node1.get_projmatrix()

    numx_rng = mdp.numx_rand.RandomState(seed=10)
    node2 = MCANode(numx_rng=numx_rng)
    node2.train(x)
    init_v2 = node2.init_eigen_vectors
    v2 = node2.get_projmatrix()

    assert_array_equal(init_v1, init_v2)
    assert_array_equal(v1, v2)
