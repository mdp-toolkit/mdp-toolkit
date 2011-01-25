import mdp
from _tools import *

def test_RBM_sample_h():
    # number of visible and hidden units
    I, J = 2, 4

    # create RBM node
    bm = mdp.nodes.RBMNode(J, I)
    # fake training to initialize internals
    bm.train(numx.zeros((1,I)))
    # init to deterministic model
    bm.w[0,:] = [1,0,1,0]
    bm.w[1,:] = [0,1,0,1]
    bm.w *= 2e4
    bm.bv *= 0.
    bm.bh *= 0.

    # ### test 1
    v = numx.array([[0,0],[1,0],[0,1],[1,1.]])
    h = []
    for n in xrange(1000):
        prob, sample = bm.sample_h(v)
        h.append(sample)

    # check inferred probabilities
    expected_probs = numx.array([[0.5, 0.5, 0.5, 0.5],
                                  [1.0, 0.5, 1.0, 0.5],
                                  [0.5, 1.0, 0.5, 1.0],
                                  [1.0, 1.0, 1.0, 1.0]])
    assert_array_almost_equal(prob, expected_probs, 8)

    # check sampled units
    h = numx.array(h)
    for n in xrange(4):
        distr = h[:,n,:].mean(axis=0)
        assert_array_almost_equal(distr, expected_probs[n,:], 1)

    # ### test 2, with bias
    bm.bh -= 1e2
    h = []
    for n in xrange(100):
        prob, sample = bm.sample_h(v)
        h.append(sample)

    # check inferred probabilities
    expected_probs = numx.array([[0., 0., 0., 0.],
                                  [1.0, 0., 1.0, 0.],
                                  [0., 1.0, 0., 1.0],
                                  [1.0, 1.0, 1.0, 1.0]])
    assert_array_almost_equal(prob, expected_probs, 8)

    # check sampled units
    h = numx.array(h)
    for n in xrange(4):
        distr = h[:,n,:].mean(axis=0)
        assert_array_almost_equal(distr, expected_probs[n,:], 1)

def test_RBM_sample_v():
    # number of visible and hidden units
    I, J = 4, 2

    # create RBM node
    bm = mdp.nodes.RBMNode(J, I)
    # fake training to initialize internals
    bm.train(numx.zeros((1,I)))
    # init to deterministic model
    bm.w[:,0] = [1,0,1,0]
    bm.w[:,1] = [0,1,0,1]
    bm.w *= 2e4
    bm.bv *= 0
    bm.bh *= 0

    # test 1
    h = numx.array([[0,0],[1,0],[0,1],[1,1.]])
    v = []
    for n in xrange(1000):
        prob, sample = bm.sample_v(h)
        v.append(sample)

    # check inferred probabilities
    expected_probs = numx.array([[0.5, 0.5, 0.5, 0.5],
                                  [1.0, 0.5, 1.0, 0.5],
                                  [0.5, 1.0, 0.5, 1.0],
                                  [1.0, 1.0, 1.0, 1.0]])
    assert_array_almost_equal(prob, expected_probs, 8)

    # check sampled units
    v = numx.array(v)
    for n in xrange(4):
        distr = v[:,n,:].mean(axis=0)
        assert_array_almost_equal(distr, expected_probs[n,:], 1)

    # test 2, with bias
    bm.bv -= 1e2
    v = []
    for n in xrange(1000):
        prob, sample = bm.sample_v(h)
        v.append(sample)

    # check inferred probabilities
    expected_probs = numx.array([[0., 0., 0., 0.],
                                  [1.0, 0., 1.0, 0.],
                                  [0., 1.0, 0., 1.0],
                                  [1.0, 1.0, 1.0, 1.0]])
    assert_array_almost_equal(prob, expected_probs, 8)

    # check sampled units
    v = numx.array(v)
    for n in xrange(4):
        distr = v[:,n,:].mean(axis=0)
        assert_array_almost_equal(distr, expected_probs[n,:], 1)

def test_RBM_stability():
    # number of visible and hidden units
    I, J = 8, 2

    # create RBM node
    bm = mdp.nodes.RBMNode(J, I)
    bm._init_weights()
    # init to random model
    bm.w = mdp.utils.random_rot(max(I,J), dtype='d')[:I, :J]
    bm.bv = numx_rand.randn(I)
    bm.bh = numx_rand.randn(J)

    # save original weights
    real_w = bm.w.copy()
    real_bv = bm.bv.copy()
    real_bh = bm.bh.copy()

    # Gibbs sample to reach the equilibrium distribution
    N = 1e4
    v = numx_rand.randint(0,2,(N,I)).astype('d')
    for k in xrange(100):
        if k%5==0: spinner()
        p, h = bm._sample_h(v)
        p, v = bm._sample_v(h)

    # see that w remains stable after learning
    for k in xrange(100):
        if k%5==0: spinner()
        err = bm.train(v)
    bm.stop_training()

    assert_array_almost_equal(real_w, bm.w, 1)
    assert_array_almost_equal(real_bv, bm.bv, 1)
    assert_array_almost_equal(real_bh, bm.bh, 1)

def test_RBM_learning():
    # number of visible and hidden units
    I, J = 4, 2

    bm = mdp.nodes.RBMNode(J, I)
    bm.w = mdp.utils.random_rot(max(I,J), dtype='d')[:I, :J]

    # the observations consist of two disjunct patterns that
    # never appear together
    N = 1e4
    v = numx.zeros((N,I))
    for n in xrange(int(N)):
        r = numx_rand.random()
        if r>0.666: v[n,:] = [0,1,0,1]
        elif r>0.333: v[n,:] = [1,0,1,0]

    for k in xrange(1500):
        if k%5==0: spinner()

        if k>5:
            mom = 0.9
        else:
            mom = 0.5
        bm.train(v, epsilon=0.3, momentum=mom)
        if bm._train_err/N<0.1: break
        #print '-------', bm._train_err

    assert bm._train_err / N < 0.1

def test_RBMWithLabelsNode():
    I, J, L = 4, 4, 2
    bm = mdp.nodes.RBMWithLabelsNode(J,L,I)
    assert bm.input_dim == I+L

    # generate input data
    N = 2500
    v = numx.zeros((2*N,I))
    l = numx.zeros((2*N,L))
    for n in xrange(N):
        r = numx_rand.random()
        if r>0.1:
            v[n,:] = [1,0,1,0]
            l[n,:] = [1,0]
    for n in xrange(N):
        r = numx_rand.random()
        if r>0.1:
            v[n,:] = [0,1,0,1]
            l[n,:] = [1,0]

    x = numx.concatenate((v, l), axis=1)
    for k in xrange(2500):
        if k%5==0: spinner()

        if k>200:
            mom = 0.9
            eps = 0.7
        else:
            mom = 0.5
            eps = 0.2
        bm.train(v, l, epsilon=eps, momentum=mom)

        ph, sh = bm._sample_h(x)
        pv, pl, sv, sl = bm._sample_v(sh, concatenate=False)

        v_train_err = float(((v-sv)**2.).sum())
        #print '-------', k, v_train_err/(2*N)
        if v_train_err / (2*N) < 0.1:
            break

    # visible units are reconstructed
    assert v_train_err / (2*N) < 0.1

    # units with 0 input have 50/50 labels
    idxzeros = v.sum(axis=1)==0
    nzeros = idxzeros.sum()
    point5 = numx.zeros((nzeros, L)) + 0.5
    assert_array_almost_equal(pl[idxzeros], point5, 2)
