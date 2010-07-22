from _tools import *
mult = mdp.utils.mult

def testSFANode():
    dim=10000
    freqs = [2*numx.pi*1, 2*numx.pi*5]
    t =  numx.linspace(0,1,num=dim)
    mat = numx.array([numx.sin(freqs[0]*t), numx.sin(freqs[1]*t)]).T
    mat = ((mat - mean(mat[:-1,:], axis=0))
           / std(mat[:-1,:],axis=0))
    des_mat = mat.copy()
    mat = mult(mat,uniform((2,2))) + uniform(2)
    sfa = mdp.nodes.SFANode()
    sfa.train(mat)
    out = sfa.execute(mat)
    correlation = mult(des_mat[:-1,:].T,out[:-1,:])/(dim - 2)
    assert sfa.get_eta_values(t=0.5) is not None, 'get_eta is None'
    assert_array_almost_equal(abs(correlation),
                              numx.eye(2), decimal-3)
    sfa = mdp.nodes.SFANode(output_dim = 1)
    sfa.train(mat)
    out = sfa.execute(mat)
    assert out.shape[1]==1, 'Wrong output_dim'
    correlation = mult(des_mat[:-1,:1].T,out[:-1,:])/(dim - 2)
    assert_array_almost_equal(abs(correlation),
                              numx.eye(1), decimal - 3)

def testSFANode_range_argument():
    node = mdp.nodes.SFANode()
    x = numx.random.random((100,10))
    node.train(x)
    node.stop_training()
    y = node.execute(x, n=5)
    assert y.shape[1] == 5
