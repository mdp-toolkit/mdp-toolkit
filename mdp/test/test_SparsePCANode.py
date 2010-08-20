from _tools import *

def testSparsePCANode_classic():
    # this is the same test as for PCANode, adjusted to work
    # with sparse matrices
    
    N = 4
    line_x = numx.zeros((1000,N),"d")
    line_y = numx.zeros((1000,N),"d")
    line_x[:,0] = numx.linspace(-1,1,num=1000,endpoint=1)
    line_y[:,1] = numx.linspace(-0.2,0.2,num=1000,endpoint=1)
    
    # add random constant and rotate
    x = numx.concatenate((line_x,line_y))
    des_var = std(x, axis=0)
    utils.rotate(x, uniform()*2*numx.pi)
    x += uniform(N)
    
    # transform in 
    mat = mdp.numx.sparse.lil_matrix(x)
    
    pca = mdp.nodes.SparsePCANode(output_dim=2)
    pca.train(mat)
    
    act_mat = numx.array(pca.execute(mat))
    assert_array_almost_equal(mean(act_mat, axis=0),\
                              [0,0],decimal)
    assert_array_almost_equal(std(act_mat,axis=0),\
                              des_var[:2],decimal)
                              
    # test that the total_variance attribute makes sense
    est_tot_var = ((des_var**2)*2000/1999.).sum()
    assert_almost_equal(est_tot_var, pca.total_variance, decimal)
    assert_almost_equal(1, pca.explained_variance, decimal)
    
    print (des_var**2)*2000/1999.
    print pca.d
    
    # test a bug in v.1.1.1, should not crash
    pca.inverse(act_mat[:,:1])

def test_SparsePCANode_large():
    """Test SparsePCA with too-large-to-be-dense matrix."""
    N = 400
    x = mdp.numx.sparse.lil_matrix((2*N, N))
    desired_var = numx.empty((N,))
    for i in range(N):
        x[2*i,i] = float(i)+1.
        x[2*i+1,i] = -float(i)-1.
        desired_var[i] = 2*((i+1.) ** 2.) / (2.*N - 1.)
    
    # FIXME: output of pcanode whould also be sparse; what else is not sparse?
    pcanode = mdp.nodes.SparsePCANode(output_dim=3)
    pcanode.train(x)
    pcanode.stop_training()

    assert_almost_equal(numx.flipud(desired_var[-3:]), pcanode.d, decimal)
