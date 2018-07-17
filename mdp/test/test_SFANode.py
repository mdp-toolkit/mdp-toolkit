from __future__ import division
from past.utils import old_div
from ._tools import *
from mdp.utils import mult, symeig
from mdp.signal_node import NodeException

def testSFANode():
    dim=10000
    freqs = [2*numx.pi*1, 2*numx.pi*5]
    t =  numx.linspace(0,1,num=dim)
    mat = numx.array([numx.sin(freqs[0]*t), numx.sin(freqs[1]*t)]).T
    mat = (old_div((mat - mean(mat[:-1,:], axis=0)), std(mat[:-1,:],axis=0)))
    des_mat = mat.copy()
    mat = mult(mat,uniform((2,2))) + uniform(2)
    sfa = mdp.nodes.SFANode()
    sfa.train(mat)
    out = sfa.execute(mat)
    correlation = old_div(mult(des_mat[:-1,:].T,out[:-1,:]),(dim - 2))
    assert sfa.get_eta_values(t=0.5) is not None, 'get_eta is None'
    assert_array_almost_equal(abs(correlation),
                              numx.eye(2), decimal-3)
    sfa = mdp.nodes.SFANode(output_dim = 1)
    sfa.train(mat)
    out = sfa.execute(mat)
    assert out.shape[1]==1, 'Wrong output_dim'
    correlation = old_div(mult(des_mat[:-1,:1].T,out[:-1,:]),(dim - 2))
    assert_array_almost_equal(abs(correlation),
                              numx.eye(1), decimal - 3)

def testSFANode_range_argument():
    node = mdp.nodes.SFANode()
    x = numx.random.random((100,10))
    node.train(x)
    node.stop_training()
    y = node.execute(x, n=5)
    assert y.shape[1] == 5

def testSFANode_one_time_samples():
    # when training with x.shape = (1, n), stop_training
    # was failing with a ValueError: array must not contain infs or NaNs
    # because with only one samples no time difference can be computed and
    # the covmatrix is updated with zeros!
    node = mdp.nodes.SFANode()
    x = numx.random.random((1,5))
    with pytest.raises(mdp.TrainingException):
        node.train(x)

def testSFANode_include_last_sample():
    # check that the default behaviour is True
    node = mdp.nodes.SFANode()
    x = numx.random.random((100,10))
    node.train(x)
    node.stop_training()
    assert node.tlen == 100
    assert node.dtlen == 99

    # check that you can set it explicitly
    node = mdp.nodes.SFANode(include_last_sample=True)
    x = numx.random.random((100,10))
    node.train(x)
    node.stop_training()
    assert node.tlen == 100
    assert node.dtlen == 99

    # check the old behaviour
    node = mdp.nodes.SFANode(include_last_sample=False)
    x = numx.random.random((100,10))
    node.train(x)
    node.stop_training()
    assert node.tlen == 99
    assert node.dtlen == 99

    # check that we can change it during training
    node = mdp.nodes.SFANode(include_last_sample=False)
    x = numx.random.random((100,10))
    node.train(x, include_last_sample=True)
    node.stop_training()
    assert node.tlen == 100
    assert node.dtlen == 99

def testSFANode_rank_deficit():

    def test_for_data(dat, dat0, dfc, out, eq_pr_dict=None, rk_thr_dict=None,
                check_data=None, check_dfc=None):
        if eq_pr_dict is None:
            eq_pr_dict = {'reg': 5, 'pca': 5, 'svd': 5, 'ldl': 5}
        if check_data is None:
            check_data = {'reg': True, 'pca': True, 'svd': True, 'ldl': True}
        if check_dfc is None:
            check_dfc = {'reg': True, 'pca': True, 'svd': True, 'ldl': True}
        if rk_thr_dict is None:
            rk_thr_dict = { \
                    'reg': 1e-10, 'pca': 1e-10, 'svd': 1e-10, 'ldl': 1e-10}
        sfa0 = mdp.nodes.SFANode(output_dim=out)
        sfa0.train(dat0)
        sfa0.stop_training()
        sdat0 = sfa0.execute(dat0)
    
        sfa2_reg = mdp.nodes.SFANode(output_dim=out, rank_deficit_method='reg')
        sfa2_reg.rank_threshold = rk_thr_dict['reg']
        # This is equivalent to sfa2._sfa_solver = sfa2._rank_deficit_solver_reg
        sfa2_reg.train(dat)
        sfa2_reg.stop_training()
        sdat_reg = sfa2_reg.execute(dat)
    
        sfa2_pca = mdp.nodes.SFANode(output_dim=out)
        # For this test we add the rank_deficit_solver later, so we can
        # assert that ordinary SFA would actually fail on the data.
        sfa2_pca.train(dat)
        try:
            sfa2_pca.stop_training()
            # Assert that with dfc > 0 ordinary SFA wouldn't reach this line.
            assert dfc == 0
        except mdp.NodeException:
            sfa2_pca.set_rank_deficit_method('pca')
            sfa2_pca.rank_threshold = rk_thr_dict['pca']
            sfa2_pca.stop_training()
        sdat_pca = sfa2_pca.execute(dat)
        
        sfa2_svd = mdp.nodes.SFANode(output_dim=out, rank_deficit_method='svd')
        sfa2_svd.rank_threshold = rk_thr_dict['svd']
        sfa2_svd.train(dat)
        sfa2_svd.stop_training()
        sdat_svd = sfa2_svd.execute(dat)
    
        def matrix_cmp(A, B):
            assert_array_almost_equal(abs(A), abs(B))
            return True

        if check_data['reg']:
            assert_array_almost_equal(abs(sdat_reg), abs(sdat0),
                    eq_pr_dict['reg'])
        if check_data['pca']:
            assert_array_almost_equal(abs(sdat_pca), abs(sdat0),
                    eq_pr_dict['pca'])
        if check_data['svd']:
            assert_array_almost_equal(abs(sdat_svd), abs(sdat0),
                    eq_pr_dict['svd'])

        reg_dfc = sfa2_reg.rank_deficit == dfc
        pca_dfc = sfa2_pca.rank_deficit == dfc
        svd_dfc = sfa2_svd.rank_deficit == dfc

        if reg_dfc:
            assert_array_almost_equal(
                    sfa2_reg.d, sfa0.d, eq_pr_dict['reg'])
        if pca_dfc:
            assert_array_almost_equal(
                    sfa2_pca.d, sfa0.d, eq_pr_dict['pca'])
        if svd_dfc:
            assert_array_almost_equal(
                    sfa2_svd.d, sfa0.d, eq_pr_dict['svd'])

        # check that constraints are met
        idn = numx.identity(out)
        d_diag = numx.diag(sfa0.d)
        # reg ok?
        assert_array_almost_equal(
                mult(sdat_reg.T, sdat_reg)/(len(sdat_reg)-1),
                idn, eq_pr_dict['reg'])
        sdat_reg_d = sdat_reg[1:]-sdat_reg[:-1]
        assert_array_almost_equal(
                mult(sdat_reg_d.T, sdat_reg_d)/(len(sdat_reg_d)-1),
                d_diag, eq_pr_dict['reg'])
        # pca ok?
        assert_array_almost_equal(
                mult(sdat_pca.T, sdat_pca)/(len(sdat_pca)-1),
                idn, eq_pr_dict['pca'])
        sdat_pca_d = sdat_pca[1:]-sdat_pca[:-1]
        assert_array_almost_equal(
                mult(sdat_pca_d.T, sdat_pca_d)/(len(sdat_pca_d)-1),
                d_diag, eq_pr_dict['pca'])
        # svd ok?
        assert_array_almost_equal(
                mult(sdat_svd.T, sdat_svd)/(len(sdat_svd)-1),
                idn, eq_pr_dict['svd'])
        sdat_svd_d = sdat_svd[1:]-sdat_svd[:-1]
        assert_array_almost_equal(
                mult(sdat_svd_d.T, sdat_svd_d)/(len(sdat_svd_d)-1),
                d_diag, eq_pr_dict['svd'])
    
        try:
            # test ldl separately due to its requirement of SciPy >= 1.0
            sfa2_ldl = mdp.nodes.SFANode(output_dim=out,
                    rank_deficit_method='ldl')
            sfa2_ldl.rank_threshold = rk_thr_dict['ldl']
            have_ldl = True
        except NodeException:
            # No SciPy >= 1.0 available.
            have_ldl = False
        if have_ldl:
            sfa2_ldl.train(dat)
            sfa2_ldl.stop_training()
            sdat_ldl = sfa2_ldl.execute(dat)

            if check_data['ldl']:
                assert_array_almost_equal(abs(sdat_ldl), abs(sdat0),
                        eq_pr_dict['ldl'])
            ldl_dfc =  sfa2_ldl.rank_deficit == dfc
            if ldl_dfc:
                assert_array_almost_equal(
                        sfa2_ldl.d, sfa0.d, eq_pr_dict['ldl'])
    
            # check that constraints are met
            # ldl ok?
            assert_array_almost_equal(
                    mult(sdat_ldl.T, sdat_ldl)/(len(sdat_ldl)-1),
                    idn, eq_pr_dict['ldl'])
            sdat_ldl_d = sdat_ldl[1:]-sdat_ldl[:-1]
            assert_array_almost_equal(
                    mult(sdat_ldl_d.T, sdat_ldl_d)/(len(sdat_ldl_d)-1),
                    d_diag, eq_pr_dict['ldl'])
        else:
            ldl_dfc = None
        ldl_dfc2 = ldl_dfc is True or (ldl_dfc is None and not have_ldl)
        assert all((reg_dfc or not check_dfc['reg'],
                pca_dfc or not check_dfc['pca'],
                svd_dfc or not check_dfc['svd'],
                ldl_dfc2 or not check_dfc['ldl'])), \
                "Rank deficit ok? reg: %s, pca: %s, svd: %s, ldl: %s" % \
                (reg_dfc, pca_dfc, svd_dfc, ldl_dfc)
        return sfa2_pca.d

    # ============test with random data:
    dfc_max = 200
    dat_dim = 500
    dat_smpl = 10000
    dat = numx.random.rand(dat_smpl, dat_dim)     # test data
    dfc = numx.random.randint(0, dfc_max)         # rank deficit
    out = numx.random.randint(4, dat_dim-50-dfc)  # output dim

    # We add some linear redundancy to the data...
    if dfc > 0:
        # dfc is how many dimensions we overwrite with duplicates
        # This should yield an overal rank deficit of dfc
        dat0 = dat[:, :-dfc] # for use by ordinary SFA
        dat[:, -dfc:] = dat[:, :dfc]
    else:
        dat0 = dat

    test_for_data(dat, dat0, dfc, out)
    
    # We mix the redundancy a bit more with the other data and test again...
    if dfc > 0:
        # This should yield an overal rank deficit of dfc
        ovl = numx.random.randint(0, dat_dim-max(out, dfc_max))
        # We generate a random, yet orthogonal matrix M for mixing:
        M = numx.random.rand(dfc+ovl, dfc+ovl)
        _, M = symeig(M+M.T)
        dat[:, -(dfc+ovl):] = dat[:, -(dfc+ovl):].dot(M)
        
        # We test again with mixing matrix applied
        test_for_data(dat, dat0, dfc, out)


    # ============test with nasty data:
    
    # Create another set of data...
    dat = numx.random.rand(dat_smpl, dat_dim)     # test data
    dfc = numx.random.randint(0, dfc_max)         # rank deficit
    out = numx.random.randint(4, dat_dim-50-dfc)  # output dim

    # We add some linear redundancy to the data...
    if dfc > 0:
        # dfc is how many dimensions we overwrite with duplicates
        # This should yield an overal rank deficit of dfc
        dat0 = dat[:, :-dfc] # for use by ordinary SFA
        dat[:, -dfc:] = dat[:, :dfc]
    else:
        dat0 = dat
    
    # And additionally add a very slow actual feature...
    dat[:, dfc] = numx.arange(dat_smpl)

    # We mute some checks here because they sometimes fail
    check_data = {'reg': False, 'pca': False, 'svd': False, 'ldl': False}
    check_dfc = {'reg': False, 'pca': False, 'svd': False, 'ldl': False}
    # Note: In most cases accuracy is much higher than checked here.
    eq_pr_dict = {'reg': 2, 'pca': 2, 'svd': 2, 'ldl': 2}
    rk_thr_dict = {'reg': 1e-8, 'pca': 1e-7, 'svd': 1e-7, 'ldl': 1e-6}
    # Here we assert the very slow but actual feature is not filtered out:
    assert test_for_data(dat, dat0, dfc, out,
            eq_pr_dict, rk_thr_dict, check_data, check_dfc)[0] < 1e-5

    # We mix the redundancy a bit more with the other data and test again...
    if dfc > 0:
        # This should yield an overal rank deficit of dfc
        ovl = numx.random.randint(0, dat_dim-max(out, dfc_max))
        # We generate a random, yet orthogonal matrix M for mixing:
        M = numx.random.rand(dfc+ovl, dfc+ovl)
        _, M = symeig(M+M.T)
        dat[:, -(dfc+ovl):] = dat[:, -(dfc+ovl):].dot(M)

        # We test again with mixing matrix applied
        # Again we assert the very slow but actual feature is not filtered out:
        assert test_for_data(dat, dat0, dfc, out,
            eq_pr_dict, rk_thr_dict, check_data, check_dfc)[0] < 1e-5


def testSFANode_derivative_bug1D():
    # one dimensional worst case scenario
    T = 100
    x = numx.zeros((T,1))
    x[0,:] =  -1.
    x[-1,:] = +1.
    x /= x.std(ddof=1)
    sfa = mdp.nodes.SFANode(include_last_sample=True)
    sfa.train(x)
    sfa.stop_training(debug=True)
    xdot = sfa.time_derivative(x)
    tlen = xdot.shape[0]
    correct_dcov_mtx = old_div((xdot*xdot).sum(),(tlen-1))
    sfa_dcov_mtx = sfa.dcov_mtx
    # quantify the error
    error = abs(correct_dcov_mtx-sfa_dcov_mtx)[0,0]
    assert error < 10**(-decimal)
    # the bug was that we were calculating the covariance matrix
    # of the derivative, i.e.
    # sfa_dcov-mtx = (xdot*xdot).sum()/(tlen-1) - xdot.sum()**2/(tlen*(tlen-1))
    # so that the error in the estimated matrix was exactly
    # xdot.sum()**2/(tlen*(tlen-1))

def testSFANode_derivative_bug2D():
    T = 100
    x = numx.zeros((T,2))
    x[0,0] =  -1.
    x[-1,0] = +1.
    x[:,1] = numx.arange(T)
    x -= x.mean(axis=0)
    x /= x.std(ddof=1, axis=0)    
    sfa = mdp.nodes.SFANode(include_last_sample=True)
    sfa.train(x)
    sfa.stop_training(debug=True)
    xdot = sfa.time_derivative(x)
    tlen = xdot.shape[0]
    correct_dcov_mtx = old_div(mdp.utils.mult(xdot.T, xdot),(tlen-1))
    sfa_dcov_mtx = sfa.dcov_mtx
    # the bug was that we were calculating the covariance matrix
    # of the derivative, i.e.
    # sfa_dcov_mtx = mdp.utils.mult(xdot.T, xdot)/(tlen-1) - \
    #                numx.outer(xdot.sum(axis=0),
    #                           xdot.sum(axis=0))/(tlen*(tlen-1)))
    # so that the error in the estimated matrix was exactly
    # numx.outer(xdot.sum(axis=0),xdot.sum(axis=0))/(tlen*(tlen-1))
    error = abs(correct_dcov_mtx-sfa_dcov_mtx)
    assert_array_almost_equal(numx.zeros(error.shape), error, decimal)

def testSFANode_derivative_bug2D_eigen():
    # this is a copy of the previous test, where we
    # quantify the error in the estimated eigenvalues
    # and eigenvectors
    T = 100
    x = numx.zeros((T,2))
    x[0,0] =  -1.
    x[-1,0] = +1.
    x[:,1] = numx.arange(T)
    x -= x.mean(axis=0)
    x /= x.std(ddof=1, axis=0)    
    sfa = mdp.nodes.SFANode(include_last_sample=True)
    sfa.train(x)
    sfa.stop_training(debug=True)
    xdot = sfa.time_derivative(x)
    tlen = xdot.shape[0]
    correct_dcov_mtx = old_div(mdp.utils.mult(xdot.T, xdot),(tlen-1))
    eigvalues, eigvectors = sfa._symeig(correct_dcov_mtx,
                                        sfa.cov_mtx,
                                        range=None,
                                        overwrite=False)
    assert_array_almost_equal(eigvalues, sfa.d, decimal)
    assert_array_almost_equal(eigvectors, sfa.sf, decimal)
