"""These are test functions for MDP nodes.

Run them with:
>>> import mdp
>>> mdp.test("nodes")

"""
import unittest
import inspect
import mdp
import cPickle
import tempfile
import os
from mdp import utils, numx, numx_rand, numx_linalg
# !!! scipy.fft seems to be extremely slow, this is a workaround until
# !!! it's fixed
import numpy.fft as numx_fft
from testing_tools import assert_array_almost_equal, assert_array_equal, \
     assert_almost_equal, assert_equal, assert_array_almost_equal_diff, \
     assert_type_equal

mult = utils.mult
mean = numx.mean
std = numx.std
normal = numx_rand.normal
uniform = numx_rand.random
testtypes = [numx.dtype('d'), numx.dtype('f')]
testtypeschar = [t.char for t in testtypes]
testdecimals = {testtypes[0]: 12, testtypes[1]: 6}

def _rand_labels(x):
    return numx.around(uniform(x.shape[0]))

def _rand_labels_array(x):
    return numx.around(uniform(x.shape[0])).reshape((x.shape[0],1))

def _std(x):
    return x.std(axis=0)
    # standard deviation without bias
    mx = mean(x, axis=0)
    mx2 = mean(x*x, axis=0)
    return numx.sqrt((mx2-mx)/(x.shape[0]-1))
    
def _cov(x,y=None):
    #return covariance matrix for x and y
    if y is None: 
        y = x.copy()
    x = x - mean(x,0)
    x = x / _std(x)
    y = y - mean(y,0)
    y = y  / _std(y)
    #return mult(numx.transpose(x),y)/(x.shape[0]-1)
    return mult(numx.transpose(x),y)/(x.shape[0])

class NodesTestSuite(unittest.TestSuite):

    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        # constants
        self.mat_dim = (500,5)
        self.decimal = 7

        # set nodes to be tested
        self._set_nodes()
        
        # get generic tests
        self._generic_test_factory()
        # get FastICA tests
        #self._fastica_test_factory()
        # get nodes tests
        #self._nodes_test_factory()

    def _set_nodes(self):
        mn = mdp.nodes
        self._nodes = [mn.PCANode,
                       mn.WhiteningNode,
                       mn.SFANode,
                       mn.SFA2Node,
                       mn.CuBICANode,
                       mn.FastICANode,
                       mn.QuadraticExpansionNode,
                       (mn.PolynomialExpansionNode, [3], None),
                       (mn.HitParadeNode, [2, 5], None),
                       (mn.TimeFramesNode, [3, 4], None),
                       mn.EtaComputerNode,
                       mn.GrowingNeuralGasNode,
                       mn.NoiseNode,
                       (mn.FDANode, [], _rand_labels),
                       (mn.GaussianClassifierNode, [], _rand_labels),
                       mn.FANode,
                       mn.ISFANode,
                       (mn.RBMNode, [5],None),
                       (mn.RBMWithLabelsNode, [5, 1], _rand_labels_array)]
        
    def _nodes_test_factory(self):
        for methname in dir(self):
            meth = getattr(self,methname)
            if inspect.ismethod(meth) and meth.__name__[:4] == "test":
                # create a nice description
                descr = 'Test '+(meth.__name__[4:]).replace('_',' ')
                self.addTest(unittest.FunctionTestCase(meth,
                             description=descr))

    def _generic_test_factory(self):
        # generate generic test cases
        for node_class in self._nodes:
            if isinstance(node_class, tuple):
                node_class, args, sup_args_func = node_class
            else:
                args = []
                sup_args_func = None

            # generate testdtype_nodeclass test cases
            funcdesc = 'Test dtype consistency of '+node_class.__name__
            testfunc = self._get_testdtype(node_class, args, sup_args_func)
            # add to the suite
            self.addTest(unittest.FunctionTestCase(testfunc,
                                                   description=funcdesc))
            # generate single testinverse_nodeclass test cases
            funcdesc = 'Test inverse function of '+node_class.__name__
            testfunc = self._get_testinverse(node_class, args,
                                             sup_args_func)
            # add to the suite
            if testfunc is not None:
                self.addTest(unittest.FunctionTestCase(testfunc,
                                                       description=funcdesc))
            # generate testoutputdim_nodeclass test cases
            if 'output_dim' in inspect.getargspec(node_class.__init__)[0]:
                funcdesc ='Test output dim consistency of '+node_class.__name__
                testfunc = self._get_testoutputdim(node_class, args,
                                                   sup_args_func)
                # add to the suite
                self.addTest(unittest.FunctionTestCase(testfunc,
                                                       description=funcdesc))
            # generate testdimset_nodeclass test cases
            funcdesc='Test dimensions and dtype settings of '+node_class.__name__
            testfunc = self._get_testdimdtypeset(node_class, args,
                                               sup_args_func)
            # add to the suite
            self.addTest(unittest.FunctionTestCase(testfunc,
                                                   description=funcdesc))
        

    def _fastica_test_factory(self):
        # generate FastICANode testcases
        fica_parm = []
        for approach in ['symm', 'defl']:
            for g in ['pow3', 'tanh', 'gaus', 'skew']:
                for fine_g in [ None, 'pow3', 'tanh', 'gaus', 'skew']:
                    for sample_size in [ 1, 0.99999 ]:
                        for mu in [1, 0.999999 ]:
                            for stab in [False, True]:
                                if mu != 1 and stab is False:
                                    # mu != 1 implies setting stabilization
                                    continue
                                # skew nonlinearity only wroks with skewed
                                # input data
                                if g != 'skew' and fine_g == 'skew':
                                    continue
                                if g == 'skew' and fine_g != 'skew':
                                    continue
                                fica_parm.append({
                                    'approach': approach,
                                    'g': g,
                                    'fine_g': fine_g,
                                    'sample_size' : sample_size,
                                    'mu' : mu,
                                    'stabilization' : stab
                                    })
        for parms in fica_parm:
            testfunc, funcdesc = self._get_testFastICA(parms)
            self.addTest(unittest.FunctionTestCase(testfunc,
                                                   description=funcdesc))
            

    def _get_random_mix(self, mat_dim = None, type = "d", scale = 1,\
                        rand_func = uniform, avg = 0, \
                        std_dev = 1):
        if mat_dim is None: mat_dim = self.mat_dim
        d = 0
        while d < 1E-3:
            #mat = ((rand_func(size=mat_dim)-0.5)*scale).astype(type)
            mat = rand_func(size=mat_dim).astype(type)
            # normalize
            mat -= mean(mat,axis=0)
            mat /= std(mat,axis=0)
            # check that the minimum eigenvalue is finite and positive
            d1 = min(utils.symeig(mult(mat.T, mat), eigenvectors = 0))
            if std_dev is not None: mat *= std_dev
            if avg is not None: mat += avg
            mix = (rand_func(size=(mat_dim[1],mat_dim[1]))*scale).astype(type)
            matmix = mult(mat,mix)
            matmix_n = matmix - mean(matmix, axis=0)
            matmix_n /= std(matmix_n, axis=0)
            d2 = min(utils.symeig(mult(matmix_n.T,matmix_n),eigenvectors=0))
            d = min(d1, d2)
        return mat, mix, matmix

    def _train_if_necessary(self, inp, node, args, sup_args_func):
        if node.is_trainable():
            while True:
                if sup_args_func is not None:
                    # for nodes that need supervision
                    sup_args = sup_args_func(inp)
                    node.train(inp, sup_args)
                else:
                    node.train(inp)
                if node.get_remaining_train_phase() > 1:
                    node.stop_training()
                else:
                    break
                
    def _stop_training_or_execute(self, node, inp):
        if node.is_trainable():
            node.stop_training()
        else:
            out = node(inp)
    
    def _get_testinverse(self, node_class, args=[], sup_args_func=None):
        # generates testinverse_nodeclass test functions
        # only if invertible
        node_args = self._set_node_args(args)
        node = node_class(*node_args)
        if not node.is_invertible():
            return None
        def _testinverse(node_class=node_class):
            mat,mix,inp = self._get_random_mix()
            # take the first available dtype for the test
            node_args = self._set_node_args(args)
            dtype = node_class(*node_args).get_supported_dtypes()[0]
            node = node_class(dtype=dtype, *node_args)
            self._train_if_necessary(inp, node, args, sup_args_func)
            # execute the node
            out = node.execute(inp)
            # compute the inverse
            rec = node.inverse(out)
            # cast inp for comparison!
            inp = inp.astype(dtype)
            assert_array_almost_equal_diff(rec,inp,self.decimal-3)
            assert_type_equal(rec.dtype, dtype)
        return _testinverse

    def _set_node_args(self, args=None):
        # used so that node instantiation arguments can be specified as
        # functions, which can return the real arguments (in case they
        # are immutable objects, so they are recreated each time a node
        # is instantiated)
        node_args = []
        if args is not None:
            for item in args:
                if callable(item):
                    node_args.append(item())
                else:
                    node_args.append(item)
            return node_args

    def _get_testdtype(self, node_class, args=[], sup_args_func=None):
        def _testdtype(node_class=node_class):
            node_args = self._set_node_args(args)
            supported_types = node_class(*node_args).get_supported_dtypes()
            for dtype in supported_types:
                node_args = self._set_node_args(args)
                if node_class == mdp.nodes.SFA2Node:
                    freqs = [2*numx.pi*100.,2*numx.pi*200.]
                    t =  numx.linspace(0, 1, num=1000)
                    mat = numx.array([numx.sin(freqs[0]*t),
                                      numx.sin(freqs[1]*t)]).T
                    inp = mat.astype('d')
                else:
                    mat, mix, inp = self._get_random_mix(type="d")
                node = node_class(*node_args, **{'dtype':dtype})
                self._train_if_necessary(inp, node, node_args, sup_args_func)
                if node_class == mdp.nodes.RBMWithLabelsNode:
                    out = node.execute(inp, sup_args_func(inp))
                else:
                    out = node.execute(inp)
                assert_type_equal(out.dtype, dtype) 
        return _testdtype

    def _get_testoutputdim(self, node_class, args=[], sup_args_func=None):
        def _testoutputdim(node_class=node_class):
            mat,mix,inp = self._get_random_mix()
            output_dim = self.mat_dim[1]/2
            # case 1: output dim set in the constructor
            node_args = self._set_node_args(args)
            node = node_class(*node_args, **{'output_dim':output_dim})
            self._train_if_necessary(inp, node, args, sup_args_func)
            # execute the node
            out = node(inp)
            assert out.shape[1]==output_dim,"%d!=%d"%(out.shape[1],output_dim)
            assert node._output_dim==output_dim,\
                   "%d!=%d"%(node._output_dim,output_dim)
            # case 2: output_dim set explicitly
            node_args = self._set_node_args(args)
            node = node_class(*node_args)
            self._train_if_necessary(inp, node, args, sup_args_func)
            node.output_dim = output_dim
            # execute the node
            out = node(inp)
            assert out.shape[1]==output_dim, "%d!=%d"%(out.shape[1],output_dim)
            assert node._output_dim==output_dim,\
                   "%d!=%d"%(node._output_dim,output_dim)
        return _testoutputdim

    def _get_testdimdtypeset(self, node_class, args=[], sup_args_func=None):
        def _testdimdtypeset(node_class=node_class):
            mat,mix,inp = self._get_random_mix()
            node_args = self._set_node_args(args)
            node = node_class(*node_args)
            self._train_if_necessary(inp, node, args, sup_args_func)
            # execute or stop_training the node
            self._stop_training_or_execute(node, inp)
            assert node.output_dim is not None
            assert node.dtype is not None
            assert node.input_dim is not None
        return _testdimdtypeset

    def _uniform(self, min_, max_, dims):
        return uniform(dims)*(max_-min_)+min_

    def testNodecopy(self):
        test_list = [1,2,3]
        generic_node = mdp.Node()
        generic_node.dummy_attr = test_list
        copy_node = generic_node.copy()
        assert generic_node.dummy_attr == copy_node.dummy_attr,\
               'Node copy method did not work'
        copy_node.dummy_attr[0] = 10
        assert generic_node.dummy_attr != copy_node.dummy_attr,\
               'Node copy method did not work'

    def testNodesave(self):
        test_list = [1,2,3]
        generic_node = mdp.Node()
        generic_node.dummy_attr = test_list
        # test string save
        copy_node_pic = generic_node.save(None)
        copy_node = cPickle.loads(copy_node_pic)
        assert generic_node.dummy_attr == copy_node.dummy_attr,\
               'Node save (string) method did not work'
        copy_node.dummy_attr[0] = 10
        assert generic_node.dummy_attr != copy_node.dummy_attr,\
               'Node save (string) method did not work'
        # test file save
        dummy_file = os.path.join(tempfile.gettempdir(),'removeme')
        generic_node.save(dummy_file, protocol=1)
        flh = open(dummy_file, 'rb')
        copy_node = cPickle.load(flh)
        flh.close()
        os.remove(dummy_file)
        assert generic_node.dummy_attr == copy_node.dummy_attr,\
               'Node save (file) method did not work'
        copy_node.dummy_attr[0] = 10
        assert generic_node.dummy_attr != copy_node.dummy_attr,\
               'Node save (file) method did not work'
        
    def testCovarianceMatrix(self):
        mat,mix,inp = self._get_random_mix()
        des_cov = numx.cov(inp, rowvar=0)
        des_avg = mean(inp,axis=0)
        des_tlen = inp.shape[0]
        act_cov = utils.CovarianceMatrix()
        act_cov.update(inp)
        act_cov,act_avg,act_tlen = act_cov.fix()
        assert_array_almost_equal(act_tlen,des_tlen,self.decimal)
        assert_array_almost_equal(act_avg,des_avg,self.decimal)
        assert_array_almost_equal(act_cov,des_cov,self.decimal)
        
    def testDelayCovarianceMatrix(self):
        dt = 5
        mat,mix,inp = self._get_random_mix()
        des_tlen = inp.shape[0] - dt
        des_avg = mean(inp[:des_tlen,:],axis=0)
        des_avg_dt = mean(inp[dt:,:],axis=0)
        des_cov = utils.cov2(inp[:des_tlen,:], inp[dt:,:])
        act_cov = utils.DelayCovarianceMatrix(dt)
        act_cov.update(inp)
        act_cov,act_avg,act_avg_dt,act_tlen = act_cov.fix()
        assert_array_almost_equal(act_tlen,des_tlen,self.decimal-1)
        assert_array_almost_equal(act_avg,des_avg,self.decimal-1)
        assert_array_almost_equal(act_avg_dt,des_avg_dt,self.decimal-1)
        assert_array_almost_equal(act_cov,des_cov,self.decimal-1)

    def testdtypeCovarianceMatrix(self):
        for type in testtypes:
            mat,mix,inp = self._get_random_mix(type='d')
            cov = utils.CovarianceMatrix(dtype=type)
            cov.update(inp)
            cov,avg,tlen = cov.fix()
            assert_type_equal(cov.dtype,type)
            assert_type_equal(avg.dtype,type) 

    def testdtypeDelayCovarianceMatrix(self):
        for type in testtypes:
            dt = 5
            mat,mix,inp = self._get_random_mix(type='d')
            cov = utils.DelayCovarianceMatrix(dt=dt,dtype=type)
            cov.update(inp)
            cov,avg,avg_dt,tlen = cov.fix()
            assert_type_equal(cov.dtype,type)
            assert_type_equal(avg.dtype,type)
            assert_type_equal(avg_dt.dtype,type)

    def testRoundOffWarningCovMatrix(self):
        import warnings
        warnings.filterwarnings("error",'.*',mdp.MDPWarning)
        for type in ['f','d']:
            inp = uniform((1,2))
            cov = utils.CovarianceMatrix(dtype=type)
            cov._tlen = int(1e+15)
            cov.update(inp)
            try:
                cov.fix()
                assert False, 'RoundOff warning did not work'
            except mdp.MDPWarning:
                pass
        # hope to reset the previous state...
        warnings.filterwarnings("once",'.*',mdp.MDPWarning)

    def testMultipleCovarianceMatricesDtypeAndFuncs(self):
        for type in testtypes:
            dec = testdecimals[type]
            res_type = self._MultipleCovarianceMatrices_funcs(type,dec)
            assert_type_equal(type,res_type)


    def _MultipleCovarianceMatrices_funcs(self,dtype,decimals):
        def assert_all(des,act, dec=decimals):
            # check list of matrices equals multcov array 
            for x in range(nmat):
                assert_array_almost_equal_diff(des[x],act.covs[:,:,x],dec)

        def rotate(mat,angle,indices):
            # perform a givens rotation of a single matrix
            [i,j] = indices
            c, s = numx.cos(angle), numx.sin(angle)
            mat_i, mat_j = mat[:,i].copy(), mat[:,j].copy()
            mat[:,i], mat[:,j] = c*mat_i-s*mat_j, s*mat_i+c*mat_j
            mat_i, mat_j = mat[i,:].copy(), mat[j,:].copy()
            mat[i,:], mat[j,:] = c*mat_i-s*mat_j, s*mat_i+c*mat_j
            return mat.copy()
        
        def permute(mat,indices):
            # permute rows and cols of a single matrix
            [i,j] = indices
            mat_i, mat_j = mat[:,i].copy(), mat[:,j].copy()
            mat[:,i], mat[:,j] = mat_j, mat_i
            mat_i, mat_j = mat[i,:].copy(), mat[j,:].copy()
            mat[i,:], mat[j,:] = mat_j, mat_i
            return mat.copy()

        dim = 7
        nmat = 13
        # create mult cov mat
        covs = [uniform((dim,dim)).astype(dtype) for x in range(nmat)]
        mult_cov = mdp.utils.MultipleCovarianceMatrices(covs)
        assert_equal(nmat,mult_cov.ncovs)
        # test symmetrize
        sym_covs = [0.5*(x+x.T) for x in covs]
        mult_cov.symmetrize()
        assert_all(sym_covs,mult_cov)
        # test weight
        weights = uniform(nmat)
        w_covs = [weights[x]*sym_covs[x] for x in range(nmat)]
        mult_cov.weight(weights)
        assert_all(w_covs,mult_cov)
        # test rotate
        angle = uniform()*2*numx.pi
        idx = numx_rand.permutation(dim)[:2]
        rot_covs = [rotate(x,angle,idx) for x in w_covs]
        mult_cov.rotate(angle,idx)
        assert_all(w_covs,mult_cov)
        # test permute
        per_covs = [permute(x,idx) for x in rot_covs]
        mult_cov.permute(idx)
        assert_all(per_covs,mult_cov)
        # test transform
        trans = uniform((dim,dim))
        trans_covs = [mult(mult(trans.T,x),trans)\
                      for x in per_covs]
        mult_cov.transform(trans)
        assert_all(trans_covs,mult_cov)
        # test copy
        cp_mult_cov = mult_cov.copy()
        assert_array_equal(mult_cov.covs,cp_mult_cov.covs)
        # check that we didn't got a reference
        mult_cov[0][0,0] = 1000
        assert int(cp_mult_cov[0][0,0]) != 1000
        # return dtype 
        return mult_cov.covs.dtype

    def testMultipleCovarianceMatricesTransformations(self):
        def get_mult_covs(inp,nmat):
            # return delayed covariance matrices
            covs = []
            for delay in range(nmat):
                tmp = mdp.utils.DelayCovarianceMatrix(delay)
                tmp.update(inp)
                cov,avg,avg_dt,tlen = tmp.fix()
                covs.append(cov)
            return mdp.utils.MultipleCovarianceMatrices(covs)
        dim = 7
        nmat = 13
        angle = uniform()*2*numx.pi
        idx = numx_rand.permutation(dim)[:2]
        inp = uniform((100*dim,dim))
        rot_inp, per_inp = inp.copy(), inp.copy()
        # test if rotating or permuting the cov matrix is equivalent
        # to rotate or permute the sources.
        mdp.utils.rotate(rot_inp,angle,idx)
        mdp.utils.permute(per_inp,idx,rows=0,cols=1)
        mcov = get_mult_covs(inp, nmat)
        mcov2 = mcov.copy()
        mcov_rot = get_mult_covs(rot_inp, nmat)
        mcov_per = get_mult_covs(per_inp, nmat)
        mcov.rotate(angle,idx)
        mcov2.permute(idx)
        assert_array_almost_equal_diff(mcov.covs, mcov_rot.covs,self.decimal)
        assert_array_almost_equal_diff(mcov2.covs, mcov_per.covs,self.decimal)
                     
    def testPolynomialExpansionNode(self):
        def hardcoded_expansion(x, degree):
            nvars = x.shape[1]
            exp_dim = mdp.nodes._expanded_dim(degree, nvars)
            exp = numx.zeros((x.shape[0], exp_dim), 'd')
            # degree 1
            exp[:,:nvars] = x.copy()
            # degree 2
            k = nvars
            if degree>=2:
                for i in range(nvars):
                    for j in range(i,nvars):
                        exp[:,k] = x[:,i]*x[:,j]
                        k += 1
            # degree 3
            if degree>=3:
                for i in range(nvars):
                    for j in range(i,nvars):
                        for l in range(j,nvars):
                            exp[:,k] = x[:,i]*x[:,j]*x[:,l]
                            k += 1
            # degree 4
            if degree>=4:
                for i in range(nvars):
                    for j in range(i,nvars):
                        for l in range(j,nvars):
                            for m in range(l,nvars):
                                exp[:,k] = x[:,i]*x[:,j]*x[:,l]*x[:,m]
                                k += 1
            # degree 5
            if degree>=5:
                for i in range(nvars):
                    for j in range(i,nvars):
                        for l in range(j,nvars):
                            for m in range(l,nvars):
                                for n in range(m,nvars):
                                    exp[:,k] = \
                                             x[:,i]*x[:,j]*x[:,l]*x[:,m]*x[:,n]
                                    k += 1
            return exp

        for degree in range(1,6):
            for dim in range(1,5):
                expand = mdp.nodes.PolynomialExpansionNode(degree=degree)
                mat,mix,inp = self._get_random_mix((10,dim))
                des = hardcoded_expansion(inp, degree)
                exp = expand.execute(inp)
                assert_array_almost_equal(exp, des, self.decimal)


    def testPCANode(self):
        line_x = numx.zeros((1000,2),"d")
        line_y = numx.zeros((1000,2),"d")
        line_x[:,0] = numx.linspace(-1,1,num=1000,endpoint=1)
        line_y[:,1] = numx.linspace(-0.2,0.2,num=1000,endpoint=1)
        mat = numx.concatenate((line_x,line_y))
        des_var = std(mat,axis=0)
        utils.rotate(mat,uniform()*2*numx.pi)
        mat += uniform(2)
        pca = mdp.nodes.PCANode()
        pca.train(mat)
        act_mat = pca.execute(mat)
        assert_array_almost_equal(mean(act_mat,axis=0),\
                                  [0,0],self.decimal)
        assert_array_almost_equal(std(act_mat,axis=0),\
                                  des_var,self.decimal)
        # test a bug in v.1.1.1, should not crash
        pca.inverse(act_mat[:,:1])

    def testPCANode_desired_variance(self):
        mat, mix, inp = self._get_random_mix(mat_dim=(1000, 3))
        # first make them white
        pca = mdp.nodes.WhiteningNode()
        pca.train(mat)
        mat = pca.execute(mat)
        # set the variances
        mat *= [0.6,0.3,0.1]
        #mat -= mat.mean(axis=0)
        pca = mdp.nodes.PCANode(output_dim=0.8)
        pca.train(mat)
        out = pca.execute(mat)
        # check that we got exactly two output_dim:
        assert pca.output_dim == 2
        assert out.shape[1] == 2
        # check that explained variance is > 0.8 and < 1
        assert (pca.explained_variance > 0.8 and pca.explained_variance < 1)
        
    
    def testPCANode_SVD(self):
        # it should pass atleast the same test as PCANode
        line_x = numx.zeros((1000,2),"d")
        line_y = numx.zeros((1000,2),"d")
        line_x[:,0] = numx.linspace(-1,1,num=1000,endpoint=1)
        line_y[:,1] = numx.linspace(-0.2,0.2,num=1000,endpoint=1)
        mat = numx.concatenate((line_x,line_y))
        des_var = std(mat,axis=0)
        utils.rotate(mat,uniform()*2*numx.pi)
        mat += uniform(2)
        pca = mdp.nodes.PCANode(svd=True)
        pca.train(mat)
        act_mat = pca.execute(mat)
        assert_array_almost_equal(mean(act_mat,axis=0),\
                                  [0,0],self.decimal)
        assert_array_almost_equal(std(act_mat,axis=0),\
                                  des_var,self.decimal)
        # Now a more difficult test, create singular cov matrices
        # and test that PCANode crashes whereas PCASVDNode doesn't
        mat, mix, inp = self._get_random_mix(mat_dim=(1000, 100), avg=1E+8)
        # now create a degenerate input
        for i in range(1,100):
            inp[:,i] = inp[:,1].copy()
        # check that standard PCA fails
        pca = mdp.nodes.PCANode()
        pca.train(inp)
        try:
            pca.stop_training()
            raise Exception, "PCANode didn't catch singular covariance matrix"
        except mdp.NodeException:
            pass
        # now try the SVD version
        pca = mdp.nodes.PCANode(svd=True)
        pca.train(inp)
        pca.stop_training()

        # now check the undetermined case
        mat, mix, inp = self._get_random_mix(mat_dim=(500, 2))
        inp = inp.T
        pca = mdp.nodes.PCANode()
        pca.train(inp)
        try:
            pca.stop_training()
            raise Exception, "PCANode didn't catch singular covariance matrix"
        except mdp.NodeException:
            pass
        # now try the SVD version
        pca = mdp.nodes.PCANode(svd=True)
        pca.train(inp)
        pca.stop_training()

        # try using the automatic dimensionality reduction function
        mat, mix, inp = self._get_random_mix(mat_dim=(1000, 3))
        # first make them decorellated
        pca = mdp.nodes.PCANode()
        pca.train(mat)
        mat = pca.execute(mat)
        mat *= [1E+5,1E-3, 1E-4]
        mat -= mat.mean(axis=0)
        pca = mdp.nodes.PCANode(svd=True,reduce=True, var_rel=1E-2)
        pca.train(mat)
        out = pca.execute(mat)
        # check that we got the only large dimension
        assert_array_almost_equal(mat[:,0].mean(axis=0),out.mean(axis=0),
                                  self.decimal)
        assert_array_almost_equal(mat[:,0].std(axis=0),out.std(axis=0),
                                  self.decimal)

        # second test for automatic dimansionality reduction
        # try using the automatic dimensionality reduction function
        mat, mix, inp = self._get_random_mix(mat_dim=(1000, 3))
        # first make them decorellated
        pca = mdp.nodes.PCANode()
        pca.train(mat)
        mat = pca.execute(mat)
        mat *= [1E+5,1E-3, 1E-18]
        mat -= mat.mean(axis=0)
        pca = mdp.nodes.PCANode(svd=True,reduce=True,
                                   var_abs=1E-8, var_rel=1E-30)
        pca.train(mat)
        out = pca.execute(mat)
        # check that we got the only large dimension
        assert_array_almost_equal(mat[:,:2].mean(axis=0),out.mean(axis=0),
                                  self.decimal)
        assert_array_almost_equal(mat[:,:2].std(axis=0),out.std(axis=0),
                                  self.decimal)
        
    def testWhiteningNode(self):
        vars = 5
        dim = (10000,vars)
        mat,mix,inp = self._get_random_mix(mat_dim=dim,
                                           avg=uniform(vars))
        w = mdp.nodes.WhiteningNode()
        w.train(inp)
        out = w.execute(inp)
        assert_array_almost_equal(mean(out,axis=0),\
                                  numx.zeros((dim[1])),self.decimal)
        assert_array_almost_equal(std(out,axis=0),\
                                  numx.ones((dim[1])),self.decimal-3)

    def testWhiteningNode_SVD(self):
        vars = 5
        dim = (10000,vars)
        mat,mix,inp = self._get_random_mix(mat_dim=dim,
                                           avg=uniform(vars))
        w = mdp.nodes.WhiteningNode(svd=True)
        w.train(inp)
        out = w.execute(inp)
        
        assert_array_almost_equal(mean(out,axis=0),\
                                  numx.zeros((dim[1])),self.decimal)
        assert_array_almost_equal(std(out,axis=0),\
                                  numx.ones((dim[1])),self.decimal-3)

    def testSFANode(self):
        dim=10000
        freqs = [2*numx.pi*1, 2*numx.pi*5]
        t =  numx.linspace(0,1,num=dim)
        mat = numx.array([numx.sin(freqs[0]*t),numx.sin(freqs[1]*t)]).T
        mat = (mat - mean(mat[:-1,:],axis=0))\
              /std(mat[:-1,:],axis=0)
        des_mat = mat.copy()
        mat = mult(mat,uniform((2,2))) + uniform(2)
        sfa = mdp.nodes.SFANode()
        sfa.train(mat)
        out = sfa.execute(mat)
        correlation = mult(des_mat[:-1,:].T,out[:-1,:])/(dim-2)
        assert sfa.get_eta_values(t=0.5) is not None, 'get_eta is None'
        assert_array_almost_equal(abs(correlation),
                                  numx.eye(2), self.decimal-3)
        sfa = mdp.nodes.SFANode(output_dim = 1)
        sfa.train(mat)
        out = sfa.execute(mat)
        assert out.shape[1]==1, 'Wrong output_dim'
        correlation = mult(des_mat[:-1,:1].T,out[:-1,:])/(dim-2)
        assert_array_almost_equal(abs(correlation),
                                  numx.eye(1), self.decimal-3)
        

    def testSFA2Node(self):
        dim = 10000
        freqs = [2*numx.pi*100.,2*numx.pi*500.]
        t =  numx.linspace(0,1,num=dim)
        mat = numx.array([numx.sin(freqs[0]*t),numx.sin(freqs[1]*t)]).T
        mat += normal(0., 1e-10, size=(dim, 2))
        mat = (mat - mean(mat[:-1,:],axis=0))\
              /std(mat[:-1,:],axis=0)
        des_mat = mat.copy()
        mat = mult(mat,uniform((2,2))) + uniform(2)
        sfa = mdp.nodes.SFA2Node()
        sfa.train(mat)
        out = sfa.execute(mat)
        assert out.shape[1]==5, "Wrong output_dim" 
        correlation = mult(des_mat[:-1,:].T,
                           numx.take(out[:-1,:], (0,2), axis=1))/(dim-2)
        assert_array_almost_equal(abs(correlation),
                                  numx.eye(2), self.decimal-3)
        for nr in range(sfa.output_dim):
            qform = sfa.get_quadratic_form(nr)
            outq = qform.apply(mat)
            assert_array_almost_equal(outq, out[:,nr], self.decimal)

        sfa = mdp.nodes.SFANode(output_dim = 2)
        sfa.train(mat)
        out = sfa.execute(mat)
        assert out.shape[1]==2, 'Wrong output_dim'
        correlation = mult(des_mat[:-1,:1].T,out[:-1,:1])/(dim-2)
        assert_array_almost_equal(abs(correlation),
                                  numx.eye(1), self.decimal-3)


    def testSFA2Node_input_dim_bug(self):
        dim = 10000
        freqs = [2*numx.pi*100.,2*numx.pi*500.]
        t =  numx.linspace(0,1,num=dim)
        mat = numx.array([numx.sin(freqs[0]*t),numx.sin(freqs[1]*t)]).T
        mat += normal(0., 1e-10, size=(dim, 2))
        mat = (mat - mean(mat[:-1,:],axis=0))\
              /std(mat[:-1,:],axis=0)
        mat = mult(mat,uniform((2,2))) + uniform(2)
        sfa = mdp.nodes.SFA2Node(input_dim=2)
        sfa.train(mat)
        out = sfa.execute(mat)

    def testSFA2Node_output_dim_bug(self):
        dim = 10000
        freqs = [2*numx.pi*100.,2*numx.pi*500.]
        t =  numx.linspace(0,1,num=dim)
        mat = numx.array([numx.sin(freqs[0]*t),numx.sin(freqs[1]*t)]).T
        mat += normal(0., 1e-10, size=(dim, 2))
        mat = (mat - mean(mat[:-1,:],axis=0))\
              /std(mat[:-1,:],axis=0)
        des_mat = mat.copy()
        mat = mult(mat,uniform((2,2))) + uniform(2)
        sfa = mdp.nodes.SFA2Node(output_dim=3)
        sfa.train(mat)
        out = sfa.execute(mat)
        assert out.shape[1]==3, 'SFA2Node output has wrong output dimensions!'
        
    def _testICANode(self,icanode, rand_func = uniform, vars = 3):
        dim = (8000,vars) 
        mat,mix,inp = self._get_random_mix(rand_func=rand_func,mat_dim=dim)
        icanode.train(inp)
        act_mat = icanode.execute(inp)
        cov = utils.cov2((mat-mean(mat,axis=0))/std(mat,axis=0), act_mat)
        maxima = numx.amax(abs(cov), axis=0)
        assert_array_almost_equal(maxima,numx.ones(vars),3)        

    def _testICANodeMatrices(self, icanode, rand_func = uniform, vars = 3):
        dim = (8000,vars) 
        mat,mix,inp = self._get_random_mix(rand_func=rand_func,
                                           mat_dim=dim, avg = 0)
        icanode.train(inp)
        # test projection matrix
        act_mat = icanode.execute(inp)
        T = icanode.get_projmatrix()
        exp_mat = mult(inp, T)
        assert_array_almost_equal(act_mat,exp_mat,6)
        # test reconstruction matrix
        out = act_mat.copy()
        act_mat = icanode.inverse(out)
        B = icanode.get_recmatrix()
        exp_mat = mult(out, B)
        assert_array_almost_equal(act_mat,exp_mat,6)
        
    def testCuBICANodeBatch(self):
        ica = mdp.nodes.CuBICANode(limit = 10**(-self.decimal))
        ica2 = ica.copy()
        self._testICANode(ica)
        self._testICANodeMatrices(ica2)
        
    def testCuBICANodeTelescope(self):
        ica = mdp.nodes.CuBICANode(limit = 10**(-self.decimal), telescope = 1)
        ica2 = ica.copy()
        self._testICANode(ica)
        self._testICANodeMatrices(ica2)
      
    def _get_testFastICA(self, parms):
        # create a function description
        header = 'TestFastICANode:'
        app =     'Approach:     '+parms['approach']
        nl =      'Nonlinearity: '+parms['g']
        fine_nl = 'Fine-tuning:  '+str(parms['fine_g'])
        if parms['sample_size'] == 1:
            compact = 'Samples  100%, '
        else:
            compact = 'Samples <100%, '
        if parms['mu'] == 1:
            compact = compact + 'Step:  1, '
        else:
            compact = compact + 'Step: <1, '
        if parms['stabilization'] is True:
            compact = compact +'Stabilized algorithm'
        else:
            compact = compact +'Standard   algorithm'
        desc = '\n'.join([header, app, nl, fine_nl, compact])
        def _testFastICA(parms=parms):
            if parms['g'] == 'skew':
                rand_func = numx_rand.exponential
            else:
                rand_func = uniform
                
            # try two times just to clear failures due to randomness
            try:
                ica=mdp.nodes.FastICANode(limit=10**(-self.decimal),**parms)
                ica2 = ica.copy()
                self._testICANode(ica, rand_func=rand_func, vars=2)
                self._testICANodeMatrices(ica2, rand_func=rand_func, vars=2)
            except Exception:
                ica=mdp.nodes.FastICANode(limit=10**(-self.decimal),**parms)
                ica2 = ica.copy()
                self._testICANode(ica, rand_func=rand_func, vars=2)
                self._testICANodeMatrices(ica2, rand_func=rand_func, vars=2)

        return _testFastICA, desc
        

    def testOneDimensionalHitParade(self):
        signal = (uniform(300)-0.5)*2
        gap = 5
        # put some maxima and minima
        signal[0] , signal[10] , signal[50] = 1.5, 1.4, 1.3
        signal[1] , signal[11] , signal[51] = -1.5, -1.4, -1.3
        # put two maxima and two minima within the gap
        signal[100], signal[103] = 2, 3
        signal[110], signal[113] = 3.1, 2
        signal[120], signal[123] = -2, -3.1
        signal[130], signal[133] = -3, -2
        hit = mdp.nodes._OneDimensionalHitParade(5,gap)
        hit.update((signal[:100],numx.arange(100)))
        hit.update((signal[100:200],numx.arange(100,200)))
        hit.update((signal[200:300],numx.arange(200,300)))
        maxima,ind_maxima = hit.get_maxima()
        minima,ind_minima = hit.get_minima()
        assert_array_equal(maxima,[3.1,3,1.5,1.4,1.3])
        assert_array_equal(ind_maxima,[110,103,0,10,50])
        assert_array_equal(minima,[-3.1,-3,-1.5,-1.4,-1.3])
        assert_array_equal(ind_minima,[123,130,1,11,51])

    def testHitParadeNode(self):
        signal = uniform((300,3))
        gap = 5
        signal[10,0], signal[120,1], signal[230,2] = 4,3,2
        signal[11,0], signal[121,1], signal[231,2] = -4,-3,-2
        hit = mdp.nodes.HitParadeNode(1,gap,3)
        hit.train(signal[:100,:])
        hit.train(signal[100:200,:])
        hit.train(signal[200:300,:])
        maxima, max_ind = hit.get_maxima()
        minima, min_ind = hit.get_minima()
        assert_array_equal(maxima,numx.array([[4,3,2]]))
        assert_array_equal(max_ind,numx.array([[10,120,230]]))
        assert_array_equal(minima,numx.array([[-4,-3,-2]]))
        assert_array_equal(min_ind,numx.array([[11,121,231]]))
        # test integer type:
        signal = (uniform((300,3))*10).astype('i')
        gap = 5
        signal[10,0], signal[120,1], signal[230,2] = 40,30,20
        signal[11,0], signal[121,1], signal[231,2] = -40,-30,-20
        hit = mdp.nodes.HitParadeNode(1,gap,3)
        hit.train(signal[:100,:])
        hit.train(signal[100:200,:])
        hit.train(signal[200:300,:])
        maxima, max_ind = hit.get_maxima()
        minima, min_ind = hit.get_minima()
        assert_array_equal(maxima,numx.array([[40,30,20]]))
        assert_array_equal(max_ind,numx.array([[10,120,230]]))
        assert_array_equal(minima,numx.array([[-40,-30,-20]]))
        assert_array_equal(min_ind,numx.array([[11,121,231]]))
        
        
    def testTimeFramesNode(self):
        length = 14
        gap = 6
        time_frames = 3
        inp = numx.array([numx.arange(length), -numx.arange(length)]).T
        # create node to be tested
        tf = mdp.nodes.TimeFramesNode(time_frames,gap)
        out = tf.execute(inp)
        # check last element
        assert_equal(out[-1,-1], -length+1)
        # check horizontal sequence
        for i in range(1,time_frames):
            assert_array_equal(out[:,2*i],out[:,0]+i*gap)
            assert_array_equal(out[:,2*i+1],out[:,1]-i*gap)
        # check pseudo-inverse
        rec = tf.pseudo_inverse(out)
        assert_equal(rec.shape[1], inp.shape[1])
        block_size = min(out.shape[0], gap)
        for i in range(0,length,gap):
            assert_array_equal(rec[i:i+block_size], inp[i:i+block_size])

    def testEtaComputerNode(self):
        tlen = 1e5
        t = numx.linspace(0,2*numx.pi,tlen)
        inp = numx.array([numx.sin(t), numx.sin(5*t)]).T
        # create node to be tested
        ecnode = mdp.nodes.EtaComputerNode()
        ecnode.train(inp)
        #
        etas = ecnode.get_eta(t=tlen)
        # precision gets better with increasing tlen
        assert_array_almost_equal(etas, [1, 5], decimal=4)

    def testGrowingNeuralGasNode(self):
        ### test 1D distribution in a 10D space
        # line coefficients
        dim = 10
        npoints = 1000
        const = self._uniform(-100,100,[dim])
        dir = self._uniform(-1,1,[dim])
        dir /= utils.norm2(dir)
        x = self._uniform(-1,1,[npoints])
        data = numx.outer(x, dir)+const
        # train the gng network
        gng = mdp.nodes.GrowingNeuralGasNode(start_poss=[data[0,:],data[1,:]])
        gng.train(data)
        gng.stop_training()
        # control that the nodes in the graph lie on the line
        poss = gng.get_nodes_position()-const
        norms = numx.sqrt(numx.sum(poss*poss, axis=1))
        poss = (poss.T/norms).T
        assert max(numx.minimum(numx.sum(abs(poss-dir),axis=1),
                                 numx.sum(abs(poss+dir),axis=1)))<1e-7, \
               'At least one node of the graph does lies out of the line.'
        # check that the graph is linear (no additional branches)
        # get a topological sort of the graph
        topolist = gng.graph.topological_sort()
        deg = map(lambda n: n.degree(), topolist)
        assert_equal(deg[:2],[1,1])
        assert_array_equal(deg[2:], [2 for i in range(len(deg)-2)])
        # check the distribution of the nodes' position is uniform
        # this node is at one of the extrema of the graph
        x0 = numx.outer(numx.amin(x, axis=0), dir)+const
        x1 = numx.outer(numx.amax(x, axis=0), dir)+const
        linelen = utils.norm2(x0-x1)
        # this is the mean distance the node should have
        dist = linelen/poss.shape[0]
        # sort the node, depth first
        nodes = gng.graph.undirected_dfs(topolist[0])
        poss = numx.array(map(lambda n: n.data.pos, nodes))
        dists = numx.sqrt(numx.sum((poss[:-1,:]-poss[1:,:])**2, axis=1))
        assert_almost_equal(dist, mean(dists), 1)
        #
        # test the nearest_neighbor function
        start_poss = [numx.asarray([2.,0]), numx.asarray([-2.,0])]
        gng = mdp.nodes.GrowingNeuralGasNode(start_poss=start_poss)
        x = numx.asarray([[2.,0]])
        gng.train(x)
        nodes, dists = gng.nearest_neighbor(numx.asarray([[1.,0]]))
        assert_equal(dists[0],1.)
        assert_array_equal(nodes[0].data.pos,numx.asarray([2,0]))

    def testNoiseNode(self):
        def bogus_noise(mean, size=None):
            return numx.ones(size)*mean

        node = mdp.nodes.NoiseNode(bogus_noise, (1.,))
        out = node.execute(numx.zeros((100,10),'d'))
        assert_array_equal(out, numx.ones((100,10),'d'))
        node = mdp.nodes.NoiseNode(bogus_noise, (1.,), 'multiplicative')
        out = node.execute(numx.zeros((100,10),'d'))
        assert_array_equal(out, numx.zeros((100,10),'d'))

    def testFDANode(self):
        mean1 = [0., 2.]
        mean2 = [0., -2.]
        std_ = numx.array([1., 0.2])
        npoints = 50000
        rot = 45
        
        # input data: two distinct gaussians rotated by 45 deg
        def distr(size): return normal(0, 1., size=(size)) * std_
        x1 = distr((npoints,2)) + mean1
        utils.rotate(x1, rot, units='degrees')
        x2 = distr((npoints,2)) + mean2
        utils.rotate(x2, rot, units='degrees')
        x = numx.concatenate((x1, x2), axis=0)
        
        # labels
        cl1 = numx.ones((x1.shape[0],), dtype='d')
        cl2 = 2.*numx.ones((x2.shape[0],), dtype='d')
        classes = numx.concatenate((cl1, cl2))

        # shuffle the data
        perm_idx = numx_rand.permutation(classes.shape[0])
        x = numx.take(x, perm_idx, axis=0)
        
        classes = numx.take(classes, perm_idx)

        flow = mdp.Flow([mdp.nodes.FDANode()])
        try:
            flow[0].train(x, numx.ones((2,)))
            assert False, 'No exception despite wrong number of labels'
        except mdp.TrainingException:
            pass
        flow.train([[(x, classes)]])
        fda_node = flow[0]

        assert fda_node.tlens[1] == npoints
        assert fda_node.tlens[2] == npoints
        m1 = numx.array([mean1])
        m2 = numx.array([mean2])
        utils.rotate(m1, rot, units='degrees')
        utils.rotate(m2, rot, units='degrees')
        assert_array_almost_equal(fda_node.means[1], m1, 2)
        assert_array_almost_equal(fda_node.means[2], m2, 2)
       
        y = flow.execute(x)
        assert_array_almost_equal(mean(y, axis=0), [0., 0.], self.decimal-2)
        assert_array_almost_equal(std(y, axis=0), [1., 1.], self.decimal-2)
        assert_almost_equal(mult(y[:,0], y[:,1].T), 0., self.decimal-2)

        v1 = fda_node.v[:,0]/fda_node.v[0,0]
        assert_array_almost_equal(v1, [1., -1.], 2)
        v1 = fda_node.v[:,1]/fda_node.v[0,1]
        assert_array_almost_equal(v1, [1., 1.], 2)

    def testGaussianClassifier_train(self):
        nclasses = 10
        dim = 4
        npoints = 10000
        covs = []
        means = []

        node = mdp.nodes.GaussianClassifierNode()
        for i in range(nclasses):
            cov = utils.symrand(dim)
            mn = uniform((dim,))*10.

            x = normal(0., 1., size=(npoints, dim))
            x = mult(x, utils.sqrtm(cov)) + mn
            x = utils.refcast(x, 'd')
            cl = numx.ones((npoints,))*i
            
            mn_estimate = mean(x, axis=0)
            means.append(mn_estimate)
            covs.append(numx.cov(x, rowvar=0))

            node.train(x, cl)
        try:
            node.train(x, numx.ones((2,)))
            assert False, 'No exception despite wrong number of labels'
        except mdp.TrainingException:
            pass

        node.stop_training()

        for i in range(nclasses):
            lbl_idx = node.labels.index(i)
            assert_array_almost_equal_diff(means[i],
                                      node.means[lbl_idx],
                                      self.decimal-1)
            assert_array_almost_equal_diff(utils.inv(covs[i]),
                                      node.inv_covs[lbl_idx],
                                      self.decimal-2)

    def testGaussianClassifier_classify(self):
        mean1 = [0., 2.]
        mean2 = [0., -2.]
        std_ = numx.array([1., 0.2])
        npoints = 100
        rot = 45
        
        # input data: two distinct gaussians rotated by 45 deg
        def distr(size): return normal(0, 1., size=(size)) * std_
        x1 = distr((npoints,2)) + mean1
        utils.rotate(x1, rot, units='degrees')
        x2 = distr((npoints,2)) + mean2
        utils.rotate(x2, rot, units='degrees')
        x = numx.concatenate((x1, x2), axis=0)

        # labels
        cl1 = numx.ones((x1.shape[0],), dtype='d')
        cl2 = 2.*numx.ones((x2.shape[0],), dtype='d')
        classes = numx.concatenate((cl1, cl2))

        # shuffle the data
        perm_idx = numx_rand.permutation(classes.shape[0])
        x = numx.take(x, perm_idx, axis=0)
        classes = numx.take(classes, perm_idx, axis=0)
        
        node = mdp.nodes.GaussianClassifierNode()
        node.train(x, classes)
        classification = node.classify(x)

        assert_array_equal(classes, classification)

    def testFANode(self):
        d = 10
        N = 5000
        k = 4

        mu = uniform((1, d))*3.+2.
        sigma = uniform((d,))*0.01
        #A = utils.random_rot(d)[:k,:]
        A = numx_rand.normal(size=(k,d))

        # latent variables
        y = numx_rand.normal(0., 1., size=(N, k))
        # observations
        noise = numx_rand.normal(0., 1., size=(N, d)) * sigma
        
        x = mult(y, A) + mu + noise
        
        fa = mdp.nodes.FANode(output_dim=k, dtype='d')
        fa.train(x)
        fa.stop_training()

        # compare estimates to real parameters
        assert_array_almost_equal(fa.mu[0,:], mean(x, axis=0), 5)
        assert_array_almost_equal(fa.sigma, std(noise, axis=0)**2, 2)
        # FA finds A only up to a rotation. here we verify that the
        # A and its estimation span the same subspace
        AA = numx.concatenate((A,fa.A.T),axis=0)
        u,s,vh = utils.svd(AA)
        assert sum(s/max(s)>1e-2)==k, \
               'A and its estimation do not span the same subspace'

        y = fa.execute(x)
        fa.generate_input()
        fa.generate_input(10)
        fa.generate_input(y)
        fa.generate_input(y, noise=True)

        # test that noise has the right mean and variance
        est = fa.generate_input(numx.zeros((N, k)), noise=True)
        est -= fa.mu
        assert_array_almost_equal(numx.diag(numx.cov(est, rowvar=0)),
                                  fa.sigma, 3)
        assert_almost_equal(numx.amax(abs(numx.mean(est, axis=0)), axis=None), 0., 3)

        est = fa.generate_input(100000)
        assert_array_almost_equal_diff(numx.cov(est, rowvar=0),
                                       utils.mult(fa.A, fa.A.T), 1)

    def testISFANodeGivensRotations(self):
        ncovs = 5
        dim = 7
        ratio = uniform(2).tolist()
        covs = [uniform((dim,dim)) for j in range(ncovs)]
        covs= mdp.utils.MultipleCovarianceMatrices(covs)
        covs.symmetrize()
        i = mdp.nodes.ISFANode(range(1, ncovs+1),sfa_ica_coeff=ratio,
                               icaweights=uniform(ncovs),
                               sfaweights=uniform(ncovs),
                               output_dim = dim-1, dtype="d")
        i._adjust_ica_sfa_coeff()
        ratio = i._bica_bsfa
        # case 2: only one axis within output space
        # get contrast using internal function
        phi, cont1, min_, dummy =\
                   i._givens_angle_case2(dim-2,dim-1,covs,ratio,complete=1)
        # get contrast using explicit rotations
        cont2 = []
        for angle in phi:
            cp = covs.copy()
            cp.rotate(angle,[dim-2,dim-1])
            cont2.append(numx.sum(i._get_contrast(cp,ratio)))
        assert_array_almost_equal(cont1,cont2,self.decimal)
        # case 1: both axes within output space
        # get contrast using internal function
        phi,cont1, min_ , dummy =\
                   i._givens_angle_case1(0,1,covs,ratio,complete = 1)
        # get contrast using explicit rotations
        cont2 = []
        for angle in phi:
            cp = covs.copy()
            cp.rotate(angle,[0,1])
            cont2.append(numx.sum(i._get_contrast(cp,ratio)))
        assert abs(min_) < numx.pi/4, 'Estimated Minimum out of bounds'
        assert_array_almost_equal(cont1,cont2,self.decimal)

    def testISFANode_SFAPart(self):
        # create independent sources
        mat = uniform((100000,3))*2-1
        fmat = numx_fft.rfft(mat,axis=0)
        # enforce different speeds
        for i in range(3):
            fmat[(i+1)*5000:,i] = 0.
        mat = numx_fft.irfft(fmat,axis=0)
        src = mdp.sfa(mat)
        # test with unmixed signals (i.e. the node should make nothing at all)
        out = mdp.isfa(src,
                       lags=1,
                       whitened=True,
                       sfa_ica_coeff=[1.,0.])
        max_cv = numx.diag(abs(_cov(out,src)))
        assert_array_almost_equal(max_cv, numx.ones((3,)),6)
        # mix linearly the signals
        mix = mult(src,uniform((3,3))*2-1)
        out = mdp.isfa(mix,
                       lags=1,
                       whitened=False,
                       sfa_ica_coeff=[1.,0.])
        max_cv = numx.diag(abs(_cov(out,src)))
        assert_array_almost_equal(max_cv, numx.ones((3,)),6)

    def testISFANode_ICAPart(self):
        # create independent sources
        src = uniform((100000,3))*2-1
        fsrc = numx_fft.rfft(src,axis=0)
        # enforce different speeds
        for i in range(3):
            fsrc[(i+1)*5000:,i] = 0.
        src = numx_fft.irfft(fsrc,axis=0)
        # enforce time-lag-1-independence
        src = mdp.isfa(src, lags=1, sfa_ica_coeff=[1.,0.])
        out = mdp.isfa(src,
                       lags=1,
                       whitened=True,
                       sfa_ica_coeff=[0.,1.])
        max_cv = numx.diag(abs(_cov(out,src)))
        assert_array_almost_equal(max_cv, numx.ones((3,)),5)
        # mix linearly the signals
        mix = mult(src,uniform((3,3))*2-1)
        out = mdp.isfa(mix,
                       lags=1,
                       whitened=False,
                       sfa_ica_coeff=[0.,1.])
        max_cv = numx.diag(abs(_cov(out,src)))
        assert_array_almost_equal(max_cv, numx.ones((3,)),5)

    def testISFANode_3Complete(self):
        # test transition from ica to sfa behavior of isfa
        # use ad hoc sources
        lag = 25
        src = numx.zeros((1001,3),"d")
        idx = [(2,4),(80,1),(2+lag,6)]
        for i in range(len(idx)):
            i0, il = idx[i]
            src[i0:i0+il,i] = 1.
            src[i0+il:i0+2*il,i] = -1.
            src[:,i] -= mean(src[:,i])
            src[:,i] /= std(src[:,i])
        # test extreme cases
        # case 1: ICA
        out = mdp.isfa(src,
                       lags=[1,lag],
                       icaweights=[1.,1.],
                       sfaweights=[1.,0.],
                       output_dim=2,
                       whitened=True,
                       sfa_ica_coeff=[1E-4,1.])
        cv = abs(_cov(src,out))
        idx_cv = numx.argmax(cv,axis=0)
        assert_array_equal(idx_cv,[2,1])
        max_cv = numx.amax(cv,axis=0)
        assert_array_almost_equal(max_cv, numx.ones((2,)),5)
        # case 2: SFA
        out = mdp.isfa(src,
                       lags=[1,lag],
                       icaweights=[1.,1.],
                       sfaweights=[1.,0.],
                       output_dim=2,
                       whitened=True,
                       sfa_ica_coeff=[1.,0.])
        cv = abs(_cov(src,out))
        idx_cv = numx.argmax(cv,axis=0)
        assert_array_equal(idx_cv,[2,0])
        max_cv = numx.amax(cv,axis=0)
        assert_array_almost_equal(max_cv, numx.ones((2,)),5)

    def _ISFA_analytical_solution(self, nsources, nmat, dim, ica_ambiguity):
        # build a sequence of random diagonal matrices
        matrices = [numx.eye(dim, dtype='d')]*nmat
        # build first matrix:
        #   - create random diagonal with elements
        #     in [0, 1]
        diag = uniform(dim)
        #   - sort it in descending order (in absolute value)
        #     [large first]
        diag = numx.take(diag, numx.argsort(abs(diag)))[::-1]
        #   - save larger elements [sfa solution] 
        sfa_solution = diag[:nsources].copy()
        #   - modify diagonal elements order to allow for a
        #     different solution for isfa:
        #     create index array
        idx = range(0,dim)
        #     take the second slowest element and put it at the end
        idx = [idx[0]]+idx[2:]+[idx[1]]
        diag = numx.take(diag, idx)
        #   - save isfa solution
        isfa_solution = diag[:nsources]
        #   - set the first matrix
        matrices[0] = matrices[0]*diag
        # build other matrices
        diag_dim = nsources+ica_ambiguity 
        for i in range(1,nmat):
            # get a random symmetric matrix
            matrices[i] = mdp.utils.symrand(dim)
            # diagonalize the subspace diag_dim
            tmp_diag = (uniform(diag_dim)-0.5)*2
            matrices[i][:diag_dim,:diag_dim] = numx.diag(tmp_diag)
        # put everything in MultCovMat
        matrices = mdp.utils.MultipleCovarianceMatrices(matrices)
        return matrices, sfa_solution, isfa_solution

    def _ISFA_unmixing_error(self, nsources, goal, estimate):
        check = mult(goal[:nsources,:], estimate[:,:nsources])
        error = (abs(numx.sum(numx.sum(abs(check),axis=1)-1))+
                 abs(numx.sum(numx.sum(abs(check),axis=0)-1)))
        error /= nsources*nsources
        return error

    def testISFANode_AnalyticalSolution(self):
        nsources = 2
        # number of time lags
        nmat = 20
        # degree of polynomial expansion
        deg = 3
        # sfa_ica coefficient
        sfa_ica_coeff = [1., 1.]
        # how many independent subspaces in addition to the sources
        ica_ambiguity = 2
        # dimensions of expanded space
        dim = mdp.nodes._expanded_dim(deg, nsources)
        assert (nsources+ica_ambiguity) < dim, 'Too much ica ambiguity.'
        trials = 20
        for trial in range(trials):
            # get analytical solution:
            # prepared matrices, solution for sfa, solution for isf
            covs,sfa_solution,isfa_solution=self._ISFA_analytical_solution(
                nsources,nmat,dim,ica_ambiguity)
            # get contrast of analytical solution
            # sfasrc, icasrc = _get_matrices_contrast(covs, nsources, dim,
            #                                         sfa_ica_coeff)
            # set rotation matrix
            R = mdp.utils.random_rot(dim)
            covs_rot = covs.copy()
            # rotate the analytical solution
            covs_rot.transform(R)
            # find the SFA solution to initialize ISFA
            eigval, SFARP = mdp.utils.symeig(covs_rot.covs[:,:,0])
            # order SFA solution by slowness
            SFARP = SFARP[:,-1::-1]
            # run ISFA
            isfa = mdp.nodes.ISFANode(lags = covs_rot.ncovs, whitened=True,
                                      sfa_ica_coeff = sfa_ica_coeff,
                                      eps_contrast = 1e-7,
                                      output_dim = nsources,
                                      max_iter = 500,
                                      verbose = False,
                                      RP = SFARP)
            isfa.train(uniform((100,dim)))
            isfa.stop_training(covs = covs_rot.copy())
            # check that the rotation matrix found by ISFA is R
            # up to a permutation matrix.
            # Unmixing error as in Tobias paper
            error = self._ISFA_unmixing_error(nsources, R, isfa.RPC)
            if error < 1E-4:
                break
        assert error < 1E-4, 'Not one out of %d trials succeded.'%trials
            
    def testRBMSample_h(self):
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
        for n in range(1000):
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
        for n in range(4):
            distr = h[:,n,:].mean(axis=0)
            assert_array_almost_equal(distr, expected_probs[n,:], 1)

        # ### test 2, with bias
        bm.bh -= 1e4
        h = []
        for n in range(100):
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
        for n in range(4):
            distr = h[:,n,:].mean(axis=0)
            assert_array_almost_equal(distr, expected_probs[n,:], 1)

    def testRBMSample_v(self):
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
        for n in range(100):
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
        for n in range(4):
            distr = v[:,n,:].mean(axis=0)
            assert_array_almost_equal(distr, expected_probs[n,:], 1)

        # test 2, with bias
        bm.bv -= 1e4
        v = []
        for n in range(100):
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
        for n in range(4):
            distr = v[:,n,:].mean(axis=0)
            assert_array_almost_equal(distr, expected_probs[n,:], 1)

    def testRBMStability(self):
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
        for k in range(100):
            p, h = bm._sample_h(v)
            p, v = bm._sample_v(h)

        # see that w remains stable after learning
        delta = (0.,0.,0.)
        for k in range(100):
            err = bm.train(v)
        bm.stop_training()

        assert_array_almost_equal(real_w, bm.w, 1)
        assert_array_almost_equal(real_bv, bm.bv, 1)
        assert_array_almost_equal(real_bh, bm.bh, 1)

    def testRBMLearning(self):
        # number of visible and hidden units
        I, J = 4, 2
        
        bm = mdp.nodes.RBMNode(J, I)
        bm.w = mdp.utils.random_rot(max(I,J), dtype='d')[:I, :J]

        # the observations consist of two disjunct patterns that
        # never appear together
        N=10000
        v = numx.zeros((N,I))
        for n in range(N):
            r = numx_rand.random()
            if r>0.666: v[n,:] = [0,1,0,1]
            elif r>0.333: v[n,:] = [1,0,1,0]

        delta = (0.,0.,0.)
        for k in range(1500):
            mom = 0.9 if k>5 else 0.5
            bm.train(v, epsilon=0.3, momentum=mom)
            if bm._train_err/N<0.1: break
            #print '-------', bm._train_err

        assert bm._train_err/N<0.1

    def testRBMWithLabelsNode(self):
        I, J, L = 4, 4, 2
        bm = mdp.nodes.RBMWithLabelsNode(J,L,I)
        assert bm.input_dim == I+L

        # generate input data
        N = 2500
        v = numx.zeros((2*N,I))
        l = numx.zeros((2*N,L))
        for n in range(N):
            r = numx_rand.random()
            if r>0.666:
                v[n,:] = [0,1,0,0]
                l[n,:] = [1,0]
            elif r>0.333:
                v[n,:] = [1,0,0,0]
                l[n,:] = [1,0]
        for n in range(N):
            r = numx_rand.random()
            if r>0.666:
                v[N+n,:] = [0,0,0,1]
                l[N+n,:] = [0,1]
            elif r>0.333:
                v[N+n,:] = [0,0,1,0]
                l[N+n,:] = [0,1]

        x = numx.concatenate((v, l), axis=1)
        for k in range(1500):
            mom = 0.9 if k>100 else 0.5
            bm.train(v, l, momentum=mom)

            ph, sh = bm._sample_h(x)
            pv, pl, sv, sl = bm._sample_v(sh, concatenate=False)

            v_train_err = float(((v-sv)**2.).sum())
            #print '-------', k, v_train_err/(2*N)
            if v_train_err/(2*N)<0.1: break

        # visible units are reconstructed
        assert v_train_err/(2*N)<0.1
        # units with 0 input have 50/50 labels
        idxzeros = v.sum(axis=1)==0
        nzeros = idxzeros.sum()
        point5 = numx.zeros((nzeros, L)) + 0.5
        assert_array_almost_equal(pl[idxzeros], point5, 2)
        
        
def get_suite():
    return NodesTestSuite()


if __name__ == '__main__':
    numx_rand.seed(1268049219)
    unittest.TextTestRunner(verbosity=2).run(get_suite())

