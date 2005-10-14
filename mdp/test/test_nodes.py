"""These are test functions for MDP nodes.

Run them with:
>>> import mdp
>>> mdp.test.test("nodes")

"""
import unittest
import inspect
import types
import mdp
from testing_tools import assert_array_almost_equal, assert_array_equal, \
     assert_almost_equal, assert_equal, assert_array_almost_equal_diff

mult = mdp.utils.mult
numx = mdp.numx
numx_rand = mdp.numx_rand
mean = mdp.utils.mean
std = mdp.utils.std
covariance = mdp.utils.cov
tr = numx.transpose
testtypes = ['d', 'f']
testdecimals = {'d':16, 'f':7}

def assert_type_equal(act,des):
    assert act == des,' Typecode mismatch: "%s" (should be "%s") '%(act,des)

def _cov(x,y=None):
    y = y or x.copy()
    x = x - mean(x,0)
    x = x / std(x,0)
    y = y - mean(y,0)
    y = y  / std(y,0)
    return mult(tr(x),y)/(x.shape[0]-1)
    
class NodesTestSuite(unittest.TestSuite):

    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        # constants
        self.mat_dim = (100,5)
        self.decimal = 7
        mn = mdp.nodes
        self._nodes = [mn.PCANode,
                       mn.WhiteningNode,
                       mn.SFANode,
                       mn.CuBICANode,
                       mn.FastICANode,
                       mn.QuadraticExpansionNode,
                       (mn.PolynomialExpansionNode, [3]),
                       (mn.HitParadeNode, [2, 5]),
                       (mn.TimeFramesNode, [3, 4]),
                       mn.EtaComputerNode,
                       mn.GrowingNeuralGasNode,
                       mn.NoiseNode]

        # generate generic test cases
        for node_class in self._nodes:
            if type(node_class) is types.TupleType:
                node_class, args = node_class
            else:
                args = []

            # generate single testinverse_nodeclass test cases
            funcdesc = 'Test inverse function of '+node_class.__name__
            testfunc = self._get_testinverse(node_class, args)
            # add to the suite
            self.addTest(unittest.FunctionTestCase(testfunc,
                                                   description=funcdesc))
            # generate testtypecode_nodeclass test cases
            funcdesc = 'Test typecode consistency of '+node_class.__name__
            testfunc = self._get_testtypecode(node_class, args)
            # add to the suite
            self.addTest(unittest.FunctionTestCase(testfunc,
                                                   description=funcdesc))
            # generate testoutputdim_nodeclass test cases
            if hasattr(node_class, 'set_output_dim'):
                funcdesc = 'Test output dim consistency of '+node_class.__name__
                testfunc = self._get_testoutputdim(node_class, args)
                # add to the suite
                self.addTest(unittest.FunctionTestCase(testfunc,
                                                   description=funcdesc))
            
        for methname in dir(self):
            meth = getattr(self,methname)
            if inspect.ismethod(meth) and meth.__name__[:4] == "test":
                self.addTest(unittest.FunctionTestCase(meth))

    def _get_random_mix(self, mat_dim = None, type = "d", scale = 1,\
                        rand_func = numx_rand.random, avg = None, \
                        std_dev = None):
        if mat_dim == None: mat_dim = self.mat_dim
        mat = ((rand_func(mat_dim)-0.5)*scale).astype(type)
        mat -= mean(mat,axis=0)
        mat /= std(mat,axis=0)
        if std_dev is not None: mat *= std_dev
        if avg is not None: mat += avg
        mix = (rand_func((mat_dim[1],mat_dim[1]))*scale).astype(type)
        return mat,mix,mult(mat,mix)
    
    def _get_testinverse(self, node_class, args=[]):
        # generates testinverse_nodeclass test functions
        def _testinverse(node_class=node_class):
            mat,mix,inp = self._get_random_mix()
            node = node_class(*args)
            if not node.is_invertible():return
            if node.is_trainable(): node.train(inp)
            # execute the node
            out = node.execute(inp)
            # compute the inverse
            rec = node.inverse(out) 
            assert_array_almost_equal(rec,inp,self.decimal)
        return _testinverse

    def _get_testtypecode(self, node_class, args=[]):
        def _testtypecode(node_class=node_class):
            for typecode in testtypes:
                mat,mix,inp = self._get_random_mix(type="d")
                node = node_class(*args,**{'typecode':typecode})
                if node.is_trainable(): node.train(inp)
                out = node.execute(inp)
                assert_type_equal(out.typecode(),typecode) 
        return _testtypecode

    def _get_testoutputdim(self, node_class, args=[]):
        def _testoutputdim(node_class=node_class):
            mat,mix,inp = self._get_random_mix()
            output_dim = self.mat_dim[1]/2
            # case 1: output dim set in the constructor
            node = node_class(*args, **{'output_dim':output_dim})
            if node.is_trainable(): node.train(inp)
            # execute the node
            out = node(inp)
            assert out.shape[1]==output_dim
            assert node._output_dim==output_dim
            # case 2: output_dim set explicitly
            node = node_class(*args)
            if node.is_trainable(): node.train(inp)
            node.set_output_dim(output_dim)
            # execute the node
            out = node(inp)
            assert out.shape[1]==output_dim
            assert node._output_dim==output_dim
        return _testoutputdim

    def _uniform(self, min_, max_, dims):
        return numx_rand.random(dims)*(max_-min_)+min_

    def testSignalNodecopy(self):
        test_list = [1,2,3]
        generic_node = mdp.SignalNode()
        generic_node.dummy_attr = test_list
        copy_node = generic_node.copy()
        assert generic_node.dummy_attr == copy_node.dummy_attr,\
               'SignalNode copy method did not work'
        copy_node.dummy_attr[0] = 10
        assert generic_node.dummy_attr != copy_node.dummy_attr,\
               'SignalNode copy method did not work'
        
    def testCovarianceMatrix(self):
        mat,mix,inp = self._get_random_mix()
        des_cov = covariance(inp)
        des_avg = mean(inp,axis=0)
        des_tlen = inp.shape[0]
        act_cov = mdp.nodes.lcov.CovarianceMatrix()
        act_cov.update(inp)
        act_cov,act_avg,act_tlen = act_cov.fix()
        assert_array_almost_equal(act_tlen,des_tlen,self.decimal)
        assert_array_almost_equal(act_avg,des_avg,self.decimal)
        assert_array_almost_equal(act_cov,des_cov,self.decimal)
        
    def testDelayCovarianceMatrix(self):
        dt = 5
        mat,mix,inp = self._get_random_mix()
        des_tlen = inp.shape[0] - dt
        des_cov = covariance(inp[:des_tlen,:],inp[dt:,:])
        des_avg = mean(inp[:des_tlen,:],axis=0)
        des_avg_dt = mean(inp[dt:,:],axis=0)
        act_cov = mdp.nodes.lcov.DelayCovarianceMatrix(dt)
        act_cov.update(inp)
        act_cov,act_avg,act_avg_dt,act_tlen = act_cov.fix()
        assert_array_almost_equal(act_tlen,des_tlen,self.decimal)
        assert_array_almost_equal(act_avg,des_avg,self.decimal)
        assert_array_almost_equal(act_avg_dt,des_avg_dt,self.decimal)
        assert_array_almost_equal(act_cov,des_cov,self.decimal)

    def testtypecodeCovarianceMatrix(self):
        for type in testtypes:
            mat,mix,inp = self._get_random_mix(type='d')
            cov = mdp.nodes.lcov.CovarianceMatrix(typecode=type)
            cov.update(inp)
            cov,avg,tlen = cov.fix()
            assert_type_equal(cov.typecode(),type)
            assert_type_equal(avg.typecode(),type) 

    def testtypecodeDelayCovarianceMatrix(self):
        for type in testtypes:
            dt = 5
            mat,mix,inp = self._get_random_mix(type='d')
            cov = mdp.nodes.lcov.DelayCovarianceMatrix(dt=dt,typecode=type)
            cov.update(inp)
            cov,avg,avg_dt,tlen = cov.fix()
            assert_type_equal(cov.typecode(),type)
            assert_type_equal(avg.typecode(),type)
            assert_type_equal(avg_dt.typecode(),type)

    def testRoundOffWarningCovMatrix(self):
        import warnings
        warnings.filterwarnings("error",'.*',mdp.MDPWarning)
        for type in ['d','f']:
            inp = numx_rand.random((1,2))
            cov = mdp.nodes.lcov.CovarianceMatrix(typecode=type)
            cov._tlen = int(1e+15)
            cov.update(inp)
            try:
                cov.fix()
                raise Exception, 'RoundOff warning did not work'
            except mdp.MDPWarning:
                pass
        # hope to reset the previous state...
        warnings.filterwarnings("once",'.*',mdp.MDPWarning)

    def testPolynomialExpansionNode(self):
        def hardcoded_expansion(x, degree):
            nvars = x.shape[1]
            exp_dim = mdp.nodes.expansion_nodes.expanded_dim(degree, nvars)
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
        line_x[:,0] = mdp.utils.linspace(-1,1,num=1000,endpoint=1)
        line_y[:,1] = mdp.utils.linspace(-0.2,0.2,num=1000,endpoint=1)
        mat = numx.concatenate((line_x,line_y))
        des_var = std(mat,axis=0)
        mdp.utils.rotate(mat,numx_rand.random()*2*numx.pi)
        mat += numx_rand.random(2)
        pca = mdp.nodes.PCANode()
        pca.train(mat)
        act_mat = pca.execute(mat)
        assert_array_almost_equal(mean(act_mat,axis=0),\
                                  [0,0],self.decimal)
        assert_array_almost_equal(std(act_mat,axis=0),\
                                  des_var,self.decimal)
        # test a bug in v.1.1.1, should not crash
        pca.inverse(act_mat[:,:1])

    def testWhiteningNode(self):
        vars = 5
        dim = (10000,vars)
        mat,mix,inp = self._get_random_mix(mat_dim=dim,avg=numx_rand.random(vars))
        w = mdp.nodes.WhiteningNode()
        w.train(inp)
        out = w.execute(inp)
        assert_array_almost_equal(mean(out,axis=0),\
                                  numx.zeros((dim[1])),self.decimal)
        assert_array_almost_equal(std(out,axis=0)**2,\
                                  numx.ones((dim[1])),self.decimal)

    def testSFANode(self):
        dim=10000
        freqs = [2*numx.pi*1,2*numx.pi*5]
        t =  mdp.utils.linspace(0,1,num=dim)
        mat = tr(numx.array(\
            [numx.sin(freqs[0]*t),numx.sin(freqs[1]*t)]))
        mat = (mat - mean(mat[:-1,:],axis=0))\
              /std(mat[:-1,:],axis=0)
        des_mat = mat.copy()
        mat = mult(mat,numx_rand.random((2,2))) + numx_rand.random(2)
        sfa = mdp.nodes.SFANode()
        sfa.train(mat)
        out = sfa.execute(mat)
        correlation = mult(tr(des_mat[:-1,:]),out[:-1,:])/(dim-2)
        assert_array_almost_equal(abs(correlation),
                                  mdp.utils.eye(2),self.decimal)

    def _testICANode(self,icanode):
        vars = 3
        dim = (8000,vars) 
        mat,mix,inp = self._get_random_mix(mat_dim=dim)
        icanode.train(inp)
        act_mat = icanode.execute(inp)
        cov = covariance((mat-mean(mat,axis=0))\
                        /std(mat,axis=0),act_mat)
        maxima = mdp.utils.amax(abs(cov))
        assert_array_almost_equal(maxima,numx.ones(vars),3)
        
    def testCuBICANodeBatch(self):
        ica = mdp.nodes.CuBICANode(limit = 10**(-self.decimal))
        self._testICANode(ica)
        
    def testCuBICANodeTelescope(self):
        ica = mdp.nodes.CuBICANode(limit = 10**(-self.decimal), telescope = 1)
        self._testICANode(ica)
        
    def testFastICANodeSymmetric(self):
        ica = mdp.nodes.FastICANode\
              (limit = 10**(-self.decimal),approach="symm")
        self._testICANode(ica)
        
    def testFastICANodeDeflation(self):
        ica = mdp.nodes.FastICANode\
              (limit = 10**(-self.decimal), approach="defl")
        self._testICANode(ica)

    def testOneDimensionalHitParade(self):
        signal = (numx_rand.random(300)-0.5)*2
        gap = 5
        # put some maxima and minima
        signal[0] , signal[10] , signal[50] = 1.5, 1.4, 1.3
        signal[1] , signal[11] , signal[51] = -1.5, -1.4, -1.3
        # put two maxima and two minima within the gap
        signal[100], signal[103] = 2, 3
        signal[110], signal[113] = 3.1, 2
        signal[120], signal[123] = -2, -3.1
        signal[130], signal[133] = -3, -2
        hit = mdp.nodes.misc_nodes.OneDimensionalHitParade(5,gap)
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
        signal = numx_rand.random((300,3))
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

    def testTimeFramesNode(self):
        length = 14
        gap = 6
        time_frames = 3
        inp = tr(numx.array(\
            [numx.arange(length), -numx.arange(length)]))
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
        t = mdp.utils.linspace(0,2*numx.pi,tlen)
        inp = tr(numx.array([numx.sin(t), numx.sin(5*t)]))
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
        dir /= mdp.utils.norm2(dir)
        x = self._uniform(-1,1,[npoints])
        data = numx.outerproduct(x, dir)+const
        # train the gng network
        gng = mdp.nodes.GrowingNeuralGasNode(start_poss=[data[0,:],data[1,:]])
        gng.train(data)
        gng.stop_training()
        # control that the nodes in the graph lie on the line
        poss = gng.get_nodes_position()-const
        norms = numx.sqrt(numx.sum(poss*poss, axis=1))
        poss = tr(tr(poss)/norms)
        assert max(numx.minimum(numx.sum(abs(poss-dir),axis=1),
                                 numx.sum(abs(poss+dir),axis=1)))<1e-10, \
               'At least one node of the graph does lies out of the line.'
        # check that the graph is linear (no additional branches)
        # get a topological sort of the graph
        topolist = gng.graph.topological_sort()
        deg = map(lambda n: n.degree(), topolist)
        assert_equal(deg[:2],[1,1])
        assert_array_equal(deg[2:], [2 for i in range(len(deg)-2)])
        # check the distribution of the nodes' position is uniform
        # this node is at one of the extrema of the graph
        x0 = numx.outerproduct(mdp.utils.amin(x), dir)+const
        x1 = numx.outerproduct(mdp.utils.amax(x), dir)+const
        linelen = mdp.utils.norm2(x0-x1)
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
        def bogus_noise(mean, shape=None):
            return numx.ones(shape)*mean

        node = mdp.nodes.NoiseNode(bogus_noise, (1.,))
        out = node.execute(numx.zeros((100,10),'d'))
        assert_array_equal(out, numx.ones((100,10),'d'))
        node = mdp.nodes.NoiseNode(bogus_noise, (1.,), 'multiplicative')
        out = node.execute(numx.zeros((100,10),'d'))
        assert_array_equal(out, numx.zeros((100,10),'d'))
        
def get_suite():
    return NodesTestSuite()


if __name__ == '__main__':
    numx_rand.seed(1268049219, 2102953867)
    unittest.TextTestRunner(verbosity=2).run(get_suite())

