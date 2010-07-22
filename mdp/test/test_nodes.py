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
import itertools
import sys
from mdp import utils, numx, numx_rand, numx_linalg, numx_fft
from testing_tools import assert_array_almost_equal, assert_array_equal, \
     assert_almost_equal, assert_equal, assert_array_almost_equal_diff, \
     assert_type_equal

mean = numx.mean
std = numx.std
normal = numx_rand.normal
uniform = numx_rand.random
testtypes = [numx.dtype('d'), numx.dtype('f')]
testtypeschar = [t.char for t in testtypes]
testdecimals = {testtypes[0]: 12, testtypes[1]: 6}

from _tools import (BogusNode, BogusNodeTrainable,
                    BogusExceptNode, BogusMultiNode)

def _rand_labels(x):
    return numx.around(uniform(x.shape[0]))

def _rand_labels_array(x):
    return numx.around(uniform(x.shape[0])).reshape((x.shape[0],1))

def _rand_array_halfdim(x):
    return uniform(size=(x.shape[0], x.shape[1]//2))

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

#_spinner = itertools.cycle((' /\b\b', ' -\b\b', ' \\\b\b', ' |\b\b'))
_spinner = itertools.cycle((' .\b\b', ' o\b\b', ' 0\b\b', ' O\b\b',
                            ' 0\b\b', ' o\b\b'))
#_spinner = itertools.cycle([" '\b\b"]*2 + [' !\b\b']*2 + [' .\b\b']*2 +
#                           [' !\b\b']*2)

# create spinner
def spinner():
    sys.stderr.write(_spinner.next())
    sys.stderr.flush()

class NodesTestSuite(unittest.TestSuite):

    def __init__(self, testname=None):
        unittest.TestSuite.__init__(self)

        # constants
        self.mat_dim = (500,5)
        self.decimal = 7

        # set nodes to be tested
        self._set_nodes()

        if testname is not None:
            self._nodes_test_factory([testname])
        else:
            # get generic tests
            self._generic_test_factory()
            # get FastICA tests
            self._fastica_test_factory()
            # get nodes tests
            self._nodes_test_factory()

    def _set_nodes(self):
        mn = mdp.nodes
        self._nodes = [mn.PCANode,
                       mn.WhiteningNode,
                       mn.SFANode,
                       mn.SFA2Node,
                       mn.TDSEPNode,
                       mn.CuBICANode,
                       mn.FastICANode,
                       mn.QuadraticExpansionNode,
                       (mn.PolynomialExpansionNode, [3], None),
                       (mn.RBFExpansionNode, [[[0.]*5, [0.]*5], [1., 1.]], None),
                       mn.GrowingNeuralGasExpansionNode,
                       (mn.HitParadeNode, [2, 5], None),
                       (mn.TimeFramesNode, [3, 4], None),
                       mn.EtaComputerNode,
                       mn.GrowingNeuralGasNode,
                       mn.NoiseNode,
                       (mn.FDANode, [], _rand_labels),
                       (mn.GaussianClassifierNode, [], _rand_labels),
                       mn.FANode,
                       mn.ISFANode,
                       (mn.RBMNode, [5], None),
                       (mn.RBMWithLabelsNode, [5, 1], _rand_labels_array),
                       (mn.LinearRegressionNode, [], _rand_array_halfdim)]

    def _nodes_test_factory(self, methods_list=None):
        if methods_list is None:
            methods_list = dir(self)
        for methname in methods_list:
            try:
                meth = getattr(self,methname)
            except AttributeError:
                continue
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
        fica_parm = {'approach': ['symm', 'defl'],
                     'g': ['pow3', 'tanh', 'gaus', 'skew'],
                     'fine_g': [ None, 'pow3', 'tanh', 'gaus', 'skew'],
                     'sample_size': [ 1, 0.99999 ],
                     'mu': [1, 0.999999 ],
                     'stabilization': [False, True]}

        for parms in utils.orthogonal_permutations(fica_parm):
            if parms['mu'] != 1 and parms['stabilization'] is False:
                # mu != 1 implies setting stabilization
                continue
            # skew nonlinearity only wroks with skewed
            # input data
            if parms['g'] != 'skew' and parms['fine_g'] == 'skew':
                continue
            if parms['g'] == 'skew' and parms['fine_g'] != 'skew':
                continue

            testfunc, funcdesc = self._get_testFastICA(parms)
            self.addTest(unittest.FunctionTestCase(testfunc,
                                                   description=funcdesc))


    def _get_random_mix(self, mat_dim = None, type = "d", scale = 1,\
                        rand_func = uniform, avg = 0, \
                        std_dev = 1):
        if mat_dim is None: mat_dim = self.mat_dim
        T = mat_dim[0]
        N = mat_dim[1]
        d = 0
        while d < 1E-3:
            #mat = ((rand_func(size=mat_dim)-0.5)*scale).astype(type)
            mat = rand_func(size=(T,N)).astype(type)
            # normalize
            mat -= mean(mat,axis=0)
            mat /= std(mat,axis=0)
            # check that the minimum eigenvalue is finite and positive
            d1 = min(utils.symeig(mult(mat.T, mat), eigenvectors = 0))
            if std_dev is not None: mat *= std_dev
            if avg is not None: mat += avg
            mix = (rand_func(size=(N,N))*scale).astype(type)
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
                if hasattr(item, '__call__'):
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
                elif node_class == mdp.nodes.LinearRegressionNode:
                    inp = uniform(size=(1000, 5))
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
            output_dim = self.mat_dim[1]//2
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
            node.output_dim = output_dim
            self._train_if_necessary(inp, node, args, sup_args_func)
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

    def testNode_multiple_training_phases(self):
        x = uniform(size=self.mat_dim)
        node = _BogusMultiNode()
        phases = node.get_remaining_train_phase()
        for i in xrange(phases):
            assert node.get_current_train_phase() == i
            assert not node._train_phase_started
            node.train(x)
            assert node._train_phase_started
            node.stop_training()

        assert not node.is_training()

    def testNode_execution_without_training(self):
        x = uniform(size=self.mat_dim)
        # try execution without training: single train phase
        node = _BogusNodeTrainable()
        node.execute(x)
        assert hasattr(node, 'bogus_attr')
        # multiple train phases
        node = _BogusMultiNode()
        node.execute(x)
        assert node.visited == [1, 2, 3, 4]


    def _testICANode(self,icanode, rand_func = uniform, vars = 3, N=8000,
                     prec = 3):
        dim = (N,vars)
        mat,mix,inp = self._get_random_mix(rand_func=rand_func,mat_dim=dim)
        icanode.train(inp)
        act_mat = icanode.execute(inp)
        cov = utils.cov2((mat-mean(mat,axis=0))/std(mat,axis=0), act_mat)
        maxima = numx.amax(abs(cov), axis=0)
        assert_array_almost_equal(maxima,numx.ones(vars),prec)

    def _testICANodeMatrices(self, icanode, rand_func = uniform, vars = 3, N=8000):
        dim = (N,vars)
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
##         # old func description: verbose and with newlines
##         header = 'TestFastICANode:'
##         app =     '  Approach:     '+parms['approach']
##         nl =      '  Nonlinearity: '+parms['g']
##         fine_nl = '  Fine-tuning:  '+str(parms['fine_g'])
##         if parms['sample_size'] == 1:
##             compact = '  Samples  100%, '
##         else:
##             compact = '  Samples <100%, '
##         if parms['mu'] == 1:
##             compact = compact + 'Step:  1, '
##         else:
##             compact = compact + 'Step: <1, '
##         if parms['stabilization'] is True:
##             compact = compact +'Stabilized algorithm'
##         else:
##             compact = compact +'Standard   algorithm'
##         desc = '\n'.join([header, app, nl, fine_nl, compact])
        # new func description: compact and one line
        header = 'Test FastICANode'
        app =     'AP:'+parms['approach']
        nl =      'NL:'+parms['g']
        fine_nl = 'FT:'+str(parms['fine_g'])
        if parms['sample_size'] == 1:
            compact = 'SA:01 '
        else:
            compact = 'SA:<1 '
        if parms['mu'] == 1:
            compact = compact + 'S:01 '
        else:
            compact = compact + 'S:<1 '
        if parms['stabilization'] is True:
            compact = compact +'STB'
        else:
            compact = compact +'STD'
        desc = ' '.join([header, app, nl, fine_nl, compact])

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

    def _rand_with_timestruct(self, size=None):
        T, N = size
        # do something special only if T!=N, otherwise
        # we were asked to generate a mixing matrix
        if T == N:
            return uniform(size=size)
        # create independent sources
        src = uniform((T,N))*2-1
        fsrc = numx_fft.rfft(src,axis=0)
        # enforce different speeds
        for i in xrange(N):
            fsrc[(i+1)*(T//20):,i] = 0.
        src = numx_fft.irfft(fsrc,axis=0)
        return src


    def testTDSEPNode(self):
        ica = mdp.nodes.TDSEPNode(lags=20,limit = 1E-10)
        ica2 = ica.copy()
        self._testICANode(ica, rand_func=self._rand_with_timestruct,vars=2, N=2**14, prec=2)
        self._testICANodeMatrices(ica2, rand_func=self._rand_with_timestruct,vars=2,N=2**14)


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
        for i in xrange(1,time_frames):
            assert_array_equal(out[:,2*i],out[:,0]+i*gap)
            assert_array_equal(out[:,2*i+1],out[:,1]-i*gap)
        # check pseudo-inverse
        rec = tf.pseudo_inverse(out)
        assert_equal(rec.shape[1], inp.shape[1])
        block_size = min(out.shape[0], gap)
        for i in xrange(0,length,gap):
            assert_array_equal(rec[i:i+block_size], inp[i:i+block_size])

    def testTimeFramesNodeBugInputDim(self):
        mdp.nodes.TimeFramesNode(time_frames=10, gap=1, input_dim=1)

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
        assert_array_equal(deg[2:], [2 for i in xrange(len(deg)-2)])
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

    def testNormalNoiseNode(self):
        node = mdp.nodes.NormalNoiseNode(noise_args=(2.1, 0.001))
        x = numx.array([range(100), range(100)])
        node.execute(x)

    def testNoiseNodePickling(self):
        node = mdp.nodes.NoiseNode()
        node.copy()
        dummy = node.save(None)

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
        for i in xrange(nclasses):
            cov = utils.symrand(uniform((dim,))*dim+1)
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

        for i in xrange(nclasses):
            lbl_idx = node.labels.index(i)
            assert_array_almost_equal_diff(means[i],
                                      node.means[lbl_idx],
                                      self.decimal-1)
            assert_array_almost_equal_diff(utils.inv(covs[i]),
                                      node.inv_covs[lbl_idx],
                                      self.decimal-2)

    def testGaussianClassifier_labellistbug(self):
        gc = mdp.nodes.GaussianClassifierNode()
        # this was failing as of MDP-2.5-309-gefa0f9d!
        gc.train(mdp.numx_rand.random((50, 3)), [+1] * 50)


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
        cl1 = numx.ones((x1.shape[0],), dtype='i')
        cl2 = 2*numx.ones((x2.shape[0],), dtype='i')
        classes = numx.concatenate((cl1, cl2))

        # shuffle the data
        perm_idx = numx_rand.permutation(classes.shape[0])
        x = numx.take(x, perm_idx, axis=0)
        classes = numx.take(classes, perm_idx, axis=0)

        node = mdp.nodes.GaussianClassifierNode()
        node.train(x, classes)
        classification = node.label(x)

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
                                       mdp.utils.mult(fa.A, fa.A.T), 1)

    def testISFANodeGivensRotations(self):
        ncovs = 5
        dim = 7
        ratio = uniform(2).tolist()
        covs = [uniform((dim,dim)) for j in xrange(ncovs)]
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
        for i in xrange(3):
            fmat[(i+1)*5000:,i] = 0.
        mat = numx_fft.irfft(fmat,axis=0)
        src = mdp.sfa(mat)
        # test with unmixed signals (i.e. the node should make nothing at all)
        out = mdp.nodes.ISFANode(lags=1,
                                 whitened=True,
                                 sfa_ica_coeff=[1.,0.])(src)
        max_cv = numx.diag(abs(_cov(out,src)))
        assert_array_almost_equal(max_cv, numx.ones((3,)),6)
        # mix linearly the signals
        mix = mult(src,uniform((3,3))*2-1)
        out = mdp.nodes.ISFANode(lags=1,
                                 whitened=False,
                                 sfa_ica_coeff=[1.,0.])(mix)
        max_cv = numx.diag(abs(_cov(out,src)))
        assert_array_almost_equal(max_cv, numx.ones((3,)),5)

    def testISFANode_ICAPart(self):
        # create independent sources
        src = uniform((100000,3))*2-1
        fsrc = numx_fft.rfft(src,axis=0)
        # enforce different speeds
        for i in xrange(3):
            fsrc[(i+1)*5000:,i] = 0.
        src = numx_fft.irfft(fsrc,axis=0)
        # enforce time-lag-1-independence
        src = mdp.nodes.ISFANode(lags=1, sfa_ica_coeff=[1.,0.])(src)
        out = mdp.nodes.ISFANode(lags=1,
                                 whitened=True,
                                 sfa_ica_coeff=[0.,1.])(src)
        max_cv = numx.diag(abs(_cov(out,src)))
        assert_array_almost_equal(max_cv, numx.ones((3,)),5)
        # mix linearly the signals
        mix = mult(src,uniform((3,3))*2-1)
        out = mdp.nodes.ISFANode(lags=1,
                                 whitened=False,
                                 sfa_ica_coeff=[0.,1.])(mix)
        max_cv = numx.diag(abs(_cov(out,src)))
        assert_array_almost_equal(max_cv, numx.ones((3,)),5)

    def testISFANode_3Complete(self):
        # test transition from ica to sfa behavior of isfa
        # use ad hoc sources
        lag = 25
        src = numx.zeros((1001,3),"d")
        idx = [(2,4),(80,1),(2+lag,6)]
        for i in xrange(len(idx)):
            i0, il = idx[i]
            src[i0:i0+il,i] = 1.
            src[i0+il:i0+2*il,i] = -1.
            src[:,i] -= mean(src[:,i])
            src[:,i] /= std(src[:,i])
        # test extreme cases
        # case 1: ICA
        out = mdp.nodes.ISFANode(lags=[1,lag],
                                 icaweights=[1.,1.],
                                 sfaweights=[1.,0.],
                                 output_dim=2,
                                 whitened=True,
                                 sfa_ica_coeff=[1E-4,1.])(src)
        cv = abs(_cov(src,out))
        idx_cv = numx.argmax(cv,axis=0)
        assert_array_equal(idx_cv,[2,1])
        max_cv = numx.amax(cv,axis=0)
        assert_array_almost_equal(max_cv, numx.ones((2,)),5)
        # case 2: SFA
        out = mdp.nodes.ISFANode(lags=[1,lag],
                                 icaweights=[1.,1.],
                                 sfaweights=[1.,0.],
                                 output_dim=2,
                                 whitened=True,
                                 sfa_ica_coeff=[1.,0.])(src)
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
        for i in xrange(1,nmat):
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
        for trial in xrange(trials):
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

    def testRBMLearning(self):
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

        assert bm._train_err/N<0.1

    def testRBMWithLabelsNode(self):
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
            if v_train_err/(2*N)<0.1: break

        # visible units are reconstructed
        assert v_train_err/(2*N)<0.1
        # units with 0 input have 50/50 labels
        idxzeros = v.sum(axis=1)==0
        nzeros = idxzeros.sum()
        point5 = numx.zeros((nzeros, L)) + 0.5
        assert_array_almost_equal(pl[idxzeros], point5, 2)

    def testLinearRegressionNode(self):

        def train_LRNode(inp, out, with_bias):
            lrnode = mdp.nodes.LinearRegressionNode(with_bias)
            for i in xrange(len(inp)):
                lrnode.train(inp[i], out[i])
            lrnode.stop_training()
            return lrnode

        indim, outdim, tlen = 5, 3, 10000

        # 1. first, without noise
        # 1a   without bias term
        # regression coefficients
        beta = numx_rand.uniform(-10., 10., size=(indim, outdim))
        # input data
        x = numx_rand.uniform(-20., 20., size=(tlen, indim))
        # output of the linear model
        y = mult(x, beta)
        # train
        lrnode = train_LRNode([x], [y], False)
        # test results
        assert_array_almost_equal(lrnode.beta, beta, self.decimal)
        res = lrnode(x)
        assert_array_almost_equal(res, y, self.decimal)

        # 1b with bias
        beta = numx_rand.uniform(-10., 10., size=(indim+1, outdim))
        x = numx_rand.uniform(-20., 20., size=(tlen, indim))
        y = mult(x, beta[1:,:]) + beta[0,:]
        lrnode = train_LRNode([x], [y], True)
        assert_array_almost_equal(lrnode.beta, beta, self.decimal)
        res = lrnode(x)
        assert_array_almost_equal(res, y, self.decimal)

        # 2. with noise, multiple sets of input
        beta = numx_rand.uniform(-10., 10., size=(indim+1, outdim))
        inp = [numx_rand.uniform(-20., 20., size=(tlen, indim))
               for i in xrange(5)]
        out = [mult(x, beta[1:,:]) + beta[0,:] +
               numx_rand.normal(size=y.shape)*0.1 for x in inp]
        lrnode = train_LRNode(inp, out, True)
        assert_array_almost_equal(lrnode.beta, beta, 2)
        res = lrnode(inp[0])
        assert_array_almost_equal_diff(res, out[0], 2)

        # 3. test error for linearly dependent input
        beta = numx_rand.uniform(-10., 10., size=(indim, outdim))
        x = numx_rand.uniform(-20., 20., size=(tlen, indim))
        x[:,-1] = 2.*x[:,0]
        y = mult(x, beta)
        try:
            lrnode = train_LRNode([x], [y], False)
            raise Exception("LinearRegressionNode didn't raise "
                            "error for linearly dependent input")
        except mdp.NodeException:
            pass

        # 4. test wrong output size
        beta = numx_rand.uniform(-10., 10., size=(indim, outdim))
        x = numx_rand.uniform(-20., 20., size=(tlen, indim))
        x[:,-1] = 2.*x[:,0]
        y = mult(x, beta)
        y = y[:10,:]
        try:
            lrnode = train_LRNode([x], [y], False)
            raise Exception, ("LinearRegressionNode didn't raise "+
                              "error for output of wrong size")
        except mdp.TrainingException:
            pass

    def testCutoffNode(self):
        node = mdp.nodes.CutoffNode(-1.5, 1.2)
        x = numx.array([[0.1, 0, -2, 3, 1.2, -1.5, -3.33]])
        y_ref = numx.array([[0.1, 0, -1.5, 1.2, 1.2, -1.5, -1.5]])
        y = node.execute(x)
        assert numx.all(y==y_ref)

    def testHistogramNode_nofraction(self):
        """Test HistogramNode with fraction set to 1.0."""
        node = mdp.nodes.HistogramNode()
        x1 = numx.array([[0.1, 0.2], [0.3, 0.5]])
        x2 = numx.array([[0.3, 0.6], [0.2, 0.1]])
        x = numx.concatenate([x1, x2])
        node.train(x1)
        node.train(x2)
        assert numx.all(x == node.data_hist)

    def testHistogramNode_fraction(self):
        """Test HistogramNode with fraction set to 0.5."""
        node = mdp.nodes.HistogramNode(hist_fraction=0.5)
        x1 = numx_rand.random((1000, 3))
        x2 = numx_rand.random((500, 3))
        node.train(x1)
        node.train(x2)
        assert len(node.data_hist) < 1000

    def testAdaptiveCutoffNode_smalldata(self):
        """Test AdaptiveCutoffNode thoroughly on a small data set."""
        # values from 0.1 to 0.6 and 0.2 to 0.7
        x1 = numx.array([[0.1, 0.3], [0.3, 0.5], [0.5, 0.7]])
        x2 = numx.array([[0.4, 0.6], [0.2, 0.4], [0.6, 0.2]])
        x = numx.concatenate([x1, x2])
        node = mdp.nodes.AdaptiveCutoffNode(lower_cutoff_fraction= 0.2, # clip first
                                        upper_cutoff_fraction=0.4) # last two
        node.train(x1)
        node.train(x2)
        node.stop_training()
        assert numx.all(x == node.data_hist)
        # test bound values
        assert numx.all(node.lower_bounds == numx.array([0.2, 0.3]))
        assert numx.all(node.upper_bounds == numx.array([0.4, 0.5]))
        # test execute
        x_test = (numx.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
        x_clip = node.execute(x_test)
        x_goal = numx.array([[0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
        assert numx.all(x_clip == x_goal)

    def testAdaptiveCutoffNode_randomdata(self):
        """Test AdaptiveCutoffNode on a large random data."""
        node = mdp.nodes.AdaptiveCutoffNode(lower_cutoff_fraction= 0.2,
                                        upper_cutoff_fraction=0.4,
                                        hist_fraction=0.5)
        x1 = numx_rand.random((1000, 3))
        x2 = numx_rand.random((500, 3))
        x = numx.concatenate([x1, x2])
        node.train(x1)
        node.train(x2)
        node.stop_training()
        node.execute(x)

    def testRBFExpansionNode(self):
        rrep = mdp.utils.rrep
        dim, n = 2, 10
        centers = numx_rand.random((n, dim))
        # grid of points to numerically compute the integral
        grid = numx.meshgrid(numx.linspace(-3., 4., 100),
                             numx.linspace(-3., 4., 100))
        grid = numx.array([grid[0].flatten(), grid[1].flatten()]).T
        # compute covariance for each point of the grid
        grid_cov = numx.zeros((grid.shape[0], dim, dim))
        for i in xrange(dim):
            for j in xrange(dim):
                grid_cov[:,i,j] = grid[:,i]*grid[:,j]

        def check_mn_cov(rbf, real_covs):
            y = rbf(grid)
            # verify means, sizes
            for i in xrange(n):
                p = y[:,i]/y[:,i].sum()
                # check mean
                mn = (rrep(p,dim)*grid).sum(0)
                assert_array_almost_equal(mn, centers[i,:], 2)
                # check variance
                vr = ((rrep(rrep(p,2),2)*grid_cov).sum(0)
                      - numx.outer(mn, mn))
                assert_array_almost_equal(vr, real_covs[i], 2)

        def scalar_to_covs(x, n):
            if numx.isscalar(x):
                x = [x]*n
            return [numx.array([[x[i],0],[0,x[i]]]) for i in xrange(n)]

        # 1: sizes is a scalar
        sizes = 0.32
        rbf = mdp.nodes.RBFExpansionNode(centers, sizes)
        check_mn_cov(rbf, scalar_to_covs(sizes, n))

        # 2: sizes is many scalars
        sizes = 0.3 + numx_rand.random(n)*0.2
        rbf = mdp.nodes.RBFExpansionNode(centers, sizes)
        check_mn_cov(rbf, scalar_to_covs(sizes, n))

        # 3: sizes is one covariance
        sizes = mdp.utils.symrand(numx.array([0.2, 0.4]))
        rbf = mdp.nodes.RBFExpansionNode(centers, sizes)
        check_mn_cov(rbf, [sizes]*n)

        # 4: sizes is many covariances
        sizes = [mdp.utils.symrand(numx.array([0.2, 0.4]))
                 for i in xrange(n)]
        rbf = mdp.nodes.RBFExpansionNode(centers, sizes)
        check_mn_cov(rbf, sizes)


def get_suite(testname=None):
    return NodesTestSuite(testname=testname)

if __name__ == '__main__':
    numx_rand.seed(1268049219)
    unittest.TextTestRunner(verbosity=2).run(get_suite())
