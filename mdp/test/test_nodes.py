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

def get_suite(testname=None):
    return NodesTestSuite(testname=testname)

if __name__ == '__main__':
    numx_rand.seed(1268049219)
    unittest.TextTestRunner(verbosity=2).run(get_suite())
