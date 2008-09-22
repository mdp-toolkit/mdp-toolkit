"""These are test functions for MDP flows.

Run them with:
>>> import mdp
>>> mdp.test("flows")

"""
import unittest
import tempfile
import pickle
import cPickle
import os
import inspect
import mdp
from testing_tools import assert_array_almost_equal, assert_almost_equal, \
     assert_array_equal, assert_equal

mult = mdp.utils.mult
numx = mdp.numx
numx_rand = mdp.numx_rand

# CheckpointFunction used in testCheckpointFunction
class _CheckpointCollectFunction(mdp.CheckpointFunction):
    def __init__(self):
        self.classes = []

    # collect the classes of the nodes it checks
    def __call__(self, node):
        self.classes.append(node.__class__)

class _BogusNode(mdp.Node):
    def is_trainable(self): return 0
    def _execute(self,x): return 2*x
    def _inverse(self,x): return 0.5*x

class _BogusNodeTrainable(mdp.Node):
    def _train(self, x):
        pass
    def _stop_training(self):
        pass
    
class _BogusExceptNode(mdp.Node):
    def _train(self,x):
        self.bogus_attr = 1
        raise Exception, "Bogus Exception"
    
    def _execute(self,x):
        raise Exception, "Bogus Exception"

class _BogusMultiNode(mdp.Node):

    def __init__(self):
        super(_BogusMultiNode, self).__init__()
        self.visited = []
    
    def _get_train_seq(self):
        return [(self.train1, self.stop1),
                (self.train2, self.stop2)]

    def train1(self, x):
        self.visited.append(1)
    def stop1(self):
        self.visited.append(2)
    def train2(self, x):
        self.visited.append(3)
    def stop2(self):
        self.visited.append(4)

class FlowsTestSuite(unittest.TestSuite):

    def __init__(self, testname=None):
        unittest.TestSuite.__init__(self)

        if testname is not None:
            self._flows_test_factory([testname])
        else:
            # get all tests
            self._flows_test_factory()

    def _flows_test_factory(self, methods_list=None):
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
        
    def _get_random_mix(self,mat_dim = None, type = "d", scale = 1,
                        rand_func = numx_rand.random, avg = 0, std = 0):
        if mat_dim is None: mat_dim = self.mat_dim
        mat = ((rand_func(mat_dim)-0.5)*scale).astype(type)
        mat -= mat.mean(axis=0)
        mat /= mat.std(axis=0)
        if std: mat *= std
        if avg: mat += avg
        mix = (rand_func((mat_dim[1], mat_dim[1]))*scale).astype(type)
        return mat,mix,mult(mat,mix)

    def _get_default_flow(self, flow_class=mdp.Flow, node_class=_BogusNode):
        flow = flow_class([node_class(),node_class(),node_class()])
        return flow
    
    def testFlow(self):
        inp = numx.ones((100,3))
        flow = self._get_default_flow()
        for i in range(len(flow)):
            assert not flow.flow[i].is_training(), \
                   'Training of node #%d has not been closed.' % i

        out = flow(inp)
        assert_array_equal(out,(2**len(flow))*inp)
        rec = flow.inverse(out)
        assert_array_equal(rec,inp)

    def testFlow_copy(self):
        dummy_list = [1,2,3]
        flow = self._get_default_flow()
        flow[0].dummy_attr = dummy_list
        copy_flow = flow.copy()
        assert flow[0].dummy_attr == copy_flow[0].dummy_attr, \
               'Flow copy method did not work'
        copy_flow[0].dummy_attr[0] = 10
        assert flow[0].dummy_attr != copy_flow[0].dummy_attr, \
               'Flow copy method did not work'

    def testFlow_save(self):
        dummy_list = [1,2,3]
        flow = self._get_default_flow()
        flow[0].dummy_attr = dummy_list
        # test string save
        copy_flow_pic = flow.save(None)
        copy_flow = cPickle.loads(copy_flow_pic)
        assert flow[0].dummy_attr == copy_flow[0].dummy_attr, \
               'Flow save (string) method did not work'
        copy_flow[0].dummy_attr[0] = 10
        assert flow[0].dummy_attr != copy_flow[0].dummy_attr, \
               'Flow save (string) method did not work'
        # test file save
        dummy_file = os.path.join(tempfile.gettempdir(),'removeme')
        flow.save(dummy_file, protocol=1)
        flh = open(dummy_file, 'rb')
        copy_flow = cPickle.load(flh)
        flh.close()
        os.remove(dummy_file)        
        assert flow[0].dummy_attr == copy_flow[0].dummy_attr, \
               'Flow save (file) method did not work'
        copy_flow[0].dummy_attr[0] = 10
        assert flow[0].dummy_attr != copy_flow[0].dummy_attr, \
               'Flow save (file) method did not work'        
        
    def testFlow_container_privmethods(self):
        mat,mix,inp = self._get_random_mix(mat_dim=(100,3))
        flow = self._get_default_flow()
        # test __len__ 
        assert_equal(len(flow), len(flow.flow))
        # test __?etitem__, integer key
        for i in range(len(flow)):
            assert flow[i]==flow.flow[i], \
                   '__getitem__  returned wrong node %d' % i
            new_node = _BogusNode()
            flow[i] = new_node
            assert flow[i]==new_node, '__setitem__ did not set node %d' % i
        # test __?etitem__, normal slice -> this fails for python < 2.2 and
        # if Flow is a subclassed from builtin 'list'
        flowslice = flow[0:2]
        assert isinstance(flowslice,mdp.Flow), \
               '__getitem__ slice is not a Flow instance'
        assert len(flowslice) == 2, '__getitem__ returned wrong slice size'
        new_nodes_list = [_BogusNode(), _BogusNode()]
        flow[:2] = new_nodes_list
        assert (flow[0] == new_nodes_list[0]) and \
               (flow[1] == new_nodes_list[1]), '__setitem__ did not set slice'
        # test__?etitem__, extended slice
        flowslice = flow[:2:1]
        assert isinstance(flowslice,mdp.Flow), \
               '__getitem__ slice is not a Flow instance'
        assert len(flowslice) == 2, '__getitem__ returned wrong slice size'
        new_nodes_list = [_BogusNode(), _BogusNode()]
        flow[:2:1] = new_nodes_list
        assert (flow[0] == new_nodes_list[0]) and \
               (flow[1] == new_nodes_list[1]), '__setitem__ did not set slice'
        # test __delitem__, integer key
        copy_flow = mdp.Flow(flow[:])
        del copy_flow[0]
        assert len(copy_flow) == len(flow)-1, '__delitem__ did not del'
        for i in range(len(copy_flow)):
            assert copy_flow[i] == flow[i+1], '__delitem__ deleted wrong node'
        # test __delitem__, normal slice
        copy_flow = mdp.Flow(flow[:])
        del copy_flow[:2]
        assert len(copy_flow) == len(flow)-2, \
               '__delitem__ did not del normal slice'
        assert copy_flow[0] == flow[2], \
               '__delitem__ deleted wrong normal slice'
        # test __delitem__, extended slice
        copy_flow = mdp.Flow(flow[:])
        del copy_flow[:2:1]
        assert len(copy_flow) == len(flow)-2, \
               '__delitem__ did not del extended slice'
        assert copy_flow[0] == flow[2], \
               '__delitem__ deleted wrong extended slice'        
        # test __add__
        newflow = flow + flow
        assert len(newflow) == len(flow)*2, '__add__ did not work'
        
    def testFlow_container_listmethods(self):
        # for all methods try using a node with right dimensionality
        # and one with wrong dimensionality
        flow = self._get_default_flow()
        length = len(flow)
        # we test __contains__ and __iter__ with the for loop 
        for node in flow:
            node.input_dim = 10
            node.output_dim = 10
        # append
        newnode = _BogusNode(input_dim=10, output_dim=10)
        flow.append(newnode)
        assert_equal(len(flow), length+1)
        length = len(flow)
        try:
            newnode = _BogusNode(input_dim=11)
            flow.append(newnode)
            raise Exception, 'flow.append appended inconsistent node'
        except ValueError:
            assert_equal(len(flow), length)
        # extend
        newflow = flow.copy()
        flow.extend(newflow)
        assert_equal(len(flow), 2*length)
        length = len(flow)
        try:
            newflow = self._get_default_flow()
            for idx in range(len(newflow)):
                if idx == 0:
                    newflow[idx].input_dim = 11
                else:
                    newflow[idx].input_dim = 10
                newflow[idx].output_dim = 10
            flow.extend(newflow)
            raise Exception, 'flow.extend appended inconsistent flow'
        except ValueError:
            assert_equal(len(flow), length)
        # insert
        newnode = _BogusNode(input_dim=10, output_dim=None)
        flow.insert(2, newnode)
        assert_equal(len(flow), length+1)
        length = len(flow)
        try:
            newnode = _BogusNode(output_dim=11)
            flow.insert(2, newnode)
            raise Exception, 'flow.insert inserted inconsistent node'
        except ValueError:
            assert_equal(len(flow), length)
        # pop
        oldnode = flow[5]
        popnode = flow.pop(5)
        assert oldnode == popnode, 'flow.pop popped wrong node out'
        assert_equal(len(flow), length-1)
        # pop - test Flow._check_nodes_consistency
        flow = self._get_default_flow() + self._get_default_flow()
        length = len(flow)
        flow[3].output_dim = 2
        flow[4].input_dim = 2
        flow[4].output_dim = 3
        flow[5].input_dim = 3
        flow._check_nodes_consistency(flow.flow)
        try:
            nottobepopped = flow.pop(4)
            raise Exception, 'flow.pop left inconsistent flow'
        except ValueError:
            assert_equal(len(flow), length)

    def testFlow_as_sum_of_nodes(self):
        node1 = _BogusNode()
        node2 = _BogusNode()
        flow = node1+node2
        assert type(flow) is mdp.Flow
        assert len(flow) == 2
        node3 = _BogusNode()
        flow = node1+node2+node3
        assert type(flow) is mdp.Flow
        assert len(flow) == 3
        
        
    
    def testCheckpointFlow(self):
        lst = []
        # checkpoint function, it collects a '1' for each call
        def cfunc(node, lst = lst):
            lst.append(1)
        mat,mix,inp = self._get_random_mix(mat_dim=(100,3))
        flow = self._get_default_flow(flow_class = mdp.CheckpointFlow,
                                      node_class = _BogusNodeTrainable)
        flow.train(inp, cfunc)
        #
        assert len(lst)==len(flow), \
               'The checkpoint function has been called %d times instead of %d times.' % (len(lst), len(flow))
        
    def testCheckpointFunction(self):
        cfunc = _CheckpointCollectFunction()
        mat,mix,inp = self._get_random_mix(mat_dim=(100,3))
        flow = self._get_default_flow(flow_class = mdp.CheckpointFlow,
                                      node_class = _BogusNodeTrainable)
        flow.train(inp, cfunc)
        #
        for i in range(len(flow)):
            assert flow[i].__class__==cfunc.classes[i], 'Wrong class collected'

    def testCrashRecovery(self):
        flow = mdp.Flow([_BogusExceptNode()])
        flow.set_crash_recovery(1)
        try:
            flow.train(mdp.numx.zeros((1,2), 'd'))
        except Exception, e:
            assert isinstance(e,mdp.FlowExceptionCR)
            fl = file(e.filename)
            pic_flow = pickle.load(fl)
            fl.close()
            os.remove(e.filename)
            assert flow[0].bogus_attr == pic_flow[0].bogus_attr
        flow.set_crash_recovery(0)
        try:
            flow.execute([None])
        except Exception, e:
            assert isinstance(e,mdp.FlowExceptionCR)
            assert not hasattr(e,'filename')

    def testCrashRecoveryException(self):
        a = 3
        try:
            raise mdp.CrashRecoveryException, \
                  ('bogus errstr',a,StandardError())
        except mdp.CrashRecoveryException, e:
            filename1 = e.dump()
            filename2 = e.dump(os.path.join(tempfile.gettempdir(),'removeme'))
            assert isinstance(e.parent_exception, StandardError)

        for fname in [filename1,filename2]:
            fl = file(fname)
            obj = pickle.load(fl)
            fl.close()
            os.remove(fname)
            assert obj == a

    def testMultiplePhases(self):
        # test basic multiple phase sequence
        flow = mdp.Flow([_BogusMultiNode()])
        flow.train(mdp.numx.zeros((1,2), 'd'))
        assert flow[0].visited == [1,2,3,4]
        # try to use an iterator to train it, check for rewinds
        class TestIterator:
            def __init__(self):
                self.used = 0
            def __iter__(self):
                self.used += 1
                yield mdp.numx.zeros((1,2), 'd')
        flow = mdp.Flow([_BogusMultiNode()])
        iter = TestIterator()
        flow.train([iter])
        assert iter.used == 2
        # should not work with an iterator
        def testgenerator():
            yield mdp.numx.zeros((1,2), 'd')
        flow = mdp.Flow([_BogusMultiNode()])
        try:
            flow.train([testgenerator()])
            raise Exception('Expected mdp.FlowException')
        except mdp.FlowException:
            pass

def get_suite(testname=None):
    return FlowsTestSuite(testname=testname)

if __name__ == '__main__':
    numx_rand.seed(1268049219)
    unittest.TextTestRunner(verbosity=2).run(get_suite())

