
from ._tools import *
from mdp import numx
from mdp import OnlineFlow, CircularOnlineFlow


from future import standard_library
standard_library.install_aliases()
from builtins import range

import tempfile
import pickle

uniform = numx_rand.random

class BogusOnlineNode(mdp.OnlineNode):
    def _check_params(self, x):
        if not hasattr(self, 'sum'): self.sum = mdp.numx.zeros((1,self.input_dim))
    def _train(self, x): self.sum += x
    def _execute(self,x): return x + 1
    def _inverse(self,x): return x - 1

class BogusOnlineNodeReturnSum(mdp.OnlineNode):
    def _check_params(self, x):
        if not hasattr(self, 'sum'): self.sum = mdp.numx.zeros((1,self.input_dim))
    def _train(self, x): self.sum += x
    def _execute(self,x): return self.sum
    @staticmethod
    def is_invertible(): return False

class BogusOnlineDiffDimNode(mdp.OnlineNode):
    def _check_params(self, x):
        if not hasattr(self, 'sum'): self.sum = mdp.numx.zeros((1,self.input_dim))
    def _train(self, x): self.sum += x
    def _execute(self,x): return numx.dot(x,numx.ones((self.input_dim, self.output_dim)))
    @staticmethod
    def is_invertible(): return False

class BogusOnlineNodeWithTrainArgs(mdp.OnlineNode):
    def _check_params(self, x):
        if not hasattr(self, 'sum'): self.sum = mdp.numx.zeros((1,self.input_dim))
    def _train(self, x, a, b): self.sum += x
    def _execute(self,x): return x + 1
    def _inverse(self,x): return x - 1

def _get_default_flow(flow_class=OnlineFlow, node_class=BogusOnlineNode):
    flow = flow_class([node_class(),node_class(),node_class()])
    return flow

def test_flow():
    flow = _get_default_flow()
    inp = numx.ones((1,3))*2
    flow.train(inp)
    out = flow(inp)
    [assert_array_equal(f.sum,i+inp)  for i,f in enumerate(flow)]
    assert_array_equal(out,len(flow)+inp)
    rec = flow.inverse(out)
    assert_array_equal(rec,inp)

def test_flow_copy():
    dummy_list = [1,2,3]
    flow = _get_default_flow()
    flow[0].dummy_attr = dummy_list
    copy_flow = flow.copy()
    assert flow[0].dummy_attr == copy_flow[0].dummy_attr, \
           'Flow copy method did not work'
    copy_flow[0].dummy_attr[0] = 10
    assert flow[0].dummy_attr != copy_flow[0].dummy_attr, \
           'Flow copy method did not work'

def test_flow_copy_with_lambda():
    generic_node = mdp.OnlineNode()
    generic_node.lambda_function = lambda: 1
    generic_flow = mdp.OnlineFlow([generic_node])
    generic_flow.copy()

def test_flow_save():
    dummy_list = [1,2,3]
    flow = _get_default_flow()
    flow[0].dummy_attr = dummy_list
    # test string save
    copy_flow_pic = flow.save(None)
    copy_flow = pickle.loads(copy_flow_pic)
    assert flow[0].dummy_attr == copy_flow[0].dummy_attr, \
           'Flow save (string) method did not work'
    copy_flow[0].dummy_attr[0] = 10
    assert flow[0].dummy_attr != copy_flow[0].dummy_attr, \
           'Flow save (string) method did not work'
    # test file save
    dummy_file = tempfile.mktemp(prefix='MDP_', suffix=".pic")
    flow.save(dummy_file, protocol=1)
    dummy_file = open(dummy_file, 'rb')
    copy_flow = pickle.load(dummy_file)
    assert flow[0].dummy_attr == copy_flow[0].dummy_attr, \
           'Flow save (file) method did not work'
    copy_flow[0].dummy_attr[0] = 10
    assert flow[0].dummy_attr != copy_flow[0].dummy_attr, \
           'Flow save (file) method did not work'

def test_flow_container_privmethods():
    mat,mix,inp = get_random_mix(mat_dim=(100,3))
    flow = _get_default_flow()
    # test __len__
    assert_equal(len(flow), len(flow.flow))
    # test __?etitem__, integer key
    for i in range(len(flow)):
        assert flow[i]==flow.flow[i], \
               '__getitem__  returned wrong node %d' % i
        new_node = BogusOnlineNode()
        flow[i] = new_node
        assert flow[i]==new_node, '__setitem__ did not set node %d' % i
    # test __?etitem__, normal slice -> this fails for python < 2.2 and
    # if Flow is a subclassed from builtin 'list'
    flowslice = flow[0:2]
    assert isinstance(flowslice,mdp.OnlineFlow), \
           '__getitem__ slice is not an OnlineFlow instance'
    assert len(flowslice) == 2, '__getitem__ returned wrong slice size'
    new_nodes_list = [BogusOnlineNode(), BogusOnlineNode()]
    flow[:2] = new_nodes_list
    assert (flow[0] == new_nodes_list[0]) and \
           (flow[1] == new_nodes_list[1]), '__setitem__ did not set slice'
    # test__?etitem__, extended slice
    flowslice = flow[:2:1]
    assert isinstance(flowslice,mdp.OnlineFlow), \
           '__getitem__ slice is not a Flow instance'
    assert len(flowslice) == 2, '__getitem__ returned wrong slice size'
    new_nodes_list = [BogusOnlineNode(), BogusOnlineNode()]
    flow[:2:1] = new_nodes_list
    assert (flow[0] == new_nodes_list[0]) and \
           (flow[1] == new_nodes_list[1]), '__setitem__ did not set slice'
    # test __delitem__, integer key
    copy_flow = mdp.OnlineFlow(flow[:])
    del copy_flow[0]
    assert len(copy_flow) == len(flow)-1, '__delitem__ did not del'
    for i in range(len(copy_flow)):
        assert copy_flow[i] == flow[i+1], '__delitem__ deleted wrong node'
    # test __delitem__, normal slice
    copy_flow = mdp.OnlineFlow(flow[:])
    del copy_flow[:2]
    assert len(copy_flow) == len(flow)-2, \
           '__delitem__ did not del normal slice'
    assert copy_flow[0] == flow[2], \
           '__delitem__ deleted wrong normal slice'
    # test __delitem__, extended slice
    copy_flow = mdp.OnlineFlow(flow[:])
    del copy_flow[:2:1]
    assert len(copy_flow) == len(flow)-2, \
           '__delitem__ did not del extended slice'
    assert copy_flow[0] == flow[2], \
           '__delitem__ deleted wrong extended slice'
    # test __add__
    newflow = flow + flow
    assert len(newflow) == len(flow)*2, '__add__ did not work'

    # test __add__ with Node
    flow = _get_default_flow()
    newflow = flow + BogusNode()
    assert len(newflow) == len(flow)+1, '__add__ did not work'

    # test insert with Node
    flow[1] = BogusNode()
    inp = numx.ones((1,3))*2
    flow.train(inp)
    out = flow(inp)
    rec = flow.inverse(out)
    assert_array_equal(rec,inp)

    # test __add__ with TrainableNode
    flow = _get_default_flow()
    newflow = flow + BogusNodeTrainable()
    assert len(newflow) == len(flow) + 1, '__add__ did not work'

    # test insert with TrainableNode
    try:
        flow[1] = BogusNodeTrainable()
    except TypeError: pass
    else: raise Exception('This is not supposed to work!')


def test_flow_container_listmethods():
    # for all methods try using a node with right dimensionality
    # and one with wrong dimensionality
    flow = _get_default_flow()
    length = len(flow)
    # we test __contains__ and __iter__ with the for loop
    for node in flow:
        node.input_dim = 10
        node.output_dim = 10
    # append
    newnode = BogusNode(input_dim=10, output_dim=10)
    flow.append(newnode)
    assert_equal(len(flow), length+1)
    length = len(flow)
    try:
        newnode = BogusNode(input_dim=11)
        flow.append(newnode)
        raise Exception('flow.append appended inconsistent node')
    except ValueError:
        assert_equal(len(flow), length)
    # extend
    newflow = flow.copy()
    flow.extend(newflow)
    assert_equal(len(flow), 2*length)
    length = len(flow)
    try:
        newflow = _get_default_flow()
        for idx in range(len(newflow)):
            if idx == 0:
                newflow[idx].input_dim = 11
            else:
                newflow[idx].input_dim = 10
            newflow[idx].output_dim = 10
        flow.extend(newflow)
        raise Exception('flow.extend appended inconsistent flow')
    except ValueError:
        assert_equal(len(flow), length)
    # insert
    newnode = BogusNode(input_dim=10, output_dim=None)
    flow.insert(2, newnode)
    assert_equal(len(flow), length+1)
    length = len(flow)
    try:
        newnode = BogusNode(output_dim=11)
        flow.insert(2, newnode)
        raise Exception('flow.insert inserted inconsistent node')
    except ValueError:
        assert_equal(len(flow), length)
    # pop
    oldnode = flow[5]
    popnode = flow.pop(5)
    assert oldnode == popnode, 'flow.pop popped wrong node out'
    assert_equal(len(flow), length-1)
    # pop - test Flow._check_nodes_consistency
    flow = _get_default_flow() + _get_default_flow()
    length = len(flow)
    flow[3].output_dim = 2
    flow[4].input_dim = 2
    flow[4].output_dim = 3
    flow[5].input_dim = 3
    flow._check_nodes_consistency(flow.flow)
    try:
        nottobepopped = flow.pop(4)
        raise Exception('flow.pop left inconsistent flow')
    except ValueError:
        assert_equal(len(flow), length)


def test_flow_train_args():
    #check args len
    flow = _get_default_flow(node_class=BogusOnlineNodeWithTrainArgs)
    _args_needed = flow._train_args_needed_list
    _keys = flow._train_arg_keys_list
    assert(mdp.numx.all(_args_needed))
    for val in _keys:
        assert(len(val) == 2)
    flow = _get_default_flow()
    _args_needed = flow._train_args_needed_list
    _keys = flow._train_arg_keys_list
    assert (not mdp.numx.all(_args_needed))
    for val in _keys:
        assert (len(val) == 0)

    # train with args
    flow = _get_default_flow(node_class=BogusOnlineNodeWithTrainArgs)
    flow[-2] = BogusOnlineNode()

    inp = numx.ones((1,3))*2
    x = [(inp, (1,2), None, (3,4))]
    flow.train(x)
    out = flow(inp)
    [assert_array_equal(f.sum,i+inp)  for i,f in enumerate(flow)]
    assert_array_equal(out,len(flow)+inp)
    rec = flow.inverse(out)
    assert_array_equal(rec,inp)


def test_circular_flow():
    flow = CircularOnlineFlow([BogusNode(input_dim=2, output_dim=2),
                        BogusOnlineDiffDimNode(input_dim=2, output_dim=4),
                        BogusNode(input_dim=4, output_dim=4),
                        BogusOnlineDiffDimNode(input_dim=4, output_dim=2)])

    for inp_node_idx in xrange(len(flow)):
        flow.set_input_node(inp_node_idx)
        for out_node_idx in xrange(len(flow)):
            flow.set_output_node(out_node_idx)
            inp = numx.ones((1, flow[0].input_dim))
            flow.train(inp)
            out = flow(inp)
            assert(out.shape[1] == flow[out_node_idx].output_dim)
            assert(flow.get_stored_input().shape[1] == flow[3].output_dim)


    flow = CircularOnlineFlow([BogusNode(),
                                BogusOnlineNodeReturnSum(),
                                BogusNode(),
                                BogusOnlineNodeReturnSum()])

    flow.set_flow_iterations(3)
    inp = numx.ones((1, 2))
    flow.train(inp)
    out = flow(inp)
    assert_array_equal(out, inp*140)
    assert (flow[1].get_current_train_iteration() == flow._flow_iterations)

