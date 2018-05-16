
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

    for inp_node_idx in range(len(flow)):
        flow.set_input_node(inp_node_idx)
        for out_node_idx in range(len(flow)):
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

def test_online_layer():
    nodes = [BogusOnlineNode(input_dim=2, output_dim=2), BogusOnlineDiffDimNode(input_dim=4, output_dim=8), BogusNode(input_dim=3, output_dim=3)]
    layer = mdp.hinet.OnlineLayer(nodes, numx_rng=mdp.numx_rand.RandomState(seed=2))
    assert(layer.input_dim == 9)
    assert(layer.output_dim == 13)
    assert(layer.numx_rng == nodes[0].numx_rng)
    assert(layer.numx_rng == nodes[1].numx_rng)

    inp = numx.ones((1,9))
    layer.train(inp)
    out = layer(inp)
    assert_array_equal(nodes[0].sum, inp[:,:2])
    assert_array_equal(nodes[1].sum, inp[:,:4])
    assert_array_equal(out[:,:2], nodes[0](inp[:,:2]))
    assert_array_equal(out[:,2:-3], nodes[1](inp[:,:4]))
    assert_array_equal(out[:,-3:], nodes[2](inp[:,:3]))

def test_clone_online_layer():
    nodes = BogusOnlineNode(input_dim=2, output_dim=2)
    layer = mdp.hinet.CloneOnlineLayer(nodes, n_nodes=2, numx_rng=mdp.numx_rand.RandomState(seed=1))
    assert(layer.input_dim == 4)
    assert(layer.output_dim == 4)
    assert(layer.numx_rng == nodes.numx_rng)

    inp = numx.ones((1,4))
    layer.train(inp)
    out = layer(inp)
    assert_array_equal(nodes.sum, inp[:,:2]+inp[:,:2])
    assert_array_equal(out[:,:2], nodes(inp[:,:2]))
    assert_array_equal(out[:,2:4], nodes(inp[:,:2]))


def test_online_flow_node():
    rng = mdp.numx_rand.RandomState(seed=1)
    flow1 = OnlineFlow([BogusNode(input_dim=2, output_dim=2),
                        BogusOnlineNode(input_dim=2, output_dim=2),
                        BogusNode(input_dim=2, output_dim=2),
                        BogusOnlineDiffDimNode(input_dim=2, output_dim=4),
                        ])
    node1 = mdp.hinet.OnlineFlowNode(flow1, numx_rng=rng)

    flow2 = OnlineFlow([BogusNode(input_dim=2, output_dim=2),
                        BogusOnlineNode(input_dim=2, output_dim=2),
                        BogusNode(input_dim=2, output_dim=2),
                        BogusOnlineDiffDimNode(input_dim=2, output_dim=4),
                        ])
    node2 = mdp.hinet.FlowNode(flow2)

    # number of training phases = number of trainable nodes + 1(if the first node is not-trainable).
    assert(node1.get_remaining_train_phase()==3)

    inp = numx.ones((2,2))
    out1 = node1(inp)
    out2 = node2(inp)
    assert_array_equal(out1, out2)
    assert(node1.is_training())
    assert(flow1[1].is_training())
    assert(flow1[3].is_training())
    assert (not node2.is_training())
    for _n in node2.flow:
        assert(not _n.is_training())

    assert(node1.numx_rng == rng)
    assert(node1.numx_rng == node1._flow[1].numx_rng)
    assert(node1.numx_rng == node1._flow[3].numx_rng)


    flow = mdp.OnlineFlow([BogusNode(),
                    BogusOnlineNodeReturnSum(),
                    BogusOnlineNodeReturnSum(),
                    BogusNode(input_dim=5, output_dim=5)])

    node = mdp.hinet.OnlineFlowNode(flow)

    inp = numx.ones((1, 5))
    assert(flow[1].get_current_train_iteration() == 0)
    out = node(inp)
    out = node(inp)
    # check if all the node dimensions are fixed.
    for _n in flow:
        assert((_n.input_dim, _n.output_dim) == (inp.shape[1],inp.shape[1]))
    assert ((node.input_dim, node.output_dim) == (inp.shape[1], inp.shape[1]))
    # check if only training was carried out once
    assert(flow[1].get_current_train_iteration() == 1)



def test_circular_online_flow_node_default():

    # default setting (= onlineflownode)
    flow1 = mdp.CircularOnlineFlow([BogusNode(),
                                BogusOnlineNodeReturnSum(),
                                BogusNode(),
                                BogusOnlineNodeReturnSum()])

    node1 = mdp.hinet.CircularOnlineFlowNode(flow1, numx_rng=mdp.numx_rand.RandomState(seed=1))
    flow2 = mdp.CircularOnlineFlow([BogusNode(),
                                BogusOnlineNodeReturnSum(),
                                BogusNode(),
                                BogusOnlineNodeReturnSum()])

    node2 = mdp.hinet.OnlineFlowNode(flow2, numx_rng=mdp.numx_rand.RandomState(seed=1))

    assert(node1.get_remaining_train_phase()==node2.get_remaining_train_phase())
    assert(node1.get_stored_input() is None)
    inp = numx.ones((1, 2))
    out1 = node1(inp) # One train and execute
    out2 = node2(inp) # One train and execute
    assert_array_equal(out1, out2)
    assert_array_equal(node1.get_stored_input(), out2)

def test_circular_online_flow_node_different_output():

    # default setting with different output_node. Check stored_input
    flow1 = mdp.CircularOnlineFlow([BogusNode(),
                                BogusOnlineNodeReturnSum(),
                                BogusNode(),
                                BogusOnlineNodeReturnSum()])
    flow1.set_output_node(2)
    node1 = mdp.hinet.CircularOnlineFlowNode(flow1, numx_rng=mdp.numx_rand.RandomState(seed=1))
    flow2 = mdp.CircularOnlineFlow([BogusNode(),
                                BogusOnlineNodeReturnSum(),
                                BogusNode(),
                                BogusOnlineNodeReturnSum()])

    node2 = mdp.hinet.OnlineFlowNode(flow2, numx_rng=mdp.numx_rand.RandomState(seed=1))

    assert(node1.get_remaining_train_phase()==node2.get_remaining_train_phase())
    assert(node1.get_stored_input() is None)
    inp = numx.ones((1, 2))
    out1 = node1(inp) # One train and execute
    out2 = node2(inp) # One train and execute
    assert_array_equal(node1.get_stored_input(), out2)
    assert (not (out1 != out2).all())

def test_circular_online_flow_node_internal_training():

    # internal training (check errors without stored inputs)
    flow1 = mdp.CircularOnlineFlow([BogusNode(),
                                BogusOnlineNodeReturnSum(),
                                BogusNode(),
                                BogusOnlineNodeReturnSum()])
    flow1.ignore_input(True)
    node1 = mdp.hinet.CircularOnlineFlowNode(flow1, numx_rng=mdp.numx_rand.RandomState(seed=1))
    flow2 = mdp.CircularOnlineFlow([BogusNode(),
                                BogusOnlineNodeReturnSum(),
                                BogusNode(),
                                BogusOnlineNodeReturnSum()])
    node2 = mdp.hinet.OnlineFlowNode(flow2, numx_rng=mdp.numx_rand.RandomState(seed=1))

    assert(node1.get_remaining_train_phase()==node2.get_remaining_train_phase())
    assert(node1.get_stored_input() is None)
    inp = numx.ones((1, 2))
    try:
        node1.train(inp)
        raise Exception("node trained internally without any stored inputs.")
    except mdp.TrainingException: pass

def test_circular_online_flow_node_internal_stored_inputs():

    # internal training with stored inputs. (check 1 loop output with default output)
    flow1 = mdp.CircularOnlineFlow([BogusNode(),
                                BogusOnlineNodeReturnSum(),
                                BogusNode(),
                                BogusOnlineNodeReturnSum()])
    flow1.ignore_input(True)
    node1 = mdp.hinet.CircularOnlineFlowNode(flow1, numx_rng=mdp.numx_rand.RandomState(seed=1))
    flow2 = mdp.CircularOnlineFlow([BogusNode(),
                                BogusOnlineNodeReturnSum(),
                                BogusNode(),
                                BogusOnlineNodeReturnSum()])
    node2 = mdp.hinet.OnlineFlowNode(flow2, numx_rng=mdp.numx_rand.RandomState(seed=1))

    inp = numx.ones((1, 2))
    node1.set_stored_input(inp)
    out1 = node1(inp) # One train and execute
    out2 = node2(inp) # One train and execute
    assert_array_equal(out1, out2)
    assert_array_equal(node1._stored_input, out2)

def test_circular_online_flow_node_internal_multiple_iterations():

    # internal training with multiple iterations.
    flow1 = mdp.CircularOnlineFlow([BogusNode(),
                                BogusOnlineNodeReturnSum(),
                                BogusNode(),
                                BogusOnlineNodeReturnSum()])
    flow1.ignore_input(True)
    flow_iters = 5
    flow1.set_flow_iterations(flow_iters)
    node1 = mdp.hinet.CircularOnlineFlowNode(flow1, numx_rng=mdp.numx_rand.RandomState(seed=1))
    flow2 = mdp.CircularOnlineFlow([BogusNode(),
                                BogusOnlineNodeReturnSum(),
                                BogusNode(),
                                BogusOnlineNodeReturnSum()])
    node2 = mdp.hinet.OnlineFlowNode(flow2, numx_rng=mdp.numx_rand.RandomState(seed=1))

    # number of phases = flow_iters * (number of trainable nodes + 1 if the first node is not trainable)
    assert(node1.get_remaining_train_phase()==3*flow_iters)

    inp = numx.ones((1, 2))
    node1.set_stored_input(inp)
    out1 = node1(inp) # One train (includes 3 iterations) and execute

    x = inp
    for _ in range(flow_iters):
        node2.train(x)
        x = node2.execute(x)

    assert_array_equal(out1, x)
    assert_array_equal(node1.get_stored_input(), x)

def test_circular_online_flow_node_external_with_iterations():

    #external training with iterations.
    flow1 = mdp.CircularOnlineFlow([BogusNode(),
                                BogusOnlineNodeReturnSum(),
                                BogusNode(),
                                BogusOnlineNodeReturnSum()])
    flow_iters = 5
    flow1.set_flow_iterations(flow_iters)
    node1 = mdp.hinet.CircularOnlineFlowNode(flow1, numx_rng=mdp.numx_rand.RandomState(seed=1))
    flow2 = mdp.CircularOnlineFlow([BogusNode(),
                                BogusOnlineNodeReturnSum(),
                                BogusNode(),
                                BogusOnlineNodeReturnSum()])
    node2 = mdp.hinet.OnlineFlowNode(flow2, numx_rng=mdp.numx_rand.RandomState(seed=1))

    # number of phases = flow_iters * (number of trainable nodes + 1 if the first node is not trainable)
    assert(node1.get_remaining_train_phase()==3*flow_iters)

    inp = numx.ones((1, 2))
    out1 = node1(inp) # One train (includes 3 iterations) and execute
    x = inp
    for _ in range(flow_iters):
        node2.train(x)
        x = node2.execute(x)
    assert_array_equal(out1, x)
    assert_array_equal(node1.get_stored_input(), x)

