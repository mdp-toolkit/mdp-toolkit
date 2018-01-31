from builtins import range
from ._tools import *

import mdp

class BogusOnlineNode(mdp.OnlineNode):
    def _check_params(self, x):
        if not hasattr(self, 'count'): self.count = 0
    def _train(self, x): self.count += 1
    def _execute(self,x): return x + 1
    def _inverse(self,x): return x - 1

class BogusOnlineNodeMultiplePhases(mdp.OnlineNode):
    def _check_params(self, x):
        if not hasattr(self, 'count'): self.count = None
    def _get_train_seq(self):
        return [(self._train, self._stop_training, self._execute)]*5
    def _train(self, x): self.count = x
    def _execute(self,x): return x + 1
    def _inverse(self,x): return x - 1

class BogusOnlineNodeMultiplePhasesWithBlankTrainSeq(mdp.OnlineNode):
    def _check_params(self, x):
        if not hasattr(self, 'count'): self.count = None
    def _get_train_seq(self):
        seq = [(self._train, self._stop_training, self._execute)]*5
        seq[-1] = (self._train, self._stop_training, lambda x: None)
        return seq
    def _train(self, x): self.count = x
    def _execute(self,x): return x + 1
    def _inverse(self,x): return x - 1

def test_train_execute_iteration():
    node = BogusOnlineNode()
    assert (node.get_current_train_iteration() == 0)

    inp = mdp.numx.zeros([10,5])

    for i in range(5):
        node.train(inp)
        assert (node.get_current_train_iteration() == inp.shape[0]*(i+1))
        out = node.execute(inp)
        rinp = node.inverse(out)
        assert_array_equal(inp, rinp)
        assert(node.count==node.get_current_train_iteration())

    #check direct execute after creation. Should train once.
    node = BogusOnlineNode()
    assert (node.get_current_train_iteration() == 0)
    inp = mdp.numx.zeros([1,5])
    out = node(inp)
    assert (node.get_current_train_iteration() == 1)


def test_batch_training_type():
    node = BogusOnlineNode()
    node.set_training_type('batch')
    assert (node.get_current_train_iteration() == 0)

    inp = mdp.numx.zeros([10,5])
    for i in range(5):
        node.train(inp)
        assert (node.get_current_train_iteration() == inp.shape[0]*(i+1))
        out = node.execute(inp)
        rinp = node.inverse(out)
        assert_array_equal(inp, rinp)
        assert(node.count==(i+1))

def test_multiple_training_phases():
    node = BogusOnlineNodeMultiplePhases()
    inp = mdp.numx.zeros([2,5])
    node.train(inp)
    assert_array_equal(node.count, (node.get_remaining_train_phase()-1)*mdp.numx.ones([1,inp.shape[1]]))

    node = BogusOnlineNodeMultiplePhases()
    node.set_training_type('batch')
    inp = mdp.numx.zeros([2,5])
    node.train(inp)
    assert_array_equal(node.count, (node.get_remaining_train_phase()-1)*mdp.numx.ones(inp.shape))

def test_batch_multiple_training_phases_train_iteration_with_blank_execute():
    node = BogusOnlineNodeMultiplePhasesWithBlankTrainSeq()
    node.set_training_type('batch')
    inp = mdp.numx.zeros([2,5])
    node.train(inp)
    assert (node.get_current_train_iteration() == inp.shape[0])

def test_node_add_online_node():
    flow = BogusOnlineNode() + BogusOnlineNode()
    assert(isinstance(flow, mdp.OnlineFlow))
    inp = mdp.numx.zeros([2,5])
    flow.train(inp)
    out = flow.execute(inp)
    assert_array_equal(mdp.numx.ones(inp.shape)*2, out)
    for node in flow:
        assert(node.is_training())

def test_node_add_trainable_classic_node():
    flow = BogusOnlineNode() + BogusNodeTrainable()
    assert(not isinstance(flow, mdp.OnlineFlow))
    assert(isinstance(flow, mdp.Flow))
    inp = mdp.numx.zeros([2,5])
    flow.train(inp)
    out = flow.execute(inp)
    assert_array_equal(mdp.numx.ones(inp.shape), out)
    assert(not flow[1].is_training())

def test_node_add_trainable_classic_node_swapped():
    # check with swapped add members
    flow =  BogusNodeTrainable() + BogusOnlineNode()
    assert(not isinstance(flow, mdp.OnlineFlow))
    assert(isinstance(flow, mdp.Flow))
    inp = mdp.numx.zeros([2,5])
    flow.train(inp)
    out = flow.execute(inp)
    assert_array_equal(mdp.numx.ones(inp.shape), out)
    #check if all the nodes are closed.
    for node in flow:
        assert(not node.is_training())

def test_numx_rng():
    rng = mdp.numx_rand.RandomState(seed=3)
    node = BogusOnlineNode(numx_rng=rng)
    inp = mdp.numx.zeros([2,5])
    node.train(inp)
    assert(node.numx_rng == rng)


