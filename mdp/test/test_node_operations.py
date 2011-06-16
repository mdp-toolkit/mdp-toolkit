from __future__ import with_statement

import os
import tempfile
import cPickle
import mdp
from _tools import BogusNode, BogusMultiNode, BogusNodeTrainable
uniform = mdp.numx_rand.random
MAT_DIM = (500,5)

def test_Node_copy():
    test_list = [1,2,3]
    generic_node = mdp.Node()
    generic_node.dummy_attr = test_list
    copy_node = generic_node.copy()
    assert generic_node.dummy_attr == copy_node.dummy_attr,\
           'Node copy method did not work'
    copy_node.dummy_attr[0] = 10
    assert generic_node.dummy_attr != copy_node.dummy_attr,\
           'Node copy method did not work'

def test_Node_copy_with_arrays_and_subnodes():
    node = mdp.Node()
    node.node = mdp.Node()
    node.node.x = mdp.numx.zeros((2,2))
    node2 = node.copy()
    assert hasattr(node2, 'node')
    assert mdp.numx.all(node2.node.x == node.node.x) 

def test_Node_copy_with_lambdas():
    generic_node = mdp.Node()
    generic_node.lambda_function = lambda: 1
    copy_node = generic_node.copy()
    
def test_Node_save():
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
    with open(dummy_file, 'rb') as flh:
        copy_node = cPickle.load(flh)
    os.remove(dummy_file)
    assert generic_node.dummy_attr == copy_node.dummy_attr,\
           'Node save (file) method did not work'
    copy_node.dummy_attr[0] = 10
    assert generic_node.dummy_attr != copy_node.dummy_attr,\
           'Node save (file) method did not work'

def test_Node_multiple_training_phases():
    x = uniform(size=MAT_DIM)
    node = BogusMultiNode()
    phases = node.get_remaining_train_phase()
    for i in xrange(phases):
        assert node.get_current_train_phase() == i
        assert not node._train_phase_started
        node.train(x)
        assert node._train_phase_started
        node.stop_training()

    assert not node.is_training()

def test_Node_execution_without_training():
    x = uniform(size=MAT_DIM)
    # try execution without training: single train phase
    node = BogusNodeTrainable()
    node.execute(x)
    assert hasattr(node, 'bogus_attr')
    # multiple train phases
    node = BogusMultiNode()
    node.execute(x)
    assert node.visited == [1, 2, 3, 4]

class AncestorNoDoc(mdp.Node):
    def _train(self, x, foo=None):
        print 'Ancestor Foo:', foo

class ChildNoDoc(AncestorNoDoc):
    def _train(self, x, foo=None, foo2=None):
        print 'Child Foo:', foo2

class AncestorDoc(mdp.Node):
    def _train(self, x, foo=None):
        """docs ancestor"""
        print 'Ancestor Foo:', foo

class ChildDoc(AncestorDoc):
    def _train(self, x, foo=None, foo2=None):
        """docs child"""
        print 'Child Foo:', foo2

def test_Node_docs_overwriting():
    x = mdp.numx_rand.random((10,10))
    AncestorNoDoc().train(x, foo='bar')
    ChildNoDoc().train(x,foo2='bar2')
    AncestorDoc().train(x, foo='bar')
    ChildDoc().train(x,foo2='bar2')
