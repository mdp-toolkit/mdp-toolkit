
import unittest

import mdp
n = mdp.numx

from bimdp import BiNode, MSG_ID_SEP, BiFlow, BiClassifier, binode_coroutine
from bimdp.nodes import (
    IdentityBiNode, SFABiNode, FDABiNode, SignumBiClassifier
)
from testnodes import JumpBiNode


class TestBiNode(unittest.TestCase):
    
    def test_msg_parsing1(self):
        """Test the message parsing and recombination."""
        class TestBiNode(BiNode):
            def _execute(self, x, a, b, d):
                self.a = a
                self.b = b
                self.d = d
                return x, {"g": 15, "z": 3}
            def is_trainable(self): return False
        binode = TestBiNode(node_id="test")
        b_key = "test" + MSG_ID_SEP + "b"
        d_key = "test" + MSG_ID_SEP + "d"
        msg = {"c": 12, b_key: 42, "a": 13, d_key: "bla"}
        _, msg = binode.execute(None, msg)
        self.assert_("a" in msg)
        self.assert_(b_key not in msg)
        self.assert_(d_key not in msg)
        self.assert_(binode.a == 13)
        self.assert_(binode.b == 42)
        self.assert_(binode.d == "bla")
        # test the message combination
        self.assert_(msg["g"] == 15)
        self.assert_(msg["z"] == 3)
        
    def test_msg_parsing2(self):
        """Test that an adressed argument is not found."""
        class TestBiNode(BiNode):
            def _execute(self, x, a, b):
                self.a = a
                self.b = b
            def is_trainable(self): return False
        binode = TestBiNode(node_id="test")
        b_key = "test" + MSG_ID_SEP + "b"
        # check that the 'd' key which is not an arg gets removed
        d_key = "test" + MSG_ID_SEP + "d"
        msg = {"c": 12, b_key: 42, "a": 13, d_key: "bla"}
        _, out_msg = binode.execute(None, msg)
        self.assert_(d_key not in out_msg)
        
    def test_msg_magic(self):
        """Test that the magic msg argument works."""
        class TestBiNode(BiNode):
            def _execute(self, x, a, msg, b):
                self.a = a
                self.b = b
                del msg["c"]
                msg["f"] = 1
                return x, msg
            def is_trainable(self): return False
        binode = TestBiNode(node_id="test")
        b_key = "test" + MSG_ID_SEP + "b"
        msg = {"c": 12, b_key: 42, "a": 13}
        _, msg = binode.execute(None, msg)
        self.assert_("a" in msg)
        self.assert_("c" not in msg)  # was deleted in _execute  
        self.assert_(msg["f"] == 1)
        self.assert_(b_key not in msg)
        self.assert_(binode.a == 13)
        self.assert_(binode.b == 42)
        
    def test_method_magic(self):
        """Test the magic method message key."""
        class TestBiNode(BiNode):
            def _test(self, x, a, b):
                self.a = a
                self.b = b
            def is_trainable(self): return False
        binode = TestBiNode(node_id="test")
        b_key = "test" + MSG_ID_SEP + "b"
        msg = {"c": 12, "a": 13, b_key: 42, "method": "test"}
        binode.execute(None, msg)
        self.assert_("a" in msg)
        self.assert_(b_key not in msg)
        self.assert_(binode.b == 42)
        
    def test_target_magic(self):
        """Test the magic target message key."""
        class TestBiNode(BiNode):
            def _execute(self, x, a, b):
                self.a = a
                self.b = b
            def is_trainable(self): return False
        binode = TestBiNode(node_id="test")
        b_key = "test" + MSG_ID_SEP + "b"
        target_key = "test" + MSG_ID_SEP + "target"
        msg = {"c": 12, b_key: 42, "a": 13, target_key: "test2"}
        result = binode.execute(None, msg)
        self.assert_(len(result) == 3)
        self.assert_(result[2] == "test2")
        
    def test_inverse_magic1(self):
        """Test the magic inverse method argument."""
        class TestBiNode(BiNode):
            def _inverse(self, x, a, b):
                self.a = a
                self.b = b
                y = n.zeros((len(x), self.input_dim))
                return y
            def is_trainable(self): return False
        binode = TestBiNode(node_id="test", input_dim=20, output_dim=10)
        b_key = "test" + MSG_ID_SEP + "b"
        msg = {"c": 12, "a": 13, b_key: 42, "method": "inverse"}
        x = n.zeros((5, binode.output_dim))
        result = binode.execute(x, msg)
        self.assert_(len(result) == 3)
        self.assert_(result[2] == -1)
        self.assert_(result[0].shape == (5, 20))

    def test_inverse_magic2(self):
        """Test overriding the magic inverse target."""
        class TestBiNode(BiNode):
            def _inverse(self, x, a, b):
                self.a = a
                self.b = b
                y = n.zeros((len(x), self.input_dim))
                return y, None, "test2"
            def is_trainable(self): return False
        binode = TestBiNode(node_id="test", input_dim=20, output_dim=10)
        b_key = "test" + MSG_ID_SEP + "b"
        msg = {"c": 12, "a": 13, b_key: 42, "method": "inverse"}
        x = n.zeros((5, binode.output_dim))
        result = binode.execute(x, msg)
        self.assert_(result[2] == "test2")
    
    def test_stoptrain_result1(self):
        """Test that stop_result is handled correctly."""
        stop_result = ({"test": 0}, 1)
        bi_sfa_node = SFABiNode(stop_result=stop_result,
                                node_id="testing binode")
        self.assertTrue(bi_sfa_node.is_trainable())
        x = n.random.random((100,10))
        train_result = bi_sfa_node.train(x)
        self.assertTrue(train_result == None)
        self.assertTrue(bi_sfa_node.is_training())
        result = bi_sfa_node.stop_training()
        self.assertTrue(result == stop_result)
        self.assertTrue(bi_sfa_node.input_dim == 10)
        self.assertTrue(bi_sfa_node.output_dim == 10)
        self.assertTrue(bi_sfa_node.dtype == "float64")

    def test_stoptrain_result2(self):
        """Test that stop_result is handled correctly for multiple phases."""
        stop_result = [({"test": 0}, 1), ({"test2": 0}, 2)]
        binode = FDABiNode(stop_result=stop_result,
                           node_id="testing binode")
        x = n.random.random((100,10))
        msg = {"cl": n.zeros(len(x))}
        binode.train(x, msg)
        result = binode.stop_training()
        self.assertTrue(result == stop_result[0])
        binode.train(x, msg)
        result = binode.stop_training()
        self.assertTrue(result == stop_result[1])
        
    def test_stop_message_execute(self):
        """Test the magic execute method argument for stop_message."""
        class TestBiNode(BiNode):
            def _execute(self, x, a):
                self.a = a
                self.x = x
                y = n.zeros((len(x), self.output_dim))
                return y
            def is_trainable(self): return False
        binode = TestBiNode(input_dim=20, output_dim=10)
        x = n.ones((5, binode.input_dim))
        msg = {"x": x, "a": 13, "method": "execute"}
        result = binode.stop_message(msg)
        self.assert_(n.all(binode.x == x))
        self.assert_(binode.x.shape == (5, binode.input_dim))
        self.assert_(binode.a == 13)
        self.assert_(len(result) == 2)
        self.assert_(result[1] == 1)
        self.assert_(result[0]["x"].shape == (5, binode.output_dim))
        self.assertFalse(n.any(result[0]["x"]))
        
    def test_stop_message_inverse(self):
        """Test the magic inverse method argument for stop_message."""
        class TestBiNode(BiNode):
            def _inverse(self, x, a):
                self.a = a
                self.x = x
                y = n.zeros((len(x), self.input_dim))
                return y
            def is_trainable(self): return False
        binode = TestBiNode(input_dim=20, output_dim=10)
        x = n.ones((5, binode.output_dim))
        msg = {"x": x, "a": 13, "method": "inverse"}
        result = binode.stop_message(msg)
        self.assert_(n.all(binode.x == x))
        self.assert_(binode.x.shape == (5, binode.output_dim))
        self.assert_(binode.a == 13)
        self.assert_(len(result) == 2)
        self.assert_(result[1] == -1)
        self.assert_(result[0]["x"].shape == (5, binode.input_dim))
        self.assertFalse(n.any(result[0]["x"]))
        
    def test_flow_from_sum(self):
        """Test the special addition method for BiNode."""
        node1 = IdentityBiNode()
        node2 = mdp.Node()
        flow = node1 + node2
        self.assert_(type(flow) is BiFlow)
        node2 = IdentityBiNode()
        flow = node1 + node2
        self.assert_(type(flow) is BiFlow)
        self.assert_(len(flow) == 2)
        node3 = IdentityBiNode()
        flow = node1 + node2 + node3
        self.assert_(type(flow) is BiFlow)
        self.assert_(len(flow) == 3)
        node4 = IdentityBiNode()
        flow = node4 + flow
        self.assert_(type(flow) is BiFlow)
        self.assert_(len(flow) == 4)


class TestBiClassifierNode(unittest.TestCase):
    
    def test_biclassifier(self):
        """Test the BiClassifier base class."""
        class TestBiClassifier(BiClassifier):
            def _label(self, x):
                return "LABELS"
            def _prob(self, x):
                return "PROPS"
            def is_trainable(self):
                return False
        node = TestBiClassifier()
        x = n.empty((5,2))
        msg = {"return_labels": "test->",
               "return_probs": True}
        result = node.execute(x, msg)
        self.assert_(result[0] is x)
        self.assert_("labels" not in result[1])
        self.assert_(result[1]["probs"] == "PROPS")
        self.assert_(result[1][msg["return_labels"] + "labels"] == "LABELS")
        self.assert_("rank" not in result[1])
        msg = {"return_labels": None}
        result = node.execute(x,msg)
        self.assert_(result[0] is x)
        self.assert_("labels" not in result[1])
        self.assert_("prop" not in result[1])
        self.assert_("rank" not in result[1])
        
    def test_autogen_biclassifier(self):
        """Test that the autogenerated classifiers work."""
        node = SignumBiClassifier()
        msg = {"return_labels": True}
        # taken from the SignumClassifier unittest
        x = n.array([[1, 2, -3, -4], [1, 2, 3, 4]])
        result = node.execute(x, msg)
        self.assert_(result[0] is x)
        self.assert_(result[1]["labels"].tolist() == [-1, 1])
    

class TestIdentityBiNode(unittest.TestCase):
    
    def test_idnode(self):
        """Test the IdentityBiNode.
        
        Instantiation is tested and it should perform like an id node, but 
        accept msg arguments.
        """
        binode = IdentityBiNode(node_id="testing binode")
        x = n.random.random((10,5))
        msg = {"some array": n.random.random((10,3))}
        # see if msg causes no problem
        y, msg = binode.execute(x, msg)
        self.assertTrue(n.all(x==y))
        # see if missing msg causes problem
        y = binode.execute(x)
        self.assertTrue(n.all(x==y))

        
class TestJumpBiNode(unittest.TestCase):

    def test_node(self):
        """Test the JumpBiNode."""
        train_results = [[(0, "t1")], [None], [(3, "t3")]]
        stop_train_results = [None, (5, "st2"), (6, "st3")]
        execute_results = [(None, {}), None, (None, {}, "et4")]
        jumpnode = JumpBiNode(train_results=train_results, 
                              stop_train_results=stop_train_results, 
                              execute_results=execute_results)
        x = n.random.random((2,2))
        self.assertTrue(jumpnode.is_trainable())
        # training
        rec_train_results = []
        rec_stop_train_results = []
        for _ in range(len(train_results)):
            rec_train_results.append([jumpnode.train(x)])
            jumpnode.bi_reset()
            rec_stop_train_results.append(jumpnode.stop_training())
            jumpnode.bi_reset()
        self.assertTrue(not jumpnode.is_training())
        self.assertTrue(rec_train_results == train_results)
        self.assertTrue(rec_stop_train_results == rec_stop_train_results)
        # execution
        rec_execute_results = []
        for _ in range(4):  # note that this is more then the execute_targets
            rec_execute_results.append(jumpnode.execute(x))
        execute_results[1] = x
        execute_results.append(x)
        self.assertTrue((rec_execute_results == execute_results))
        self.assertTrue(jumpnode.loop_counter == 4)
        
    def test_node_bi(self):
        """Test the message and stop_message of JumpBiNode."""
        tmsg = {"test value": n.zeros((10))}  # test msg
        execute_results = [(tmsg, "test", tmsg, "test"), None, 
                           (tmsg, "e3", tmsg, "e4")]
        stop_message_results = [(tmsg, "test"), None]
        jumpnode = JumpBiNode(
                        execute_results=execute_results, 
                        stop_message_results=stop_message_results)
        x = n.random.random((10,5))
        self.assertTrue(not jumpnode.is_trainable())
        # stop_message results
        result = jumpnode.stop_message()
        self.assertTrue(result == stop_message_results[0])
        result = jumpnode.stop_message(tmsg)
        self.assertTrue(result is None)
        jumpnode.bi_reset()
        

class TestBiNodeCoroutine(unittest.TestCase):
    """Test the coroutine decorator and the related BiNode functionality."""
    
    def test_codecorator(self):
        """Test basic codecorator functionality."""
        
        class CoroutineBiNode(BiNode):
        
            def is_trainable(self):
                return False
            
            @binode_coroutine(["x", "alpha", "beta"])
            def _execute(self, x, alpha):
                """Blabla."""
                x, alpha, beta = yield (x, {"alpha": alpha, "beta": 2},
                                        self.node_id) 
                x, alpha, beta = yield (x, {"alpha": alpha+1, "beta": beta+2},
                                        self.node_id) 
                yield x, {"alpha": alpha, "beta": beta}
            
        node = CoroutineBiNode(node_id="conode")
        flow = BiFlow([node])
        x = n.random.random((3,2))
        y, msg = flow.execute(x, {"alpha": 3})
        self.assertEqual(msg["alpha"], 4)
        self.assertEqual(msg["beta"], 4)
        self.assertEqual(node.execute.__doc__, """Blabla.""")
        
    def test_codecorator2(self):
        """Test codecorator functionality with StopIteration."""
        
        class CoroutineBiNode(BiNode):
        
            def is_trainable(self):
                return False
            
            @binode_coroutine(["x", "alpha", "beta"])
            def _execute(self, x, alpha):
                x, alpha, beta = yield (x, {"alpha": alpha, "beta": 2},
                                        self.node_id) 
                x, alpha, beta = yield (x, {"alpha": alpha+1, "beta": beta+2},
                                        self.node_id) 
                raise StopIteration(x, {"alpha": alpha, "beta": beta})
            
        node = CoroutineBiNode(node_id="conode")
        flow = BiFlow([node])
        x = n.random.random((3,2))
        y, msg = flow.execute(x, {"alpha": 3})
        self.assertEqual(msg["alpha"], 4)
        self.assertEqual(msg["beta"], 4)
        
    def test_codecorator_defaults(self):
        """Test codecorator argument default values."""
        
        class CoroutineBiNode(BiNode):
        
            def is_trainable(self):
                return False
            
            @binode_coroutine(["x", "alpha", "beta"],
                                          defaults=(7,8))
            def _execute(self, x):
                x, alpha, beta = yield (x, None, self.node_id) 
                raise StopIteration(x, {"alpha": alpha, "beta": beta})
            
        node = CoroutineBiNode(node_id="conode")
        flow = BiFlow([node])
        x = n.random.random((3,2))
        y, msg = flow.execute(x)
        self.assertEqual(msg["alpha"], 7)
        self.assertEqual(msg["beta"], 8)
        
    def test_codecorator_stop_message(self):
        """Test codecorator functionality for stop_message phase."""
        
        # use this class to initialize the stop_messsage phase
        class DummyNode(BiNode):
            def _train(self, x):
                return None
            def _stop_training(self, alpha):
                return {"alpha": alpha}, 1
        
        class CoroutineBiNode(BiNode):
        
            def is_trainable(self):
                return False
            
            @binode_coroutine(["alpha", "beta"], stop_message=True)
            def _stop_message(self, alpha):
                alpha, beta = yield ({"alpha": alpha, "beta": 2}, self.node_id) 
                alpha, beta = yield ({"alpha": alpha+1, "beta": beta+2},
                                     self.node_id)
                # this data should be ignored, since no target is given
                self.alpha = alpha
                self.beta = beta
                raise StopIteration()
        
        node1 = DummyNode()
        node2 = CoroutineBiNode(node_id="conode")
        flow = node1 + node2
        x = n.random.random((3,2))
        flow.train([[x],[None]], stop_messages=[{"alpha": 3},None])
        self.assertEqual(node2.alpha, 4)
        self.assertEqual(node2.beta, 4)
        
    def test_codecorator_no_iteration(self):
        """Test codecorator corner case with no iterations."""
        
        class CoroutineBiNode(BiNode):
        
            def is_trainable(self):
                return False
            
            @binode_coroutine(["x"])
            def _execute(self, x):
                # at least one yield must be in a coroutine
                if False:
                    yield None
                raise StopIteration(None, {"a": 1}, self.node_id) 
        
        node1 = CoroutineBiNode()
        x = n.random.random((3,2))
        result = node1.execute(x)
        self.assertEqual(result, (None, {"a": 1}, None))
        
    def test_codecorator_reset1(self):
        """Test that codecorator correctly resets after termination."""
        
        class CoroutineBiNode(BiNode):
        
            def is_trainable(self):
                return False
            
            @binode_coroutine(["x"])
            def _execute(self, x, a, msg=None):
                # note that the a argument is required, drop message
                for _ in range(2):
                    x = yield x
                raise StopIteration(x) 
        
        node1 = CoroutineBiNode()
        x = n.random.random((3,2))
        # this inits the coroutine, a argument is needed
        node1.execute(x, {"a": 2})
        node1.execute(x)
        node1.execute(x)
        self.assert_(node1._coroutine_instances == {})
        # couroutine should be reset, a argument is needed again
        self.assertRaises(TypeError, lambda: node1.execute(x))
        
    def test_codecorator_reset2(self):
        """Test that codecorator correctly resets without yields."""
        
        class CoroutineBiNode(BiNode):
        
            def is_trainable(self):
                return False
            
            @binode_coroutine(["x"])
            def _execute(self, x, a, msg=None):
                if False:
                    yield
                raise StopIteration(x) 
        
        node1 = CoroutineBiNode()
        x = n.random.random((3,2))
        node1.execute(x, {"a": 2})
        self.assert_(node1._coroutine_instances == {})
        

def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBiNode))
    suite.addTest(unittest.makeSuite(TestBiClassifierNode))
    suite.addTest(unittest.makeSuite(TestIdentityBiNode))
    suite.addTest(unittest.makeSuite(TestJumpBiNode))
    suite.addTest(unittest.makeSuite(TestBiNodeCoroutine))
    return suite
            
if __name__ == '__main__':
    unittest.main() 
