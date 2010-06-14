
import unittest
import numpy as np

import bimdp


class TestCodecorator(unittest.TestCase):
    
    def test_codecorator(self):
        """Test basic codecorator functionality."""
        
        class CoroutineBiNode(bimdp.nodes.CoroutineBiNodeMixin, bimdp.BiNode):
        
            def is_trainable(self):
                return False
            
            @bimdp.nodes.binode_coroutine(["x", "alpha", "beta"])
            def _execute(self, x, alpha):
                """Blabla."""
                x, alpha, beta = yield (x, {"alpha": alpha, "beta": 2},
                                        self.node_id) 
                x, alpha, beta = yield (x, {"alpha": alpha+1, "beta": beta+2},
                                        self.node_id) 
                yield x, {"alpha": alpha, "beta": beta}
            
        node = CoroutineBiNode(node_id="conode")
        flow = bimdp.BiFlow([node])
        x = np.random.random((3,2))
        y, msg = flow.execute(x, {"alpha": 3})
        self.assertEqual(msg["alpha"], 4)
        self.assertEqual(msg["beta"], 4)
        self.assertEqual(node.execute.__doc__, """Blabla.""")
        
    def test_codecorator2(self):
        """Test codecorator functionality with StopIteration."""
        
        class CoroutineBiNode(bimdp.nodes.CoroutineBiNodeMixin, bimdp.BiNode):
        
            def is_trainable(self):
                return False
            
            @bimdp.nodes.binode_coroutine(["x", "alpha", "beta"])
            def _execute(self, x, alpha):
                x, alpha, beta = yield (x, {"alpha": alpha, "beta": 2},
                                        self.node_id) 
                x, alpha, beta = yield (x, {"alpha": alpha+1, "beta": beta+2},
                                        self.node_id) 
                raise StopIteration(x, {"alpha": alpha, "beta": beta})
            
        node = CoroutineBiNode(node_id="conode")
        flow = bimdp.BiFlow([node])
        x = np.random.random((3,2))
        y, msg = flow.execute(x, {"alpha": 3})
        self.assertEqual(msg["alpha"], 4)
        self.assertEqual(msg["beta"], 4)
        
    def test_codecorator_defaults(self):
        """Test codecorator argument default values."""
        
        class CoroutineBiNode(bimdp.nodes.CoroutineBiNodeMixin, bimdp.BiNode):
        
            def is_trainable(self):
                return False
            
            @bimdp.nodes.binode_coroutine(["x", "alpha", "beta"],
                                          defaults=(7,8))
            def _execute(self, x):
                x, alpha, beta = yield (x, None, self.node_id) 
                raise StopIteration(x, {"alpha": alpha, "beta": beta})
            
        node = CoroutineBiNode(node_id="conode")
        flow = bimdp.BiFlow([node])
        x = np.random.random((3,2))
        y, msg = flow.execute(x)
        self.assertEqual(msg["alpha"], 7)
        self.assertEqual(msg["beta"], 8)
        
    def test_codecorator_stop_message(self):
        """Test codecorator functionality for stop_message phase."""
        
        # use this class to initialize the stop_messsage phase
        class DummyNode(bimdp.BiNode):
            def _train(self, x):
                return None
            def _stop_training(self, alpha):
                return {"alpha": alpha}, 1
        
        class CoroutineBiNode(bimdp.nodes.CoroutineBiNodeMixin, bimdp.BiNode):
        
            def is_trainable(self):
                return False
            
            @bimdp.nodes.binode_coroutine(["alpha", "beta"], stop_message=True)
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
        x = np.random.random((3,2))
        flow.train([[x],[None]], stop_messages=[{"alpha": 3},None])
        self.assertEqual(node2.alpha, 4)
        self.assertEqual(node2.beta, 4)
        
    def test_codecorator_no_iteration(self):
        """Test codecorator corner case with no iterations."""
        
        class CoroutineBiNode(bimdp.nodes.CoroutineBiNodeMixin, bimdp.BiNode):
        
            def is_trainable(self):
                return False
            
            @bimdp.nodes.binode_coroutine(["x"])
            def _execute(self, x):
                # at least one yield must be in a coroutine
                if False:
                    yield None
                raise StopIteration(None, {"a": 1}, self.node_id) 
        
        node1 = CoroutineBiNode()
        x = np.random.random((3,2))
        result = node1.execute(x)
        self.assertEqual(result, (None, {"a": 1}, None))
 

def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCodecorator))
    return suite
            
if __name__ == '__main__':
    unittest.main() 