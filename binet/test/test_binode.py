
import unittest

import numpy as n

import mdp

import hiphi.binet as binet


class TestBiNode(unittest.TestCase):

    def test_idnode(self):
        """Perform a basic test on the IdentityBiNode.
        
        Instanciation is tested and it should perform like an id node, but 
        accept msg arguments.
        """
        binode = binet.IdentityBiNode(node_id="testing binode")
        x = n.random.random((10,5))
        msg = {"some array": n.random.random((10,3))}
        # see if msg causes no problem
        y, msg = binode.execute(x, msg)
        self.assertTrue(n.all(x==y))
        # see if missing msg causes problem
        y = binode.execute(x)
        self.assertTrue(n.all(x==y))
        
        
class TestStopTrainBiNode(unittest.TestCase):

    def test_node(self):
        """Test a derived node of StopTrainJumpBiNode."""
        # define class for testing
        class BiSFANode(binet.BiNode, mdp.nodes.SFANode):
            pass
        stop_msg = {"test": 0}
        bi_sfa_node = BiSFANode(stop_msg=stop_msg,
                                  node_id="testing binode")
        self.assertTrue(bi_sfa_node.is_trainable())
        x = n.random.random((100,10))
        train_result = bi_sfa_node.train(x)
        self.assertTrue(train_result == None)
        self.assertTrue(bi_sfa_node.is_training())
        msg = bi_sfa_node.stop_training()
        self.assertTrue(msg == stop_msg)
        self.assertTrue(bi_sfa_node.input_dim == 10)
        self.assertTrue(bi_sfa_node.output_dim == 10)
        self.assertTrue(bi_sfa_node.dtype == "float64")
        
        
class TestJumpBiNode(unittest.TestCase):

    def test_node(self):
        """Test the JumpBiNode."""
        train_results = [(0, "t1"), None, (3, "t3")]
        stop_train_results = [None, (5, "st2"), (6, "st3")]
        execute_results = [(0, "et1"), None, (3, "et3", 4, "et4")]
        jumpnode = binet.JumpBiNode(train_results=train_results, 
                                    stop_train_results=stop_train_results, 
                                    execute_results=execute_results)
        x = n.random.random((10,5))
        self.assertTrue(jumpnode.is_trainable())
        # training
        rec_train_results = []
        rec_stop_train_results = []
        for _ in range(len(train_results)):
            rec_train_results.append(jumpnode.train(x))
            rec_stop_train_results.append(jumpnode.stop_training())
        self.assertTrue(not jumpnode.is_training())
        self.assertTrue(rec_train_results == train_results)
        self.assertTrue(rec_stop_train_results == rec_stop_train_results)
        # execution
        rec_execute_results = []
        for _ in range(5):  # note that this is more then the execute_targets
            result = jumpnode.execute(x)
            if type(result) == tuple:
                # skip y value
                result = result[1:]
            elif type(result) == mdp.numx.ndarray:
                result = None
            rec_execute_results.append(result) 
        self.assertTrue((rec_execute_results == execute_results + [None, None]))
        self.assertTrue(jumpnode.loop_counter == 5)
        
    def test_node_bi(self):
        """Test the message and stop_message of JumpBiNode."""
        tmsg = {"test value": n.zeros((10))}  # test msg
        execute_results = [(tmsg, "test", tmsg, "test"), None, 
                           (tmsg, "e3", tmsg, "e4")]
        message_results = [None, (tmsg, "bt1")]
        stop_message_results = [(tmsg, "test"), None]
        jumpnode = binet.JumpBiNode(
                        execute_results=execute_results, 
                        message_results=message_results,
                        stop_message_results=stop_message_results)
        x = n.random.random((10,5))
        self.assertTrue(not jumpnode.is_trainable())
        # execution with message called on self
        result = jumpnode.execute(x)
        self.assertTrue(result == (x,) + execute_results[0])
        result = jumpnode.message(result[3])
        self.assertTrue(result == message_results[1])
        result = jumpnode.execute(x)
        self.assertTrue(result == (x,) + execute_results[2])
        jumpnode.bi_reset()
        # stop_message results
        result = jumpnode.stop_message()
        self.assertTrue(result == stop_message_results[0])
        result = jumpnode.stop_message(tmsg)
        self.assertTrue(result == stop_message_results[0][0])
        jumpnode.bi_reset()


def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBiNode))
    suite.addTest(unittest.makeSuite(TestStopTrainBiNode))
    suite.addTest(unittest.makeSuite(TestJumpBiNode))
    return suite
            
if __name__ == '__main__':
    unittest.main() 