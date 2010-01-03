
import unittest

import mdp
n = mdp.numx

import binet


class TestBiNode(unittest.TestCase):

    def test_idnode(self):
        """Perform a basic test on the IdentityBiNode.
        
        Instantiation is tested and it should perform like an id node, but 
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
        stop_result = ({"test": 0}, 1)
        bi_sfa_node = BiSFANode(stop_result=stop_result,
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
            elif type(result) == n.ndarray:
                result = None
            rec_execute_results.append(result) 
        self.assertTrue((rec_execute_results == execute_results + [None, None]))
        self.assertTrue(jumpnode.loop_counter == 5)
        
    def test_node_bi(self):
        """Test the message and stop_message of JumpBiNode."""
        tmsg = {"test value": n.zeros((10))}  # test msg
        execute_results = [(tmsg, "test", tmsg, "test"), None, 
                           (tmsg, "e3", tmsg, "e4")]
        stop_message_results = [(tmsg, "test"), None]
        jumpnode = binet.JumpBiNode(
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

#class _DummyUDNode(binet.UpDownBiNode):
#    def __init__(self, **kwargs):
#        super(_DummyUDNode, self).__init__(**kwargs)
#        self._up, self._down = 0, 0
#        self._trained = False
#    def _up_pass(self, msg=None):
#        #print 'up', self._node_id
#        self._up += 1
#    def _down_pass(self, y, top=False, msg=None):
#        #print '\ndown', self._node_id
#        self._down += 1
#        self._down_y = y
#        self._is_top = top
#        return y
#    def _train(self, x):
#        #print 'train'
#        self._trained = True

#class TestUpDownNode(unittest.TestCase):
#
#    def test_updown(self):
#        NUPDOWN = 3
#        flow = binet.BiFlow([_DummyUDNode(node_id='bottom'),
#                             _DummyUDNode(node_id='top'),
#                             binet.TopUpDownBiNode(bottom_id='bottom',
#                                                   top_id='top')])
#        x = mdp.numx_rand.random((10,2))
#        flow.train([x, x, [x]*NUPDOWN])
#        y = flow(x)
#        
#        assert n.all(x==y)        
#        for i in range(len(flow)-1):
#            # check that the nodes are trained
#            assert not flow[i].is_training()
#            assert not flow[i].is_bi_training()
#            assert flow[i]._trained
#            # check the number of up-down phases
#            assert flow[i]._up == NUPDOWN
#            assert flow[i]._down == NUPDOWN
#            # check that the input-output is not saved at the end of
#            # the global training
#            assert not hasattr(flow[i], '_save_x')
#            assert not hasattr(flow[i], '_save_y')
#            # check that during the down phase one receives the
#            # output of the network
#            assert n.all(flow[i]._down_y==y)
#        # check that the 'top' message arrives at destination
#        assert not flow[0]._is_top
#        assert flow[-2]._is_top

def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBiNode))
    suite.addTest(unittest.makeSuite(TestStopTrainBiNode))
    suite.addTest(unittest.makeSuite(TestJumpBiNode))
    #suite.addTest(unittest.makeSuite(TestUpDownNode))
    return suite
            
if __name__ == '__main__':
    unittest.main() 
