
import unittest

import mdp
from mdp import utils, numx, numx_rand


class TestMDPExtensions(unittest.TestCase):
    
    def test_Extension(self):
        """Test a new extension."""
        
        class TestExtensionNode(mdp.ExtensionNode):
            extension_name = "test"
            def test(self):
                pass
            
        class TestSFANode(TestExtensionNode, mdp.nodes.SFANode):
            def test(self):
                return 42
        
        sfa_node = mdp.nodes.SFANode()
        mdp.activate_extension("test")
        self.assert_(sfa_node.test() == 42) 
        mdp.deactivate_extension("test")
        self.assert_(not hasattr(mdp.nodes.SFANode, "test")) 
        del mdp.get_extensions()["test"]
        
        
def get_suite(testname=None):
    # this suite just ignores the testname argument
    # you can't select tests by name here!
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMDPExtensions))
    return suite
            
if __name__ == '__main__':
    unittest.main() 

