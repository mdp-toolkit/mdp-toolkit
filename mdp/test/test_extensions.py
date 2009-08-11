
import unittest

import mdp


class TestMDPExtensions(unittest.TestCase):
    
    def testSimpleExtension(self):
        """Basic test with a single new extension."""
        class TestExtensionNode(mdp.ExtensionNode):
            extension_name = "__test"
            def _testtest(self):
                pass
        class TestSFANode(TestExtensionNode, mdp.nodes.SFANode):
            def _testtest(self):
                return 42
        sfa_node = mdp.nodes.SFANode()
        mdp.activate_extension("__test")
        self.assert_(sfa_node._testtest() == 42) 
        mdp.deactivate_extension("__test")
        self.assert_(not hasattr(mdp.nodes.SFANode, "_testtest")) 
        del mdp.get_extensions()["__test"]
        
    def testExtCollision(self):
        """Test check for method name collision."""
        class Test1ExtensionNode(mdp.ExtensionNode, mdp.Node):
            extension_name = "__test1"
            def _testtest(self):
                pass
        class Test2ExtensionNode(mdp.ExtensionNode, mdp.Node):
            extension_name = "__test2"
            def _testtest(self):
                pass
        self.assertRaises(mdp.ExtensionException,
                    lambda: mdp.activate_extensions(["__test1", "__test2"]))
        # none of the extension should be active after the exception
        self.assert_(not hasattr(mdp.Node, "_testtest"))
        del mdp.get_extensions()["__test1"]
        del mdp.get_extensions()["__test2"]
        
        
def get_suite(testname=None):
    # this suite just ignores the testname argument
    # you can't select tests by name here!
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMDPExtensions))
    return suite
            
if __name__ == '__main__':
    unittest.main() 

