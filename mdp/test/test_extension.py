
import unittest

import mdp
import sys


class TestMDPExtensions(unittest.TestCase):
    
    def tearDown(self):
        """Deactivate all extensions and remove testing extensions."""
        mdp.deactivate_extensions(mdp.get_active_extensions())
        for key in mdp.get_extensions().copy():
            if key.startswith("__test"):
                del mdp.get_extensions()[key]
    
    def testSimpleExtension(self):
        """Test for a single new extension."""
        class TestExtensionNode(mdp.ExtensionNode):
            extension_name = "__test"
            def _testtest(self):
                pass
            _testtest_attr = 1337
        class TestSFANode(TestExtensionNode, mdp.nodes.SFANode):
            def _testtest(self):
                return 42
            _testtest_attr = 1338
        sfa_node = mdp.nodes.SFANode()
        mdp.activate_extension("__test")
        self.assert_(sfa_node._testtest() == 42) 
        self.assert_(sfa_node._testtest_attr == 1338) 
        mdp.deactivate_extension("__test")
        self.assert_(not hasattr(mdp.nodes.SFANode, "_testtest")) 
        
    def testDecoratorExtension(self):
        """Test extension decorator with a single new extension."""
        class TestExtensionNode(mdp.ExtensionNode):
            extension_name = "__test"
            def _testtest(self):
                pass
        @mdp.extension_method("__test", mdp.nodes.SFANode, "_testtest")
        def _sfa_testtest(self):
            return 42
        @mdp.extension_method("__test", mdp.nodes.SFA2Node)
        def _testtest(self):
            return 42 + _sfa_testtest(self)
        sfa_node = mdp.nodes.SFANode()
        sfa2_node = mdp.nodes.SFA2Node()
        mdp.activate_extension("__test")
        self.assert_(sfa_node._testtest() == 42) 
        self.assert_(sfa2_node._testtest() == 84)
        mdp.deactivate_extension("__test")
        self.assert_(not hasattr(mdp.nodes.SFANode, "_testtest")) 
        self.assert_(not hasattr(mdp.nodes.SFA2Node, "_testtest")) 
        
    def testDecoratorInheritance(self):
        """Test inhertiance with decorators for a single new extension."""
        class TestExtensionNode(mdp.ExtensionNode):
            extension_name = "__test"
            def _testtest(self):
                pass
        @mdp.extension_method("__test", mdp.nodes.SFANode, "_testtest")
        def _sfa_testtest(self):
            return 42
        @mdp.extension_method("__test", mdp.nodes.SFA2Node)
        def _testtest(self):
            return 42 + super(mdp.nodes.SFA2Node, self)._testtest()
        sfa_node = mdp.nodes.SFANode()
        sfa2_node = mdp.nodes.SFA2Node()
        mdp.activate_extension("__test")
        self.assert_(sfa_node._testtest() == 42) 
        self.assert_(sfa2_node._testtest() == 84)
    
    def testExtensionInheritance(self):
        """Test inheritance of extension nodes."""
        class TestExtensionNode(mdp.ExtensionNode):
            extension_name = "__test"
            def _testtest(self):
                pass
        class TestSFANode(TestExtensionNode, mdp.nodes.SFANode):
            def _testtest(self):
                return 42
            _testtest_attr = 1337
        class TestSFA2Node(TestSFANode, mdp.nodes.SFA2Node):
            def _testtest(self):
                if sys.version_info[0] < 3:
                    return TestSFANode._testtest.__func__(self)
                else:
                    return TestSFANode._testtest(self)
        sfa2_node = mdp.nodes.SFA2Node()
        mdp.activate_extension("__test")
        self.assert_(sfa2_node._testtest() == 42)
        self.assert_(sfa2_node._testtest_attr == 1337)
        
    def testExtensionInheritance2(self):
        """Test inheritance of extension nodes, using super."""
        class TestExtensionNode(mdp.ExtensionNode):
            extension_name = "__test"
            def _testtest(self):
                pass
        class TestSFANode(TestExtensionNode, mdp.nodes.SFANode):
            def _testtest(self):
                return 42
        class TestSFA2Node(mdp.nodes.SFA2Node, TestSFANode):
            def _testtest(self):
                return super(mdp.nodes.SFA2Node, self)._testtest()
        sfa2_node = mdp.nodes.SFA2Node()
        mdp.activate_extension("__test")
        self.assert_(sfa2_node._testtest() == 42)
        
    def testExtensionInheritance3(self):
        """Test explicit use of extension nodes and inheritance."""
        class TestExtensionNode(mdp.ExtensionNode):
            extension_name = "__test"
            def _testtest(self):
                pass
        class TestSFANode(TestExtensionNode, mdp.nodes.SFANode):
            def _testtest(self):
                return 42
        # Note the inheritance order, otherwise this would not work.
        class TestSFA2Node(mdp.nodes.SFA2Node, TestSFANode):
            def _testtest(self):
                return super(mdp.nodes.SFA2Node, self)._testtest()
        sfa2_node = TestSFA2Node()
        self.assert_(sfa2_node._testtest() == 42) 
        
    def testMultipleExtensions(self):
        """Test behavior of multiple extensions."""
        class Test1ExtensionNode(mdp.ExtensionNode, mdp.Node):
            extension_name = "__test1"
            def _testtest1(self):
                pass
        class Test2ExtensionNode(mdp.ExtensionNode, mdp.Node):
            extension_name = "__test2"
            def _testtest2(self):
                pass
        mdp.activate_extension("__test1")
        node = mdp.Node()
        node._testtest1()
        mdp.activate_extension("__test2")
        node._testtest2()
        mdp.deactivate_extension("__test1")
        self.assert_(not hasattr(mdp.nodes.SFANode, "_testtest1"))
        mdp.activate_extension("__test1")
        node._testtest1()
        mdp.deactivate_extensions(["__test1", "__test2"])
        self.assert_(not hasattr(mdp.nodes.SFANode, "_testtest1"))
        self.assert_(not hasattr(mdp.nodes.SFANode, "_testtest2"))
        
    def testExtCollision(self):
        """Test the check for method name collision."""
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

    def testExtensionInheritanceInjection(self):
        """Test the injection of inherited methods"""
        class TestNode(object):
            def _test1(self):
                return 0 
        class TestExtensionNode(mdp.ExtensionNode):
            extension_name = "__test"
            def _test1(self):
                return 1
            def _test2(self):
                return 2
            def _test3(self):
                return 3
        class TestNodeExt(TestExtensionNode, TestNode): 
            def _test2(self):
                return "2b"
        @mdp.extension_method("__test", TestNode)
        def _test4(self):
            return 4
        test_node = TestNode()
        mdp.activate_extension("__test")
        self.assert_(test_node._test1() == 1)
        self.assert_(test_node._test2() == "2b")
        self.assert_(test_node._test3() == 3)
        self.assert_(test_node._test4() == 4)
        mdp.deactivate_extension("__test") 
        self.assert_(test_node._test1() == 0)
        self.assert_(not hasattr(test_node, "_test2"))
        self.assert_(not hasattr(test_node, "_test3"))
        self.assert_(not hasattr(test_node, "_test4"))
 
    def testExtensionInheritanceInjectionNonExtension(self):
        """Test non_extension method injection."""
        class TestExtensionNode(mdp.ExtensionNode):
            extension_name = "__test"
            def _execute(self):
                return 0
        class TestNode(mdp.Node):
            # no _execute method
            pass
        class ExtendedTestNode(TestExtensionNode, TestNode):
            pass
        test_node = TestNode()
        mdp.activate_extension('__test')
        self.assert_(hasattr(test_node, "_non_extension__execute"))
        mdp.deactivate_extension('__test')
        self.assert_(not hasattr(test_node, "_non_extension__execute"))
        self.assert_(not hasattr(test_node, "_extension_for__execute"))
        # test that the non-native _execute has been completely removed
        self.assert_("_execute" not in test_node.__class__.__dict__)
        
    def testExtensionInheritanceInjectionNonExtension2(self):
        """Test non_extension method injection."""
        class TestExtensionNode(mdp.ExtensionNode):
            extension_name = "__test"
            def _execute(self):
                return 0
        class TestNode(mdp.Node):
            def _execute(self):
                return 1
        class ExtendedTestNode(TestExtensionNode, TestNode):
            pass
        test_node = TestNode()
        mdp.activate_extension('__test')
        # test that non-extended attribute has been added as well
        self.assert_(hasattr(test_node, "_non_extension__execute"))
        mdp.deactivate_extension('__test')
        self.assert_(not hasattr(test_node, "_non_extension__execute"))
        self.assert_(not hasattr(test_node, "_extension_for__execute"))
        # test that the native _execute has been preserved
        self.assert_("_execute" in test_node.__class__.__dict__)
        
    def testExtensionInheritanceTwoExtensions(self):
        """Test non_extension injection for multiple extensions."""
        class Test1ExtensionNode(mdp.ExtensionNode):
            extension_name = "__test1"
            def _execute(self):
                return 1
        class Test2ExtensionNode(mdp.ExtensionNode):
            extension_name = "__test2"
        class Test3ExtensionNode(mdp.ExtensionNode):
            extension_name = "__test3"
            def _execute(self):
                return "3a"
        class TestNode1(mdp.Node):
            pass
        class TestNode2(TestNode1):
            pass
        class ExtendedTest1Node2(Test1ExtensionNode, TestNode2):
            pass
        class ExtendedTest2Node1(Test2ExtensionNode, TestNode1):
            def _execute(self):
                return 2
        class ExtendedTest3Node1(Test3ExtensionNode, TestNode1):
            def _execute(self):
                return "3b"
        test_node = TestNode2()
        mdp.activate_extension('__test2')
        self.assert_(test_node._execute() == 2)
        mdp.deactivate_extension('__test2')
        # in this order TestNode2 should get execute from __test1,
        # the later addition by __test1 to TestNode1 doesn't matter
        mdp.activate_extensions(['__test1', '__test2'])
        self.assert_(test_node._execute() == 1)
        mdp.deactivate_extensions(['__test2', '__test1'])
        # now activate in inverse order
        # TestNode2 already gets _execute from __test2, but that is still
        # overriden by __test1, thats how its registered in _extensions
        mdp.activate_extensions(['__test2', '__test1'])
        self.assert_(test_node._execute() == 1)
        mdp.deactivate_extensions(['__test2', '__test1'])
        ## now the same with extension 3
        mdp.activate_extension('__test3')
        self.assert_(test_node._execute() == "3b")
        mdp.deactivate_extension('__test3')
        # __test3 does not override, since the _execute slot for Node2
        # was first filled by __test1
        mdp.activate_extensions(['__test3', '__test1'])
        self.assert_(test_node._execute() == 1)
        mdp.deactivate_extensions(['__test3', '__test1'])
        # inverse order
        mdp.activate_extensions(['__test1', '__test3'])
        self.assert_(test_node._execute() == 1)
        mdp.deactivate_extensions(['__test2', '__test1'])

def get_suite(testname=None):
    # this suite just ignores the testname argument
    # you can't select tests by name here!
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMDPExtensions))
    return suite
            
if __name__ == '__main__':
    unittest.main() 

