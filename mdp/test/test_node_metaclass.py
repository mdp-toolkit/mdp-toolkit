from __future__ import with_statement

import mdp
X = mdp.numx_rand.random(size=(500,5))

def test_docstrings():
    # first try on a subclass of Node if
    # the docstring is exported to the public method
    class AncestorNode(mdp.Node):
        def _train(self, x):
            """doc ancestor"""
            pass
    assert AncestorNode.train.__doc__ == "doc ancestor"

    # now try on a subclass of it
    class ChildNode(AncestorNode):
        def _train(self, x):
            """doc child"""
            pass
    assert ChildNode.train.__doc__ == "doc child"

def test_signatures_no_doc():
    # first try on a subclass of Node if
    # the signature is exported to the public method
    class AncestorNode(mdp.Node):
        def _train(self, x, foo=None):
            pass
    AncestorNode().train(X, foo='abc')

    # now try on a subclass of it
    class ChildNode(AncestorNode):
        def _train(self, x, foo2=None):
            pass
    ChildNode().train(X, foo2='abc')

def test_signatures_with_doc():
    # first try on a subclass of Node if
    # the signature and the docstring are exported to
    # the public method
    class AncestorNode(mdp.Node):
        def _train(self, x, foo=None):
            """doc ancestor"""
            pass
    assert AncestorNode.train.__doc__ == "doc ancestor"
    AncestorNode().train(X, foo='abc')

    # now try on a subclass of it
    class ChildNode(AncestorNode):
        def _train(self, x, foo2=None):
            """doc child"""
            pass
    assert ChildNode.train.__doc__ == "doc child"
    ChildNode().train(X, foo2='abc')
