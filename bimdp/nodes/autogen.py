"""
Module to automatically create a module with BiMDP versions of MDP nodes.

Run this module to overwrite the autogen_binodes module with a new version.
"""

import inspect
import mdp
from cStringIO import StringIO

# Blacklist of nodes that cause problems with autogeneration
NOAUTOGEN_MDP_NODES = [
    "NoiseNode"  # function default value causes trouble
]
NOAUTOGEN_MDP_CLASSIFIERS = []

def _get_node_subclasses(node_class=mdp.Node, module=mdp.nodes):
    """
    Return all node classes in module which are subclasses of node_class.
    """
    node_subclasses = []
    for node_subclass in (getattr(module, name) for name in dir(module)):
        if (isinstance(node_subclass, type) and
            issubclass(node_subclass, node_class)):
            node_subclasses.append(node_subclass)
    return node_subclasses

def _binode_code(fid, node_class, modulename, base_classname="BiNode",
                 old_classname="Node"):
    """Write code for BiMDP versions of normal node classes into module file.
    
    It preserves the signature, which is useful for introspection (this is
    used by the ParallelNode _default_fork implementation).

    fid -- File handle of the module file.
    node_class -- Node class for which the new node class will be created.
    modulename -- Name of the module where the node is from.
    base_classname -- Base class to be used for the new nodes.
    old_classname -- Name of the original base class, which will be replaced
        in the new class name.
    """
    node_name = node_class.__name__
    binode_name = node_name[:-len(old_classname)] + base_classname
    fid.write('class %s(%s, %s.%s):' %
              (binode_name, base_classname, modulename, node_name))
    docstring = ("Automatically created %s version of %s." %
                 (base_classname, node_name))
    fid.write('\n    """%s"""' % docstring)
    ## define the init method explicitly to preserve the signature
    docstring = node_class.__init__.__doc__
    args, varargs, varkw, defaults = inspect.getargspec(node_class.__init__)
    args.remove('self')
    args += ('node_id', 'stop_result')
    defaults += (None, None)
    if defaults is None:
        defaults = []
    first_default = len(args) - len(defaults)
    fid.write('\n    def __init__(self')
    fid.write(''.join(', ' + arg for arg in args[:-len(defaults)]))
    fid.write(''.join(', ' + arg + '=' + repr(defaults[i_arg])
                      for i_arg, arg in enumerate(args[first_default:])))
    if varargs:
        fid.write(', *%s' % varargs)
    # always support kwargs, to prevent multiple-inheritance issues 
    if not varkw:
        varkw = "kwargs"
    fid.write(', **%s' % varkw)
    fid.write('):')
    if docstring:
        fid.write('\n        """%s"""' % docstring)
    fid.write('\n        super(%s, self).__init__(' % binode_name)
    fid.write(', '.join('%s=%s' % (arg, arg) for arg in args))
    if varargs:
        if args:
            fid.write(', ')
        fid.write('*%s' % varargs)
    if args or varargs:
        fid.write(', ')
    fid.write('**%s' % varkw)
    fid.write(')\n\n')

def _binode_module(fid, node_classes, modulename="mdp.nodes",
                   base_classname="BiNode", old_classname="Node",
                   base_import="from bimdp import BiNode"):
    """Write code for BiMDP versions of normal node classes into module file.

    fid -- File handle of the module file.
    node_classes -- List of node classes for which binodes are created.
    modulename -- Name of the module where the node is from.
    base_classname -- Base class to be used for the new nodes.
    old_classname -- Name of the original base class, which will be replaced
        in the new class name.
    base_import -- Inmport line for the base_class.
    """
    fid.write('"""\nAUTOMATICALLY GENERATED CODE, DO NOT MODIFY!\n\n')
    fid.write('Edit and run autogen.py instead to overwrite this module.\n"""')
    fid.write('\n\nimport %s\n' % modulename)
    fid.write(base_import + '\n\n')
    for node_class in node_classes:
        _binode_code(fid, node_class, modulename,
                     base_classname=base_classname,
                     old_classname=old_classname)
        
def binodes_code():
    """Generate and import the BiNode wrappers for MDP Nodes."""
    fid = StringIO()
    nodes = (node for node in
                _get_node_subclasses(node_class=mdp.Node, module=mdp.nodes)
             if not issubclass(node, mdp.ClassifierNode) and
                 node.__name__ not in NOAUTOGEN_MDP_NODES)
    _binode_module(fid, nodes)
    return fid.getvalue()
    
def biclassifiers_code():
    """Generate and import the BiClassifier wrappers for ClassifierNodes."""
    fid = StringIO()
    nodes = (node for node in
                _get_node_subclasses(node_class=mdp.ClassifierNode,
                                     module=mdp.nodes)
             if node.__name__ not in NOAUTOGEN_MDP_CLASSIFIERS)
    _binode_module(fid, nodes, base_classname="BiClassifier",
                   old_classname="Classifier",
                   base_import="from bimdp import BiClassifier")
    return fid.getvalue()
