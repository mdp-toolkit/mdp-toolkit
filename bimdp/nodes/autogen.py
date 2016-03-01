"""
Module to automatically create a module with BiMDP versions of MDP nodes.
"""
from future import standard_library
standard_library.install_aliases()
from builtins import str

import inspect
import mdp
from io import StringIO

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


def _binode_code(write, node_class, modulename, base_classname="BiNode",
                 old_classname="Node"):
    """Write code for BiMDP versions of normal node classes into module file.
    
    It preserves the signature, which is useful for introspection (this is
    used by the ParallelNode _default_fork implementation).

    write -- Function to write out code.
    node_class -- Node class for which the new node class will be created.
    modulename -- Name of the module where the node is from.
    base_classname -- Base class to be used for the new nodes.
    old_classname -- Name of the original base class, which will be replaced
        in the new class name.
    """
    node_name = node_class.__name__
    binode_name = node_name[:-len(old_classname)] + base_classname
    write('class %s(%s, %s.%s):' %
          (binode_name, base_classname, modulename, node_name))
    docstring = ("Automatically created %s version of %s." %
                 (base_classname, node_name))
    write('\n    """%s"""' % docstring)
    ## define the init method explicitly to preserve the signature
    docstring = node_class.__init__.__doc__
    args, varargs, varkw, defaults = inspect.getargspec(node_class.__init__)
    args.remove('self')
    args += ('node_id', 'stop_result')
    defaults += (None, None)
    if defaults is None:
        defaults = []
    first_default = len(args) - len(defaults)
    write('\n    def __init__(self')
    write(''.join(', ' + arg for arg in args[:-len(defaults)]))
    write(''.join(', ' + arg + '=' + repr(defaults[i_arg])
                  for i_arg, arg in enumerate(args[first_default:])))
    if varargs:
        write(', *%s' % varargs)
    # always support kwargs, to prevent multiple-inheritance issues
    if not varkw:
        varkw = "kwargs"
    write(', **%s' % varkw)
    write('):')
    if docstring:
        write('\n        """%s"""' % docstring)
    write('\n        super(%s, self).__init__(' % binode_name)
    write(', '.join('%s=%s' % (arg, arg) for arg in args))
    if varargs:
        if args:
            write(', ')
        write('*%s' % varargs)
    if args or varargs:
        write(', ')
    write('**%s' % varkw)
    write(')\n\n')


def _binode_module(write, node_classes, modulename="mdp.nodes",
                   base_classname="BiNode", old_classname="Node",
                   base_import="from bimdp import BiNode"):
    """Write code for BiMDP versions of normal node classes into module file.

    write -- Function to write out code.
    node_classes -- List of node classes for which binodes are created.
    modulename -- Name of the module where the node is from.
    base_classname -- Base class to be used for the new nodes.
    old_classname -- Name of the original base class, which will be replaced
        in the new class name.
    base_import -- Inmport line for the base_class.
    """
    write('"""\nAUTOMATICALLY GENERATED CODE, DO NOT MODIFY!\n\n')
    write('Edit and run autogen.py instead to overwrite this module.\n"""')
    write('\n\nimport %s\n' % modulename)
    write(base_import + '\n\n')
    for node_class in node_classes:
        _binode_code(write, node_class, modulename,
                     base_classname=base_classname,
                     old_classname=old_classname)


def _get_unicode_write(fid):
    def write(txt):
        if type(txt) is str:
            fid.write(txt)
        else:
            fid.write(str(txt, encoding='utf-8'))
    return write


def binodes_code():
    """Generate and import the BiNode wrappers for MDP Nodes."""
    fid = StringIO()
    nodes = (node for node in
                _get_node_subclasses(node_class=mdp.Node, module=mdp.nodes)
             if not issubclass(node, mdp.ClassifierNode) and
                 node.__name__ not in NOAUTOGEN_MDP_NODES)
    _binode_module(_get_unicode_write(fid), nodes)
    return fid.getvalue()
    
def biclassifiers_code():
    """Generate and import the BiClassifier wrappers for ClassifierNodes."""
    fid = StringIO()
    def write(txt):
        if type(txt) is str:
            fid.write(txt)
        else:
            fid.write(str(txt, encoding='utf-8'))
    nodes = (node for node in
                _get_node_subclasses(node_class=mdp.ClassifierNode,
                                     module=mdp.nodes)
             if node.__name__ not in NOAUTOGEN_MDP_CLASSIFIERS)
    _binode_module(_get_unicode_write(fid), nodes,
                   base_classname="BiClassifier",
                   old_classname="Classifier",
                   base_import="from bimdp import BiClassifier")
    return fid.getvalue()
