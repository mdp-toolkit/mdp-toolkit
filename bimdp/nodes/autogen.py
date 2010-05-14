"""
Module to automatically create a module with BiMDP versions of MDP nodes.

Run this module to overwrite the autogen_binodes module with a new version.
"""

import inspect
import mdp

# Note: 'NoiseNode' was removed because the function argument default value
#    causes problems.
AUTOMATIC_MDP_NODES = [
    'AdaptiveCutoffNode', 'CuBICANode', 'CutoffNode', 'EtaComputerNode',
    'FANode', 'FDANode', 'FastICANode', 'GrowingNeuralGasExpansionNode',
    'GrowingNeuralGasNode', 'HLLENode', 'HistogramNode', 'HitParadeNode',
    'ICANode', 'ISFANode', 'IdentityNode', 'JADENode', 'LLENode',
    'LinearRegressionNode', 'NIPALSNode', 'NormalNoiseNode',
    'PCANode', 'PerceptronClassifier', 'PolynomialExpansionNode',
    'QuadraticExpansionNode', 'RBFExpansionNode', 'RBMNode',
    'RBMWithLabelsNode', 'SFA2Node', 'SFANode',
    'SignumClassifier', 'SimpleMarkovClassifier', 'TDSEPNode',
    'TimeFramesNode', 'WhiteningNode', 'XSFANode',
]

# TODO: use a special wrapper for classifier nodes, these are currently ignored
AUTOMATIC_MDP_CLASSIFIERS = [
     'SignumClassifier', 'PerceptronClassifier',
     'SimpleMarkovClassifier', 'DiscreteHopfieldClassifier',
     'KMeansClassifier',
     # 'LibSVMClassifier', 'ShogunSVMClassifier'
]

# this function is currently not needed, but can be used for node_classes
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

def _write_single_node(fid, node_class, modulename, base_classname="BiNode",
                       old_classname="Node"):
    """Write code for BiMDP versions of normal node classes into module file.
    
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
        err = ("varargs are not supported by autogen, " +
               "please disable the automatic node creation for class " +
               node_name)
        raise Exception(err)
    if varkw:
        fid.write(', *%s' % varkw)
    fid.write('):')
    if docstring:
        fid.write('\n        """%s"""' % docstring)
    fid.write('\n        super(%s, self).__init__(' % binode_name)
    fid.write(', '.join('%s=%s' % (arg, arg) for arg in args))
    if varkw:
        if not args:
            fid.write(', ')
        fid.write('**%s' % varkw)
    fid.write(')\n\n')

def _write_node_file(fid, node_classes, modulename="mdp.nodes",
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
        _write_single_node(fid, node_class, modulename,
                           base_classname=base_classname,
                           old_classname=old_classname)

if __name__ == "__main__":
    ## create file with binode classes
    filename = "autogen_binodes.py"
    autogen_file = open(filename, 'w')
    try:
        _write_node_file(autogen_file,
                         (getattr(mdp.nodes, node_name)
                          for node_name in AUTOMATIC_MDP_NODES))
    finally:
        autogen_file.close()
    print "wrote auto-generated code into file %s" % filename
    ## create file with classifier classes
    filename = "autogen_biclassifiers.py"
    autogen_file = open(filename, 'w')
    try:
        _write_node_file(autogen_file,
                (getattr(mdp.nodes, node_name)
                 for node_name in AUTOMATIC_MDP_CLASSIFIERS),
                base_classname="BiClassifier",
                old_classname="Classifier",
                base_import="from bimdp import BiClassifier")
    finally:
        autogen_file.close()
    print "wrote auto-generated code into file %s" % filename
