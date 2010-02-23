"""
Module to automatically create a module with BiMDP versions of MDP nodes.

Run this module to overwrite the autogen_binodes module with a new version.
"""

import inspect
import mdp

# trick to load BiNode without loading BiMDP (which would try to import the
# autogen_binodes module, leading to a chicken-egg situation).
import sys
bimdp_path = __file__
bimdp_path = bimdp_path[:bimdp_path.rfind("nodes")-1]
sys.path.append(bimdp_path)
from binode import BiNode, BiNodeException

# Note: 'NoiseNode' was removed because the function argument default value
#    causes problems.
AUTOMATIC_MDP_NODES = [
    'AdaptiveCutoffNode', 'CuBICANode', 'CutoffNode', 'EtaComputerNode',
    'FANode', 'FDANode', 'FastICANode', 'GrowingNeuralGasExpansionNode',
    'GrowingNeuralGasNode', 'HLLENode', 'HistogramNode', 'HitParadeNode',
    'ICANode', 'ISFANode', 'IdentityNode', 'JADENode', 'LLENode', 'LibSVMNode',
    'LinearRegressionNode', 'NIPALSNode', 'NormalNoiseNode',
    'PCANode', 'PerceptronClassifier', 'PolynomialExpansionNode',
    'QuadraticExpansionNode', 'RBFExpansionNode', 'RBMNode',
    'RBMWithLabelsNode', 'SFA2Node', 'SFANode', 'ShogunSVMNode',
    'SignumClassifier', 'SimpleMarkovClassifier', 'TDSEPNode',
    'TimeFramesNode', 'WhiteningNode', 'XSFANode',
]

# TODO: use a special wrapper for classifier nodes, these are currently ignored
AUTOMATIC_MDP_CLASSIFIERS = [
     'GaussianClassifierNode', 'NaiveBayesClassifier'                     
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
    
def _write_binode_code(fid, node_classes, from_module=mdp.nodes):
    """Write code for BiNode versions of normal node classes into module file.
    
    fid -- File handle of the module file.
    node_classes -- List of node classes for which binodes are created,
        if None then simply all nodes in mdp.nodes are used.
    from_module -- Module in which the node_classes are.
    """
    fid.write('"""\nAUTOMATICALLY GENERATED CODE, DO NOT MODIFY!\n\n')
    fid.write('Edit and run autogen.py instead to overwrite this module.\n"""')
    fid.write('\n\nimport %s\n' % from_module.__name__)
    fid.write('from ..binode import BiNode\n\n')
    for node_class in node_classes:
        node_name = node_class.__name__
        binode_name = node_name[:-4] + "BiNode"
        fid.write('class %s(BiNode, %s.%s):' %
                  (binode_name, from_module.__name__, node_name))
        docstring = "Automatically created BiNode version of %s." % node_name
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
            err = ("varargs are not supported by the BiNode class, " +
                   "please disable the automatic binode creation for class " +
                   node_name)
            raise BiNodeException(err)
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
        

if __name__ == "__main__":
    filename = "autogen_binodes.py"
    autogen_file = open(filename, 'w')
    try:
        _write_binode_code(autogen_file,
                           (getattr(mdp.nodes, node_name)
                            for node_name in AUTOMATIC_MDP_NODES))
    finally:
        autogen_file.close()
    print "wrote auto-generated code into file %s" % filename