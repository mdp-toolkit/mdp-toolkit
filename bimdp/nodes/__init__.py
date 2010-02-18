
from miscnodes import IdentityBiNode, SenderBiNode

del miscnodes


## automatically create BiNode versions of all Nodes in mdp.nodes ##

# TODO: use a special wrapper for classifier nodes

# Note: Using mdp.NodeMetaclass.__new__ instead of exec makes the classes
#    appear as if they were defined in mdp.nodes, breaking pickle. 

import sys
import inspect
import mdp
from ..binode import BiNode, BiNodeException

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

AUTOMATIC_MDP_CLASSIFIERS = [
     'GaussianClassifierNode', 'NaiveBayesClassifier'                     
]

def _get_node_subclasses(node_class=mdp.Node, module=mdp.nodes):
    """Return all node classes in module which are subclasses of node_class.
    """
    node_subclasses = []
    for node_subclass in (getattr(module, name) for name in dir(module)):
        if (isinstance(node_subclass, type) and
            issubclass(node_subclass, node_class)):
            node_subclasses.append(node_subclass)
    return node_subclasses
    
def create_binodes(node_classes, from_module=mdp.nodes, to_module=None):
    """Automatically create BiNode versions of normal node classes.
    
    node_classes -- List of node classes for which binodes are created,
        if None then simply all nodes in mdp.nodes are used.
    from_module -- Module in which the node_classes are.
    to_module -- Module to which the new classes are added, if None then then
        the current module is used.
    """
    if not to_module:
        to_module = sys.modules[__name__]
    for node_class in node_classes:
        node_name = node_class.__name__
        binode_name = node_name[:-4] + "BiNode"
        node_code = ('class %s(BiNode, %s.%s):' %
                     (binode_name, from_module.__name__, node_name))
        docstring = "Automatically created BiNode version of %s." % node_name
        node_code += '\n    """%s"""' % docstring
        ## define the init method explicitly to preserve the signature
        docstring = node_class.__init__.__doc__
        args, varargs, varkw, defaults = inspect.getargspec(node_class.__init__)
        args.remove('self')
        args += ('node_id', 'stop_result')
        defaults += (None, None)
        if defaults is None:
            defaults = []
        first_default = len(args) - len(defaults)
        node_code += '\n    def __init__(self, '
        node_code += ''.join(arg + ', ' for arg in args[:-len(defaults)])
        node_code += ''.join(arg + '=' + repr(defaults[i_arg]) + ', '
                             for i_arg, arg in enumerate(args[first_default:]))
        if varargs:
            err = ("varargs are not supported by the BiNode class, " +
                   "please disable the automatic binode creation for class " +
                   node_name)
            raise BiNodeException(err)
        if varkw:
            node_code += '*%s, ' % varkw
        if node_code.endswith(", "):
            node_code = node_code[:-2]
        node_code += '):'
        if docstring:
            node_code += '\n        """%s"""' % docstring
        node_code += '\n        super(%s, self).__init__(' % binode_name
        node_code += ''.join('%s=%s, ' % (arg, arg) for arg in args)
        if varkw:
            node_code += '**%s, ' % varkw
        if node_code.endswith(", "):
            node_code = node_code[:-2]
        node_code += ')'
        exec node_code in to_module.__dict__
        
create_binodes(getattr(mdp.nodes, node_name)
               for node_name in AUTOMATIC_MDP_NODES)
