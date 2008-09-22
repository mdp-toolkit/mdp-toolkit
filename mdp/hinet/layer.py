"""
Module for Layers.

Note that additional args and kwargs for train or execute are currently not 
supported. 
"""


# TODO: late initialization of dimensions for Layer and SameInputLayer?
# TODO: Test if the nodes are compatible (somewhat done, could go further)

import mdp
from mdp import numx


class Layer(mdp.Node):
    """Layers are nodes which consist of multiple horizontally parallel nodes.

    Since they are nodes layers may be stacked in a flow (e.g. to build a
    layered network).
    
    If one would like to use flows instead of nodes inside of a layer one can
    use a FlowNode.
    """
    
    def __init__(self, nodes, dtype=None):
        """Setup the layer with the given list of nodes.
        
        The input and output dimensions for the nodes must be already set 
        (the output dimensions for simplicity reasons). The training phases for 
        the nodes are allowed to differ.
        
        Keyword arguments:
        nodes -- List of the nodes to be used.
        """
        self.nodes = nodes
        # check nodes properties and get the dtype
        dtype = self._check_props(dtype)
        # calculate the the dimensions
        self.node_input_dims = [0] * len(self.nodes)
        input_dim = 0
        output_dim = 0
        for index, node in enumerate(nodes):
            input_dim += node.input_dim
            output_dim += node.output_dim
            self.node_input_dims[index] = node.input_dim
        super(Layer, self).__init__(input_dim=input_dim,
                                    output_dim=output_dim,
                                    dtype=dtype)
                
    def _check_props(self, dtype):
        """Check the compatibility of the properties of the internal nodes.
        
        Return the found dtype and check the dimensions.
        
        dtype -- The specified layer dtype.
        """
        dtype_list = []  # the dtypes for all the nodes
        for i, node in enumerate(self.nodes):
            # input_dim and output_dim for each node must be set
            cond = node.input_dim and node.output_dim
            if cond is None:
                msg = ('input_dim and output_dim must be set for every node. ' +
                       'Node #%d (%s) does not comply!' % (i, node))
                raise mdp.NodeException(msg)
            if node.dtype is not None:
                dtype_list.append(node.dtype)
        # check that the dtype is None or the same for every node
        nodes_dtype = None
        nodes_dtypes = set(dtype_list)
        nodes_dtypes.discard(None)
        if len(nodes_dtypes) > 1:
            msg = ('All nodes must have the same dtype (found: %s).' % 
                   nodes_dtypes)
            raise mdp.NodeException(msg)
        elif len(nodes_dtypes) == 1:
            nodes_dtype = list(nodes_dtypes)[0]
        # check that the nodes dtype matches the specified dtype
        if nodes_dtype and dtype:
            if not numx.dtype(nodes_dtype) == numx.dtype(dtype):
                msg = ('Cannot set dtype to %s: ' %
                       numx.dtype(nodes_dtype).name +
                       'an internal node requires %s' % numx.dtype(dtype).name)
                raise mdp.NodeException(msg)
        elif nodes_dtype and not dtype:
            dtype = nodes_dtype
        return dtype
            
    def _set_dtype(self, t):
        for node in self.nodes:
            node.dtype = t
        self._dtype = t

    def _get_supported_dtypes(self):
        # we supported the minimal common dtype set
        types = set(mdp.utils.get_dtypes('All'))
        for node in self.nodes:
            types = types.intersection(node.get_supported_dtypes())
        return list(types)

    def is_trainable(self):
        for node in self.nodes:
            if node.is_trainable():
                return True
        return False
    
    def is_invertible(self): 
        for node in self.nodes:
            if not node.is_invertible():
                return False
        return True
    
    def _get_train_seq(self):
        """Return the train sequence.
        
        The length is set by the node with maximum length.
        """
        max_train_length = 0
        for node in self.nodes:
            node_length = len(node._get_train_seq())
            if node_length > max_train_length:
                max_train_length = node_length
        return ([[self._train, self._stop_training]] * max_train_length)
    
    def _train(self, x):
        """Perform single training step by training the internal nodes."""
        start_index = 0
        stop_index = 0
        for node in self.nodes:
            start_index = stop_index
            stop_index += node.input_dim
            if node.is_training():
                node.train(x[:, start_index : stop_index])

    def _stop_training(self):
        """Stop training of the internal nodes."""
        for node in self.nodes:
            if node.is_training():
                node.stop_training()
    
    def _execute(self, x):
        """Process the data through the internal nodes."""
        in_start = 0
        in_stop = 0
        out_start = 0
        out_stop = 0
        result = numx.zeros([x.shape[0], self.output_dim], dtype=x.dtype)
        for node in self.nodes:
            out_start = out_stop
            out_stop += node.output_dim
            in_start = in_stop
            in_stop += node.input_dim
            result[:, out_start : out_stop] = \
                 node.execute(x[:, in_start : in_stop])
        return result
    
    def _inverse(self, x):
        """Combine the inverse of all the internal nodes."""
        in_start = 0
        in_stop = 0
        out_start = 0
        out_stop = 0
        # compared with execute, input and output are switched
        result = numx.zeros([x.shape[0], self.input_dim], dtype=x.dtype)
        for node in self.nodes:
            out_start = out_stop
            out_stop += node.input_dim
            in_start = in_stop
            in_stop += node.output_dim
            result[:, out_start : out_stop] = \
                 node.inverse(x[:, in_start : in_stop])
        return result


class CloneLayer(Layer):
    """Layer with a single node instance that is used multiple times.
    
    The same single node instance is used to build the layer, so 
    Clonelayer(node, 3) executes in the same way as Layer([node]*3). 
    But Layer([node]*3) would have a problem when closing a training phase, 
    so one has to use CloneLayer.
    
    A CloneLayer can be used for weight sharing in the training phase. It might
    be also useful for reducing the memory footprint use during the execution
    phase (since only a single node instance is needed).
    """
    
    def __init__(self, node, n_nodes=1, dtype=None):
        """Setup the layer with the given list of nodes.
        
        Keyword arguments:
        node -- Node to be cloned.
        n_nodes -- Number of repetitions/clones of the given node.
        """
        super(CloneLayer, self).__init__((node,) * n_nodes, dtype=dtype)
        self.node = node  # attribute for convenience
        
    def _stop_training(self):
        """Stop training of the internal node."""
        if self.node.is_training():
            self.node.stop_training()
            
            
class SameInputLayer(Layer):
    """SameInputLayer is a layer were all nodes receive the full input.
    
    So instead of splitting the input according to node dimensions, all nodes
    receive the complete input data.
    """ 
    
    def __init__(self, nodes, dtype=None):
        """Setup the layer with the given list of nodes.
        
        The input dimensions for the nodes must all be equal, the output
        dimensions can differ (but must be set as well for simplicity reasons).
        
        Keyword arguments:
        nodes -- List of the nodes to be used.
        """
        self.nodes = nodes
        # check node properties and get the dtype
        dtype = self._check_props(dtype)
        # check that the input dimensions are all the same
        input_dim = self.nodes[0].input_dim
        output_dim = 0
        for node in self.nodes:
            output_dim += node.output_dim
            if not node.input_dim == input_dim:
                msg = 'The nodes have different input dimensions.'
                raise mdp.NodeException(msg)
        # TODO: use super, but somehow skip Layer
        mdp.Node.__init__(self, input_dim=input_dim, output_dim=output_dim,
                       dtype=dtype)
                
    def is_invertible(self):
        return False
    
    def _train(self, x):
        """Perform single training step by training the internal nodes."""
        for node in self.nodes:
            if node.is_training():
                node.train(x)
                
    def _execute(self, x):
        """Process the data through the internal nodes."""
        out_start = 0
        out_stop = 0
        result = numx.zeros([x.shape[0], self.output_dim], dtype=x.dtype)
        for node in self.nodes:
            out_start = out_stop
            out_stop += node.output_dim
            result[:, out_start : out_stop] = node.execute(x)
        return result