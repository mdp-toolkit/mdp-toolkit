"""Hierarchical Networks Package.

This package makes it possible to construct graph-like Node structures,
especially hierarchical networks.

The most important building block is the new Layer node, which works as an 
horizontal version of flow. It encapsulates a list of Nodes, which are trained
and executed in parallel. 
For example we can take two Nodes with 100 dimensional input to
construct a layer with a 200 dimensional input. The first half of the input
data is automatically fed into the first Node, the second half into the second
Node.

Since one might also want to use Flows (i.e. vertical stacks of Nodes) in a
Layer, a wrapper class for Nodes is provided.
The FlowNode class wraps any Flow into a Node, which can then be used like any 
other Node. Together with the Layer this allows you to combine Nodes both
horizontally and vertically. Thereby one can in principle realize
any feed-forward network topology.

For a hierarchical networks one might want to route the different parts of the
data to different Nodes in a Layer in complicated ways. This is done by a
Switchboard that handles all the routing.
Defining the routing manually can be quite tedious, so one can derive subclasses
for special routing situations. One such subclass for 2d image data is provided. 
It maps the data according to rectangular overlapping 2d input areas. One can 
then feed the output into a Layer and each Node will get the correct input.
"""

# TODO: late initialization of dimensions for Layer and SameInputLayer?
# TODO: Test if the nodes are compatible (somewhat done, could go further)
# TODO: add ChannelSwitchboard with get_out_channel_node?

from operator import isSequenceType

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
        # TODO: use SameInputLayer instead of Layer, modify our usage of super?
        super(Layer, self).__init__(input_dim=input_dim,
                                    output_dim=output_dim,
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
          

class CloneLayer(Layer):
    """Layer with a single node that is used in parallel multiple times.
    
    The same single node is used multiple times, so Clonelayer(node, 3)
    executes in the same way as Layer([node]*3). But Layer([node]*3) would
    lead to problems when closing a training phase, which can be avoided
    by using CloneLayer.
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
    
    
class FlowNode(mdp.Node):
    """FlowNode wraps a Flow of Nodes into a single Node.
    
    This is necessary if one wants to use a flow where a Node is demanded.
    """
    
    def __init__(self, flow, input_dim=None, output_dim=None, dtype=None):
        """Wrap the given flow into this node.
        
        Pretrained nodes are allowed, but the internal _flow must not 
        be modified after the FlowNode was created. 
        The node dimensions do not have to be specified.
        """
        self._flow = flow
        # set properties if needed:
        if input_dim is None:
            input_dim = self._flow[0].input_dim
        if output_dim is None:
            output_dim = self._flow[-1].output_dim
        if dtype is None:
            dtype = self._flow[-1].dtype
        # store which nodes are pretrained up to what phase
        self._pretrained_phase = [node.get_current_train_phase()
                                  for node in flow]
        super(FlowNode, self).__init__(input_dim=input_dim,
                                       output_dim=output_dim, dtype=dtype)
        
    def _set_input_dim(self, n):
        # try setting the input_dim of the first node
        self._flow[0].input_dim = n
        # let a consistency check run
        self._flow._check_nodes_consistency()
        # if we didn't fail here, go on
        self._input_dim = n

    def _set_output_dim(self, n):
        # try setting the output_dim of the last node
        self._flow[-1].output_dim = n
        # let a consistency check run
        self._flow._check_nodes_consistency()
        # if we didn't fail here, go on
        self._output_dim = n

    def _set_dtype(self, t):
        # dtype can not be set for sure in arbitrary flows
        # but here we want to be sure that FlowNode *can*
        # offer a dtype that is consistent
        for node in self._flow:
            node.dtype = t
        self._dtype = t

    def _get_supported_dtypes(self):
        # we support the minimal common dtype set
        types = set(mdp.utils.get_dtypes('All'))
        for node in self._flow:
            types = types.intersection(node.get_supported_dtypes())
        return list(types)
    
    def is_trainable(self):
        for node in self._flow:
            if node.is_trainable():
                return True
        return False

    def is_invertible(self):
        for node in self._flow:
            if not node.is_invertible():
                return False
        return True
        
    def _get_train_seq(self):
        """Return a training sequence containing all training phases."""
        def get_train_function(_i_node, _node):
            # This internal function is needed to channel the data through
            # the nodes in front of the current nodes.
            # using nested scopes here instead of default args, see pep-0227
            def _train(x):
                if i_node > 0:
                    # do not use _execute_seq, no crash recovery here
                    prior_flow = self._flow[:_i_node]
                    _node.train(prior_flow.execute(x))
                else:
                    _node.train(x)
            return _train
        train_seq = []
        for i_node, node in enumerate(self._flow):
            if node.is_trainable():
                remaining_len = (len(node._get_train_seq())
                                 - self._pretrained_phase[i_node])
                train_seq += ([(get_train_function(i_node, node), 
                                node.stop_training)] 
                              * remaining_len)
        # If the last node is trainable,
        # then we have to set the output dimensions of the FlowNode.
        if self._flow[-1].is_trainable():
            train_seq[-1] = (train_seq[-1][0],
                             self._get_stop_training_wrapper(self._flow[-1],
                                                             train_seq[-1][1]))
        return train_seq

    def _get_stop_training_wrapper(self, node, func):
        """Return wrapper for stop_training to set FlowNode outputdim."""
        def _stop_training_wrapper(*args, **kwargs):
            func(*args, **kwargs)
            self.output_dim = node.output_dim
        return _stop_training_wrapper
        
    def _execute(self, x):
        return self._flow.execute(x)
        
    def _inverse(self, x):
        return self._flow.inverse(x)


class SwitchboardException(mdp.NodeException):
    """Exception for routing problems in the Switchboard class."""
    pass


class Switchboard(mdp.Node):
    """Does the routing associated with the connections between layers.
    
    It may be directly used as a layer/node, routing all the data at once. If 
    the routing/mapping is not injective the processed data may be quite large 
    and probably contains many redundant copies of the input data. 
    So is this case one may instead use nodes for individual output
    channels and put each in a MultiNode.
    
    SwitchboardLayer is the most general version of a switchboard layer, since
    there is no imposed rule for the connection topology. For practical 
    applications should often derive more specialized classes.
    """
    
    def __init__(self, input_dim, connections):
        """Create a generic switchboard.
        
        The input and output dimension as well as dtype have to be fixed
        at initialization time.
       
        Keyword arguments:
        input_dim -- Dimension of the input data (number of connections).
        connections -- 1d Array or sequence with an entry for each output 
            connection, containing the corresponding index of the 
            input connection.
        """
        # check connections for inconsistencies
        if len(connections) == 0:
            raise SwitchboardException("Received empty connection list.")
        if numx.nanmax(connections) >= input_dim:
            raise SwitchboardException("One or more switchboard connection " +
                                       "indices exceed the input dimension.")
        # checks passed
        self.connections = numx.array(connections)
        output_dim = len(connections)
        super(Switchboard, self).__init__(input_dim=input_dim,
                                          output_dim=output_dim)
        # try to invert connections
        if (self.input_dim == self.output_dim and
            len(numx.unique(self.connections)) == self.input_dim):
            self.inverse_connections = numx.argsort(self.connections)
        else:
            self.inverse_connections = None
            
    def _execute(self, x):
        return x[:,self.connections]
        
    def is_trainable(self): 
        return False
    
    def is_invertible(self):
        if self.inverse_connections is None:
            return False
        else:
            return True
    
    def _inverse(self, x):
        if self.inverse_connections is None:
            raise SwitchboardException("Connections are not invertible.")
        else:
            return x[:,self.inverse_connections]
    

class Rectangular2dSwitchboardException(SwitchboardException):
    """Exception for routing problems in the Rectangular2dSwitchboard class."""
    pass


class Rectangular2dSwitchboard(Switchboard):
    """Switchboard for a 2-dimensional topology.
    
    This is a specialized version of SwitchboardLayer that makes it easy to
    implement connection topologies which are based on a 2-dimensional network
    layers.
    
    The input connections are assumed to be grouped into so called channels, 
    which are considered as lying in a two dimensional rectangular plane. 
    Each output channel corresponds to a 2d rectangular field in the 
    input plane. The fields can overlap.
    
    The coordinates follow the standard image convention (see the above 
    CoordinateTranslator class).
    """
    
    def __init__(self, x_in_channels, y_in_channels, 
                 x_field_channels, y_field_channels,
                 x_field_spacing=1, y_field_spacing=1, 
                 in_channel_dim=1, ignore_cover=False):
        """Calculate the connections.
        
        Keyword arguments:
        x_in_channels -- Number of input channels in the x-direction.
            This has to be specified, since the actual input is only one
            1d array.
        y_in_channels -- Number of input channels in the y-direction
        in_channel_dim -- Number of connections per input channel
        x_field_channels -- Number of channels in each field in the x-direction
        y_field_channels -- Number of channels in each field in the y-direction
        x_field_spacing -- Offset between two fields in the x-direction.
        y_field_spacing -- Offset between two fields in the y-direction.
        ignore_cover -- Boolean value defines if an 
            Rectangular2dSwitchboardException is raised when the fields do not
            cover all input channels. Set this to True if you are willing to
            risk loosing input channels at the border.
        """
        ## count channels and stuff
        self.in_channel_dim = in_channel_dim
        self.x_in_channels = x_in_channels
        self.y_in_channels = y_in_channels
        self.in_channels = x_in_channels * y_in_channels
        self.x_field_channels = x_field_channels
        self.y_field_channels = y_field_channels
        self.out_channel_dim = (in_channel_dim * 
                                x_field_channels * y_field_channels)
        self.x_field_spacing = x_field_spacing
        self.y_field_spacing = y_field_spacing
        ## check parameters for inconsistencies ##
        if (x_field_channels > x_in_channels):
            raise Rectangular2dSwitchboardException("Number of field channels" +
                    "exceeds the number of input channels in x-direction. " +
                    "This would lead to an empty connection list.")
        if (y_field_channels > y_in_channels):
            raise Rectangular2dSwitchboardException("Number of field channels" +
                    "exceeds the number of input channels in y-direction. " +
                    "This would lead to an empty connection list.")
        # number of output channels in x-direction
        self.x_out_channels = \
            (x_in_channels - x_field_channels) // x_field_spacing + 1
        if (((x_in_channels - x_field_channels) < 0 or
             (x_in_channels - x_field_channels) % x_field_spacing)
             and not ignore_cover):
            raise Rectangular2dSwitchboardException("Channel fields do not " + 
                                    "cover all input channels in x-direction.")
        # number of output channels in y-direction                       
        self.y_out_channels = \
            (y_in_channels - y_field_channels) // y_field_spacing + 1
        if (((y_in_channels - y_field_channels) < 0 or
             (y_in_channels - y_field_channels) % y_field_spacing)
             and not ignore_cover):
            raise Rectangular2dSwitchboardException("Channel fields do not " + 
                                    "cover all input channels in y-direction.")
        ## end of parameters checks ##
        self.output_channels = self.x_out_channels * self.y_out_channels
        input_dim = self.in_channels * in_channel_dim
        output_dim = self.output_channels * self.out_channel_dim
        self.in_trans = CoordinateTranslator(x_in_channels, y_in_channels)
        self.out_trans = CoordinateTranslator(self.x_out_channels, 
                                              self.y_out_channels)
        self.field_trans = CoordinateTranslator(self.x_field_channels, 
                                                self.y_field_channels)
        # input-output mapping of connections
        # connections has an entry for each output connection, 
        # containing the index of the input connection.
        connections = numx.zeros([output_dim], dtype=numx.int32)
        for x_out_chan in range(self.x_out_channels):
            for y_out_chan in range(self.y_out_channels):
                # inner loop over perceptive field
                x_start_chan = x_out_chan * x_field_spacing
                y_start_chan = y_out_chan * y_field_spacing
                base_out_con = \
                    self.out_trans.image_to_index(x_out_chan, y_out_chan) \
                    * self.out_channel_dim
                for ix, x_in_chan in enumerate(range(x_start_chan, 
                                               x_start_chan+x_field_channels)):
                    for iy, y_in_chan in enumerate(range(y_start_chan, 
                                               y_start_chan+y_field_channels)):
                        # array index of the first input connection 
                        # for this input channel
                        first_in_con = \
                            self.in_trans.image_to_index(x_in_chan, y_in_chan) \
                            * self.in_channel_dim
                        field_out_con = \
                            self.field_trans.image_to_index(ix, iy) \
                            * self.in_channel_dim
                        first_out_con = base_out_con + field_out_con
                        connections[first_out_con : first_out_con +
                                    in_channel_dim] \
                            = range(first_in_con, first_in_con +in_channel_dim)
        Switchboard.__init__(self, input_dim=input_dim, connections=connections)
    
    def get_out_channel_node(self, channel):
        """Return a Switchboard that does the routing to a specific 
        output channel.
        
        One can use the resulting nodes in a SameInputLayer for cases where the 
        memory footprint of the output is a problem.
        
        channel -- The index of the required channel. Can be also a tuple with
            the image coordinates of the channel.
        """
        if isSequenceType(channel):
            index = self.out_trans.image_to_index(channel[0], channel[1])
        else:
            index = channel
        # construct connection table for the channel
        index *= self.out_channel_dim
        return Switchboard(self.input_dim, 
                    self.connections[index : index+self.out_channel_dim])
        
        
# utility class for Rectangular2dSwitchboard

class CoordinateTranslator(object):
    """Translate between image (PIL) and numpy array coordinates.
    
    PIL image coordinates go from 0..width-1 . The first coordinate is x.
    Array coordinates also start from 0, but the first coordinate is the row.
    As depicted below we have x = column, y = row. The entry index numbers are
    also shown ( 
    
      +------> x
      | 1 2
      | 3 4 
    y v
    
    array[y][x] 
    """
    
    def __init__(self, x_image_dim, y_image_dim):
        self.x_image_dim = x_image_dim
        self.y_image_dim = y_image_dim

    def image_to_array(self, x, y):
        return y, x
    
    def image_to_index(self, x, y):
        return y * self.x_image_dim + x
    
    def array_to_image(self, row, col):
        return col, row
        
    def array_to_index(self, row, col):
        return row * self.x_image_dim + col
    
    def index_to_array(self, index):
        return index // self.x_image_dim, index % self.x_image_dim
    
    def index_to_image(self, index):
        return index % self.x_image_dim, index // self.x_image_dim
