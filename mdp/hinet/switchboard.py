"""
Module for Switchboards.

Note that additional args and kwargs for train or execute are currently not 
supported. 
"""



# TODO: add ChannelSwitchboard with get_out_channel_node?

from operator import isSequenceType

import mdp
from mdp import numx


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
            err = "Received empty connection list."
            raise SwitchboardException(err)
        if numx.nanmax(connections) >= input_dim:
            err = "One or more switchboard connection "
            "indices exceed the input dimension."
            raise SwitchboardException(err)
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
        return x[:, self.connections]
        
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
            return x[:, self.inverse_connections]
    

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
            err = "Number of field channels"
            "exceeds the number of input channels in x-direction. "
            "This would lead to an empty connection list."
            raise Rectangular2dSwitchboardException(err)
        if (y_field_channels > y_in_channels):
            err = "Number of field channels"
            "exceeds the number of input channels in y-direction. "
            "This would lead to an empty connection list."
            raise Rectangular2dSwitchboardException(err)
        # number of output channels in x-direction
        self.x_out_channels = ((x_in_channels - x_field_channels) //
                               x_field_spacing + 1)
        if (((x_in_channels - x_field_channels) < 0 or
             (x_in_channels - x_field_channels) % x_field_spacing)
             and not ignore_cover):
            err = "Channel fields do not "
            "cover all input channels in x-direction."
            raise Rectangular2dSwitchboardException(err)
        # number of output channels in y-direction                       
        self.y_out_channels = ((y_in_channels - y_field_channels) //
                               y_field_spacing + 1)
        if (((y_in_channels - y_field_channels) < 0 or
             (y_in_channels - y_field_channels) % y_field_spacing)
             and not ignore_cover):
            err = "Channel fields do not "
            "cover all input channels in y-direction."
            raise Rectangular2dSwitchboardException(err)
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
                base_out_con = (self.out_trans.image_to_index(x_out_chan,
                                                              y_out_chan)
                                * self.out_channel_dim)
                for ix, x_in_chan in enumerate(range(x_start_chan, 
                                               x_start_chan+x_field_channels)):
                    for iy, y_in_chan in enumerate(range(y_start_chan, 
                                               y_start_chan+y_field_channels)):
                        # array index of the first input connection 
                        # for this input channel
                        first_in_con = (self.in_trans.image_to_index(x_in_chan,
                                                                     y_in_chan)
                                        * self.in_channel_dim)
                        field_out_con = (self.field_trans.image_to_index(ix,
                                                                         iy)
                                         * self.in_channel_dim)
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
