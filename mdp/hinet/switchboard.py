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
            err = ("One or more switchboard connection "
                   "indices exceed the input dimension.")
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
        self.x_unused_channels = 0  # number of channels which are not covered
        self.y_unused_channels = 0
        ## check parameters for inconsistencies
        if (x_field_channels > x_in_channels):
            err = ("Number of field channels"
                   "exceeds the number of input channels in x-direction. "
                   "This would lead to an empty connection list.")
            raise Rectangular2dSwitchboardException(err)
        if (y_field_channels > y_in_channels):
            err = ("Number of field channels"
                   "exceeds the number of input channels in y-direction. "
                   "This would lead to an empty connection list.")
            raise Rectangular2dSwitchboardException(err)
        # number of output channels in x-direction
        self.x_out_channels = ((x_in_channels - x_field_channels) //
                               x_field_spacing + 1)
        self.x_unused_channels = x_in_channels - x_field_channels
        if self.x_unused_channels > 0:
            self.x_unused_channels %= x_field_spacing
        elif self.x_unused_channels < 0:
            self.x_unused_channels = x_in_channels
        if self.x_unused_channels and not ignore_cover:
            err = ("Channel fields do not "
                   "cover all input channels in x-direction.")
            raise Rectangular2dSwitchboardException(err)
        # number of output channels in y-direction                       
        self.y_out_channels = ((y_in_channels - y_field_channels) //
                               y_field_spacing + 1)
        self.y_unused_channels = y_in_channels - y_field_channels
        if self.y_unused_channels > 0:
            self.y_unused_channels %= y_field_spacing
        elif self.y_unused_channels < 0:
            self.y_unused_channels = y_in_channels
        if self.y_unused_channels and not ignore_cover:
            err = ("Channel fields do not "
                   "cover all input channels in y-direction.")
            raise Rectangular2dSwitchboardException(err)
        ## end of parameters checks
        self.output_channels = self.x_out_channels * self.y_out_channels
        self.in_trans = CoordinateTranslator(x_in_channels, y_in_channels)
        self.out_trans = CoordinateTranslator(self.x_out_channels, 
                                              self.y_out_channels)
        # input-output mapping of connections
        # connections has an entry for each output connection, 
        # containing the index of the input connection.
        connections = numx.zeros([self.output_channels * self.out_channel_dim],
                                 dtype=numx.int32)
        first_out_con = 0
        for y_out_chan in range(self.y_out_channels):
            for x_out_chan in range(self.x_out_channels):
                # inner loop over field
                x_start_chan = x_out_chan * x_field_spacing
                y_start_chan = y_out_chan * y_field_spacing
                for x_in_chan in range(x_start_chan,
                                       x_start_chan + self.x_field_channels):
                    for y_in_chan in range(y_start_chan,
                                        y_start_chan + self.y_field_channels):
                        first_in_con = (self.in_trans.image_to_index(
                                                    x_in_chan, y_in_chan) *
                                        self.in_channel_dim)
                        connections[first_out_con:
                                    first_out_con + self.in_channel_dim] = \
                            range(first_in_con,
                                  first_in_con + self.in_channel_dim)
                        first_out_con += self.in_channel_dim
        Switchboard.__init__(self, input_dim= self.in_channels * in_channel_dim,
                             connections=connections)

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
        

class DoubleRect2dSwitchboard(Switchboard):
    """Special 2d Switchboard where each inner point is covered twice.
    
    First the input is covered with non-overlapping rectangular fields.
    Then the input is covered with fields of the same size that are shifted
    in the x and y direction by half the field size (we call this the
    uneven fields).
    
    Note that the output of this switchboard cannot be interpreted as
    a rectangular grid, because the uneven fields are shifted. Instead it is
    a rhombic grid (it is not a hexagonal grid because the distances of the
    field centers do not satisfy the necessary relation).
    See http://en.wikipedia.org/wiki/Lattice_(group)
    
    Example for a 6x4 input and a field size of 2 in both directions:
    
    even fields:
    
    1 1 2 2 3 3
    1 1 2 2 3 3
    4 4 5 5 6 6
    4 4 5 5 6 6
    
    uneven fields:
    
    * * * * * *
    * 7 7 8 8 *
    * 7 7 8 8 *
    * * * * * *
    
    Note that the uneven connections come after all the even connections in
    the connections sequence.
    """
    
    def __init__(self, x_in_channels, y_in_channels, 
                 x_field_channels, y_field_channels,
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
        x_field_spacing = x_field_channels / 2
        y_field_spacing = y_field_channels / 2
        self.x_unused_channels = 0  # number of channels which are not covered
        self.y_unused_channels = 0
        ## check parameters for inconsistencies
        if (x_field_channels > x_in_channels):
            err = ("Number of field channels"
                   "exceeds the number of input channels in x-direction. "
                   "This would lead to an empty connection list.")
            raise Rectangular2dSwitchboardException(err)
        if (y_field_channels > y_in_channels):
            err = ("Number of field channels"
                   "exceeds the number of input channels in y-direction. "
                   "This would lead to an empty connection list.")
            raise Rectangular2dSwitchboardException(err)
        # number of output channels in x-direction
        self.x_out_channels = ((x_in_channels - x_field_channels) //
                               x_field_spacing + 1)
        self.x_unused_channels = x_in_channels - x_field_channels
        if self.x_unused_channels > 0:
            self.x_unused_channels %= x_field_spacing
        elif self.x_unused_channels < 0:
            self.x_unused_channels = x_in_channels
        if self.x_unused_channels and not ignore_cover:
            err = ("Channel fields do not "
                   "cover all input channels in x-direction.")
            raise Rectangular2dSwitchboardException(err)
        # number of output channels in y-direction                       
        self.y_out_channels = ((y_in_channels - y_field_channels) //
                               y_field_spacing + 1)
        self.y_unused_channels = y_in_channels - y_field_channels
        if self.y_unused_channels > 0:
            self.y_unused_channels %= y_field_spacing
        elif self.y_unused_channels < 0:
            self.y_unused_channels = y_in_channels
        if self.y_unused_channels and not ignore_cover:
            err = ("Channel fields do not "
                   "cover all input channels in y-direction.")
            raise Rectangular2dSwitchboardException(err)
        ## end of parameters checks
        self.output_channels = ((self.x_out_channels * self.y_out_channels -
                                 1) / 2 + 1)
        self.in_trans = CoordinateTranslator(x_in_channels, y_in_channels)
        # input-output mapping of connections
        # connections has an entry for each output connection, 
        # containing the index of the input connection.
        connections = numx.zeros([self.output_channels * self.out_channel_dim],
                                 dtype=numx.int32)
        first_out_con = 0
        ## first create the even connections
        even_x_out_channels = x_in_channels / (2 * x_field_spacing)
        even_y_out_channels = y_in_channels / (2 * y_field_spacing)
        for y_out_chan in range(even_y_out_channels):
            for x_out_chan in range(even_x_out_channels):
                # inner loop over field
                x_start_chan = x_out_chan * (2 * x_field_spacing)
                y_start_chan = y_out_chan * (2 * y_field_spacing)
                for y_in_chan in range(y_start_chan,
                                       y_start_chan + self.y_field_channels):
                    for x_in_chan in range(x_start_chan,
                                       x_start_chan + self.x_field_channels):
                        first_in_con = (self.in_trans.image_to_index(
                                                    x_in_chan, y_in_chan) *
                                        self.in_channel_dim)
                        connections[first_out_con:
                                    first_out_con + self.in_channel_dim] = \
                            range(first_in_con,
                                  first_in_con + self.in_channel_dim)
                        first_out_con += self.in_channel_dim
        ## create the uneven connections
        for y_out_chan in range(even_y_out_channels - 1):
            for x_out_chan in range(even_x_out_channels - 1):
                # inner loop over field
                x_start_chan = (x_out_chan * (2 * x_field_spacing) +
                                x_field_spacing)
                y_start_chan = (y_out_chan * (2 * y_field_spacing) +
                                y_field_spacing)
                for y_in_chan in range(y_start_chan,
                                       y_start_chan + self.y_field_channels):
                    for x_in_chan in range(x_start_chan,
                                       x_start_chan + self.x_field_channels):
                        first_in_con = (self.in_trans.image_to_index(
                                                    x_in_chan, y_in_chan) *
                                        self.in_channel_dim)
                        connections[first_out_con:
                                    first_out_con + self.in_channel_dim] = \
                            range(first_in_con,
                                  first_in_con + self.in_channel_dim)
                        first_out_con += self.in_channel_dim
        Switchboard.__init__(self, input_dim=self.in_channels * in_channel_dim,
                             connections=connections)
        

class DoubleRhomb2dSwitchboardException(SwitchboardException):
    """Exception for routing problems in the DoubleRhomb2dSwitchboard class."""
    pass


class DoubleRhomb2dSwitchboard(Switchboard):
    """Rectangular lattice switchboard covering a rhombic lattice.
    
    All inner points of the rhombic lattice are covered twice. The rectangular
    fields are rotated by 45 degree.
    """
    
    def __init__(self, x_even_in_channels, y_even_in_channels,
                 diag_field_channels, in_channel_dim=1):
        """Calculate the connections.
        
        Note that the incoming data will be interpreted as a rhombic grid,
        as it is produced by DoubleRect2dSwitchboard.
        
        Keyword arguments:
        x_even_in_channels -- Number of even input channels in the x-direction.
        y_even_in_channels -- Number of even input channels in the y-direction
        diag_field_channels -- Field edge size (before the rotation).
        in_channel_dim -- Number of connections per input channel
        """
        ## check parameters for inconsistencies ##
        if diag_field_channels % 2:
            err = ("diag_field_channels must be even (for double cover)")
            raise DoubleRhomb2dSwitchboardException(err)
        if (x_even_in_channels - 1) % (diag_field_channels // 2):
            err = ("diag_field_channels value is not compatible with "
                   "x_even_in_channels")
            raise DoubleRhomb2dSwitchboardException(err)
        if (y_even_in_channels - 1) % (diag_field_channels // 2):
            err = ("diag_field_channels value is not compatible with "
                   "y_even_in_channels")
            raise DoubleRhomb2dSwitchboardException(err)
        ## count channels and stuff
        self.in_channel_dim = in_channel_dim
        input_dim = ((2 * x_even_in_channels * y_even_in_channels
                     - x_even_in_channels - y_even_in_channels + 1) *
                     in_channel_dim)
        self.out_channel_dim = in_channel_dim * diag_field_channels**2
        self.x_out_channels = x_even_in_channels // diag_field_channels
        self.y_out_channels = y_even_in_channels // diag_field_channels
        self.output_channels = self.x_out_channels * self.y_out_channels
        output_dim = self.output_channels * self.out_channel_dim
        ## prepare iteration over fields
        even_in_trans = CoordinateTranslator(x_even_in_channels,
                                             y_even_in_channels)
        uneven_in_trans = CoordinateTranslator(x_even_in_channels - 1,
                                               y_even_in_channels - 1)
        uneven_in_offset = x_even_in_channels * y_even_in_channels
        # input-output mapping of connections
        # connections has an entry for each output connection, 
        # containing the index of the input connection.
        connections = numx.zeros([output_dim], dtype=numx.int32)
        first_out_con = 0
        for x_out_chan in range(self.x_out_channels):
            for y_out_chan in range(self.y_out_channels):
                # inner loop over perceptive field
                # TODO: Fix ambivalent offset issue
                x_start_chan = x_out_chan * (diag_field_channels // 2) + 1
                y_start_chan = y_out_chan * (diag_field_channels // 2)
                # pick the inital offset to
                # iterate over both even and uneven lines
                for iy, y_in_chan in enumerate(range(y_start_chan,
                                y_start_chan + (2 * diag_field_channels - 1))):
                    # half width of the field in the given row
                    if iy <= (diag_field_channels - 1):
                        field_width = iy + 1
                    else:
                        field_width = (diag_field_channels - 1 -
                                       (iy % diag_field_channels))
                        
                    #print "w: %d" % field_width
                        
                    for x_in_chan in range(x_start_chan - field_width // 2,
                                           x_start_chan + field_width // 2
                                                + field_width % 2):
                        # array index of the first input connection
                        # for this input channel
                        if not y_in_chan % 2:
                            first_in_con = (
                                even_in_trans.image_to_index(
                                                x_in_chan, y_in_chan // 2) *
                                                        self.in_channel_dim)
                        else:
                            first_in_con = (
                                (uneven_in_trans.image_to_index(
                                                x_in_chan, y_in_chan // 2)
                                 + uneven_in_offset) * self.in_channel_dim)
                        connections[first_out_con:
                                    first_out_con + self.in_channel_dim] = \
                            range(first_in_con,
                                  first_in_con + self.in_channel_dim)
                        first_out_con += self.in_channel_dim
                        
        #print connections
                        
        Switchboard.__init__(self, input_dim=input_dim, connections=connections)
        

# utility class for Rectangular2dSwitchboard

class CoordinateTranslator(object):
    """Translate between image (PIL) and numpy array coordinates.
    
    PIL image coordinates go from 0..width-1 . The first coordinate is x.
    Array coordinates also start from 0, but the first coordinate is the row.
    As depicted below we have x = column, y = row. The entry index numbers are
    also shown.
    
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
