
import mdp
import mdp.hinet as hinet
n = mdp.numx

from ..binode import BiNode

# TODO: can hugely simplify this whole class, just implement inverse!
#    also update the unittests

        
class BiSwitchboard(BiNode, hinet.Switchboard):
    """BiNet version of the normal Switchboard.
    
    In addition to the feed-forward routing it also allows top-down routing.
    This is the default behavior for message.
    """
    
    def __init__(self, **kwargs):
        """Initialize BiSwitchboard.
        
        args and kwargs are forwarded via super to the next __init__ method
        in the MRO.
        """
        super(BiSwitchboard, self).__init__(**kwargs)
        if self.inverse_connections is not None:
            self.down_connections = self.inverse_connections
        else:
            # a stable (order preserving where possible) sort is
            # necessary here, therefore use mergesort
            # otherwise channels (e.g. for a Rectangular2d...) are mixed up 
            self.down_connections = n.argsort(self.connections, 
                                              kind="mergesort")
    
    def _execute(self, x, msg=None, y=None, send_down=False):
        """Return the routed input data.
        
        y -- Provide y instead of x for inverse routing. If y is given then
            the x return value is None and msg contains x instead.
        """
        if (y is not None) or send_down:
            msg = self._down_execute_msg(msg)
            if y is not None:
                x = self._down_execute(y)
                msg["y"] = x
            result = (None, msg)
        else:
            y = super(BiSwitchboard, self)._execute(x)
            msg = self._execute_msg(msg)
            result = (y, msg)
        if result[1] is None:
            result = result[0]
        return result
    
    def _stop_message(self, msg=None, send_down=False, target=None):
        return self._message(self, msg, send_down, target)
    
    def is_bi_training(self):
        return False
    
    ## Helper methods ##
            
    def _down_execute(self, x):
        """Return the top-down routed x."""
        return x[:,self.inverse_connections]
    
    def _down_execute_msg(self, msg):
        """Top-down routing for msg."""
        if not msg:
            return None
        out_msg = {}
        for (key, value) in msg.items():
            if type(value) is n.ndarray:
                out_msg[key] = self._down_execute(value)
            else:
                out_msg[key] = value
        return out_msg
    
    def _execute_msg(self, msg):
        """Feed-forward routing for msg."""
        if not msg:
            return None
        out_msg = {}
        for (key, value) in msg.items():
            if type(value) is n.ndarray:
                out_msg[key] = super(BiSwitchboard, self)._execute(value)
            else:
                out_msg[key] = value
        return out_msg
    

class MeanBiSwitchboard(BiSwitchboard):
    """Variant of BiSwitchboard with modified down-execute."""    
    
    def _down_execute(self, x):
        """Take the mean of overlapping values."""
        # note that x refers to signal on top
        n_y_cons = n.bincount(self.connections)  # n. connections to y_i
        y_cons = n.argsort(self.connections)  # x indices for y_i
        y = n.zeros((len(x), self.input_dim))
        i_x_counter = 0  # counter for processed x indices
        i_y = 0  # current y index
        while True:
            n_cons = n_y_cons[i_y]
            if n_cons > 0:
                y[:,i_y] = n.sum(x[:,y_cons[i_x_counter:
                                            i_x_counter + n_cons]],
                                 axis=1) / n_cons
                i_x_counter += n_cons
                if i_x_counter >= self.output_dim:
                    break
            i_y += 1
        return y
    
    def is_invertible(self):
        return True
    
    def _inverse(self, x):
        return self._down_execute(x)
        

class Rectangular2dBiSwitchboardException(
                                    hinet.Rectangular2dSwitchboardException):
    pass


class Rectangular2dBiSwitchboard(BiSwitchboard,
                                 hinet.Rectangular2dSwitchboard):
    """BiNet version of the Rectangular2dSwitchboard."""
    pass

    
class Rectangular2dMeanBiSwitchboard(MeanBiSwitchboard,
                                     Rectangular2dBiSwitchboard):
    """Combinations of Rectangular2dBiSwitchboard with MeanBiSwitchboard."""
    pass


class FactoryRectangular2dBiSwitchboard(
                                mdp.hinet.FactoryExtensionChannelSwitchboard,
                                Rectangular2dBiSwitchboard):
    
    free_parameters = ["field_size_xy", "field_step_xy", "ignore_cover",
                       "node_id"]

    @classmethod
    def _create_switchboard(cls, free_params, prev_switchboard,
                            prev_output_dim):
        in_channel_dim = (prev_output_dim // prev_switchboard.output_channels)
        if not "ignore_cover" in free_params:
            free_params["ignore_cover"] = True
        return cls(node_id=free_params["node_id"],
                   x_in_channels=prev_switchboard.x_out_channels, 
                   y_in_channels=prev_switchboard.y_out_channels, 
                   x_field_channels=free_params["field_size_xy"][0], 
                   y_field_channels=free_params["field_size_xy"][1],
                   x_field_spacing=free_params["field_step_xy"][0], 
                   y_field_spacing=free_params["field_step_xy"][1],
                   in_channel_dim=in_channel_dim,
                   ignore_cover=free_params["ignore_cover"])
