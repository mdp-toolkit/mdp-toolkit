
import mdp
import mdp.hinet as hinet
n = mdp.numx

from ..binode import BiNode

        
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
    
    def _execute(self, x, msg=None):
        """Return the routed input data."""
        if x is not None:
            y = super(BiSwitchboard, self)._execute(x)
        else:
            y = None
        msg = self._execute_msg(msg)
        if not msg:
            return y
        else:
            return y, msg
    
    def _inverse(self, x, msg=None):
        """Return the routed input data."""
        if x is not None:
            y = super(BiSwitchboard, self)._inverse(x)
        else:
            y = None
        msg = self._inverse_msg(msg)
        if not msg:
            return y
        else:
            return y, msg
    
    def _stop_message(self, msg=None):
        return self._execute_msg(msg)
    
    def is_bi_training(self):
        return False
    
    ## Helper methods ##
            
    def _inverse_msg(self, msg):
        """Inverse routing for msg."""
        if not msg:
            return None
        out_msg = {}
        for (key, value) in msg.items():
            if type(value) is n.ndarray:
                out_msg[key] = super(BiSwitchboard, self)._inverse(value)
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


class Rectangular2dBiSwitchboardException(
                                    hinet.Rectangular2dSwitchboardException):
    pass


class Rectangular2dBiSwitchboard(BiSwitchboard,
                                 hinet.Rectangular2dSwitchboard):
    pass

    
class Rectangular2dMeanBiSwitchboard(Rectangular2dBiSwitchboard,
                                     hinet.MeanInverseSwitchboard):
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
