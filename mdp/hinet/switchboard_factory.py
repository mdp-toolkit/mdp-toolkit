"""
Extension for building switchboards in a 2d hierarchical network.
"""

# TODO: add unittests and maybe mention it in the tutorial

# TODO: maybe integrate all this into the original switchboard classes?

import mdp
from mdp.hinet import (
    ChannelSwitchboard, Rectangular2dSwitchboard, DoubleRect2dSwitchboard,
    DoubleRhomb2dSwitchboard
)

def get_2d_image_switchboard(image_size_xy):
    """Return a Rectangular2dSwitchboard representing an image.
    
    This can then be used as the prev_switchboard.
    """
    if isinstance(image_size_xy, int):
        image_size_x = image_size_xy
        image_size_y = image_size_xy
    else:
        image_size_x, image_size_y = image_size_xy
    return Rectangular2dSwitchboard(
                        x_in_channels=image_size_x,
                        y_in_channels=image_size_y, 
                        x_field_channels=1, y_field_channels=1,
                        x_field_spacing=1, y_field_spacing=1)


class FactoryExtensionChannelSwitchboard(mdp.ExtensionNode,
                                         ChannelSwitchboard):
    """Extension node for the assembly of channel switchboards.
    
    data attributes:
    free_parameters -- List of parameters that do not depend on the previous
        layer. Note that there might still be restrictions imposed by the
        switchboard.
        By convention parameters that end with '_xy' can either be a single
        int or a 2-tuple (sequence) of ints.
    compatible_pre_switchboards -- List of compatible base classes for
        prev_switchboard. 
    """
    
    extension_name = "switchboard_factory"
    
    free_parameters = []
    compatible_pre_switchboards = [ChannelSwitchboard]
    
    @classmethod
    def create_switchboard(cls, free_params, prev_switchboard,
                           prev_output_dim):
        """Return a new instance of this switchboard.
        
        free_params -- Parameters as specified by free_parameters.
        prev_switchboard -- Instance of the previous switchboard.
        prev_output_dim -- Output dimension of the previous layer.
        
        This template method checks the compatibility of the prev_switchboard
        and sanitizes '_xy' in free_params.
        """
        compatible = False
        for base_class in cls.compatible_pre_switchboards:
            if isinstance(prev_switchboard, base_class):
                compatible = True
        if not compatible:
            err = ("The prev_switchboard class '%s'" %
                        prev_switchboard.__class__.__name__ +
                   " is not compatible with this switchboard class" +
                   " '%s'." % cls.__name__)
            raise mdp.hinet.SwitchboardException(err)
        for key, value in free_params.items():
            if key.endswith('_xy') and isinstance(value, int):
                free_params[key] = (value, value)
        kwargs = cls._get_switchboard_kwargs(free_params, prev_switchboard,
                                             prev_output_dim)
        return cls(**kwargs)

    @staticmethod
    def _get_switchboard_kwargs(free_params, prev_switchboard,
                                prev_output_dim):
        """Return the kwargs for the cls '__init__' method.
        
        Reference implementation, merges input into one single channel.
        Override this method for other switchboard classes.
        """
        in_channel_dim = prev_output_dim // prev_switchboard.output_channels
        return {"input_dim": prev_output_dim,
                "connections": range(prev_output_dim),
                "out_channel_dim": prev_output_dim,
                "in_channel_dim": in_channel_dim}


class FactoryRectangular2dSwitchboard(FactoryExtensionChannelSwitchboard,
                                      Rectangular2dSwitchboard):
    
    free_parameters = ["field_size_xy", "field_step_xy", "ignore_cover"]
    compatible_pre_switchboards = [Rectangular2dSwitchboard,
                                   DoubleRhomb2dSwitchboard]
    
    @staticmethod
    def _get_switchboard_kwargs(free_params, prev_switchboard,
                                prev_output_dim):
        in_channel_dim = (prev_output_dim // prev_switchboard.output_channels)
        if not "ignore_cover" in free_params:
            free_params["ignore_cover"] = True
        return {"x_in_channels": prev_switchboard.x_out_channels, 
                "y_in_channels": prev_switchboard.y_out_channels, 
                "x_field_channels": free_params["field_size_xy"][0], 
                "y_field_channels": free_params["field_size_xy"][1],
                "x_field_spacing": free_params["field_step_xy"][0], 
                "y_field_spacing": free_params["field_step_xy"][1],
                "in_channel_dim": in_channel_dim,
                "ignore_cover": free_params["ignore_cover"]}
    
    
class FactoryDoubleRect2dSwitchboard(FactoryExtensionChannelSwitchboard,
                                     DoubleRect2dSwitchboard):
    
    free_parameters = ["field_size_xy", "ignore_cover"]
    compatible_pre_switchboards = [Rectangular2dSwitchboard,
                                   DoubleRhomb2dSwitchboard]
    
    @staticmethod
    def _get_switchboard_kwargs(free_params, prev_switchboard,
                                prev_output_dim):
        in_channel_dim = (prev_output_dim // prev_switchboard.output_channels)
        if not "ignore_cover" in free_params:
            free_params["ignore_cover"] = True
        return {"x_in_channels": prev_switchboard.x_out_channels, 
                "y_in_channels": prev_switchboard.y_out_channels, 
                "x_field_channels": free_params["field_size_xy"][0], 
                "y_field_channels": free_params["field_size_xy"][1],
                "in_channel_dim": in_channel_dim,
                "ignore_cover": free_params["ignore_cover"]}
   
   
class FactoryDoubleRhomb2dSwitchboard(FactoryExtensionChannelSwitchboard,
                                      DoubleRhomb2dSwitchboard):
    
    free_parameters = ["field_size"]
    compatible_pre_switchboards =  [DoubleRect2dSwitchboard]
    
    @staticmethod
    def _get_switchboard_kwargs(free_params, prev_switchboard,
                                prev_output_dim):
        in_channel_dim = (prev_output_dim // prev_switchboard.output_channels)
        return {"x_long_in_channels": prev_switchboard.x_long_out_channels, 
                "y_long_in_channels": prev_switchboard.y_long_out_channels, 
                "diag_field_channels": free_params["field_size"], 
                "in_channel_dim": in_channel_dim}

