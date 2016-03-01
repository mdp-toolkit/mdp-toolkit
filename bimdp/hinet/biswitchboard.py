import sys
import mdp
import mdp.hinet as hinet
n = mdp.numx

from bimdp import BiNode


class BiSwitchboard(BiNode, hinet.Switchboard):
    """BiMDP version of the normal Switchboard.

    It adds support for stop_message and also tries to apply the switchboard
    mapping to arrays in the message. The mapping is only applied if the
    array is at least two dimensional and the second dimension matches the
    switchboard dimension.
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

    def is_bi_training(self):
        return False

    ## Helper methods ##

    def _inverse_msg(self, msg):
        """Inverse routing for msg."""
        if not msg:
            return None
        out_msg = {}
        for (key, value) in list(msg.items()):
            if (type(value) is n.ndarray and
                len(value.shape) >= 2 and value.shape[1] == self.output_dim):
                out_msg[key] = super(BiSwitchboard, self)._inverse(value)
            else:
                out_msg[key] = value
        return out_msg

    def _execute_msg(self, msg):
        """Feed-forward routing for msg."""
        if not msg:
            return None
        out_msg = {}
        for (key, value) in list(msg.items()):
            if (type(value) is n.ndarray and
                len(value.shape) >= 2 and value.shape[1] == self.input_dim):
                out_msg[key] = super(BiSwitchboard, self)._execute(value)
            else:
                out_msg[key] = value
        return out_msg


## create BiSwitchboard versions of the standard MDP switchboards ##

# corresponding methods for the switchboard_factory extension are
# created as well

@classmethod
def _binode_create_switchboard(cls, free_params, prev_switchboard,
                               prev_output_dim, node_id):
    """Modified version of create_switchboard to support node_id.

    This method can be used as a substitute when using the switchboard_factory
    extension.
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
    for key, value in list(free_params.items()):
        if key.endswith('_xy') and isinstance(value, int):
            free_params[key] = (value, value)
    kwargs = cls._get_switchboard_kwargs(free_params, prev_switchboard,
                                         prev_output_dim)
    return cls(node_id=node_id, **kwargs)


# TODO: Use same technique as for binodes?
#    But have to take care of the switchboard_factory extension.

# use a function to avoid poluting the namespace
def _create_bi_switchboards():
    switchboard_classes = [
        mdp.hinet.ChannelSwitchboard,
        mdp.hinet.Rectangular2dSwitchboard,
        mdp.hinet.DoubleRect2dSwitchboard,
        mdp.hinet.DoubleRhomb2dSwitchboard,
    ]
    current_module = sys.modules[__name__]
    for switchboard_class in switchboard_classes:
        node_name = switchboard_class.__name__
        binode_name = node_name[:-len("Switchboard")] + "BiSwitchboard"
        docstring = ("Automatically created BiSwitchboard version of %s." %
                     node_name)
        docstring = "Automatically created BiNode version of %s." % node_name
        exec(('class %s(BiSwitchboard, mdp.hinet.%s): "%s"' %
              (binode_name, node_name, docstring)), current_module.__dict__)
        # create appropriate FactoryExtension nodes
        mdp.extension_method("switchboard_factory",
                             current_module.__dict__[binode_name],
                             "create_switchboard")(_binode_create_switchboard)

_create_bi_switchboards()
