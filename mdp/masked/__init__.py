"""MDP extension to support masked node data.
"""
__docformat__ = "restructuredtext en"

from ..utils import MaskedCovarianceMatrix
from ..extension import ExtensionNode, activate_extension, deactivate_extension
from ..signal_node import Node


__all__ = [
    'activate_masked',
    'deactivate_masked',
    ]

_masked_active_global = False
_masked_classes = []
_masked_instances = []


class MaskedCovarianceNode(ExtensionNode, Node):
    """MDP extension for masked training data

    Adapt a node to handle masked training data if:

    a. the extension is activated in global mode,
    b. the Node subclass is registered to be masked, or
    c. the instance is registered to be masked

    See `activate_masked`, `deactivate_masked`, and the `masked`
    context manager to learn about how to activate the masked
    mechanism and its options.
    """

    extension_name = 'masked'

    def is_masked(self):
        """Return True if the node is masked."""
        global _masked_active_global
        global _masked_classes
        global _masked_instances
        return (_masked_active_global
                or self.__class__ in _masked_classes
                or self in _masked_instances)

    def set_instance_masked(self, active=True):
        """Add or remove this instance from masked.

        The global masked and class masked options still have priority over
        the instance masked option.
        """
        # add to global dictionary
        global _masked_instances
        if active:
            _masked_instances.append(self)
        else:
            if self in _masked_instances:
                _masked_instances.remove(self)

    def _new_covariance_matrix(self, *args, **kwargs):
        if not self.is_masked():
            return self._non_extension__new_covariance_matrix(*args, **kwargs)
        return MaskedCovarianceMatrix(dtype=self.dtype)


def activate_masked(masked_classes=None, masked_instances=None):
    """Activate masked extension

    By default, the masked is activated globally (i.e., for all
    instances of Node). If masked_classes or masked_instances are
    specified, the masked is activated only for those classes and
    instances.

    :Parameters:
     masked_classes
      A list of Node subclasses for which masked is activated.
      Default value: None
     masked_classes
      A list of Node instances for which masked is activated.
      Default value: None
    """
    global _masked_active_global
    global _masked_classes
    global _masked_instances

    _masked_active_global = (masked_classes is None and
                             masked_instances is None)

    # active masked for specific classes and instances
    if masked_classes is not None:
        _masked_classes = list(masked_classes)
    if masked_instances is not None:
        _masked_instances = list(masked_instances)

    activate_extension('masked')


def deactivate_masked():
    "De-activate masked extension"
    global _masked_active_global
    global _masked_classes
    global _masked_instances
    __masked_active_global = False
    _masked_classes = []
    _masked_instances = []
    deactivate_extension('masked')
