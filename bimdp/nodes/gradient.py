"""
Extension to get the total derivative / gradient / Jacobian matrix.
"""
from builtins import str
from builtins import range

import mdp
import bimdp
np = mdp.numx

class NotDifferentiableException(mdp.NodeException):
    """Exception if the total derivative does not exist."""
    pass


# Default implementation is needed to satisfy the "method" request.
class GradientExtensionNode(mdp.ExtensionNode, mdp.Node):
    """Base node of the extension to calculate the gradient at a certain point.

    To get the gradient simply put 'method': 'gradient' into the msg dict.

    The grad array is three dimensional, with shape
    (len(x), self.output_dim, self.input_dim).
    The matrix formed by the last two indices is also called the Jacobian
    matrix.

    Nodes which have no well defined total derivative should raise the
    NotDifferentiableException.
    """

    extension_name = "gradient"

    def _gradient(self, x, grad=None):
        """Calculate the contribution to the grad for this node at point x.

        The contribution is then combined with the given gradient, to get
        the gradient for the original x.

        This is a template function, derived classes should override _get_grad.
        """
        if self.is_training():
            raise mdp.TrainingException("The training is not completed yet.")
        if grad is None:
            grad = np.zeros((len(x), self.input_dim, self.input_dim))
            diag_indices = np.arange(self.input_dim)
            grad[:,diag_indices,diag_indices] = 1.0
        new_grad = self._get_grad(x)
        # combine the gradients
        grad = np.asarray([np.dot(new_grad[i], grad[i])
                           for i in range(len(new_grad))])
        # update the x value for the next node
        result = self._execute(x)
        if isinstance(result, tuple):
            x = result[0]
            msg = result[1]
        else:
            x = result
            msg = {}
        msg.update({"grad": grad})
        return x, msg

    def _get_grad(self, x):
        """Return the grad for the given points.

        Override this method.
        """
        err = "Gradient not implemented for class %s." % str(self.__class__)
        raise NotImplementedError(err)

    def _stop_gradient(self, x, grad=None):
        """Helper method to make gradient available for stop_message."""
        result = self._gradient(x, grad)
        # FIXME: Is this really correct? x should be updated!
        #    Could remove this once we have the new stop signature.
        return result[1], 1


## Implementations for specific nodes. ##

# TODO: cache the gradient for linear nodes?
#    If there was a linear base class one could integrate this?

# TODO: add at least a PCA gradient implementation

@mdp.extension_method("gradient", mdp.nodes.IdentityNode, "_get_grad")
def _identity_grad(self, x):
    grad = np.zeros((len(x), self.output_dim, self.input_dim))
    diag_indices = np.arange(self.input_dim)
    grad[:,diag_indices,diag_indices] = 1.0
    return grad

@mdp.extension_method("gradient", mdp.nodes.SFANode, "_get_grad")
def _sfa_grad(self, x):
    # the gradient is constant, but have to give it for each x point
    return np.repeat(self.sf.T[np.newaxis,:,:], len(x), axis=0)

@mdp.extension_method("gradient", mdp.nodes.QuadraticExpansionNode,
                      "_get_grad")
def _quadex_grad(self, x):
    # the exapansion is:
    # [x1, x2, x3, x1x1, x1x2, x1x3, x2x2, x2x3, x3,x3]
    dim = self.input_dim
    grad = np.zeros((len(x), self.output_dim, dim))
    # constant part
    diag_indices = np.arange(dim)
    grad[:,diag_indices,diag_indices] = 1.0
    # quadratic part
    i_start = dim
    for i in range(dim):
        grad[:, i_start:i_start+dim-i, i] = x[:,i:]
        diag_indices = np.arange(dim - i)
        grad[:, diag_indices+i_start, diag_indices+i] += x[:,i,np.newaxis]
        i_start += (dim - i)
    return grad

@mdp.extension_method("gradient", mdp.nodes.SFA2Node, "_get_grad")
def _sfa2_grad(self, x):
    quadex_grad = self._expnode._get_grad(x)
    sfa_grad = _sfa_grad(self, x)
    return np.asarray([np.dot(sfa_grad[i], quadex_grad[i])
                       for i in range(len(sfa_grad))])

## mdp.hinet nodes ##

@mdp.extension_method("gradient", mdp.hinet.Layer, "_get_grad")
def _layer_grad(self, x):
    in_start = 0
    in_stop = 0
    out_start = 0
    out_stop = 0
    grad = None
    for node in self.nodes:
        out_start = out_stop
        out_stop += node.output_dim
        in_start = in_stop
        in_stop += node.input_dim
        if grad is None:
            node_grad = node._get_grad(x[:, in_start:in_stop])
            grad = np.zeros([node_grad.shape[0], self.output_dim,
                             self.input_dim],
                            dtype=node_grad.dtype)
            # note that the gradient is block-diagonal
            grad[:, out_start:out_stop, in_start:in_stop] = node_grad
        else:
            grad[:, out_start:out_stop, in_start:in_stop] = \
                node._get_grad(x[:, in_start:in_stop])
    return grad

# this is an optimized implementation, the original implementation is
# used for reference in the unittest
@mdp.extension_method("gradient", mdp.hinet.Switchboard, "_gradient")
def _switchboard_gradient(self, x, grad=None):
    if grad is None:
        grad = np.zeros((len(x), self.input_dim, self.input_dim))
        diag_indices = np.arange(self.input_dim)
        grad[:,diag_indices,diag_indices] = 1.0
    ## custom implementation for greater speed 
    grad =  grad[:, self.connections]
    # update the x value for the next node
    result = self._execute(x)
    if isinstance(result, tuple):
        x = result[0]
        msg = result[1]
    else:
        x = result
        msg = {}
    msg.update({"grad": grad})
    return x, msg
