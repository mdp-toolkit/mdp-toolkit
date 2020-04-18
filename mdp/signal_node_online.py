from builtins import range
import mdp
from mdp import NodeException, IsNotTrainableException
from mdp import TrainingException, TrainingFinishedException, IsNotInvertibleException
from mdp import Node

__docformat__ = "restructuredtext en"


class OnlineNodeException(NodeException):
    """Base class for exceptions in `OnlineNode` subclasses."""
    pass


class OnlineNode(Node):

    """An online Node (OnlineNode) is the basic building block of
        an online MDP application.

        It represents a data processing element, for example a learning
        algorithm, a data filter, or a visualization step.
        Each node has a training phase (optional), during which the
        internal structures are updated incrementally (or in batches)
        from training data (e.g. the weights of a neural network are adapted).

        Each node can also be executed at any point in time, where new
        data can be processed forwards (or backwards by applying the inverse
        of the transformation computed by the node if defined).

        Unlike a Node, an OnlineNode's execute (or inverse) call __does not__ end the
        training phase of the node. The training phase can only be stopped through
        an explicit 'stop_training' call. Once stopped, an OnlineNode becomes
        functionally equivalent to a trained Node.

        OnlineNode also supports multiple training phases. However, all
        the training_phases are active for each data point. See _train_seq
        documentation for more information.

        The training type of an OnlineNode can be set either to 'incremental'
        or 'batch'. When set to 'incremental', the input data array is passed for
        training sample by sample, while for 'batch' the entire data array is passed.

        An `OnlineNode` inherits all the Node's utility methods, for example
        `copy` and `save`, that returns an exact copy of a node and saves it
        in a file, respectively. Additional methods may be present, depending
        on the algorithm.

        OnlineNodes also support using a pre-seeded random number generator
        through a 'numx_rng' argument. This can be useful to replicate
        results.

        `OnlineNode` subclasses should take care of overwriting (if necessary)
        the functions `_train`, `_stop_training`, `_execute`, 'is_trainable',
        `is_invertible`, `_inverse`, `_get_supported_dtypes` and '_get_supported_training_types'.
        If you need to overwrite the getters and setters of the
        node's properties refer to the docstring of `get_input_dim`/`set_input_dim`,
        `get_output_dim`/`set_output_dim`, `get_dtype`/`set_dtype`, 'get_numx_rng'/'set_numx_rng'.
    """

    def __init__(self, input_dim=None, output_dim=None, dtype=None, numx_rng=None):
        """If the input dimension and the output dimension are
        unspecified, they will be set when the `train` or `execute`
        method is called for the first time.
        If dtype is unspecified, it will be inherited from the data
        it receives at the first call of `train` or `execute`.
        Every subclass must take care of up- or down-casting the internal
        structures to match this argument (use `_refcast` private
        method when possible).
        If numx_rng is unspecified, it will be set to a random number generator
        with a random seed.
        """
        super(OnlineNode, self).__init__(input_dim, output_dim, dtype)
        # this var stores the index of the current training iteration
        self._train_iteration = 0
        # this var stores random number generator
        self._numx_rng = None
        self.set_numx_rng(numx_rng)
        # this var stores the training type. By default, the supported types are 'incremental' and 'batch'.
        # incremental - data is passed through the _train_seq sample by sample
        # batch - data is passed through the _train_seq in one shot (block-incremental training)
        # this variable can only be set using set_training_type() method.
        self._training_type = None

    # properties in addition to the Node properties

    def get_numx_rng(self):
        """Return input dimensions."""
        return self._numx_rng

    def set_numx_rng(self, rng):
        """Set numx random number generator.
        Note that subclasses should overwrite `self._set_numx_rng` when needed.
        """
        if rng is None:
            pass
        elif not isinstance(rng, mdp.numx_rand.mtrand.RandomState):
            raise OnlineNodeException('numx_rng should be of type %s but given %s'
                                      % (str(mdp.numx_rand.mtrand.RandomState), str(type(rng))))
        else:
            self._set_numx_rng(rng)

    def _set_numx_rng(self, rng):
        self._numx_rng = rng

    numx_rng = property(get_numx_rng,
                        set_numx_rng,
                        doc="Numpy seeded random number generator")

    @property
    def training_type(self):
        """Training type (Read only)"""
        return self._training_type

    def _get_supported_training_types(self):
        """Return the list of training types supported by this node.
        """
        return ['incremental', 'batch']

    def set_training_type(self, training_type):
        """Sets the training type
        """
        if training_type in self._get_supported_training_types():
            self._training_type = training_type
        else:
            raise OnlineNodeException("Unknown training type specified %s. Supported types "
                                      "%s" % (str(training_type), str(self._get_supported_training_types())))

    # Each element in the _train_seq contains three sub elements
    # (training-phase, stop-training-phase, execution-phase)
    # as opposed to the two elements for the Node (training-phase, stop-training-phase).
    # Execution-phases come in handy for online training to transfer
    # data between the active training phases.
    # For eg. in a flow of OnlineNodes = [node1, node2, node3],
    # where each phase is assigned to each node.
    # The execution phases enable the output of node1 to
    # be fed to node2 and the output of node2 to node3. Therefore,
    # requiring only two execute calls for the given input x.
    # node1 -> execute -> node2 -> execute -> node3.
    # Whereas, using the original _train_seq, we have
    # node1 -> execute -> node2
    # node1 -> execute -> node2 -> execute -> node3,
    # requiring three execute calls. This difference increases
    # with more number of nodes and is not efficient if the execute calls of the nodes
    # are computationally expensive.
    # Having said that, the modified _train_seq only makes sense for online training
    # and not when the nodes are trained sequentially (with only one active training phase)
    # in an offline sense.
    # The default behavior is kept functionally identical to the
    # original _train_seq. Check OnlineFlowNodes where the execution phases are used.

    _train_seq = property(lambda self: self._get_train_seq(),
                          doc="""\
        List of tuples::

          [(training-phase1, stop-training-phase1, execution-phase1),
           (training-phase2, stop_training-phase2, execution-phase2),
           ...]

        By default::

          _train_seq = [(self._train, self._stop_training, Identity-function)]
        """)

    def _get_train_seq(self):
        return [(self._train, self._stop_training, lambda x, *args, **kwargs: x)]

    # additional OnlineNode states

    def get_current_train_iteration(self):
        """Return the index of the current training iteration."""
        return self._train_iteration

    # check functions

    # The stop-training operation is removed to support continual training
    # and execution phases.
    def _pre_execution_checks(self, x):
        """This method contains all pre-execution checks.
        It can be used when a subclass defines multiple execution methods.
        """
        # if training has not started yet, assume we want to train the node
        if (self.get_current_train_phase() == 0) and not self._train_phase_started:
            self.train(x)

        # control the dimension x
        self._check_input(x)

        # check/set params
        self._check_params(x)

        # set the output dimension if necessary
        if self.output_dim is None:
            self.output_dim = self.input_dim

    # The stop-training operation is removed to support continual training
    # and inversion phases.
    def _pre_inversion_checks(self, y):
        """This method contains all pre-inversion checks.

        It can be used when a subclass defines multiple inversion methods.
        """
        if not self.is_invertible():
            raise IsNotInvertibleException("This node is not invertible.")

        # set the output dimension if necessary
        if self.output_dim is None:
            # if the input_dim is not defined, raise an exception
            if self.input_dim is None:
                errstr = ("Number of input dimensions undefined. Inversion"
                          "not possible.")
                raise NodeException(errstr)
            self.output_dim = self.input_dim

        # control the dimension of y
        self._check_output(y)

    def _check_input(self, x):
        super(OnlineNode, self)._check_input(x)

        # set numx_rng if necessary
        if self.numx_rng is None:
            self.numx_rng = mdp.numx_rand.RandomState()

        # set training type if necessary
        if self.training_type is None:
            # set the first supported training type as default
            self._training_type = self._get_supported_training_types()[0]

    # Additional methods to be implemented by the user

    # these are the methods the user has to overwrite
    # they receive the data already casted to the correct type

    def _check_params(self, x):
        # overwrite to check if the learning parameters of the node are properly defined.
        pass

    # User interface to the overwritten methods

    def train(self, x, *args, **kwargs):
        """Update the internal structures according to the input data `x`.

        `x` is a matrix having different variables on different columns
        and observations on the rows.

        By default, subclasses should overwrite `_train` to implement their
        training phase. The docstring of the `_train` method overwrites this
        docstring.

        Note: a subclass supporting multiple training phases should implement
        the *same* signature for all the training phases and document the
        meaning of the arguments in the `_train` method doc-string. Having
        consistent signatures is a requirement to use the node in a flow.
        """

        if not self.is_trainable():
            raise IsNotTrainableException("This node is not trainable.")

        if not self.is_training():
            err_str = "The training phase has already finished."
            raise TrainingFinishedException(err_str)

        self._check_input(x)
        self._check_train_args(x, *args, **kwargs)
        self._check_params(x)

        x = self._refcast(x)
        self._train_phase_started = True

        if self.training_type == 'incremental':
            x = x[:, None, :]  # to train sample by sample with 2D shape
            for _x in x:
                for _phase in range(len(self._train_seq)):
                    self._train_seq[_phase][0](_x, *args, **kwargs)
                    # legacy support for _train_seq.
                    if len(self._train_seq[_phase]) > 2:
                        _x = self._train_seq[_phase][2](_x, *args, **kwargs)
                self._train_iteration += 1
        else:
            _x = x
            for _phase in range(len(self._train_seq)):
                self._train_seq[_phase][0](_x, *args, **kwargs)
                # legacy support for _train_seq.
                if len(self._train_seq[_phase]) > 2:
                    _x = self._train_seq[_phase][2](_x, *args, **kwargs)
            self._train_iteration += x.shape[0]

    def stop_training(self, *args, **kwargs):
        """Stop the training phase.

        By default, subclasses should overwrite `_stop_training` to implement
        this functionality. The docstring of the `_stop_training` method
        overwrites this docstring.
        """
        if self.is_training() and self._train_phase_started is False:
            raise TrainingException("The node has not been trained.")

        if not self.is_training():
            err_str = "The training phase has already finished."
            raise TrainingFinishedException(err_str)

        # close the current phase.
        for _phase in range(len(self._train_seq)):
            self._train_seq[_phase][1](*args, **kwargs)
        self._train_phase = len(self._train_seq)
        self._train_phase_started = False
        # check if we have some training phase left
        if self.get_remaining_train_phase() == 0:
            self._training = False

    # adding an OnlineNode returns an OnlineFlow
    # adding a trainable/non-trainable Node returns a Flow (to keep it consistent with Node add)
    # adding an OnlineFlow returns an OnlineFlow
    # adding a Flow returns a Flow.
    def __add__(self, other):
        # check other is a node
        if isinstance(other, OnlineNode):
            return mdp.OnlineFlow([self, other])
        elif isinstance(other, Node):
            return mdp.Flow([self, other])
        elif isinstance(other, mdp.Flow):
            flow_copy = other.copy()
            flow_copy.insert(0, self)
            return flow_copy.copy()
        else:
            err_str = ('can only concatenate node'
                       ' (not \'%s\') to node' % type(other).__name__)
            raise TypeError(err_str)

    # string representation

    def __repr__(self):
        # print input_dim, output_dim, dtype and numx_rng
        name = type(self).__name__
        inp = "input_dim=%s" % str(self.input_dim)
        out = "output_dim=%s" % str(self.output_dim)
        if self.dtype is None:
            typ = 'dtype=None'
        else:
            typ = "dtype='%s'" % self.dtype.name
        numx_rng = "numx_rng=%s" % str(self.numx_rng)
        args = ', '.join((inp, out, typ, numx_rng))
        return name + '(' + args + ')'


class PreserveDimOnlineNode(OnlineNode, mdp.PreserveDimNode):
    """Abstract base class with ``output_dim == input_dim``.

    If one dimension is set then the other is set to the same value.
    If the dimensions are set to different values, then an
    `InconsistentDimException` is raised.
    """

    def _set_input_dim(self, n):
        if (self._output_dim is not None) and (self._output_dim != n):
            err = "input_dim must be equal to output_dim for this node."
            raise mdp.InconsistentDimException(err)
        self._input_dim = n
        self._output_dim = n

    def _set_output_dim(self, n):
        if (self._input_dim is not None) and (self._input_dim != n):
            err = "output_dim must be equal to input_dim for this node."
            raise mdp.InconsistentDimException(err)
        self._input_dim = n
        self._output_dim = n
