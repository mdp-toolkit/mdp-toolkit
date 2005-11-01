import cPickle as _cPickle
import warnings as _warnings
import mdp

# import numeric module (scipy, Numeric or numarray)
numx = mdp.numx

class NodeException(mdp.MDPException):
    """Base class for exceptions in Node subclasses."""
    pass

# deprecated alias
class SignalNodeException(NodeException):
    """Deprecated, use NodeException instead."""
    def __init__(self, str = ''):
        wrnstr = "The alias 'SignalNodeException' is deprecated and won't " + \
        "be continued in future releases. Use 'NodeException' instead."
        _warnings.warn(wrnstr, DeprecationWarning)
        super(SignalNodeException, self).__init__(str)

class TrainingException(NodeException):
    """Base class for exceptions in the training phase."""
    pass

class TrainingFinishedException(TrainingException):
    """Raised when the 'train' function is called although the
    training phase is closed."""
    pass

class IsNotTrainableException(TrainingException):
    """Raised when the 'train' function is called although the
    node is not trainable."""
    pass

class IsNotInvertibleException(NodeException):
    """Raised when the 'inverse' function is called although the
    node is not invertible."""
    pass


class Node(object):
    """A Node corresponds to a learning algorithm or to a generic
    data processing unit. Each Node can have a training phase,
    during which the internal structures are learned from training data
    (e.g. the weights of a neural network are adapted or the covariance
    matrix is estimated) and an execution phase, where new data can be
    processed forwards (by processing the data through the node) or
    backwards (by applying the inverse of the transformation computed by
    the node if defined). The Node class is designed to make the
    implementation of new algorithms easy and intuitive, for example by
    setting automatically input and output dimension and by casting the
    data to match the typecode (e.g. float or double) of the internal
    structures. Node was designed to be applied to arbitrarily
    long sets of data: the internal structures can be updated successively
    by sending chunks of the input data (this is equivalent to online
    learning if the chunks consists of single observations, or to batch
    learning if the whole data is sent in a single chunk).

    A Node can be anything taking a multidimensional input signal
    and returning a multidimensional output signal. It can have two phases: a
    'training' phase, were statistics about the data are collected by calling
    the function 'train', and an 'execution' phase, where the input
    data is processed somehow and output data is returned by calling
    the function 'execute'.

    Node subclasses should take care of redefining (if necessary)
    the functions is_trainable, train, stop_training, execute, is_invertible,
    inverse, and get_supported_typecodes."""

    def __init__(self, input_dim = None, output_dim = None, typecode = None):
        """If the input dimension and the output dimension are
        unspecified, they will be set when the 'train' or 'execute'
        function is called for the first time.
        If the typecode is unspecified, it will be inherited from the data
        it receives at the first call of 'train' or 'execute'. Every subclass
        must take care of up- or down-casting the input and its internal
        structures to match this argument (use _refcast and _scast private
        methods when possible).
        """
        # initialize basic attributes
        self._input_dim = None
        self._output_dim = None
        self._typecode = None
        # call set functions for properties
        self.set_input_dim(input_dim)
        self.set_output_dim(output_dim)
        self.set_typecode(typecode)
 
        # skip the training phase if the node is not trainable
        if not self.is_trainable():
            self._training = False
            self._train_phase = -1
            self._train_phase_started = False
        else:
            # this var stores at which point in the training sequence we are
            self._train_phase = 0          
            # this var is False if the training of the current phase hasn't
            #  started yet, True otherwise
            self._train_phase_started = False
            # this var is False if the complete training is finished
            self._training = True

    ### properties

    def get_input_dim(self):
        """Return input dimensions."""
        return self._input_dim

    def set_input_dim(self, n):
        """Set input dimensions.
        Performs sanity checks and then calls self._set_input_dim(n), which
        is responsible for setting the internal attribute self._input_dim.
        Note that self._set_input_dim can be overriden by subclasses,
        """
        if n is None:
            pass
        elif (self._input_dim is not None) and (self._input_dim !=  n):
            msg = "Input dim are set already (%d)!"%(self.input_dim)
            raise NodeException, msg
        else:
            self._set_input_dim(n)

    def _set_input_dim(self, n):
        self._input_dim = n

    input_dim = property(get_input_dim,
                         set_input_dim,
                         doc = "Input dimensions")

    def get_output_dim(self):
        """Return output dimensions."""
        return self._output_dim

    def set_output_dim(self, n):
        """Set output dimensions.
        Performs sanity checks and then calls self._set_output_dim(n), which
        is responsible for setting the internal attribute self._output_dim.
        Note that self._set_output_dim can be overriden by subclasses,
        """
        if n is None:
            pass
        elif (self._output_dim is not None) and (self._output_dim != n):
            msg = "Output dim are set already (%d)!"%(self.output_dim)
            raise NodeException, msg
        else:
            self._set_output_dim(n)

    def _set_output_dim(self, n):
        self._output_dim = n

    output_dim = property(get_output_dim,
                          set_output_dim,
                          doc = "Output dimensions")

    def get_typecode(self):
        """Return typecode."""
        return self._typecode
    
    def set_typecode(self, t):
        """Set Node's internal structures typecode.
        Performs sanity checks and then calls self._set_typecode(n), which
        is responsible for setting the internal attribute self._typecode.
        Note that self._set_typecode can be overriden by subclasses,
        """
        if t is None:
            pass
        elif (self._typecode is not None) and (self._typecode != t):
            errstr = "Typecode is already set to '%s' " %(self.typecode)
            raise NodeException, errstr
        elif t not in self.get_supported_typecodes():
            errstr = "\nTypecode '%s' is not supported.\n"%t+ \
                      "Supported typecodes: %s" \
                      %(str(self.get_supported_typecodes()))
            raise NodeException, errstr
        else:
            self._set_typecode(t)

    def _set_typecode(self, t):
        self._typecode = t
        
    typecode = property(get_typecode,
                        set_typecode,
                        doc = "Typecode")



    _train_seq = property(lambda self: self._get_train_seq(),
                          doc = "List of tuples: [(training-phase1, " +\
                          "stop-training-phase1), (training-phase2, " +\
                          "stop_training-phase2), ... ].\n" +\
                          " By default _train_seq = [(self._train," +\
                          " self._stop_training]")

    def _get_train_seq(self):
        return [(self._train, self._stop_training)]

    ### Node states
    def is_training(self):
        """Return True if the node is in the training phase,
        False otherwise."""
        return self._training

    def get_current_train_phase(self):
        """Return the index of the current training phase. The training phases
        are defined in the list self._train_seq."""
        return self._train_phase

    def get_remaining_train_phase(self):
        """Return the number of training phases still to accomplish."""
        return len(self._train_seq) - self._train_phase

    ### Node capabilities
    def is_trainable(self):
        """Return True if the node can be trained, False otherwise."""
        return True

    def is_invertible(self):
        """Return True if the node can be inverted, False otherwise."""
        return True

    def get_supported_typecodes(self):
        """Return the list of typecodes supported by this node."""
        return ['i','l','f','d','F','D']

    ### check functions
    def _check_input(self, x):
        # check input rank
        if not numx.rank(x) == 2:
            error_str = "x has rank %d, should be 2"\
                        %(numx.rank(x))
            raise NodeException, error_str

        # set the input dimension if necessary
        if self.input_dim is None:
            self.input_dim = x.shape[1]

        # set the typecode if necessary
        if self.typecode is None:
            self.typecode = x.typecode()

        # check the input dimension
        if not x.shape[1] == self.input_dim:
            error_str = "x has dimension %d, should be %d" \
                        % (x.shape[1], self.input_dim)
            raise NodeException, error_str

        if x.shape[0] == 0:
            error_str = "x must have at least one observation (zero given)"
            raise NodeException, error_str
        
    def _check_output(self, y):
        # check output rank
        if not numx.rank(y) == 2:
            error_str = "y has rank %d, should be 2"\
                        %(numx.rank(y))
            raise SignalNodeException, error_str

        # check the output dimension
        if not y.shape[1] == self.output_dim:
            error_str = "y has dimension %d, should be %d" \
                        % (y.shape[1], self.output_dim)
            raise NodeException, error_str

    def _if_training_stop_training(self):
        if self.is_training():
            self.stop_training()
            # if there is some training phases left
            # we shouldn't be here!
            if self.get_remaining_train_phase() > 0:
                raise TrainingException, \
                      "The training phases are not completed yet."

    def _pre_execution_checks(self, x):
        """This method contains all pre-execution checks.
        It can be used when a subclass defines multiple execution methods."""
        
        self._if_training_stop_training()
        
        # control the dimension x
        self._check_input(x)

        # set the output dimension if necessary
        if self.output_dim is None:
            self.output_dim = self.input_dim

    def _check_train_args(self, x, *args):
        # implemented by subclasses if needed
        pass

    ### casting helper functions

    def _refcast(self, x):
        """Helper function to cast arrays to the internal typecode."""
        return mdp.utils.refcast(x, self.typecode)

    def _scast(self, scalar):
        """Helper function to cast scalars to the internal typecode."""
        # if numeric finally becomes scipy_base we will remove this.
        return mdp.utils.scast(scalar, self.typecode)
    
    ### Methods to be implemented by the user

    # this are the methods the user has to overwrite
    # they receive the data already casted to the correct type
    
    def _train(self, x, *args):
        if self.is_trainable():
            raise NotImplementedError
        else:
            pass

    def _stop_training(self):
        if self.is_trainable():
            raise NotImplementedError
        else:
            pass

    def _execute(self, x):
        return x
        
    def _inverse(self, x):
        if self.is_invertible():
            return x
        else:
            pass

    ### User interface to the overwritten methods
    
    def train(self, x, *args):
        """Update the internal structures according to the input data 'x'.
        
        'x' is a matrix having different variables on different columns
        and observations on the rows."""

        if not self.is_trainable():
            raise IsNotTrainableException, "This node is not trainable."

        if not self.is_training():
            raise TrainingFinishedException, \
                  "The training phase has already finished."

        self._check_input(x)
        self._check_train_args(x, *args)        
        
        self._train_phase_started = True
        self._train_seq[self._train_phase][0](self._refcast(x), *args)

    def stop_training(self):
        """Stop the training phase."""
        if self.is_trainable() and self._train_phase_started == False:
            raise TrainingException, \
                  "The node has not been trained."
        
        if not self.is_training():
            raise TrainingFinishedException, \
                  "The training phase has already finished."

        # close the current phase.
        self._train_seq[self._train_phase][1]()
        self._train_phase += 1
        self.train_phase_started = False
        # check if we have some training phase left
        if self.get_remaining_train_phase() == 0:
            self._training = False

    def execute(self, x, *args, **kargs):
        """Process the data contained in 'x'.
        
        If the object is still in the training phase, the function
        'stop_training' will be called.
        'x' is a matrix having different variables on different columns
        and observations on the rows."""
        self._pre_execution_checks(x)
        return self._execute(self._refcast(x), *args, **kargs)

    def inverse(self, y, *args, **kargs):
        """Invert 'y'.
        
        If the node is invertible, compute the input x such that
        y = execute(x)."""
        
        if not self.is_invertible():
            raise IsNotInvertibleException, "This node is not invertible."

        self._if_training_stop_training()

        # set the output dimension if necessary
        if self.output_dim is None:
            # if the input_dim is not defined, raise an exception
            if self.input_dim is None:
                errstr = "Number of input dimensions undefined. Inversion"+\
                         "not possible."
                raise NodeException, errstr
            self.output_dim = self.input_dim
        
        # control the dimension of y
        self._check_output(y)

        return self._inverse(self._refcast(y), *args, **kargs)

    def __call__(self, x):
        """Calling an instance if Node is equivalent to call
        its 'execute' method."""
        return self.execute(x)

    ###### string representation
    
    def __str__(self):
        return str(type(self).__name__)
    
    def __repr__(self):
        # print input_dim, output_dim, typecode 
        name = type(self).__name__
        inp = "input_dim=%s"%str(self.input_dim)
        out = "output_dim=%s"%str(self.output_dim)
        typ = "typecode='%s'"%self.typecode
        args = ', '.join((inp, out, typ))
        return name+'('+args+')'

    def copy(self, protocol = -1):
        """Return a deep copy of the node.
        Protocol is the pickle protocol."""
        as_str = _cPickle.dumps(self, protocol)
        return _cPickle.loads(as_str)

class Cumulator(Node):
    """A Cumulator is a Node whose training phase simply cumulates
    all input data.
    This makes it possible to implement batch-mode learning.
    """

    def __init__(self, input_dim = None, output_dim = None, typecode = None):
        super(Cumulator, self).__init__(input_dim, output_dim, typecode)
        self.data = []
        self.tlen = 0

    def _train(self, x):
        """Cumulate all imput data in a one dimensional list.
        """
        self.tlen += x.shape[0]
        self.data.extend(numx.ravel(x).tolist())

    def _stop_training(self):
        """Cast the data list to array type and reshape it.
        """
        self._training = False
        self.data = numx.array(self.data, typecode = self.typecode)
        self.data.shape = (self.tlen, self.input_dim)

# deprecated alias
class SignalNode(Node):
    def __init__(self, input_dim = None, output_dim = None, typecode = None):
        wrnstr = "The alias 'SignalNode' is deprecated and won't be " + \
        "continued in future releases. Use 'Node' instead."
        _warnings.warn(wrnstr, DeprecationWarning)
        super(SignalNode, self).__init__(input_dim, output_dim, typecode)
