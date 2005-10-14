import cPickle as _cPickle
import inspect as _inspect
import mdp
from mdp.utils import refcast, scast

# import numeric module (scipy, Numeric or numarray)
numx = mdp.numx

class SignalNodeException(mdp.MDPException):
    """Base class for exceptions in SignalNode subclasses."""
    pass

class TrainingException(SignalNodeException):
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

class IsNotInvertibleException(SignalNodeException):
    """Raised when the 'inverse' function is called although the
    node is not invertible."""
    pass

class SignalNode(object):
    """A SignalNode corresponds to a learning algorithm or to a generic
    data processing unit. Each SignalNode can have a training phase,
    during which the internal structures are learned from training data
    (e.g. the weights of a neural network are adapted or the covariance
    matrix is estimated) and an execution phase, where new data can be
    processed forwards (by processing the data through the node) or
    backwards (by applying the inverse of the transformation computed by
    the node if defined). The Signalnode class is designed to make the
    implementation of new algorithms easy and intuitive, for example by
    setting automatically input and output dimension and by casting the
    data to match the typecode (e.g. float or double) of the internal
    structures. SignalNode was designed to be applied to arbitrarily
    long sets of data: the internal structures can be updated successively
    by sending chunks of the input data (this is equivalent to online
    learning if the chunks consists of single observations, or to batch
    learning if the whole data is sent in a single chunk).

    A SignalNode can be anything taking a multidimensional input signal
    and returning a multidimensional output signal. It can have two phases: a
    'training' phase, were statistics about the data are collected by calling
    the function 'train', and an 'execution' phase, where the input
    data is processed somehow and output data is returned by calling
    the function 'execute'.

    SignalNode subclasses should take care of redefining (if necessary)
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
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._typecode = None
        if typecode is not None:
            self._set_typecode(typecode)
            
        self.__training = -1

        # skip the training phase if the node is not trainable
        if not self.is_trainable():
            self.stop_training()

    ### getters

    def is_training(self):
        """Return 1 if the node is in the training phase, -1 if the training
        phase has not yet started, and 0 otherwise."""
        return self.__training

    def is_trainable(self):
        """Return 1 if the node can be trained, 0 otherwise."""
        return 1

    def is_invertible(self):
        """Return 1 if the node can be inverted, 0 otherwise."""
        return 1

    def get_input_dim(self):
        """Return input dimensions."""
        return self._input_dim

    def get_output_dim(self):
        """Return output dimensions."""
        return self._output_dim

    def _set_typecode(self,typecode):
        if self._typecode is not None:
            errstr = "Typecode is already set to '%s' " %(self._typecode)
            raise SignalNodeException, errstr
        
        if typecode in self.get_supported_typecodes():
            self._typecode = typecode
        else:
            errstr = "\nTypecode '%s' is not supported.\n"%typecode+ \
                      "Supported typecodes: %s" \
                      %(str(self.get_supported_typecodes()))
            raise SignalNodeException, errstr

    def get_supported_typecodes(self):
        """Return the list of typecodes supported by this node."""
        return ['i','l','f','d','F','D']

    def get_typecode(self):
        """Return typecode."""
        return self._typecode

    ### default settings- and check- functions

    def _set_default_inputdim(self, nvariables):
        self._input_dim = nvariables
        
    def _set_default_outputdim(self, nvariables):
        self._output_dim = nvariables
        
    def _check_input(self, x):
        # check input rank
        if not numx.rank(x) == 2:
            error_str = "x has rank %d, should be 2"\
                        %(numx.rank(x))
            raise SignalNodeException, error_str

        # set the input dimension if necessary
        if not self._input_dim:
            self._set_default_inputdim(x.shape[1])

        # set the typecode if necessary
        if not self._typecode:
            self._set_typecode(x.typecode())

        # check the input dimension
        if not x.shape[1]==self._input_dim:
            error_str = "x has dimension %d, should be %d" \
                        % (x.shape[1], self._input_dim)
            raise SignalNodeException, error_str

        if x.shape[0]==0:
            error_str = "x must have at least one observation (zero given)"
            raise SignalNodeException, error_str
        
    def _check_if_training(self):
        #If training stop training.
        if self.is_training():
            self.stop_training()

    def _check_output(self, y):
        # check output rank
        if not numx.rank(y) == 2:
            error_str = "y has rank %d, should be 2"\
                        %(numx.rank(y))
            raise SignalNodeException, error_str

        # check the output dimension
        if not y.shape[1]==self._output_dim:
            error_str = "y has dimension %d, should be %d" \
                        % (y.shape[1], self._output_dim)
            raise SignalNodeException, error_str

    def _refcast(self, x):
        """Helper function to cast arrays to the internal typecode."""
        return refcast(x,self._typecode)

    def _scast(self, scalar):
        """Helper function to cast scalars to the internal typecode."""
        # if numeric finally becomes scipy_base we will remove this.
        return scast(scalar, self._typecode)
    
    ### main functions
    
    def train(self, x):
        """Update the internal structures according to the input data 'x'.
        
        'x' is a matrix having different variables on different columns
        and observations on the rows."""

        if not self.is_trainable():
            raise IsNotTrainableException, "This node is not trainable."

        if not self.is_training():
            raise TrainingFinishedException, \
                  "The training phase has already finished."

        # control the dimension of x
        self._check_input(x)

        self.__training = 1
        
    def stop_training(self):
        """Stop the training phase."""
        if self.is_trainable() and self.__training == -1:
            raise TrainingException, \
                  "The node has not been trained."
        
        if not self.is_training():
            raise TrainingFinishedException, \
                  "The training phase has already finished."
        self.__training = 0
        
    def execute(self, x):
        """Process the data contained in 'x'.
        
        If the object is still in the training phase, the function
        'stop_training' will be called.
        'x' is a matrix having different variables on different columns
        and observations on the rows."""
        
        # if training stop training
        self._check_if_training()
        
        # control the dimension x
        self._check_input(x)

        # set the output dimension if necessary
        if not self._output_dim:
            self._set_default_outputdim(self._input_dim)

    def inverse(self, y):
        """Invert 'x'.
        
        If the node is invertible, compute the input x such that
        y = execute(x)."""
        
        # if training stop training 
        self._check_if_training()

        # set the output dimension if necessary
        if not self._output_dim:
            # if the input_dim is not defined, raise an exception
            if not self._input_dim:
                errstr = "Number of input dimensions undefined. Inversion"+\
                         "not possible."
                raise SignalNodeException, errstr
            self._set_default_outputdim(self._input_dim)
        
        # control the dimension of y
        self._check_output(y)

        if not self.is_invertible():
            raise IsNotInvertibleException, "This node is not invertible."

    def __call__(self,x):
        """Calling an instance if SignalNode is equivalent to call
        its 'execute' method."""
        return self.execute(x)

    ###### string representation
    def __str__(self):
        return str(type(self).__name__)
    
    def __repr__(self):
        # print input_dim, output_dim, typecode 
        name = type(self).__name__
        inp = "input_dim=%s"%str(self._input_dim)
        out = "output_dim=%s"%str(self._output_dim)
        typ = "typecode='%s'"%self._typecode
        args = ', '.join((inp, out, typ))
        return name+'('+args+')'

    def copy(self, protocol = -1):
        """Return a deep copy of the node.
        Protocol is the pickle protocol."""
        as_str = _cPickle.dumps(self, protocol)
        return _cPickle.loads(as_str)

class Cumulator(SignalNode):
    """A Cumulator is a SignalNode whose training phase simply cumulates
    all input data.
    This makes it possible to implement batch-mode learning.
    """

    def __init__(self, input_dim = None, output_dim = None, typecode = None):
        super(Cumulator, self).__init__(input_dim, output_dim, typecode)
        self.data = []
        self.tlen = 0

    def train(self,x):
        """Cumulate all imput data in a one dimensional list.
        """
        super(Cumulator, self).train(x)
        x = self._refcast(x)
        self.tlen += x.shape[0]
        self.data.extend(numx.ravel(x).tolist())

    def stop_training(self):
        """Cast the data list to array type and reshape it.
        """
        super(Cumulator, self).stop_training()
        self.data = numx.array(self.data, typecode = self._typecode)
        self.data.shape = (self.tlen, self._input_dim)

class IdentityNode(SignalNode):
    """An IdentityNode is a SignalNode that performs the identity operation,
    i.e. returns the input when 'execute' and 'inverse' methods are called.
    It is useful to generate subclasses of nodes that only perform analysis
    on input data, without modifying it. 
    """

    def execute(self,x):
        super(IdentityNode, self).execute(x)
        return self._refcast(x)

    def inverse(self,x):
        super(IdentityNode, self).inverse(x)
        return self._refcast(x)
