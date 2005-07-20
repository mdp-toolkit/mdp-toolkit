"""This demo shows how to write subclasses of SignalNode.

This is what you want to read if you intend to expand the MDP nodes library.
"""
import mdp

# Writing your own nodes: subclassing SignalNode
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# MDP tries to make it easy to write new data processing elements
# that fit with the existing elements. To expand the MDP library of
# implemented nodes with your own nodes you can subclass
# the SignalNode class, overriding some of the methods according
# to your needs.
# We'll see in the following some examples:
# - We start by defining a node that multiplies its input by 2.
#
#   Define the class as a subclass of SignalNode:
class TimesTwoNode(mdp.SignalNode):
    #   This node cannot be trained. To define this, one has to overwrite
    #   the ``is_trainable`` method to return 0:
    def is_trainable(self): return 0
    #   Execute has in principle only to multiply x by 2
    def execute(self, x):    
        #   However, we must first call the method of the parent class
        #   that performs some tests, for example to make
        #   sure that ``x`` has the right rank and dimensionality:
        super(TimesTwoNode, self).execute(x)
        #   Each subclass has to handle the typecode defined by the user
        #   or inherited by the input data, and make sure that internal
        #   structures are stored consistently.
        #   This often means that input data has to be cast. SignalNode
        #   contains a helper function that casts the array only if
        #   necessary:
        x = self._refcast(x)
        #   Finally we can compute the result.
        #   Note that we have to cast the scalar to be sure that
        #   if we use some of the numeric extension (e.g. ``Numeric``),
        #   the result of the multiplication is not upcasted
        #   Use the internal helper function to do so:
        return self._scast(2)*x
    #   The inverse of the multiplication by 2 is of course the division by 2:
    def inverse(self, y):
        #   The inverse of the multiplication by 2 is of course the
        #   division by 2.  As in execute, we first have to call the
        #    parent class and cast the input vector and the scalar:
        super(TimesTwoNode, self).inverse(y)
        return self._refcast(y/self._scast(2))
#
#   The same definition without comments:
class TimesTwoNode(mdp.SignalNode):
    def is_trainable(self): return 0
    def execute(self, x):    
        super(TimesTwoNode, self).execute(x)
        x = self._refcast(x)
        return self._scast(2)*x
    def inverse(self, y):
        super(TimesTwoNode, self).inverse(y)
        return self._refcast(y/self._scast(2))
#
#   Test the new node:
node = TimesTwoNode(typecode = 'i')
x = mdp.numx.array([[1.0, 2.0, 3.0]])
y = node(x)
print x, '* 2 =  ', y
print y, '/ 2 =', node.inverse(y)
#
# - We then define a node that raises the input to the power specified
#   at the instance's creation
#
class PowerNode(mdp.SignalNode):
    #   We redefine the init method to take the power as first argument.
    #   In general one should always give the possibility to set the typecode
    #   and the input dimensions. The default value is None, which means that
    #   the exact value is going to be inherited from the input data:
    def __init__(self, power, input_dim=None, typecode=None):
        #   Initialize the parent class:
        super(PowerNode, self).__init__(input_dim=input_dim, typecode=typecode)
        #   Store the power:
        self.power = power
    #   ``PowerNode`` is not trainable...
    def is_trainable(self): return 0
    #   ... nor invertible:
    def is_invertible(self): return 0
    #   It is possible to overwrite the function ``get_supported_typecodes``
    #   to return a list of typecodes supported by the node:
    def get_supported_typecodes(self):
        return ['f', 'd']
    #   The ``execute`` method:
    def execute(self, x):
        super(PowerNode, self).execute(x)
        return self._refcast(x**self._scast(self.power))
#
#   The same definition without comments:
class PowerNode(mdp.SignalNode):
    def __init__(self, power, input_dim=None, typecode=None):
        super(PowerNode, self).__init__(input_dim=input_dim, typecode=typecode)
        self.power = power
    def is_trainable(self): return 0
    def is_invertible(self): return 0
    def get_supported_typecodes(self):
        return ['f', 'd']
    def execute(self, x):
        super(PowerNode, self).execute(x)
        return self._refcast(x**self._scast(self.power))
#
#     Test the new node
node = PowerNode(3)
x = mdp.numx.array([[1.0, 2.0, 3.0]])
y = node.execute(x)
print x, '**', node.power, '=', node(x)
#
# - We now define a node that needs to be trained. The ``MeanfreeNode``
#   computes the mean of its training data and subtracts it from the input
#   during execution:
#
class MeanFreeNode(mdp.SignalNode):
    def __init__(self, input_dim=None, typecode=None):
        super(MeanFreeNode, self).__init__(input_dim=input_dim,
                                           typecode=typecode)
        #   Mean of the input data. We initialize it to None since we still
        #   don't know how large is an input vector:
        self.avg = None
        #   Number of training points:
        self.tlen = 0
    #   The ``train`` method receives the input data:
    def train(self, x):
        super(MeanFreeNode, self).train(x)
        x = self._refcast(x)
        #   Initialize the mean vector with the right size and
        #   typecode if necessary:
        if self.avg is None:
            self.avg = mdp.numx.zeros(self.get_input_dim(),
                                      typecode=self.get_typecode())
        #   Update the mean with the sum of the new data:
        self.avg += sum(x, 0)
        #   Count the number of points processed:
        self.tlen += x.shape[0]
    #   The ``stop_training`` function is called when the training phase is over:
    def stop_training(self):
        super(MeanFreeNode, self).stop_training()
        #   When the training is over, divide the sum of the training
        #   data by the number of training vectors to obtain the mean:
        self.avg /= self._scast(self.tlen)
    #   The ``execute`` and ``inverse`` methods:
    def execute(self, x):
        super(MeanFreeNode, self).execute(x)
        return self._refcast(x - self.avg)
    def inverse(self, y):
        super(MeanFreeNode, self).inverse(y)
        return self._refcast(y + self.avg)
#
#   The same definition without comments:
class MeanFreeNode(mdp.SignalNode):
    def __init__(self, input_dim=None, typecode=None):
        super(MeanFreeNode, self).__init__(input_dim=input_dim,
                                           typecode=typecode)
        self.avg = None
        self.tlen = 0
    def train(self, x):
        super(MeanFreeNode, self).train(x)
        x = self._refcast(x)
        if self.avg is None:
            self.avg = mdp.numx.zeros(self.get_input_dim(),
                                      typecode=self.get_typecode())
        self.avg += sum(x, 0)
        self.tlen += x.shape[0]
    def stop_training(self):
        super(MeanFreeNode, self).stop_training()
        self.avg /= self._scast(self.tlen)
    def execute(self, x):
        super(MeanFreeNode, self).execute(x)
        return self._refcast(x - self.avg)
    def inverse(self, y):
        super(MeanFreeNode, self).inverse(y)
        return self._refcast(y + self.avg)
#
#   Test the new node:
node = MeanFreeNode()
x = mdp.numx_rand.random((10,4))
node.train(x)
y = node.execute(x)
print 'Mean of y (should be zero): ', mdp.utils.mean(y, 0)
#
# - In our last example we'll define a node that repeats its input twice,
#   returning an input that has twice as many dimensions:
#
class TwiceNode(mdp.SignalNode):
    def is_trainable(self): return 0
    def is_invertible(self): return 0
    #   When ``SignalNode`` inherits the input and output dimension from
    #   the input data, it calls the ``_set_default_inputdim`` and
    #   ``_set_default_outputdim`` functions. Here we overwrite the
    #   ``_set_default_outputdim`` to set the output dimension to be twice the
    #   input dimension:
    def _set_default_outputdim(self, nvariables):
        self._output_dim = 2*nvariables
    #   The ``execute`` method:
    def execute(self, x):
        super(TwiceNode, self).execute(x)
        x = self._refcast(x)
        return mdp.numx.concatenate((x, x),1)
#
#   The same definition without comments:
class TwiceNode(mdp.SignalNode):
    def is_trainable(self): return 0
    def is_invertible(self): return 0
    def _set_default_outputdim(self, nvariables):
        self._output_dim = 2*nvariables
    def execute(self, x):
        super(TwiceNode, self).execute(x)
        x = self._refcast(x)
        return mdp.numx.concatenate((x, x),1)
#
#   Test the new node
node = TwiceNode()
x = mdp.numx.zeros((5,2))
print 'x\n', x
print 'twice x\n', node.execute(x)
