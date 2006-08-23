# Modular toolkit for Data Processing (MDP)
"""
Modular toolkit for Data Processing (MDP) is a data processing
framework written in Python.

From the user's perspective, MDP consists of a collection of trainable
supervised and unsupervised algorithms or other data processing units
(nodes) that can be combined into data processing flows. Given a
sequence of input data, MDP takes care of successively training or
executing all nodes in the flow. This structure allows to specify
complex algorithms as a sequence of simpler data processing steps in a
natural way. Training can be performed using small chunks of input
data, so that the use of very large data sets becomes possible while
reducing the memory requirements. Memory usage can also be minimized
by defining the internals of the nodes to be single precision.

The base of readily available algorithms includes Principal Component
Analysis, two flavors of Independent Component Analysis, Slow Feature
Analysis, Gaussian Classifiers, Growing Neural Gas, Fisher
Discriminant Analysis, and Factor Analysis.

From the developer's perspective, MDP is a framework to make the
implementation of new algorithms easier. The basic class 'Node' takes
care of tedious tasks like numerical type and dimensionality checking,
leaving the developer free to concentrate on the implementation of the
training and execution phases. The node then automatically integrates
with the rest of the library and can be used in a flow together with
other nodes. A node can have multiple training phases and even an
undetermined number of phases. This allows for example the
implementation of algorithms that need to collect some statistics on
the whole input before proceeding with the actual training, or others
that need to iterate over a training phase until a convergence
criterion is satisfied. The ability to train each phase using chunks
of input data is maintained if the chunks are generated with
iterators. Moreover, crash recovery is optionally available: in case
of failure, the current state of the flow is saved for later
inspection.

MDP has been written in the context of theoretical research in
neuroscience, but it has been designed to be helpful in any context
where trainable data processing algorithms are used. Its simplicity on
the user side together with the reusability of the implemented nodes
make it also a valid educational tool.

As its user base is steadily increasing, MDP appears as a good
candidate for becoming a common repository of user-supplied, freely
available, Python implemented data processing algorithms.

http://mdp-toolkit.sourceforge.net
"""


import numpy as numx
import numpy.random as numx_rand
import numpy.linalg as numx_linalg

numx_description = 'numpy'

class MDPException(Exception):
    """Base class for exceptions in MDP."""
    pass

class MDPWarning(UserWarning):
    """Base class for warnings in MDP."""
    pass

# import the utils module (used by other modules)
# here we set scipy_emulation if needed.
import utils

# import exceptions from nodes and flows
from signal_node import NodeException, TrainingException, \
     TrainingFinishedException, IsNotTrainableException, \
     IsNotInvertibleException
from linear_flows import CrashRecoveryException, FlowException, FlowExceptionCR

# import base node and flow classes.
from signal_node import Node, Cumulator
from linear_flows import Flow, CheckpointFlow, \
     CheckpointFunction, CheckpointSaveFunction

# import helper functions:
from helper_funcs import pca, whitening, fastica, cubica, sfa, get_eta, \
                         sfa2, factor_analysis

# import test functions:
from test import test

# import our modules
import nodes

__version__ = '2.0RC2'
