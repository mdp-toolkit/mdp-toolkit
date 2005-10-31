# Modular toolkit for Data Processing (MDP)
"""
Modular toolkit for Data Processing (MDP) is a Python library to
implement data processing elements (nodes) and to combine them into
data processing sequences (flows).

A node is the basic unit in MDP, and it represents a data processing
element, like for example a learning algorithm, a filter, a
visualization step etc. Each node can have a training phase, during
which the internal structures are learned from training data (e.g. the
weights of a neural network are adapted or the covariance matrix is
estimated) and an execution phase, where new data can be processed
forwards (by processing the data through the node) or backwards (by
applying the inverse of the transformation computed by the node if
defined). MDP is designed to make the implementation of new algorithms
easy and intuitive, for example by setting automatically input and
output dimension and by casting the data to match the typecode
(e.g. float or double precision) of the internal structures. Most of
the nodes were designed to be applied to arbitrarily long sets of
data: the internal structures can be updated successively by sending
chunks of the input data (this is equivalent to online learning if the
chunks consists of single observations, or to batch learning if the
whole data is sent in a single chunk). Already implemented nodes
include Principal Component Analysis (PCA), Independent Component
Analysis (ICA), Slow Feature Analysis (SFA), and Growing Neural Gas
Network.

A flow consists in an acyclic graph of nodes (currently only node
sequences are implemented). The data is sent to an input node and is
successively processed by the following nodes on the graph. The
general flow implementation automatizes the training, execution and
inverse execution (if defined) of the whole graph. Crash recovery is
optionally available: in case of failure, the current state of the flow
is saved for later inspection. A subclass of the basic flow class
allows user-supplied checkpoint functions to be executed at the end of
each phase, for example to save the internal structures of a node for
later analysis.

MDP supports the most common numerical extensions to Python and the
symeig package (a Python wrapper for the LAPACK functions to solve
the standard and generalized eigenvalue problems for symmetric
(hermitian) positive definite matrices). MDP also includes graph
(a lightweight package to handle graphs).

When used together with SciPy (the scientific Python library) and symeig,
MDP gives to the scientific programmer the full power of well-known C and
FORTRAN data processing libraries.  MDP helps the programmer to
exploit Python object oriented design with C and FORTRAN efficiency.

MDP has been written for research in neuroscience, but it has been
designed to be helpful in any context where trainable data processing
algorithms are used.  Its simplicity on the user side together with
the reusability of the implemented nodes could make it also a valid
educational tool.

http://mdp-toolkit.sourceforge.net
"""

# The following code is for importing a numeric extension module for
# Python. We support Numeric, numarray and scipy.
# Note that numarray is much slower than Numeric and scipy, and
# scipy contains more functions and wrappers to FORTRAN LAPACK
# routines, i.e. scipy is almost always faster than Numeric.
# We use some scipy routines anyway. The ones that are not found in
# Numeric and numarray are in utils.scipy_emulation.

import os as _os
import warnings as _warnings

# workaround to import module.submodule
# see http://docs.python.org/lib/built-in-funcs.html
def _name_import(name):
    mod = __import__(name)
    components = name.split('.')
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

# list of supported numerical extensions
_NUMX_LABELS = ['symeig', 'scipy', 'Numeric', 'numarray']

#  label : (numx_module, numx_rand_module, numx_linalg_module)              
_NUMX_PKGS = {_NUMX_LABELS[0] : ('scipy', 'scipy.stats', 'scipy.linalg'),
              _NUMX_LABELS[1] : ('scipy', 'scipy.stats', 'scipy.linalg'),
              _NUMX_LABELS[2] : ('Numeric', 'RandomArray', 'LinearAlgebra'),
              _NUMX_LABELS[3] : ('numarray', 'numarray.random_array',
                                 'numarray.linear_algebra')
              }


# To force MDP to use one specific extension module
# set the environment variable MDPNUMX
# Mainly useful for testing

_USR_LABEL = _os.getenv('MDPNUMX')
if _USR_LABEL in _NUMX_LABELS:
    _NUMX_LABELS = [_USR_LABEL]
elif _USR_LABEL is None:
    pass
else:
    _wrnstr = "\nExtension '%s' not supported. "%(_USR_LABEL) + \
              "Supported extensions:\n %s"%(str(_NUMX_LABELS))
    
    _warnings.warn(_wrnstr, UserWarning)
    del _wrnstr

# try to load in sequence: symeig+scipy, scipy, Numeric, numarray
numx_description = None
for _label in _NUMX_LABELS:
    try:
        _dumb = _name_import(_label)
        _packages = _NUMX_PKGS[_label]
        numx = _name_import(_packages[0])
        numx_rand = _name_import(_packages[1])
        numx_linalg = _name_import(_packages[2])
        numx_description = _label
        break
    except ImportError:
        pass
    
if numx_description is None:
    raise ImportError, \
          "Could not find any of the numeric modules "+ \
          "scipy, Numeric, or numarray"

# clean up
del _os, _warnings, _NUMX_LABELS, _NUMX_PKGS, _USR_LABEL
del _name_import, _label, _dumb, _packages

# define our exceptions and warnings.
# ?? in python 2.4 MDPException can be made new style
# MDPException(Exception, object)
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
from linear_flows import FlowException, FlowExceptionCR

# import base node and flow classes.
from signal_node import Node, SignalNode, Cumulator
from linear_flows import Flow, SimpleFlow, CheckpointFlow, \
     CheckpointFunction, CheckpointSaveFunction

# import helper functions:
from helper_funcs import pca, whitening, fastica, cubica, sfa, get_eta

# import our modules
import nodes
import test

__version__ = '1.2.0.bore_revision'
