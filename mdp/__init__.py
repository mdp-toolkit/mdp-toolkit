# Modular toolkit for Data Processing (MDP)
"""
The Modular toolkit for Data Processing (MDP) is a library of widely
used data processing algorithms that can be combined according to a
pipeline analogy to build more complex data processing software.

From the user's perspective, MDP consists of a collection of
supervised and unsupervised learning algorithms, and other data
processing units (nodes) that can be combined into data processing
sequences (flows) and more complex feed-forward network
architectures. Given a set of input data, MDP takes care of
successively training or executing all nodes in the network. This
allows the user to specify complex algorithms as a series of simpler
data processing steps in a natural way.

The base of available algorithms is steadily increasing and includes,
to name but the most common, Principal Component Analysis (PCA and
NIPALS), several Independent Component Analysis algorithms (CuBICA,
FastICA, TDSEP, JADE, and XSFA), Slow Feature Analysis, Gaussian
Classifiers, Restricted Boltzmann Machine, and Locally Linear
Embedding.

Particular care has been taken to make computations efficient in terms
of speed and memory.  To reduce memory requirements, it is possible to
perform learning using batches of data, and to define the internal
parameters of the nodes to be single precision, which makes the usage
of very large data sets possible.  Moreover, the 'parallel' subpackage
offers a parallel implementation of the basic nodes and flows.

From the developer's perspective, MDP is a framework that makes the
implementation of new supervised and unsupervised learning algorithms
easy and straightforward.  The basic class, 'Node', takes care of
tedious tasks like numerical type and dimensionality checking, leaving
the developer free to concentrate on the implementation of the
learning and execution phases. Because of the common interface, the
node then automatically integrates with the rest of the library and
can be used in a network together with other nodes. A node can have
multiple training phases and even an undetermined number of phases.
This allows the implementation of algorithms that need to collect some
statistics on the whole input before proceeding with the actual
training, and others that need to iterate over a training phase until
a convergence criterion is satisfied. The ability to train each phase
using chunks of input data is maintained if the chunks are generated
with iterators. Moreover, crash recovery is optionally available: in
case of failure, the current state of the flow is saved for later
inspection.

MDP has been written in the context of theoretical research in
neuroscience, but it has been designed to be helpful in any context
where trainable data processing algorithms are used. Its simplicity on
the user side together with the reusability of the implemented nodes
make it also a valid educational tool.

http://mdp-toolkit.sourceforge.net
"""

class MDPException(Exception):
    """Base class for exceptions in MDP."""
    pass

class MDPWarning(UserWarning):
    """Base class for warnings in MDP."""
    pass

import os as _os

# list of supported numerical extensions
_NUMX_LABELS = ['scipy', 'numpy']

# To force MDP to use one specific extension module
# set the environment variable MDPNUMX
# Mainly useful for testing
_USR_LABEL = _os.getenv('MDPNUMX')
if _USR_LABEL in _NUMX_LABELS:
    _NUMX_LABELS = [_USR_LABEL]
elif _USR_LABEL is None:
    pass
else:
    err = ("\nExtension '%s' not supported. " 
           "Supported extensions:\n %s" % (_USR_LABEL,str(_NUMX_LABELS)))
    raise ImportError(err)

# try to load in sequence: scipy, numpy
numx_description = None
for _label in _NUMX_LABELS:
    try:
        if _label == 'scipy':
            import scipy, scipy.linalg, scipy.fftpack, scipy.version
            numx = scipy
            numx_rand = scipy.random
            numx_linalg = scipy.linalg
            numx_fft = scipy.fftpack
            numx_description = 'scipy'
            numx_version = scipy.version.version
            del scipy
            break
        else:
            import numpy
            import numpy as numx
            import numpy.random as numx_rand
            import numpy.linalg as numx_linalg
            import numpy.fft as numx_fft
            numx_description = 'numpy'
            numx_version = numpy.version.version
            del numpy
            break
    except ImportError:
        pass
        
if numx_description is None:
    raise ImportError("Could not find any of the numeric modules "
                      "scipy or numpy")

del _os, _NUMX_LABELS, _USR_LABEL, _label

# import the utils module (used by other modules)
# here we set scipy_emulation if needed.
import utils

# import exceptions from nodes and flows
from signal_node import (NodeException, TrainingException,
                         TrainingFinishedException, IsNotTrainableException,
                         IsNotInvertibleException)
from linear_flows import CrashRecoveryException, FlowException, FlowExceptionCR

# import base node and flow classes.
from signal_node import NodeMetaclass, Node, Cumulator
from linear_flows import (Flow, CheckpointFlow,
                          CheckpointFunction, CheckpointSaveFunction)

# import helper functions:
from helper_funcs import pca, whitening, fastica, sfa, get_eta

# import our modules
import nodes
import hinet
import parallel

# import test functions:
from test import test

# clean up namespace
del signal_node
del linear_flows

# explicitly set __all__, mainly needed for epydoc
__all__ = ['CheckpointFlow', 'CheckpointFunction', 'CheckpointSaveFunction',
           'CrashRecoveryException', 'Cumulator', 'Flow', 'FlowException',
           'FlowExceptionCR', 'IsNotInvertibleException',
           'IsNotTrainableException', 'MDPException', 'MDPWarning', 'Node',
           'NodeException', 'TrainingException', 'TrainingFinishedException',
           'contrib', 'get_eta', 'graph', 'helper_funcs', 'hinet', 'nodes', 
           'numx_description', 'pca', 'sfa', 'test', 'utils', 'whitening',
           'parallel', 'numx_version']

__version__ = '2.6'
__revision__ = utils.get_svn_revision()
__authors__ = 'Pietro Berkes, Niko Wilbert, and Tiziano Zito'
__copyright__ = '(c) 2003-2009 Pietro Berkes, Niko Wilbert, Tiziano Zito'
__license__ = 'LGPL v3, http://www.gnu.org/licenses/lgpl.html'
__contact__ = 'mdp-toolkit-users AT lists.sourceforge.net'
