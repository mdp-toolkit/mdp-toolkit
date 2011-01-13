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

The base of available algorithms is steadily increasing and includes
signal processing methods (Principal Component Analysis,
Independent Component Analysis, Slow Feature Analysis),
manifold learning methods ([Hessian] Locally Linear Embedding),
several classifiers, probabilistic methods (Factor Analysis, RBM),
data pre-processing methods, and many others.

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
__docformat__ = "restructuredtext en"

__short_description__ = (
    "MDP is a Python library of widely used data processing algorithms"
    "that can be combined according to a pipeline analogy to build more"
    "complex data processing software. The base of available algorithms"
    "includes signal processing methods (Principal Component Analysis,"
    "Independent Component Analysis, Slow Feature Analysis),"
    "manifold learning methods ([Hessian] Locally Linear Embedding),"
    "several classifiers, probabilistic methods (Factor Analysis, RBM),"
    "data pre-processing methods, and many others."
    )

class MDPException(Exception):
    """Base class for exceptions in MDP."""
    pass

class MDPWarning(UserWarning):
    """Base class for warnings in MDP."""
    pass

class MDPDeprecationWarning(DeprecationWarning, MDPWarning):
    """Warn about deprecated MDP API."""
    pass


import configuration

config = configuration.config
(numx_description, numx, numx_linalg, numx_fft,
 numx_rand, numx_version) = configuration.get_numx()

configuration.set_configuration()

# import the utils module (used by other modules)
import utils
# set symeig
utils.symeig = configuration.get_symeig(numx_linalg)

__version__ = '3.0'
__revision__ = utils.get_git_revision()
__authors__ = 'MDP Developers'
__copyright__ = '(c) 2003-2011 mdp-toolkit-devel@lists.sourceforge.net'
__license__ = 'BSD License, see COPYRIGHT'
__contact__ = 'mdp-toolkit-users@lists.sourceforge.net'
__homepage__ = 'http://mdp-toolkit.sourceforge.net'


# import exceptions from nodes and flows
from signal_node import (NodeException, InconsistentDimException,
                         TrainingException,
                         TrainingFinishedException, IsNotTrainableException,
                         IsNotInvertibleException)
from linear_flows import CrashRecoveryException, FlowException, FlowExceptionCR

# import base nodes and flow classes
from signal_node import (NodeMetaclass, Node, PreserveDimNode,
                         Cumulator, VariadicCumulator)

from linear_flows import (Flow, CheckpointFlow,
                          CheckpointFunction, CheckpointSaveFunction)

# import helper functions:
from helper_funcs import pca, fastica

# import extension mechanism
from extension import (ExtensionException, extension_method,
                       ExtensionNodeMetaclass,
                       ExtensionNode, get_extensions,
                       get_active_extensions, with_extension,
                       activate_extension, deactivate_extension,
                       activate_extensions, deactivate_extensions,
                       extension)

# import classifier node
from classifier_node import (ClassifierNode, ClassifierCumulator)

# import our modules
import nodes
import hinet
import parallel
from test import test

# clean up namespace
del signal_node
del linear_flows
del classifier_node
del helper_funcs
del configuration

# explicitly set __all__, mainly needed for epydoc
__all__ = ['config',
           'CheckpointFlow',
           'CheckpointFunction',
           'CheckpointSaveFunction',
           'ClassifierCumulator',
           'ClassifierNode',
           'CrashRecoveryException',
           'Cumulator',
           'ExtensionNode',
           'ExtensionNodeMetaclass',
           'Flow',
           'FlowException',
           'FlowExceptionCR',
           'IsNotInvertibleException',
           'IsNotTrainableException',
           'MDPException',
           'MDPWarning',
           'Node',
           'NodeException',
           'TrainingException',
           'TrainingFinishedException',
           'VariadicCumulator',
           'activate_extension',
           'activate_extensions',
           'deactivate_extension',
           'deactivate_extensions',
           'extension_method',
           'get_extensions',
           'graph',
           'hinet',
           'nodes',
           'parallel',
           'pca',
           'fastica',
           'utils',
           'with_extension',
           ]

if config.has_joblib:
    import caching
    __all__ += ['caching']

utils.fixup_namespace(__name__, __all__,
                      ('signal_node',
                       'linear_flows',
                       'helper_funcs',
                       'extension',
                       'classifier_node',
                       'configuration',
                       ))
