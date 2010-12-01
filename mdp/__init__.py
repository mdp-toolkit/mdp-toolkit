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


class ExternalDep(object):
    def __init__(self, version=None, failmsg=None):
        self.version = version
        self.failmsg = failmsg

    def __repr__(self):
        if self:
            return "%s" % self.version
        else:
            return "NOT AVAILABLE: %s" % self.failmsg

class ExternalDepFail(ExternalDep):
    def __init__(self, failmsg):
        super(ExternalDepFail, self).__init__(failmsg=failmsg)

    def __nonzero__(self):
        return False

class ExternalDepFound(ExternalDep):
    def __init__(self, version):
        super(ExternalDepFound, self).__init__(version=version)

    def __nonzero__(self):
        return True


import os as _os
import functools

# To force MDP to use one specific extension module
# set the environment variable MDPNUMX
# Mainly useful for testing
_USR_LABEL = _os.getenv('MDPNUMX')
if _USR_LABEL and _USR_LABEL not in ('numpy', 'scipy'):
    err = """
Numerical backend '%s' not supported.
Supported backends: numpy, scipy.""" % _USR_LABEL
    raise ImportError(err)


numx_description = None
numx_exceptions = {}

if not _USR_LABEL or _USR_LABEL=='scipy':
    try:
        import scipy_fail
        import scipy as numx
        from scipy import (linalg as numx_linalg,
                           fftpack as numx_fft,
                           random as numx_rand,
                           version as numx_version)
        numx_description = 'scipy'
    except ImportError, exc:
        numx_exceptions['scipy'] = exc

if numx_description is None and (not _USR_LABEL or _USR_LABEL=='numpy'):
    try:
        import numpy_fail
        import numpy as numx
        from numpy import (linalg as numx_linalg,
                           fft as numx_fft,
                           random as numx_rand)
        from numpy.version import version as numx_version
        numx_description = 'numpy'
    except ImportError, exc:
        numx_exceptions['numpy'] = exc

if numx_description is None:
    # The test is for numx_description, not numx, because numx could
    # be imported sucessfuly, but e.g. numx_rand could later fail.
    msg = ("Could not import any of the numeric backends.\n"
           "Import errors:\n"
           + '\n'.join(label+': '+str(exc)
                       for label, exc in numx_exceptions.iteritems()))
    raise ImportError(msg)
else:
    # we have numx, we don't need the exceptions anymore
    del numx_exceptions

del _os, _USR_LABEL


from utils import get_git_revision

__version__ = '2.6'
__revision__ = get_git_revision()
__authors__ = 'Pietro Berkes, Rike-Benjamin Schuppner, Niko Wilbert, and Tiziano Zito'
__copyright__ = '(c) 2003-2010 Pietro Berkes, Rike-Benjamin Schuppner, Niko Wilbert, Tiziano Zito'
__license__ = 'Modified BSD License (GPL compatible), see COPYRIGHT'
__contact__ = 'mdp-toolkit-users@lists.sourceforge.net'


from utils import OrderedDict

import sys

class MDPConfiguration(object):
    """MDPConfiguration() does checks on the available libraries
    and auto-generates a list of features for inclusion in debug output.
    """
    # TODO: Needs more love with version checking.
    def __getitem__(self, key):

        if key in self._features:
            return self._features[key]
        raise KeyError("'%s'" % key)

    def add_feature(self, feature, is_available, doc=True):
        self._features[feature] = is_available
        if doc:
            self._doc.add(feature)

    def check_feature(self, feature, prop, doc=True):
        self._features[feature] = prop()
        if doc:
            self._doc.add(feature)

    def add_info(self, feature, val, doc=True):
        self._features[feature] = val
        if doc:
            self._doc.add(feature)

    def __init__(self):
        self._features = OrderedDict()
        self._doc = set()

        self.add_info('MDP Version', __version__)
        self.add_info('MDP Revision', __revision__)
        self.add_info('Python Version', '.'.join([str(x) for x in
                                                  sys.version_info]))
        
        self.add_feature("numpy", numx_description == 'numpy', doc=False)
        self.add_feature("scipy", numx_description == 'scipy', doc=False)

        self.check_feature("Numerical Backend", self.numerical_backend)
        self.check_feature("Parallel Python", self.has_parallel_python)
        self.check_feature("shogun", self.has_shogun)
        self.check_feature("LibSVM", self.has_libsvm)
        self.check_feature("Symeig Backend", self.symeig)

    def module_exists(self, module_name):
        """Returns True if a given module exists."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    def has(self, dep):
        """Checks if a dependency is available."""
        return self[dep]

    def has_parallel_python(self):
        try:
            import pp
        except ImportError, msg:
            return ExternalDepFail(msg)
        return ExternalDepFound(pp.version)

    def has_shogun(self):
        try:
            import shogun.Kernel as _sgKernel
            import shogun.Classifier as _sgClassifier
        except ImportError, msg:
            return ExternalDepFail(msg)
        # We need to have at least SHOGUN 0.9, as we rely on
        # SHOGUN's CClassifier::classify() method.
        # (It makes our code much nicer, by the way.)
        #
        if not hasattr(_sgClassifier.Classifier, 'classify'):
            return ExternalDepFail("CClassifier::classify not found")
        try:
            version = _sgKernel._Kernel.Version_get_version_release()
        except AttributeError:
            version = ""

        if not (version.startswith('v0.9') or version.startswith('v1.')):
            return ExternalDepFail("We need at least SHOGUN version 0.9.")
        return ExternalDepFound(version)

    def has_libsvm(self):
        try:
            import svm
        except ImportError, exc:
            return ExternalDepFail(exc)
        return ExternalDepFound(svm.libsvm._name)

    def has_symeig(self):
        return self._has_symeig

    def numerical_backend(self):
        return (numx_description, numx_version)

    def symeig(self):
        import utils
        self._has_symeig = False
        # check what symeig are we using
        if utils.symeig is utils.wrap_eigh:
            SYMEIG = 'scipy.linalg.eigh'
            self._has_symeig = True
        else:
            try:
                import symeig
                if utils.symeig is symeig.symeig:
                    SYMEIG = 'symeig'
                    self._has_symeig = True
                elif utils.symeig is utils._symeig_fake:
                    SYMEIG = 'symeig_fake'
                else:
                    SYMEIG = 'unknown'
            except ImportError:
                if utils.symeig is utils._symeig_fake:
                    SYMEIG = 'symeig_fake'
                else:
                    SYMEIG = 'unknown'
        return SYMEIG

    def info(self):
        """Return nicely formatted info about MDP."""
        listable_features = [f for f in self._features if f in self._doc]
        maxlen = max(len(f) for f in listable_features)
        l = []
        for feature in listable_features:
            s = "%*s: %s" % (maxlen+1, feature, self[feature])
            l.append(s)
        return "\n".join(l)

config = MDPConfiguration()

# import the utils module (used by other modules)
# here we set scipy_emulation if needed.
import utils

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
from helper_funcs import pca, whitening, fastica, sfa, get_eta

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
del sys

# explicitly set __all__, mainly needed for epydoc
__all__ = ['CheckpointFlow', 'CheckpointFunction', 'CheckpointSaveFunction',
           'CrashRecoveryException', 'Cumulator', 'VariadicCumulator', 'Flow',
           'FlowException',
           'FlowExceptionCR', 'IsNotInvertibleException',
           'IsNotTrainableException', 'MDPException', 'MDPWarning', 'Node',
           'NodeException', 'TrainingException', 'TrainingFinishedException',
           'ClassifierNode', 'ClassifierCumulator',
           'get_eta', 'graph', 'helper_funcs', 'hinet', 'nodes',
           'numx_description', 'pca', 'sfa', 'utils', 'whitening',
           'parallel', 'numx_version',
           'extension_method', 'ExtensionNodeMetaclass', 'ExtensionNode',
           'get_extensions', 'with_extension',
           'activate_extension', 'deactivate_extension', 'activate_extensions',
           'deactivate_extensions',
           'ClassifierNode',
           'config'
           ]

if config.module_exists('joblib'):
    import caching
    __all__ += ['caching']
