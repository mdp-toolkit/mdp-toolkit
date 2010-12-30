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
from __future__ import with_statement
__docformat__ = "restructuredtext en"

class MDPException(Exception):
    """Base class for exceptions in MDP."""
    pass

class MDPWarning(UserWarning):
    """Base class for warnings in MDP."""
    pass

class MDPDeprecationWarning(DeprecationWarning, MDPWarning):
    """Warn about deprecated MDP API."""
    pass

class config(object):
    """Provide information about optional dependencies.

    This class should not be instantiated, it serves as a namespace
    for dependency information. This information is encoded as a
    series of attributes called 'has_<dependency>'.

    Dependency parameters are object which have a a boolean value
    (``True`` if the dependency is available). If False, they contain
    an error string which will be used in `mdp.config.info()`
    output. If ``True``, they contain information about the available
    version of the dependency. After the dependency object is created
    its value should be set by calling `ExternalDep.found` or
    `ExternalDep.failed`. Until one of those two methods is called,
    the object is presented as SKIPPED.

    Dependency parameters are numbered in the order of creation,
    so the output is predictable.

    >>> config.has_python
    True
    """

    _HAS_NUMBER = 0
    class ExternalDep(object):
        def __init__(self, name):
            """
            :Parameters:
              name
                identifier of the optional dependency. This should
                be a valid python identifier, because it will be
                accessible as ``mdp.config.has_<name>`` attribute.
            """

            self.version = None
            self.failmsg = None

            global config
            self.order = config._HAS_NUMBER
            config._HAS_NUMBER += 1
            setattr(config, 'has_' + name, self)

        def __nonzero__(self):
            return self.version is not None

        def __repr__(self):
            if self.version is not None:
                return self.version
            elif self.failmsg is not None:
                return "NOT AVAILABLE: " + str(self.failmsg)
            else:
                return "SKIPPED"

        def found(self, version):
            """Inform that an optional dependency was found.

            :Parameters:
              version
                an object convertible to ``str``, which will be displayed in
                ``mdp.config.info()`` output. Something like ``'0.4.3'``.
            """
            if version is None:
                raise ValueError('version cannot be None')
            if self.failmsg is not None:
                raise ValueError("failmsg was previously set to '%s'" % self.failmsg)
            self.version = str(version)

        def failed(self, failmsg):
            """Inform that an optional dependency was not found.

            The value of this `ExternalDep` object stored in `config`
            will be ``False``.

            :Parameters:
              failmsg
                an object convertible to ``str``, which will be displayed in
                ``mdp.config.info()`` output. This will usually be either an
                exception (e.g. ``ImportError``), or a message string.
            """
            if failmsg is None:
                raise ValueError('failmsg cannot be None')
            if self.version is not None:
                raise ValueError("version was previously set to '%s'" % self.version)
            self.failmsg = failmsg

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            if type is not None:
                self.failed(value)
                return True

    @classmethod
    def info(cls):
        """Return nicely formatted info about MDP.

        >>> print mdp.config.info()                           # doctest: +SKIP
                    python: 2.6.6.final.0
                     scipy: 0.7.2
                     numpy: NOT AVAILABLE: MDPNUMX set to scipy
              scipy signal: 0.7.2
           parallel python: 1.6.0
                    shogun: v0.9.3_r4889_2010-05-27_20:52_
                    libsvm: NOT AVAILABLE: No module named svm
                    symeig: wrap_eigh
                new symeig: numx_linalg.eigh
         standalone symeig: SKIPPED
                    joblib: 0.4.6

        This function is used to provide the py.test report header and
        footer.
        """
        listable_features = [(f[4:].replace('_', ' '), getattr(cls, f))
                             for f in dir(cls) if f.startswith('has_')]
        maxlen = max(len(f[0]) for f in listable_features)
        listable_features = sorted(listable_features, key=lambda f: f[1].order)
        return '\n'.join('%*s: %r' % (maxlen+1, f[0], f[1])
                         for f in listable_features)

import sys, os

config.ExternalDep('python').found('.'.join([str(x) for x in sys.version_info]))

# To force MDP to use one specific extension module
# set the environment variable MDPNUMX
# Mainly useful for testing
_USR_LABEL = os.getenv('MDPNUMX')
if _USR_LABEL and _USR_LABEL not in ('numpy', 'scipy'):
    err = """
Numerical backend '%s' not supported.
Supported backends: numpy, scipy.""" % _USR_LABEL
    raise ImportError(err)

numx_description = None

with config.ExternalDep('scipy') as dep:
    if _USR_LABEL and _USR_LABEL != 'scipy':
        dep.failed('MDPNUMX set to %s' % _USR_LABEL)
    else:
        import scipy as numx
        from scipy import (linalg as numx_linalg,
                           fftpack as numx_fft,
                           random as numx_rand,
                           version as numx_version)
        numx_description = 'scipy'
        dep.found(numx_version.version)

with config.ExternalDep('numpy') as dep:
    if _USR_LABEL and _USR_LABEL != 'numpy':
        dep.failed('MDPNUMX set to %s' % _USR_LABEL)
    elif numx_description is None:
        import numpy as numx
        from numpy import (linalg as numx_linalg,
                           fft as numx_fft,
                           random as numx_rand,
                           version as numx_version)
        numx_description = 'numpy'
        dep.found(numx_version.version)
    else:
        dep.failed('scipy is preferred')

assert bool(config.has_scipy) != bool(config.has_numpy)

if numx_description is None:
    # The test is for numx_description, not numx, because numx could
    # be imported sucessfuly, but e.g. numx_rand could later fail.
    msg = ("Could not import any of the numeric backends.\n"
           "Import errors:\n"
           "numpy:" + str(config.has_numpy.failmsg) + "\n"
           "scipy:" + str(config.has_scipy.failmsg) + "\n")
    raise ImportError(msg)

del _USR_LABEL

with config.ExternalDep('scipy_signal') as dep:
    if config.has_scipy:
        import scipy.signal
        dep.found(scipy.version.version)
    else:
        dep.failed('scipy not available')

# import the utils module (used by other modules)
# here we set scipy_emulation if needed.
import utils

__version__ = '2.6'
__revision__ = utils.get_git_revision()
__authors__ = 'Pietro Berkes, Rike-Benjamin Schuppner, Niko Wilbert, and Tiziano Zito'
__copyright__ = '(c) 2003-2010 Pietro Berkes, Rike-Benjamin Schuppner, Niko Wilbert, Tiziano Zito'
__license__ = 'Modified BSD License (GPL compatible), see COPYRIGHT'
__contact__ = 'mdp-toolkit-users@lists.sourceforge.net'

with config.ExternalDep('parallel_python') as dep:
    import pp
    dep.found(pp.version)

with config.ExternalDep('shogun') as dep:
    from shogun import (Kernel as sgKernel,
                        Features as sgFeatures,
                        Classifier as sgClassifier)
    # We need to have at least SHOGUN 0.9, as we rely on
    # SHOGUN's CClassifier::classify() method.
    # (It makes our code much nicer, by the way.)
    #
    if not hasattr(sgClassifier.Classifier, 'classify'):
        raise ValueError('CClassifier::classify not found')
    version = sgKernel._Kernel.Version_get_version_release()
    if not (version.startswith('v0.9') or version.startswith('v1.')):
        raise ValueError('We need at least SHOGUN version 0.9.')
    dep.found(version)

with config.ExternalDep('libsvm') as dep:
    import svm as libsvm
    dep.found(libsvm.libsvm._name)

import inspect as _inspect
with config.ExternalDep('symeig') as dep:
    with config.ExternalDep('new_symeig') as dep2:
        # check if scipy.linalg.eigh is the new version
        # if yes, just wrap it
        args = _inspect.getargspec(numx_linalg.eigh)[0]
        if len(args) > 4:
            dep.found('wrap_eigh')
            dep2.found('numx_linalg.eigh')
            from utils._symeig import (wrap_eigh as symeig,
                                       SymeigException)
        else:
            dep2.failed('numx_linalg.eigh version too old')

    with config.ExternalDep('standalone_symeig') as dep2:
        if not config.has_new_symeig:
            from symeig import symeig, SymeigException
            dep.found(symeig.__name__)
            dep2.found(symeig.__name__)

    del dep2

    if not config.has_new_symeig and not config.has_standalone_symeig:
        dep.failed('using fake symeig')
        from utils._symeig import (_symeig_fake as symeig,
                                   SymeigException)


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
del os, sys

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

with config.ExternalDep('joblib') as dep:
    import joblib
    __all__ += ['joblib']
    dep.found(joblib.__version__)
del dep

if config.has_joblib:
    import caching
    __all__ += ['caching']
