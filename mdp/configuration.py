__docformat__ = "restructuredtext en"
import sys
import os
import inspect

class config(object):
    """Provide information about optional dependencies.

    This class should not be instantiated, it serves as a namespace
    for dependency information. This information is encoded as a
    series of attributes called 'has_<dependency>'.

    Dependency parameters are object which have a a boolean value
    (``True`` if the dependency is available). If False, they contain an
    error string which will be used in ``mdp.config.info()`` output. If
    ``True``, they contain information about the available version of
    the dependency. Those objects should be created by using the helper
    class methods `ExternalDepFound` and `ExternalDepFailed`.

    Dependency parameters are numbered in the order of creation,
    so the output is predictable::

    >>> config.has_python
    True

    The loading of a dependency can be inhibited by setting the
    environment variable MDP_DISABLE_DEPNAME.
    """

    _HAS_NUMBER = 0

    class _ExternalDep(object):
        def __init__(self, name, version=None, failmsg=None):
            assert (version is not None) + (failmsg is not None) == 1

            self.version = str(version) # convert e.g. exception to str
            self.failmsg = str(failmsg) if failmsg is not None else None

            global config
            self.order = config._HAS_NUMBER
            config._HAS_NUMBER += 1
            setattr(config, 'has_' + name, self)

        def __nonzero__(self):
            return self.failmsg is None

        def __repr__(self):
            if self:
                return self.version
            else:
                return "NOT AVAILABLE: " + self.failmsg

    @classmethod
    def ExternalDepFailed(cls, name, failmsg):
        """Inform that an optional dependency was not found.

        A new `_ExternalDep` object will be created and stored
        in `config`.

        :Parameters:
          name
            identifier of the optional dependency. This should
            be a valid python identifier, because it will be
            accessible as ``mdp.config.has_<name>`` attribute.
          failmsg
            an object convertible to ``str``, which will be displayed in
            ``mdp.config.info()`` output. This will usually be either an
            exception (e.g. ``ImportError``), or a message string.
        """
        return cls._ExternalDep(name, failmsg=failmsg)

    @classmethod
    def ExternalDepFound(cls, name, version):
        """Inform that an optional dependency was found.

        A new `_ExternalDep` object will be created and stored
        in `config`.

        :Parameters:
          name
            identifier of the optional dependency. This should
            be a valid python identifier, because it will be
            accessible as ``mdp.config.has_<name>`` attribute.
          version
            an object convertible to ``str``, which will be displayed in
            ``mdp.config.info()`` output. Something like ``'0.4.3'``.
        """
        return cls._ExternalDep(name, version=version)

    @classmethod
    def info(cls):
        """Return nicely formatted info about MDP.

        >>> print mdp.config.info()                           # doctest: +SKIP
                  python: 2.6.6.final.0
                   scipy: 0.7.2
            scipy signal: 0.7.2
         parallel python: 1.6.0
                  shogun: v0.9.3_r4889_2010-05-27_20:52_
                  libsvm: NOT AVAILABLE: No module named svm
              new symeig: NOT AVAILABLE: symeig version too old
                  symeig: wrap_eigh
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

def get_numx():
    # find out the numerical extension
    # To force MDP to use one specific extension module
    # set the environment variable MDPNUMX
    # Mainly useful for testing
    USR_LABEL = os.getenv('MDPNUMX')

    # check if the variable is properly set
    if USR_LABEL and USR_LABEL not in ('numpy', 'scipy'):
        err = ("Numerical backend '%s'" % USR_LABEL +
               "not supported. Supported backends: numpy, scipy.")
        raise ImportError(err)

    numx_description = None
    numx_exception = {}

    # if variable is not set or the user wants scipy
    if USR_LABEL is None or USR_LABEL == 'scipy':
        try:
            import scipy as numx
            from scipy import (linalg as numx_linalg,
                               fftpack as numx_fft,
                               random as numx_rand,
                               version as numx_version)
            numx_description = 'scipy'
            config.ExternalDepFound('numx', 'scipy ' + numx_version.version)
        except ImportError, exc:
            if USR_LABEL:
                raise ImportError(exc)
            else:
                numx_exception['scipy'] = exc

    # if the user wants numpy or scipy was not available
    if USR_LABEL == 'numpy' or numx_description is None:
        try:
            import numpy as numx
            from numpy import (linalg as numx_linalg,
                               fft as numx_fft,
                               random as numx_rand,
                               version as numx_version)
            numx_description = 'numpy'
            config.ExternalDepFound('numx', 'numpy ' + numx_version.version)
        except ImportError, exc:
            config.ExternalDepFailed('numx', exc)
            numx_exception['numpy'] = exc

    # fail if neither scipy nor numpy could be imported
    # the test is for numx_description, not numx, because numx could
    # be imported successfully, but e.g. numx_rand could later fail.
    if numx_description is None:
        msg = ("Could not import any of the numeric backends.\n"
               "Import errors:\n"
               + '\n'.join(label+': '+str(exc)
                           for label, exc in numx_exception.items()))
        raise ImportError(msg)

    return (numx_description, numx, numx_linalg,
            numx_fft, numx_rand, numx_version)

def get_symeig(numx_linalg):
    # if we have scipy, check if the version of
    # scipy.linalg.eigh supports the rich interface
    args = inspect.getargspec(numx_linalg.eigh)[0]
    if len(args) > 4:
        # if yes, just wrap it
        from utils._symeig import wrap_eigh as symeig
        config.ExternalDepFound('symeig', 'scipy.linalg.eigh')
    else:
        # either we have numpy, or we have an old scipy
        # we need to use our own rich wrapper
        from utils._symeig import _symeig_fake as symeig
        config.ExternalDepFound('symeig', 'symeig_fake')
    return symeig

def set_configuration():
    # set python version
    config.ExternalDepFound('python', '.'.join([str(x)
                                                for x in sys.version_info]))

    # parallel python dependency
    try:
        import pp
    except ImportError, exc:
        config.ExternalDepFailed('parallel_python', exc)
    else:
        if os.getenv('MDP_DISABLE_PARALLEL_PYTHON'):
            config.ExternalDepFailed('parallel_python', 'disabled')
        else:
            config.ExternalDepFound('parallel_python', pp.version)

    # shogun
    try:
        import shogun
        from shogun import (Kernel as sgKernel,
                            Features as sgFeatures,
                            Classifier as sgClassifier)
    except ImportError, exc:
        config.ExternalDepFailed('shogun', exc)
    else:
        # We need to have at least SHOGUN 0.9, as we rely on
        # SHOGUN's CClassifier::classify() method.
        # (It makes our code much nicer, by the way.)
        #
        if not hasattr(sgClassifier.Classifier, 'classify'):
            config.ExternalDepFailed('shogun', "CClassifier::classify not found")
        try:
            version = sgKernel._Kernel.Version_get_version_release()
        except AttributeError, msg:
            config.ExternalDepFailed('shogun', msg)
        else:
            if os.getenv('MDP_DISABLE_SHOGUN'):
                config.ExternalDepFailed('shogun', 'disabled')
            elif not (version.startswith('v0.9') or version.startswith('v1.')):
                config.ExternalDepFailed('shogun',
                                         'We need at least SHOGUN version 0.9.')
            else:
                config.ExternalDepFound('shogun', version)

    # libsvm
    try:
        import svm as libsvm
    except ImportError, exc:
        config.ExternalDepFailed('libsvm', exc)
    else:
        if os.getenv('MDP_DISABLE_LIBSVM'):
            config.ExternalDepFailed('libsvm', 'disabled')
        else:
            config.ExternalDepFound('libsvm', libsvm.libsvm._name)

    # joblib
    try:
        import joblib
    except ImportError, exc:
        config.ExternalDepFailed('joblib', exc)
    else:
        if os.getenv('MDP_DISABLE_JOBLIB'):
            config.ExternalDepFailed('joblib', 'disabled')
        elif joblib.__version__ < '0.4.3':
            config.ExternalDepFail('joblib',
                                   'version %s is too old' % joblib.__version__)
        else:
            config.ExternalDepFound('joblib', joblib.__version__)

    # scikits.learn
    try:
        import scikits.learn
    except ImportError, exc:
        config.ExternalDepFailed('scikits', exc)
    else:
        if os.getenv('MDP_DISABLE_SCIKITS_LEARN'):
            config.ExternalDepFailed('scikits', 'disabled')
        else:
            config.ExternalDepFound('scikits', scikits.learn.__version__)

