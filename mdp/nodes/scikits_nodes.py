# -*- coding:utf-8; -*-
"""Wraps the algorithms defined in scikits.learn in MDP Nodes.
This module is based on the 0.6.X branch of scikits.learn .
"""
from __future__ import print_function
__docformat__ = "restructuredtext en"

try:
    import sklearn
    _sklearn_prefix = 'sklearn'
except ImportError:
    import scikits.learn as sklearn
    _sklearn_prefix = 'scikits.learn'

import inspect
import re

import mdp

class ScikitsException(mdp.NodeException):
    """Base class for exceptions in nodes wrapping scikits algorithms."""
    pass

# import all submodules of sklearn (to work around lazy import)
from mdp.configuration import _version_too_old
if _version_too_old(sklearn.__version__, (0, 8)):
    scikits_modules = ['ann', 'cluster', 'covariance', 'feature_extraction',
                       'feature_selection', 'features', 'gaussian_process', 'glm',
                       'linear_model', 'preprocessing', 'svm',
                       'pca', 'lda', 'hmm', 'fastica', 'grid_search', 'mixture',
                       'naive_bayes', 'neighbors', 'qda']
elif _version_too_old(sklearn.__version__, (0, 9)):
    # package structure has been changed in 0.8
    scikits_modules = ['svm', 'linear_model', 'naive_bayes', 'neighbors',
                       'mixture', 'hmm', 'cluster', 'decomposition', 'lda',
                       'covariance', 'cross_val', 'grid_search',
                       'feature_selection.rfe', 'feature_extraction.image',
                       'feature_extraction.text', 'pipelines', 'pls',
                       'gaussian_process', 'qda']
elif _version_too_old(sklearn.__version__, (0, 11)):
    # from release 0.9 cross_val becomes cross_validation and hmm is deprecated
    scikits_modules = ['svm', 'linear_model', 'naive_bayes', 'neighbors',
                       'mixture', 'cluster', 'decomposition', 'lda',
                       'covariance', 'cross_validation', 'grid_search',
                       'feature_selection.rfe', 'feature_extraction.image',
                       'feature_extraction.text', 'pipelines', 'pls',
                       'gaussian_process', 'qda', 'ensemble', 'manifold',
                       'metrics', 'preprocessing', 'tree']    
elif _version_too_old(sklearn.__version__, (0, 17)):
    scikits_modules = ['svm', 'linear_model', 'naive_bayes', 'neighbors',
                       'mixture', 'cluster', 'decomposition', 'lda',
                       'covariance', 'cross_validation', 'grid_search',
                       'feature_selection', 'feature_extraction',
                       'pipeline', 'pls', 'gaussian_process', 'qda',
                       'ensemble', 'manifold', 'metrics', 'preprocessing',
                       'semi_supervised', 'tree', 'hmm']

else:
    scikits_modules = ['calibration', 'cluster', 'covariance', 'cross_decomposition',
                       'cross_validation', 'decomposition', 'discriminant_analysis',
                       'ensemble', 'feature_extraction', 'feature_selection',
                       'gaussian_process', 'grid_search', 'isotonic', 'kernel_approximation',
                       'kernel_ridge', 'learning_curve', 'linear_model', 'manifold',
                       'metrics', 'mixture', 'multiclass', 'naive_bayes', 'neighbors',
                       'neural_network', 'preprocessing', 'random_projection',
                       'semi_supervised', 'svm', 'tree']


for name in scikits_modules:
    # not all modules may be available due to missing dependencies
    # on the user system.
    # we just ignore failing imports
    try:
        __import__(_sklearn_prefix + '.' + name)
    except ImportError:
        pass


_WS_LINE_RE = re.compile(r'^\s*$')
_WS_PREFIX_RE = re.compile(r'^(\s*)')
_HEADINGS_RE = re.compile(r'''^(Parameters|Attributes|Methods|Examples|Notes)\n
                           (----+|====+)''', re.M + re.X)
_UNDERLINE_RE = re.compile(r'----+|====+')
_VARWITHUNDER_RE = re.compile(r'(\s|^)([a-zA-Z_][a-zA-Z0-9_]*_)(\s|$|[,.])')

_HEADINGS = set(['Parameters', 'Attributes', 'Methods', 'Examples',
                 'Notes', 'References'])

_DOC_TEMPLATE = """
%s
This node has been automatically generated by wrapping the ``%s.%s`` class
from the ``sklearn`` library.  The wrapped instance can be accessed
through the ``scikits_alg`` attribute.
%s
"""

def _gen_docstring(object, docsource=None):
    module = object.__module__
    name = object.__name__
    if docsource is None:
        docsource = object
    docstring = docsource.__doc__
    if docstring is None:
        return None

    lines = docstring.strip().split('\n')
    for i,line in enumerate(lines):
        if _WS_LINE_RE.match(line):
            break
    header = [line.strip() for line in lines[:i]]

    therest = [line.rstrip() for line in lines[i+1:]]
    body = []

    if therest:
        prefix = min(len(_WS_PREFIX_RE.match(line).group(1))
                     for line in therest if line)
        quoteind = None
        for i, line in enumerate(therest):
            line = line[prefix:]
            if line in _HEADINGS:
                body.append('**%s**' % line)
            elif _UNDERLINE_RE.match(line):
                body.append('')
            else:
                line = _VARWITHUNDER_RE.sub(r'\1``\2``\3', line)
                if quoteind:
                    if len(_WS_PREFIX_RE.match(line).group(1)) >= quoteind:
                        line = quoteind * ' ' + '- ' + line[quoteind:]
                    else:
                        quoteind = None
                        body.append('')
                body.append(line)

            if line.endswith(':'):
                body.append('')
                if i+1 < len(therest):
                    next = therest[i+1][prefix:]
                    quoteind = len(_WS_PREFIX_RE.match(next).group(1))

    return _DOC_TEMPLATE % ('\n'.join(header), module, name, '\n'.join(body))

# TODO: generalize dtype support
# TODO: have a look at predict_proba for Classifier.prob
# TODO: inverse <-> generate/rvs
# TODO: deal with input_dim/output_dim
# TODO: change signature of overwritten functions
# TODO: wrap_scikits_instance
# TODO: add sklearn availability to test info strings
# TODO: which tests ? (test that particular algorithm are / are not trainable)
# XXX: if class defines n_components, allow output_dim, otherwise throw exception
#      also for classifiers (overwrite _set_output_dim)
#      Problem: sometimes they call it 'k' (e.g., algorithms in sklearn.cluster)

def apply_to_scikits_algorithms(current_module, action,
                                processed_modules=None,
                                processed_classes=None):
    """Function that traverses a module to find scikits algorithms.
    'sklearn' algorithms are identified by the 'fit' 'predict',
    or 'transform' methods. The 'action' function is applied to each found
    algorithm.
    
    :param action: A function that is called with as action(class_), where
        'class_' is a class that defines the 'fit' or 'predict' method.
    :type action: function
    """

    # only consider modules and classes once
    if processed_modules is None:
        processed_modules = []
    if processed_classes is None:
        processed_classes = []

    if current_module in processed_modules:
        return
    processed_modules.append(current_module)

    for member_name, member in list(current_module.__dict__.items()):
        if not member_name.startswith('_'):

            # classes
            if (inspect.isclass(member) and
                member not in processed_classes):
                processed_classes.append(member)
                if (hasattr(member, 'fit')
                    or hasattr(member, 'predict')
                    or hasattr(member, 'transform')):
                    action(member)

            # other modules
            elif (inspect.ismodule(member) and
                  member.__name__.startswith(_sklearn_prefix)):
                apply_to_scikits_algorithms(member, action, processed_modules,
                                            processed_classes)
    return processed_classes


_OUTPUTDIM_ERROR = """'output_dim' keyword not supported.
Please set the output dimensionality using sklearn keyword
arguments (e.g., 'n_components', or 'k'). See the docstring of this
class for details."""

def wrap_scikits_classifier(scikits_class):
    """Wrap a sklearn classifier as an MDP Node subclass.
    The wrapper maps these MDP methods to their sklearn equivalents:

    - _stop_training -> fit
    - _label -> predict

    """

    newaxis = mdp.numx.newaxis

    # create a wrapper class for a sklearn classifier
    class ScikitsNode(mdp.ClassifierCumulator):

        def __init__(self, input_dim=None, output_dim=None, dtype=None,
                     **kwargs):
            """
            Initializes an object of type 'ScikitsNode'.

            :param input_dim: Dimensionality of the input.
                Default is None.
            :type input_dim: int
        
            :param output_dim: Dimensionality of the output.
                Default is None.
            :type output_dim: int
        
            :param dtype: Datatype of the input.
                Default is None.
            :type dtype: numpy.dtype or str
            """
            if output_dim is not None:
                # output_dim and n_components cannot be defined at the same time
                if 'n_components' in kwargs:
                    msg = ("Dimensionality set both by "
                           "output_dim=%d and n_components=%d""")
                    raise ScikitsException(msg % (output_dim,
                                                  kwargs['n_components']))


            super(ScikitsNode, self).__init__(input_dim=input_dim,
                                              output_dim=output_dim,
                                              dtype=dtype)
            self.scikits_alg = scikits_class(**kwargs)

        # ---- re-direct training and execution to the wrapped algorithm

        def _stop_training(self, **kwargs):
            super(ScikitsNode, self)._stop_training(self)
            return self.scikits_alg.fit(self.data, self.labels, **kwargs)

        def _label(self, x):
            return self.scikits_alg.predict(x)[:, newaxis]

        # ---- administrative details

        @staticmethod
        def is_invertible():
            return False

        @staticmethod
        def is_trainable():
            """Return True if the node can be trained, False otherwise.

            :return: A boolean indicating whether the node can be trained.
            :rtype: bool
            """
            return hasattr(scikits_class, 'fit')

        # NOTE: at this point scikits nodes can only support up to
        # 64-bits floats because some call numpy.linalg.svd, which for
        # some reason does not support higher precisions
        def _get_supported_dtypes(self):
            """Return the list of dtypes supported by this node.
            The types can be specified in any format allowed by numpy.dtype.

            :return: The list of dtypes supported by this node.
            :rtype: list
            """
            return ['float32', 'float64']

    # modify class name and docstring
    ScikitsNode.__name__ = scikits_class.__name__ + 'ScikitsLearnNode'
    ScikitsNode.__doc__ = _gen_docstring(scikits_class)

    # change the docstring of the methods to match the ones in sklearn

    # methods_dict maps ScikitsNode method names to sklearn method names
    methods_dict = {'__init__': '__init__',
                    'stop_training': 'fit',
                    'label': 'predict'}
    if hasattr(scikits_class, 'predict_proba'):
        methods_dict['prob'] = 'predict_proba'

    for mdp_name, scikits_name in list(methods_dict.items()):
        mdp_method = getattr(ScikitsNode, mdp_name)
        scikits_method = getattr(scikits_class, scikits_name)
        if hasattr(scikits_method, '__func__'):
            mdp_method.__func__.__doc__ = _gen_docstring(scikits_class,
                                                        scikits_method.__func__)

    if scikits_class.__init__.__doc__ is None:
        try:
            ScikitsNode.__init__.__func__.__doc__ = _gen_docstring(scikits_class)
        except AttributeError:
            # we are in Python3
            ScikitsNode.__init__.__doc__ = _gen_docstring(scikits_class)

    return ScikitsNode


def wrap_scikits_transformer(scikits_class):
    """Wrap a sklearn transformer as an MDP Node subclass.
    The wrapper maps these MDP methods to their sklearn equivalents:

    - _stop_training -> fit
    - _execute -> transform

    """

    # create a wrapper class for a sklearn transformer
    class ScikitsNode(mdp.Cumulator):

        def __init__(self, input_dim=None, output_dim=None, dtype=None, **kwargs):
            """
            Initializes an object of type 'ScikitsNode'.

            :param input_dim: Dimensionality of the input.
                Default is None.
            :type input_dim: int
        
            :param output_dim: Dimensionality of the output.
                Default is None.
            :type output_dim: int
        
            :param dtype: Datatype of the input.
                Default is None.
            :type dtype: numpy.dtype or str
            """
            if output_dim is not None:
                raise ScikitsException(_OUTPUTDIM_ERROR)
            super(ScikitsNode, self).__init__(input_dim=input_dim,
                                              output_dim=output_dim,
                                              dtype=dtype)
            self.scikits_alg = scikits_class(**kwargs)

        # ---- re-direct training and execution to the wrapped algorithm

        def _stop_training(self, **kwargs):
            super(ScikitsNode, self)._stop_training(self)
            return self.scikits_alg.fit(self.data, **kwargs)

        def _execute(self, x):
            return self.scikits_alg.transform(x)

        # ---- administrative details

        @staticmethod
        def is_invertible():
            return False

        @staticmethod
        def is_trainable():
            """Return True if the node can be trained, False otherwise.

            :return: A boolean indication whether the node can be trained.
            :rtype: bool
            """
            return hasattr(scikits_class, 'fit')

        # NOTE: at this point scikits nodes can only support up to
        # 64-bits floats because some call numpy.linalg.svd, which for
        # some reason does not support higher precisions
        def _get_supported_dtypes(self):
            """Return the list of dtypes supported by this node.
            The types can be specified in any format allowed by numpy.dtype.

            :return: The list of dtypes supported by this node.
            :rtype: list
            """
            return ['float32', 'float64']

    # modify class name and docstring
    ScikitsNode.__name__ = scikits_class.__name__ + 'ScikitsLearnNode'
    ScikitsNode.__doc__ = _gen_docstring(scikits_class)

    # change the docstring of the methods to match the ones in sklearn

    # methods_dict maps ScikitsNode method names to sklearn method names
    methods_dict = {'__init__': '__init__',
                    'stop_training': 'fit',
                    'execute': 'transform'}

    for mdp_name, scikits_name in list(methods_dict.items()):
        mdp_method = getattr(ScikitsNode, mdp_name)
        scikits_method = getattr(scikits_class, scikits_name, None)
        if hasattr(scikits_method, '__func__'):
            mdp_method.__func__.__doc__ = _gen_docstring(scikits_class,
                                                        scikits_method.__func__)

    if scikits_class.__init__.__doc__ is None:
        try:
            ScikitsNode.__init__.__func__.__doc__ = _gen_docstring(scikits_class)
        except AttributeError:
            # we are in Python3
            ScikitsNode.__init__.__doc__ = _gen_docstring(scikits_class)

    return ScikitsNode


def wrap_scikits_predictor(scikits_class):
    """Wrap a sklearn transformer as an MDP Node subclass.
    The wrapper maps these MDP methods to their sklearn equivalents:

    - _stop_training -> fit
    - _execute -> predict
    
    """

    # create a wrapper class for a sklearn predictor
    class ScikitsNode(mdp.Cumulator):

        def __init__(self, input_dim=None, output_dim=None, dtype=None, **kwargs):
            """
            Initializes an object of type 'ScikitsNode'.

            :param input_dim: Dimensionality of the input.
                Default is None.
            :type input_dim: int
        
            :param output_dim: Dimensionality of the output.
                Default is None.
            :type output_dim: int
        
            :param dtype: Datatype of the input.
                Default is None.
            :type dtype: numpy.dtype or str
            """
            if output_dim is not None:
                raise ScikitsException(_OUTPUTDIM_ERROR)
            super(ScikitsNode, self).__init__(input_dim=input_dim,
                                              output_dim=output_dim,
                                              dtype=dtype)
            self.scikits_alg = scikits_class(**kwargs)

        # ---- re-direct training and execution to the wrapped algorithm

        def _stop_training(self, **kwargs):
            super(ScikitsNode, self)._stop_training(self)
            return self.scikits_alg.fit(self.data, **kwargs)

        def _execute(self, x):
            return self.scikits_alg.predict(x)

        # ---- administrative details

        @staticmethod
        def is_invertible():
            return False

        @staticmethod
        def is_trainable():
            """Return True if the node can be trained, False otherwise.

            :return: A boolean indicating whether the node can be trained.
            :rtype: bool"""
            return hasattr(scikits_class, 'fit')

        # NOTE: at this point scikits nodes can only support up to 64-bits floats
        # because some call numpy.linalg.svd, which for some reason does not
        # support higher precisions
        def _get_supported_dtypes(self):
            """Return the list of dtypes supported by this node.
            The types can be specified in any format allowed by numpy.dtype."""
            return ['float32', 'float64']

    # modify class name and docstring
    ScikitsNode.__name__ = scikits_class.__name__ + 'ScikitsLearnNode'
    ScikitsNode.__doc__ = _gen_docstring(scikits_class)

    # change the docstring of the methods to match the ones in sklearn

    # methods_dict maps ScikitsNode method names to sklearn method names
    methods_dict = {'__init__': '__init__',
                    'stop_training': 'fit',
                    'execute': 'predict'}

    for mdp_name, scikits_name in list(methods_dict.items()):
        mdp_method = getattr(ScikitsNode, mdp_name)
        scikits_method = getattr(scikits_class, scikits_name)
        if hasattr(scikits_method, '__func__'):
            mdp_method.__func__.__doc__ = _gen_docstring(scikits_class,
                                                        scikits_method.__func__)

    if scikits_class.__init__.__doc__ is None:
        try:
            ScikitsNode.__init__.__func__.__doc__ = _gen_docstring(scikits_class)
        except AttributeError:
            # we are in Python3
            ScikitsNode.__init__.__doc__ = _gen_docstring(scikits_class)

    return ScikitsNode


#list candidate nodes
def print_public_members(class_):
    """Print methods of sklearn algorithm.
    """
    print('\n', '-' * 15)
    print('%s (%s)' % (class_.__name__, class_.__module__))
    for attr_name in dir(class_):
        attr = getattr(class_, attr_name)
        #print attr_name, type(attr)
        if not attr_name.startswith('_') and inspect.ismethod(attr):
            print(' -', attr_name)

#apply_to_scikits_algorithms(sklearn, print_public_members)


def wrap_scikits_algorithms(scikits_class, nodes_list):
    """NEED DOCSTRING."""

    name = scikits_class.__name__
    if (name[:4] == 'Base' or name == 'LinearModel'
        or name.startswith('EllipticEnvelop')
        or name.startswith('ForestClassifier')):
        return

    if issubclass(scikits_class, sklearn.base.ClassifierMixin) and \
        hasattr(scikits_class, 'fit'):
        nodes_list.append(wrap_scikits_classifier(scikits_class))
    # Some (abstract) transformers do not implement fit.
    elif hasattr(scikits_class, 'transform') and hasattr(scikits_class, 'fit'):
        nodes_list.append(wrap_scikits_transformer(scikits_class))
    elif hasattr(scikits_class, 'predict') and hasattr(scikits_class, 'fit'):
        nodes_list.append(wrap_scikits_predictor(scikits_class))

scikits_nodes = []
apply_to_scikits_algorithms(sklearn,
                            lambda c: wrap_scikits_algorithms(c, scikits_nodes))

# add scikit nodes to dictionary
#scikits_module = new.module('scikits')
DICT_ = {}
for wrapped_c in scikits_nodes:
    #print wrapped_c.__name__
    DICT_[wrapped_c.__name__] = wrapped_c