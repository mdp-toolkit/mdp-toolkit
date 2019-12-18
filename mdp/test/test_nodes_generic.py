from builtins import range
from builtins import object
import pytest
# python 2/3 compatibility
try:
    from inspect import getfullargspec as getargs
except ImportError:
    from inspect import getargspec as getargs
import inspect
from mdp import (config, nodes, ClassifierNode,
                 PreserveDimNode, InconsistentDimException)
from ._tools import *

uniform = numx_rand.random


def _rand_labels(x):
    return numx_rand.randint(0, 2, size=(x.shape[0],))


def _rand_labels_array(x):
    return numx_rand.randint(0, 2, size=(x.shape[0], 1))


def _rand_classification_labels_array(x):
    labels = numx_rand.randint(0, 2, size=(x.shape[0],))
    labels[labels == 0] = -1
    return labels


def _dumb_quadratic_expansion(x):
    dim_x = x.shape[1]
    return numx.asarray([(x[i].reshape(dim_x, 1) *
                          x[i].reshape(1, dim_x)).flatten()
                         for i in range(len(x))])


def _rand_array_halfdim(x):
    return uniform(size=(x.shape[0], x.shape[1]//2))


class Iter(object):
    pass


def _rand_array_single_rows():
    x = uniform((500, 4))

    class _Iter(Iter):
        def __iter__(self):
            for row in range(x.shape[0]):
                yield x[numx.newaxis, row, :]
    return _Iter()


def _contrib_get_random_mix():
    return get_random_mix(type='d', mat_dim=(100, 3))[2]


def _positive_get_random_mix():
    return abs(get_random_mix()[2])


def _train_if_necessary(inp, node, sup_arg_gen):
    if node.is_trainable():
        while True:
            if sup_arg_gen is not None:
                # for nodes that need supervision
                node.train(inp, sup_arg_gen(inp))
            else:
                # support generators
                if isinstance(inp, Iter):
                    for x in inp:
                        node.train(x)
                else:
                    node.train(inp)
            if node.get_remaining_train_phase() > 1:
                node.stop_training()
            else:
                break


def _stop_training_or_execute(node, inp):
    if node.is_trainable():
        node.stop_training()
    else:
        if isinstance(inp, Iter):
            for x in inp:
                node.execute(x)
        else:
            node.execute(inp)


def pytest_generate_tests(metafunc):
    generic_test_factory(NODES, metafunc)


def generic_test_factory(big_nodes, metafunc):
    """Generator creating a test for each of the nodes
    based upon arguments in a list of nodes in big_nodes.

    Format of big_nodes:
    each item in the list can be either a
    - class name, in this case the class instances are initialized
      without arguments and default arguments are used during
      the training and execution phases.
    - dict containing items which can override the initialization
      arguments, provide extra arguments for training and/or
      execution.

    Available keys in the configuration dict:
    `klass`
      Mandatory.
      The type of Node.

    `init_args=()`
      A sequence used to provide the initialization arguments to node
      constructor. Before being used, the items in this sequence are
      executed if they are callable. This allows one to create fresh
      instances of nodes before each Node initalization.

    `inp_arg_gen=...a call to get_random_mix('d')`
      Used to construct the `inp` data argument used for training and
      execution. It can be an iterable.

    `sup_arg_gen=None`
      A function taking a single argument (`inp`)
      Used to contruct extra arguments passed to `train`.

    `execute_arg_gen=None`
      A function similar to `sup_arg_gen` but used during execution.
      The return value is unpacked and used as additional arguments to
      `execute`.
    """
    ids = []
    argvalues = []
    for nodetype in big_nodes:

        if not isinstance(nodetype, dict):
            nodetype = dict(klass=nodetype)

        if hasattr(metafunc.function, 'only_if_node_condition'):
            # A TypeError can be thrown by the condition checking
            # function (e.g. when nodetype.is_trainable() is not a staticmethod).
            condition = metafunc.function.only_if_node_condition
            try:
                if not condition(nodetype['klass']):
                    continue
            except TypeError:
                continue

        argv = [(), lambda: get_random_mix(type='d')
                [2], None, None, None]

        # add whatever is in nodetype to the argumentlist
        for key in nodetype:
            if key == 'init_args':
                argv[0] = nodetype[key]
            elif key == 'inp_arg_gen':
                argv[1] = nodetype[key]
            elif key == 'sup_arg_gen':
                argv[2] = nodetype[key]
            elif key == 'execute_arg_gen':
                argv[3] = nodetype[key]
            elif key == 'klass':
                argv[4] = nodetype[key]

        # make list with different inputs
        argvalues.append(tuple(argv))
        ids.append(nodetype['klass'].__name__)

    # argnames are the same
    argnames = 'init_args,inp_arg_gen,sup_arg_gen,execute_arg_gen,klass'
    metafunc.parametrize(argnames=argnames, argvalues=argvalues, ids=ids)


def only_if_node(condition):
    """Execute the test only if condition(nodetype) is True.

    If condition(nodetype) throws TypeError, just assume False.
    """
    def f(func):
        func.only_if_node_condition = condition
        return func
    return f


def call_init_args(init_args):
    return [item() if hasattr(item, '__call__') else item
            for item in init_args]


def test_dtype_consistency(klass, init_args, inp_arg_gen,
                           sup_arg_gen, execute_arg_gen):
    args = call_init_args(init_args)
    supported_types = klass(*args).get_supported_dtypes()
    for dtype in supported_types:
        inp = inp_arg_gen()
        # See https://github.com/mdp-toolkit/mdp-toolkit/pull/47
        # and https://github.com/mdp-toolkit/mdp-toolkit/issues/62.
        if klass.__name__ in ABSINPUT_NODES:
            inp = numx.absolute(inp)
        args = call_init_args(init_args)
        node = klass(dtype=dtype, *args)
        _train_if_necessary(inp, node, sup_arg_gen)

        extra = [execute_arg_gen(inp)] if execute_arg_gen else []
        # support generators
        if isinstance(inp, Iter):
            for x in inp:
                out = node.execute(x, *extra)
        else:
            out = node.execute(inp, *extra)
        assert out.dtype == dtype


def test_outputdim_consistency(klass, init_args, inp_arg_gen,
                               sup_arg_gen, execute_arg_gen):
    args = call_init_args(init_args)
    inp = inp_arg_gen()
    # See https://github.com/mdp-toolkit/mdp-toolkit/pull/47
    # and https://github.com/mdp-toolkit/mdp-toolkit/issues/62.
    if klass.__name__ in ABSINPUT_NODES:
        inp = numx.absolute(inp)
    # support generators
    if isinstance(inp, Iter):
        for x in inp:
            pass
        output_dim = x.shape[1] // 2
    else:
        output_dim = inp.shape[1] // 2
    extra = [execute_arg_gen(inp)] if execute_arg_gen else []

    def _test(node):
        _train_if_necessary(inp, node, sup_arg_gen)
        # support generators
        if isinstance(inp, Iter):
            for x in inp:
                out = node.execute(x)
        else:
            out = node.execute(inp, *extra)
        assert out.shape[1] == output_dim
        assert node._output_dim == output_dim

    # check if the node output dimension can be set or must be determined
    # by the node
    if (not issubclass(klass, PreserveDimNode) and
            'output_dim' in getargs(klass.__init__)[0]):
        # case 1: output dim set in the constructor
        node = klass(output_dim=output_dim, *args)
        _test(node)

        # case 2: output_dim set explicitly
        node = klass(*args)
        node.output_dim = output_dim
        _test(node)
    else:
        if issubclass(klass, PreserveDimNode):
            # check that constructor allows to set output_dim
            assert 'output_dim' in getargs(klass.__init__)[0]
            # check that setting the input dim, then incompatible output dims
            # raises an appropriate error
            # case 1: both in the constructor
            pytest.raises(InconsistentDimException,
                          'klass(input_dim=inp.shape[1], output_dim=output_dim, *args)')
            # case 2: first input_dim, then output_dim
            node = klass(input_dim=inp.shape[1], *args)
            pytest.raises(InconsistentDimException,
                          'node.output_dim = output_dim')
            # case 3: first output_dim, then input_dim
            node = klass(output_dim=output_dim, *args)
            node.output_dim = output_dim
            pytest.raises(InconsistentDimException,
                          'node.input_dim = inp.shape[1]')

        # check that output_dim is set to whatever the output dim is
        node = klass(*args)
        _train_if_necessary(inp, node, sup_arg_gen)
        # support generators
        if isinstance(inp, Iter):
            for x in inp:
                out = node.execute(x, *extra)
        else:
            out = node.execute(inp, *extra)

        assert out.shape[1] == node.output_dim


def test_dimdtypeset(klass, init_args, inp_arg_gen,
                     sup_arg_gen, execute_arg_gen):
    init_args = call_init_args(init_args)
    inp = inp_arg_gen()
    # See https://github.com/mdp-toolkit/mdp-toolkit/pull/47
    # and https://github.com/mdp-toolkit/mdp-toolkit/issues/62.
    if klass.__name__ in ABSINPUT_NODES:
        inp = numx.absolute(inp)
    node = klass(*init_args)
    _train_if_necessary(inp, node, sup_arg_gen)
    _stop_training_or_execute(node, inp)
    assert node.output_dim is not None
    assert node.dtype is not None
    assert node.input_dim is not None


@only_if_node(lambda nodetype: nodetype.is_invertible())
def test_inverse(klass, init_args, inp_arg_gen,
                 sup_arg_gen, execute_arg_gen):
    args = call_init_args(init_args)
    inp = inp_arg_gen()
    # take the first available dtype for the test
    dtype = klass(*args).get_supported_dtypes()[0]
    args = call_init_args(init_args)
    node = klass(dtype=dtype, *args)
    _train_if_necessary(inp, node, sup_arg_gen)
    extra = [execute_arg_gen(inp)] if execute_arg_gen else []
    out = node.execute(inp, *extra)
    # compute the inverse
    rec = node.inverse(out)
    # cast inp for comparison!
    inp = inp.astype(dtype)
    assert_array_almost_equal_diff(rec, inp, decimal-4)
    assert rec.dtype == dtype


def SFA2Node_inp_arg_gen():
    freqs = [2*numx.pi*100., 2*numx.pi*200.]
    t = numx.linspace(0, 1, num=1000)
    mat = numx.array([numx.sin(freqs[0]*t),
                      numx.sin(freqs[1]*t)]).T
    inp = mat.astype('d')
    return inp


def NeuralGasNode_inp_arg_gen():
    return numx.asarray([[2., 0, 0], [-2, 0, 0], [0, 0, 0]])


def LinearRegressionNode_inp_arg_gen():
    return uniform(size=(1000, 5))


def iGSFANode_inp_arg_gen():
    return uniform(size=(1000, 4)) * 0.01


def _rand_1d(x):
    return uniform(size=(x.shape[0],))


def CCIPCANode_inp_arg_gen():
    line_x = numx.zeros((1000, 2), "d")
    line_y = numx.zeros((1000, 2), "d")
    line_x[:, 0] = numx.linspace(-1, 1, num=1000, endpoint=1)
    line_y[:, 1] = numx.linspace(-0.2, 0.2, num=1000, endpoint=1)
    mat = numx.concatenate((line_x, line_y))
    utils.rotate(mat, uniform() * 2 * numx.pi)
    mat += uniform(2)
    mat -= mat.mean(axis=0)
    return mat


NODES = [
    dict(klass='NeuralGasNode',
         init_args=[3, NeuralGasNode_inp_arg_gen()],
         inp_arg_gen=NeuralGasNode_inp_arg_gen),
    dict(klass='SFA2Node',
         inp_arg_gen=SFA2Node_inp_arg_gen),
    dict(klass='PolynomialExpansionNode',
         init_args=[3]),
    dict(klass='RBFExpansionNode',
         init_args=[[[0.]*5, [0.]*5], [1., 1.]]),
    dict(klass='GeneralExpansionNode',
         init_args=[[lambda x:x, lambda x: x**2, _dumb_quadratic_expansion]]),
    dict(klass='HitParadeNode',
         init_args=[2, 5]),
    dict(klass='TimeFramesNode',
         init_args=[3, 4]),
    dict(klass='TimeDelayNode',
         init_args=[3, 4]),
    dict(klass='TimeDelaySlidingWindowNode',
         init_args=[3, 4],
         inp_arg_gen=_rand_array_single_rows),
    dict(klass='FDANode',
         sup_arg_gen=_rand_labels),
    dict(klass='GaussianClassifier',
         sup_arg_gen=_rand_labels),
    dict(klass='NearestMeanClassifier',
         sup_arg_gen=_rand_labels),
    dict(klass='KNNClassifier',
         sup_arg_gen=_rand_labels),
    dict(klass='RBMNode',
         init_args=[5]),
    dict(klass='RBMWithLabelsNode',
         init_args=[5, 1],
         sup_arg_gen=_rand_labels_array,
         execute_arg_gen=_rand_labels_array),
    dict(klass='LinearRegressionNode',
         sup_arg_gen=_rand_array_halfdim),
    dict(klass='Convolution2DNode',
         init_args=[mdp.numx.array([[[1.]]]), (5, 1)]),
    dict(klass='JADENode',
         inp_arg_gen=_contrib_get_random_mix),
    dict(klass='NIPALSNode',
         inp_arg_gen=_contrib_get_random_mix),
    dict(klass='XSFANode',
         inp_arg_gen=_contrib_get_random_mix,
         init_args=[(nodes.PolynomialExpansionNode, (1,), {}),
                    (nodes.PolynomialExpansionNode, (1,), {}),
                    True]),
    dict(klass='iGSFANode',
         init_args=[None, None, None, None, None,
                    None, None, 0.5, False, False],
         inp_arg_gen=iGSFANode_inp_arg_gen),
    dict(klass='LLENode',
         inp_arg_gen=_contrib_get_random_mix,
         init_args=[3, 0.001, True]),
    dict(klass='HLLENode',
         inp_arg_gen=_contrib_get_random_mix,
         init_args=[10, 0.001, True]),
    dict(klass='KMeansClassifier',
         init_args=[2, 3]),
    dict(klass='CCIPCANode',
         inp_arg_gen=CCIPCANode_inp_arg_gen),
    dict(klass='CCIPCAWhiteningNode',
         inp_arg_gen=CCIPCANode_inp_arg_gen),
    dict(klass='MCANode',
         inp_arg_gen=CCIPCANode_inp_arg_gen),
    dict(klass='IncSFANode',
         inp_arg_gen=CCIPCANode_inp_arg_gen),
    dict(klass='PerceptronClassifier',
         sup_arg_gen=_rand_classification_labels_array),
    dict(klass='SimpleMarkovClassifier',
         sup_arg_gen=_rand_classification_labels_array),
    dict(klass='ShogunSVMClassifier',
         sup_arg_gen=_rand_labels_array,
         init_args=["libsvmmulticlass", (), None, "GaussianKernel"]),
    dict(klass='LibSVMClassifier',
         sup_arg_gen=_rand_labels_array,
         init_args=["LINEAR", "C_SVC"]),
    dict(klass='MultinomialNBScikitsLearnNode',
         inp_arg_gen=_positive_get_random_mix,
         sup_arg_gen=_rand_labels),
    dict(klass='NeighborsScikitsLearnNode',
         sup_arg_gen=_rand_1d),
]

# LabelSpreadingScikitsLearnNode is broken in sklearn version 0.11
# It works fine in version 0.12
EXCLUDE_NODES = ('ICANode', 'LabelSpreadingScikitsLearnNode',
                 'OutputCodeClassifierScikitsLearnNode',
                 'OneVsOneClassifierScikitsLearnNode',
                 'OneVsRestClassifierScikitsLearnNode',
                 'VotingClassifierScikitsLearnNode',
                 'StackingClassifierScikitsLearnNode')

# The following Nodes require their input to be made positive.
# We do this using inp = numx.absolute(inp) for these nodes.
# See https://github.com/mdp-toolkit/mdp-toolkit/pull/47
# and https://github.com/mdp-toolkit/mdp-toolkit/issues/62.
ABSINPUT_NODES = ('ComplementNBScikitsLearnNode',
                  'CategoricalNBScikitsLearnNode')


def generate_nodes_list(nodes_dicts):
    nodes_list = []
    # append nodes with additional arguments or supervised if they exist
    visited = []
    excluded = []
    for dct in nodes_dicts:
        klass = dct['klass']
        if type(klass) is __builtins__['str']:
            # some of the nodes on the list may be optional
            if not hasattr(nodes, klass):
                continue
            # transform class name into class (needed by automatic tests)
            klass = getattr(nodes, klass)
            dct['klass'] = klass
        # only append to list if the node is present in MDP
        # in case some of the nodes in NODES are optional
        if hasattr(nodes, klass.__name__):
            nodes_list.append(dct)
            visited.append(klass)
    for node_name in EXCLUDE_NODES:
        if hasattr(nodes, node_name):
            excluded.append(getattr(nodes, node_name))
    # append sklearn nodes if supported
    # XXX
    # remove all non classifier nodes from the scikits nodes
    # they do not have a common API that would allow
    # automatic testing
    # XXX
    for node_name in mdp.nodes.__dict__:
        node = mdp.nodes.__dict__[node_name]
        if (inspect.isclass(node)
            and node_name.endswith('ScikitsLearnNode')
            and (node not in visited)
                and (node not in excluded)):
            if issubclass(node, ClassifierNode):
                nodes_list.append(dict(klass=node,
                                       sup_arg_gen=_rand_labels))
                visited.append(node)
            else:
                excluded.append(node)

    # append all other nodes in mdp.nodes
    for attr in dir(nodes):
        if attr[0] == '_':
            continue
        attr = getattr(nodes, attr)
        if (inspect.isclass(attr)
            and issubclass(attr, mdp.Node)
            and attr not in visited
                and attr not in excluded):
            nodes_list.append(attr)
    return nodes_list


NODES = generate_nodes_list(NODES)
