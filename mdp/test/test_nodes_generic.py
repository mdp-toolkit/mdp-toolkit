import py.test
import inspect
from mdp import nodes, numx, numx_rand
from _tools import *
uniform = numx_rand.random

def _rand_labels(x):
    return numx.around(uniform(x.shape[0]))

def _rand_labels_array(x):
    return numx.around(uniform(x.shape[0])).reshape((x.shape[0],1))

def _rand_array_halfdim(x):
    return uniform(size=(x.shape[0], x.shape[1]//2))

def _train_if_necessary(inp, node, sup_arg_gen):
    if node.is_trainable():
        while True:
            if sup_arg_gen is not None:
                # for nodes that need supervision
                node.train(inp, sup_arg_gen(inp))
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
        node.execute(inp)

def pytest_generate_tests(metafunc):
    _generic_test_factory(NODES, metafunc)

def _generic_test_factory(big_nodes, metafunc):
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
      execution.

    `sup_arg_gen=None`
      A function taking a single argument (`inp`)
      Used to contruct extra arguments passed to `train`.

    `execute_arg_gen=None`
      A function similar to `sup_arg_gen` but used during execution.
      The return value is unpacked and used as additional arguments to
      `execute`.
    """
    for nodetype in big_nodes:
        if not isinstance(nodetype, dict):
            nodetype = dict(klass=nodetype)
        funcargs = dict(
            init_args=(),
            inp_arg_gen=lambda: get_random_mix(type='d')[2],
            sup_arg_gen=None,
            execute_arg_gen=None)
        funcargs.update(nodetype)

        if hasattr(metafunc.function, 'only_if_node_condition'):
            # A TypeError can be thrown by the condition checking
            # function (e.g. when nodetype.is_trainable() is not a staticmethod).
            condition = metafunc.function.only_if_node_condition
            try:
                if not condition(nodetype['klass']):
                    continue
            except TypeError:
                continue

        theid = metafunc.function.id_format.format(**funcargs)
        metafunc.addcall(funcargs, id=theid)

def id_format(format):
    """Specify the name of the test"""
    def f(func):
        func.id_format = format
        return func
    return f

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

@id_format("{klass.__name__}")
def test_dtype_consistency(klass, init_args, inp_arg_gen,
                           sup_arg_gen, execute_arg_gen):
    args = call_init_args(init_args)
    supported_types = klass(*args).get_supported_dtypes()
    for dtype in supported_types:
        inp = inp_arg_gen()
        args = call_init_args(init_args)
        node = klass(*args, dtype=dtype)
        _train_if_necessary(inp, node, sup_arg_gen)

        extra = [execute_arg_gen(inp)] if execute_arg_gen else []
        out = node.execute(inp, *extra)
        assert out.dtype == dtype


@id_format("{klass.__name__}")
def test_outputdim_consistency(klass, init_args, inp_arg_gen,
                               sup_arg_gen, execute_arg_gen):
    args = call_init_args(init_args)
    inp = inp_arg_gen()
    output_dim = inp.shape[1] // 2
    extra = [execute_arg_gen(inp)] if execute_arg_gen else []

    def _test(node):
        _train_if_necessary(inp, node, sup_arg_gen)
        out = node.execute(inp, *extra)
        assert out.shape[1] == output_dim
        assert node._output_dim == output_dim

    if 'output_dim' in inspect.getargspec(klass.__init__)[0]:
        # case 1: output dim set in the constructor
        node = klass(*args, output_dim=output_dim)
        _test(node)

        # case 2: output_dim set explicitly
        node = klass(*args)
        node.output_dim = output_dim
        _test(node)
    else:
        node = klass(*args)
        _train_if_necessary(inp, node, sup_arg_gen)
        out = node.execute(inp, *extra)
        assert out.shape[1] == node.output_dim

@id_format("{klass.__name__}")
def test_dimdtypeset(klass, init_args, inp_arg_gen,
                     sup_arg_gen, execute_arg_gen):
    init_args = call_init_args(init_args)
    inp = inp_arg_gen()
    node = klass(*init_args)
    _train_if_necessary(inp, node, sup_arg_gen)
    _stop_training_or_execute(node, inp)
    assert node.output_dim is not None
    assert node.dtype is not None
    assert node.input_dim is not None

@id_format("{klass.__name__}")
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
    assert_array_almost_equal_diff(rec, inp, decimal-3)
    assert rec.dtype == dtype

def SFA2Node_inp_arg_gen():
    freqs = [2*numx.pi*100.,2*numx.pi*200.]
    t =  numx.linspace(0, 1, num=1000)
    mat = numx.array([numx.sin(freqs[0]*t),
                      numx.sin(freqs[1]*t)]).T
    inp = mat.astype('d')
    return inp

def LinearRegressionNode_inp_arg_gen():
    return uniform(size=(1000, 5))

NODES = [
    nodes.PCANode,
    nodes.WhiteningNode,
    nodes.SFANode,
    dict(klass=nodes.SFA2Node,
         inp_arg_gen=SFA2Node_inp_arg_gen),
    nodes.TDSEPNode,
    nodes.CuBICANode,
    nodes.FastICANode,
    nodes.QuadraticExpansionNode,
    dict(klass=nodes.PolynomialExpansionNode,
         init_args=[3]),
    dict(klass=nodes.RBFExpansionNode,
         init_args=[[[0.]*5, [0.]*5], [1., 1.]]),
    nodes.GrowingNeuralGasExpansionNode,
    dict(klass=nodes.HitParadeNode,
         init_args=[2, 5]),
    dict(klass=nodes.TimeFramesNode,
         init_args=[3, 4]),
    nodes.EtaComputerNode,
    nodes.GrowingNeuralGasNode,
    nodes.NoiseNode,
    dict(klass=nodes.FDANode,
         sup_arg_gen=_rand_labels),
    dict(klass=nodes.GaussianClassifierNode,
         sup_arg_gen=_rand_labels),
    nodes.FANode,
    nodes.ISFANode,
    dict(klass=nodes.RBMNode,
         init_args=[5]),
    dict(klass=nodes.RBMWithLabelsNode,
         init_args=[5, 1],
         sup_arg_gen=_rand_labels_array,
         execute_arg_gen=_rand_labels_array),
    dict(klass=nodes.LinearRegressionNode,
         sup_arg_gen=_rand_array_halfdim),
    ]
