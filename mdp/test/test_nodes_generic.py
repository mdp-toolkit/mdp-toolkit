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

# ./test_example.py
def pytest_generate_tests(metafunc):
    _generic_test_factory(NODES, metafunc)

def _generic_test_factory(big_nodes, metafunc):
    """Generator creating a test for each of the nodes
    based upon arguments passwd in the tuple.
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
        theid = metafunc.function.id_format.format(**funcargs)
        metafunc.addcall(funcargs, id=theid)

def id_format(format):
    def f(func):
        func.id_format = format
        return func
    return f

@id_format("{klass.__name__}")
def test_dtype_consistency(klass, init_args, inp_arg_gen,
                           sup_arg_gen, execute_arg_gen):
    #init_args = self._set_node_args(init_args)
    supported_types = klass(*init_args).get_supported_dtypes()
    for dtype in supported_types:
        inp = inp_arg_gen()
        node = klass(*init_args, dtype=dtype)
        _train_if_necessary(inp, node, sup_arg_gen)

        extra = [execute_arg_gen(inp)] if execute_arg_gen else []
        out = node.execute(inp, *extra)
        assert out.dtype == dtype


@id_format("{klass.__name__}")
def test_outputdim_consistency(klass, init_args, inp_arg_gen,
                               sup_arg_gen, execute_arg_gen):
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
        node = klass(*init_args, output_dim=output_dim)
        _test(node)

        # case 2: output_dim set explicitly
        node = klass(*init_args)
        node.output_dim = output_dim
        _test(node)
    else:
        node = klass(*init_args)
        _train_if_necessary(inp, node, sup_arg_gen)
        out = node.execute(inp, *extra)
        assert out.shape[1] == node.output_dim


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
    #(nodes.RBFExpansionNode, [[[0.]*5, [0.]*5], [1., 1.]], None),
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
