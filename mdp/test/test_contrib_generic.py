import mdp.contrib as mc
from _tools import *
from test_nodes_generic import (generic_test_factory,
                                test_dtype_consistency,
                                test_outputdim_consistency,
                                test_dimdtypeset,
                                test_inverse,
                                )

def contrib_get_random_mix():
    return get_random_mix(type='d', mat_dim=(100, 3))[2]

NODES = [dict(klass=mc.JADENode,
              inp_arg_gen=contrib_get_random_mix,
              ),
         dict(klass=mc.NIPALSNode,
              inp_arg_gen=contrib_get_random_mix,
              ),
         dict(klass=mc.XSFANode,
              inp_arg_gen=contrib_get_random_mix,
              init_args=[(mdp.nodes.PolynomialExpansionNode, (1,), {}),
                         (mdp.nodes.PolynomialExpansionNode, (1,), {}),
                         True]),
         dict(klass=mc.LLENode,
              inp_arg_gen=contrib_get_random_mix,
              init_args=[3, 0.001, True]),
         dict(klass=mc.HLLENode,
              inp_arg_gen=contrib_get_random_mix,
              init_args=[10, 0.001, True]),
         ]

def pytest_generate_tests(metafunc):
    generic_test_factory(NODES, metafunc)
