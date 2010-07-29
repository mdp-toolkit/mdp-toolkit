from mdp.test.test_namespace_fixups import (generate_calls,
                                            test_dunder_module_dunder)

MODULES = ['bimdp',
           'bimdp.nodes',
           'bimdp.hinet',
           'bimdp.parallel',
           ]

def pytest_generate_tests(metafunc):
    generate_calls(MODULES, metafunc)

