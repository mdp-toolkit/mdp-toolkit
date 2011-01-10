from mdp.test.test_namespace_fixups import (generate_calls,
                                            test_exports)

MODULES = ['bimdp',
           'bimdp.nodes',
           'bimdp.hinet',
           'bimdp.parallel',
           ]

def pytest_generate_tests(metafunc):
    generate_calls(MODULES, metafunc)

