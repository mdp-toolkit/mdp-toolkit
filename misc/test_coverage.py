"""Collect coverage information while running the MDP tests."""

import figleaf
figleaf.start()

try:
    import mdp
    mdp.test()
finally:
    figleaf.stop()
    figleaf.write_coverage('mdp_test.figleaf')
