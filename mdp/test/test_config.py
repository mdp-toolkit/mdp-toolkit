"""Test the configuration object."""

import mdp

def test_config_module_exists():
    assert mdp.config.module_exists('numpy')
    assert not mdp.config.module_exists('__this_module_does_not_exists')
