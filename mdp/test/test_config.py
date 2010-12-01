"""Test the configuration object."""

from mdp import config

def test_config_numpy_or_scipy():
    assert config.has_scipy() or config.has_numpy()
