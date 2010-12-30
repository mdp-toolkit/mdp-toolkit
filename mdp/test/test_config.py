"""Test the configuration object."""

from mdp import config

def test_config_numpy_or_scipy():
    assert bool(config.has_scipy) != bool(config.has_numpy)

class TestConfig(object):
    def teardown_method(self, method):
        delattr(config, 'has_test_property')

    def test_config_depfound(self):
        dep = config.ExternalDep('test_property')
        dep.found(0.777)
        assert bool(dep) == True
        assert config.has_test_property
        info = config.info()
        assert 'test property' in info
        assert '0.777' in info

    def test_config_depfound_string(self):
        dep = config.ExternalDep('test_property')
        dep.found('0.777')
        assert bool(dep) == True
        assert config.has_test_property
        info = config.info()
        assert 'test property' in info
        assert '0.777' in info

    def test_config_depfailed_exc(self):
        dep = config.ExternalDep('test_property')
        dep.failed(ImportError('GOOGOO'))
        assert bool(dep) == False
        assert not config.has_test_property
        info = config.info()
        assert 'test property' in info
        assert 'GOOGOO' in info

    def test_config_depfailed_string(self):
        dep = config.ExternalDep('test_property')
        dep.failed('GOOGOO')
        assert bool(dep) == False
        assert not config.has_test_property
        info = config.info()
        assert 'test property' in info
        assert 'GOOGOO' in info
