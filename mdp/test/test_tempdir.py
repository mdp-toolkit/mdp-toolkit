import tempfile
import os
import pytest

def test_tmpdir_exists():
    assert os.path.exists(pytest.mdp_tempdirname)

def test_tmpdir_writable1():
    with open(os.path.join(pytest.mdp_tempdirname, 'empty'), 'w'):
        pass

def test_tmpdir_writable2():
    with tempfile.NamedTemporaryFile(prefix='MDP_', suffix='.testfile',
                                     dir=pytest.mdp_tempdirname):
        pass
