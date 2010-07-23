"""Test caching extension."""
import mdp
from mdp import caching
import _tools

requires_joblib = _tools.skip_on_condition(
    "not mdp.config.module_exists('joblib')",
    "This test requires the 'joblib' module.")


_counter = 0
class _CounterNode(mdp.Node):
    def __init__(self):
        super(_CounterNode, self).__init__()

    def is_trainable(self):
        return False

    def _execute(self, x):
        """The execute method has the side effect of increasing
        a global counter by one."""
        global _counter
        _counter += 1
        return x

@requires_joblib
def test_caching_extension():
    """Test that the caching extension is working."""

    global _counter
    _counter = 0
    node = _CounterNode()

    # before decoration the global counter is incremented at every call
    k = 0
    for i in range(3):
        x = mdp.numx.array([[i]], dtype='d')
        for j in range(2):
            k += 1
            assert mdp.numx.all(node.execute(x) == x)
            assert _counter == k

    # reset counter
    _counter = 0
    # activate the extension
    caching.activate_caching()
    assert mdp.get_active_extensions() == ['cache_execute']

    # after decoration the global counter is incremented for each new 'x'
    for i in range(3):
        x = mdp.numx.array([[i]], dtype='d')
        for _ in range(2):
            assert mdp.numx.all(node.execute(x) == x)
            assert _counter == i+1

    # after deactivation
    caching.deactivate_caching()
    assert mdp.get_active_extensions() == []
    # reset counter
    _counter = 0

    k = 0
    for i in range(3):
        x = mdp.numx.array([[i]], dtype='d')
        for j in range(2):
            k += 1
            assert mdp.numx.all(node.execute(x) == x)
            assert _counter == k

@requires_joblib
def test_different_instances_same_content():
    global _counter
    x = mdp.numx.array([[100.]], dtype='d')

    caching.activate_caching()
    node = _CounterNode()
    # make one fake execution to avoid that automatic setting of
    # attributes (e.g. dtype interferes with cache)
    node.execute(mdp.numx.array([[0.]], dtype='d'))
    _counter = 0

    # add attribute to make instance unique
    node.attr = 'unique'

    # cache x
    node.execute(x)
    assert _counter == 1
    # should be cached now
    node.execute(x)
    print _counter
    assert _counter == 1

    # create new instance, make is also unique and check that
    # result is still cached
    _counter = 0
    node = _CounterNode()
    node.attr = 'unique'
    node.execute(x)
    assert _counter == 1

    caching.deactivate_caching()

@requires_joblib
def test_caching_context_manager():
    global _counter
    node = _CounterNode()
    # make one fake execution to avoid that automatic setting of
    # attributes (e.g. dtype interferes with cache)
    node.execute(mdp.numx.array([[0.]], dtype='d'))
    _counter = 0

    assert mdp.get_active_extensions() == []
    with caching.cache():
        assert mdp.get_active_extensions() == ['cache_execute']

        for i in range(3):
            x = mdp.numx.array([[i]], dtype='d')
            for _ in range(2):
                assert mdp.numx.all(node.execute(x) == x)
                assert _counter == i+1
    assert mdp.get_active_extensions() == []
