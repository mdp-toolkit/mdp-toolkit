from builtins import range
from builtins import object
import mdp
n = mdp.numx

import py.test

from bimdp import BiNode, MSG_ID_SEP, BiFlow, BiClassifier, binode_coroutine
from bimdp.nodes import (
    IdentityBiNode, SFABiNode, FDABiNode, SignumBiClassifier
)
from ._tools import JumpBiNode


class TestBiNode(object):

    def test_msg_parsing1(self):
        """Test the message parsing and recombination."""
        class TestBiNode(BiNode):
            def _execute(self, x, a, b, d):
                self.a = a
                self.b = b
                self.d = d
                return x, {"g": 15, "z": 3}
            @staticmethod
            def is_trainable(): return False
        binode = TestBiNode(node_id="test")
        b_key = "test" + MSG_ID_SEP + "b"
        d_key = "test" + MSG_ID_SEP + "d"
        msg = {"c": 12, b_key: 42, "a": 13, d_key: "bla"}
        _, msg = binode.execute(None, msg)
        assert "a" in msg
        assert b_key not in msg
        assert d_key not in msg
        assert binode.a == 13
        assert binode.b == 42
        assert binode.d == "bla"
        # test the message combination
        assert msg["g"] == 15
        assert msg["z"] == 3

    def test_msg_parsing2(self):
        """Test that an adressed argument is not found."""
        class TestBiNode(BiNode):
            def _execute(self, x, a, b):
                self.a = a
                self.b = b
            @staticmethod
            def is_trainable(): return False
        binode = TestBiNode(node_id="test")
        b_key = "test" + MSG_ID_SEP + "b"
        # check that the 'd' key which is not an arg gets removed
        d_key = "test" + MSG_ID_SEP + "d"
        msg = {"c": 12, b_key: 42, "a": 13, d_key: "bla"}
        _, out_msg = binode.execute(None, msg)
        assert d_key not in out_msg

    def test_msg_magic(self):
        """Test that the magic msg argument works."""
        class TestBiNode(BiNode):
            def _execute(self, x, a, msg, b):
                self.a = a
                self.b = b
                del msg["c"]
                msg["f"] = 1
                return x, msg
            @staticmethod
            def is_trainable(): return False
        binode = TestBiNode(node_id="test")
        b_key = "test" + MSG_ID_SEP + "b"
        msg = {"c": 12, b_key: 42, "a": 13}
        _, msg = binode.execute(None, msg)
        assert "a" in msg
        assert "c" not in msg  # was deleted in _execute
        assert msg["f"] == 1
        assert b_key not in msg
        assert binode.a == 13
        assert binode.b == 42

    def test_method_magic(self):
        """Test the magic method message key."""
        class TestBiNode(BiNode):
            def _test(self, x, a, b):
                self.a = a
                self.b = b
            @staticmethod
            def is_trainable(): return False
        binode = TestBiNode(node_id="test")
        b_key = "test" + MSG_ID_SEP + "b"
        msg = {"c": 12, "a": 13, b_key: 42, "method": "test"}
        binode.execute(None, msg)
        assert "a" in msg
        assert b_key not in msg
        assert binode.b == 42

    def test_target_magic(self):
        """Test the magic target message key."""
        class TestBiNode(BiNode):
            def _execute(self, x, a, b):
                self.a = a
                self.b = b
            @staticmethod
            def is_trainable(): return False
        binode = TestBiNode(node_id="test")
        b_key = "test" + MSG_ID_SEP + "b"
        target_key = "test" + MSG_ID_SEP + "target"
        msg = {"c": 12, b_key: 42, "a": 13, target_key: "test2"}
        result = binode.execute(None, msg)
        assert len(result) == 3
        assert result[2] == "test2"

    def test_inverse_magic1(self):
        """Test the magic inverse method argument."""
        class TestBiNode(BiNode):
            def _inverse(self, x, a, b):
                self.a = a
                self.b = b
                y = n.zeros((len(x), self.input_dim))
                return y
            @staticmethod
            def is_trainable(): return False
        binode = TestBiNode(node_id="test", input_dim=20, output_dim=10)
        b_key = "test" + MSG_ID_SEP + "b"
        msg = {"c": 12, "a": 13, b_key: 42, "method": "inverse"}
        x = n.zeros((5, binode.output_dim))
        result = binode.execute(x, msg)
        assert len(result) == 3
        assert result[2] == -1
        assert result[0].shape == (5, 20)

    def test_inverse_magic2(self):
        """Test overriding the magic inverse target."""
        class TestBiNode(BiNode):
            def _inverse(self, x, a, b):
                self.a = a
                self.b = b
                y = n.zeros((len(x), self.input_dim))
                return y, None, "test2"
            @staticmethod
            def is_trainable(): return False
        binode = TestBiNode(node_id="test", input_dim=20, output_dim=10)
        b_key = "test" + MSG_ID_SEP + "b"
        msg = {"c": 12, "a": 13, b_key: 42, "method": "inverse"}
        x = n.zeros((5, binode.output_dim))
        result = binode.execute(x, msg)
        assert result[2] == "test2"

    def test_stoptrain_result1(self):
        """Test that stop_result is handled correctly."""
        stop_result = ({"test": 0}, 1)
        bi_sfa_node = SFABiNode(stop_result=stop_result,
                                node_id="testing binode")
        assert bi_sfa_node.is_trainable()
        x = n.random.random((100,10))
        train_result = bi_sfa_node.train(x)
        assert train_result == None
        assert bi_sfa_node.is_training()
        result = bi_sfa_node.stop_training()
        assert result == (None,) + stop_result
        assert bi_sfa_node.input_dim == 10
        assert bi_sfa_node.output_dim == 10
        assert bi_sfa_node.dtype == "float64"

    def test_stoptrain_result2(self):
        """Test that stop_result is handled correctly for multiple phases."""
        stop_result = [({"test": 0}, 1), ({"test2": 0}, 2)]
        binode = FDABiNode(stop_result=stop_result,
                           node_id="testing binode")
        x = n.random.random((100,10))
        msg = {"labels": n.zeros(len(x))}
        binode.train(x, msg)
        result = binode.stop_training()
        assert result == (None,) + stop_result[0]
        binode.train(x, msg)
        result = binode.stop_training()
        assert result == (None,) + stop_result[1]

    def test_stop_training_execute(self):
        """Test the magic execute method argument for stop_training."""
        class TestBiNode(BiNode):
            def _train(self, x): pass
            def _execute(self, x, a):
                self.a = a
                self.x = x
                y = n.zeros((len(x), self.output_dim))
                return y

        binode = TestBiNode(input_dim=20, output_dim=10)
        x = n.ones((5, binode.input_dim))
        binode.train(x)
        msg = {"x": x, "a": 13, "method": "execute"}
        result = binode.stop_training(msg)
        assert n.all(binode.x == x)
        assert binode.x.shape == (5, binode.input_dim)
        assert binode.a == 13
        assert len(result) == 2
        assert result[0].shape == (5, binode.output_dim)
        assert not n.any(result[0])

    def test_stop_training_inverse(self):
        """Test the magic inverse method argument for stop_training."""
        class TestBiNode(BiNode):
            def _train(self, x): pass
            def _inverse(self, x, a):
                self.a = a
                self.x = x
                y = n.zeros((len(x), self.input_dim))
                return y

        binode = TestBiNode(input_dim=20, output_dim=10)
        binode.train(n.ones((5, binode.input_dim)))
        x = n.ones((5, binode.output_dim))
        msg = {"x": x, "a": 13, "method": "inverse"}
        result = binode.stop_training(msg)
        assert n.all(binode.x == x)
        assert binode.x.shape == (5, binode.output_dim)
        assert binode.a == 13
        assert len(result) == 3
        assert result[2] == -1
        assert result[0].shape == (5, binode.input_dim)
        assert not n.any(result[0])

    def test_flow_from_sum(self):
        """Test the special addition method for BiNode."""
        node1 = IdentityBiNode()
        node2 = mdp.Node()
        flow = node1 + node2
        assert type(flow) is BiFlow
        node2 = IdentityBiNode()
        flow = node1 + node2
        assert type(flow) is BiFlow
        assert len(flow) == 2
        node3 = IdentityBiNode()
        flow = node1 + node2 + node3
        assert type(flow) is BiFlow
        assert len(flow) == 3
        node4 = IdentityBiNode()
        flow = node4 + flow
        assert type(flow) is BiFlow
        assert len(flow) == 4


class TestBiClassifierNode(object):

    def test_biclassifier(self):
        """Test the BiClassifier base class."""
        class TestBiClassifier(BiClassifier):
            def _label(self, x):
                return "LABELS"
            def _prob(self, x):
                return "PROPS"
            @staticmethod
            def is_trainable():
                return False
        node = TestBiClassifier()
        x = n.empty((5,2))
        msg = {"return_labels": "test->",
               "return_probs": True}
        result = node.execute(x, msg)
        assert result[0] is x
        assert "labels" not in result[1]
        assert result[1]["probs"] == "PROPS"
        assert result[1][msg["return_labels"] + "labels"] == "LABELS"
        assert "rank" not in result[1]
        msg = {"return_labels": None}
        result = node.execute(x,msg)
        assert result[0] is x
        assert "labels" not in result[1]
        assert "prop" not in result[1]
        assert "rank" not in result[1]

    def test_autogen_biclassifier(self):
        """Test that the autogenerated classifiers work."""
        node = SignumBiClassifier()
        msg = {"return_labels": True}
        # taken from the SignumClassifier unittest
        x = n.array([[1, 2, -3, -4], [1, 2, 3, 4]])
        result = node.execute(x, msg)
        assert result[0] is x
        assert result[1]["labels"].tolist() == [-1, 1]


class TestIdentityBiNode(object):

    def test_idnode(self):
        """Test the IdentityBiNode.

        Instantiation is tested and it should perform like an id node, but
        accept msg arguments.
        """
        binode = IdentityBiNode(node_id="testing binode")
        x = n.random.random((10,5))
        msg = {"some array": n.random.random((10,3))}
        # see if msg causes no problem
        y, msg = binode.execute(x, msg)
        assert n.all(x==y)
        # see if missing msg causes problem
        y = binode.execute(x)
        assert n.all(x==y)


class TestJumpBiNode(object):

    def test_node(self):
        """Test the JumpBiNode."""
        train_results = [[(0, "t1")], [None], [(3, "t3")]]
        stop_train_results = [None, (5, "st2"), (6, "st3")]
        execute_results = [(None, {}), None, (None, {}, "et4")]
        jumpnode = JumpBiNode(train_results=train_results,
                              stop_train_results=stop_train_results,
                              execute_results=execute_results)
        x = n.random.random((2,2))
        assert jumpnode.is_trainable()
        # training
        rec_train_results = []
        rec_stop_train_results = []
        for _ in range(len(train_results)):
            rec_train_results.append([jumpnode.train(x)])
            jumpnode.bi_reset()
            rec_stop_train_results.append(jumpnode.stop_training())
            jumpnode.bi_reset()
        assert not jumpnode.is_training()
        assert rec_train_results == train_results
        assert rec_stop_train_results == rec_stop_train_results
        # execution
        rec_execute_results = []
        for _ in range(4):  # note that this is more then the execute_targets
            rec_execute_results.append(jumpnode.execute(x))
        execute_results[1] = x
        execute_results.append(x)
        assert (rec_execute_results == execute_results)
        assert jumpnode.loop_counter == 4


class TestBiNodeCoroutine(object):
    """Test the coroutine decorator and the related BiNode functionality."""

    def test_codecorator(self):
        """Test basic codecorator functionality."""

        class CoroutineBiNode(BiNode):

            @staticmethod
            def is_trainable():
                return False

            @binode_coroutine(["alpha", "beta"])
            def _execute(self, x, alpha):
                """Blabla."""
                x, alpha, beta = yield (x, {"alpha": alpha, "beta": 2},
                                        self.node_id)
                x, alpha, beta = yield (x, {"alpha": alpha+1, "beta": beta+2},
                                        self.node_id)
                yield x, {"alpha": alpha, "beta": beta}

        node = CoroutineBiNode(node_id="conode")
        flow = BiFlow([node])
        x = n.random.random((3,2))
        y, msg = flow.execute(x, {"alpha": 3})
        assert msg["alpha"] == 4
        assert msg["beta"] == 4
        assert node.execute.__doc__ == """Blabla."""

    def test_codecorator2(self):
        """Test codecorator functionality with StopIteration."""

        class CoroutineBiNode(BiNode):

            @staticmethod
            def is_trainable():
                return False

            @binode_coroutine(["alpha", "beta"])
            def _execute(self, x, alpha):
                x, alpha, beta = yield (x, {"alpha": alpha, "beta": 2},
                                        self.node_id)
                x, alpha, beta = yield (x, {"alpha": alpha+1, "beta": beta+2},
                                        self.node_id)
                raise StopIteration(x, {"alpha": alpha, "beta": beta})

        node = CoroutineBiNode(node_id="conode")
        flow = BiFlow([node])
        x = n.random.random((3,2))
        y, msg = flow.execute(x, {"alpha": 3})
        assert msg["alpha"] == 4
        assert msg["beta"] == 4

    def test_codecorator_defaults(self):
        """Test codecorator argument default values."""

        class CoroutineBiNode(BiNode):

            @staticmethod
            def is_trainable():
                return False

            @binode_coroutine(["alpha", "beta"], defaults=(7,8))
            def _execute(self, x):
                x, alpha, beta = yield (x, None, self.node_id)
                raise StopIteration(x, {"alpha": alpha, "beta": beta})

        node = CoroutineBiNode(node_id="conode")
        flow = BiFlow([node])
        x = n.random.random((3,2))
        y, msg = flow.execute(x)
        assert msg["alpha"] == 7
        assert msg["beta"] == 8

    def test_codecorator_no_iteration(self):
        """Test codecorator corner case with no iterations."""

        class CoroutineBiNode(BiNode):

            @staticmethod
            def is_trainable():
                return False

            @binode_coroutine()
            def _execute(self, x):
                # at least one yield must be in a coroutine
                if False:
                    yield None
                raise StopIteration(None, {"a": 1}, self.node_id)

        node1 = CoroutineBiNode()
        x = n.random.random((3,2))
        result = node1.execute(x)
        assert result == (None, {"a": 1}, None)

    def test_codecorator_reset1(self):
        """Test that codecorator correctly resets after termination."""

        class CoroutineBiNode(BiNode):

            @staticmethod
            def is_trainable():
                return False

            @binode_coroutine()
            def _execute(self, x, a, msg=None):
                # note that the a argument is required, drop message
                for _ in range(2):
                    x = yield x
                raise StopIteration(x)

        node1 = CoroutineBiNode()
        x = n.random.random((3,2))
        # this inits the coroutine, a argument is needed
        node1.execute(x, {"a": 2})
        node1.execute(x)
        node1.execute(x)
        assert node1._coroutine_instances == {}
        # couroutine should be reset, a argument is needed again
        py.test.raises(TypeError, node1.execute, x)

    def test_codecorator_reset2(self):
        """Test that codecorator correctly resets without yields."""

        class CoroutineBiNode(BiNode):

            @staticmethod
            def is_trainable():
                return False

            @binode_coroutine()
            def _execute(self, x, a, msg=None):
                if False:
                    yield
                raise StopIteration(x)

        node1 = CoroutineBiNode()
        x = n.random.random((3,2))
        node1.execute(x, {"a": 2})
        assert node1._coroutine_instances == {}
