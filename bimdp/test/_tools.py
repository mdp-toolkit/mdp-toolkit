"""
Classes for tracing BiNode behavior in flows.
"""
from __future__ import print_function
from builtins import range

import mdp
from bimdp.nodes import IdentityBiNode


class JumpBiNode(IdentityBiNode):
    """BiNode which can perform all kinds of jumps.

    This is useful for testing or flow control. It can also be used
    together with BiNodes as simple jump targets.
    """

    def __init__(self, train_results=None, stop_train_results=None,
                 execute_results=None, *args, **kwargs):
        """Initialize this BiNode.

        Note that this node has an internal variable self.loop_counter which is
        used by execute, message and stop_message (and incremented by each).

        train_results -- List of lists of results for the training phases.
            First index for training phase, second for loop counter.
        stop_train_results -- List of results for the training phases.
        execute_results -- Single result tuple starting at msg or list of
            results, which are used according to the loop counter. The list
            entries can also be None (then x is simply forwarded).
        stop_message_results -- Like execute_results.
        """
        self.loop_counter = 0 # counter for execution phase
        self._train_results = train_results
        self._stop_train_results = stop_train_results
        self._execute_results = execute_results
        super(JumpBiNode, self).__init__(*args, **kwargs)

    def is_trainable(self):
        if self._train_results:
            return True
        else:
            return False

    @staticmethod
    def is_invertible():
        return False

    def _get_train_seq(self):
        """Return a train_seq which returns the predefined values."""
        # wrapper function for _train, using local scopes
        def get_train_function(i_phase):
            def train_function(x):
                self.loop_counter += 1
                if self.loop_counter-1 >= len(self._train_results[i_phase]):
                    return None
                return self._train_results[i_phase][self.loop_counter-1]
            return train_function
        # wrapper function for _stop_training
        def get_stop_training(i_phase):
            def stop_training():
                return self._stop_train_results[i_phase]
            return stop_training
        # now wrap the training sequence
        train_seq = []
        if not self._train_results:
            return train_seq
        for i_phase in range(len(self._train_results)):
            train_seq.append((get_train_function(i_phase),
                              get_stop_training(i_phase)))
        return train_seq

    def _execute(self, x):
        """Return the predefined values for the current loop count value."""
        self.loop_counter += 1
        if not self._execute_results:
            return x
        if self.loop_counter-1 >= len(self._execute_results):
            return x
        result = self._execute_results[self.loop_counter-1]
        if result is None:
            return x
        else:
            return result

    def _bi_reset(self):
        """Reset the loop counter."""
        self.loop_counter = 0

    def is_bi_learning(self):
        return False


class TraceJumpBiNode(JumpBiNode):
    """Node for testing, that logs when and how it is called."""

    def __init__(self, tracelog, log_data=False, verbose=False,
                 *args, **kwargs):
        """Initialize the node.

        tracelog -- list to which to append the log entries
        log_data -- if true the data will be logged as well
        """
        self._tracelog = tracelog
        self._log_data = log_data
        self._verbose = verbose
        super(TraceJumpBiNode, self).__init__(*args, **kwargs)

    def train(self, x, msg=None):
        if self._log_data:
            self._tracelog.append((self._node_id, "train", x, msg))
        else:
            self._tracelog.append((self._node_id, "train"))
        if self._verbose:
            print(self._tracelog[-1])
        return super(TraceJumpBiNode, self).train(x, msg)

    def execute(self, x, msg=None):
        if self._log_data:
            self._tracelog.append((self._node_id, "execute", x, msg))
        else:
            self._tracelog.append((self._node_id, "execute"))
        if self._verbose:
            print(self._tracelog[-1])
        return super(TraceJumpBiNode, self).execute(x, msg)

    def stop_training(self, msg=None):
        self._tracelog.append((self._node_id, "stop_training"))
        if self._verbose:
            print(self._tracelog[-1])
        return super(TraceJumpBiNode, self).stop_training(msg)

    def _bi_reset(self):
        self._tracelog.append((self._node_id, "bi_reset"))
        if self._verbose:
            print(self._tracelog[-1])
        return super(TraceJumpBiNode, self)._bi_reset()


class ParallelTraceJumpBiNode(TraceJumpBiNode):

    def _fork(self):
        return self.copy()

    def _join(self):
        pass


class IdNode(mdp.Node):
    """Non-bi identity node for testing."""

    @staticmethod
    def is_trainable():
        return False
