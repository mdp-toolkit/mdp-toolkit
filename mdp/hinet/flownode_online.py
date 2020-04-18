"""
Module for the OnlineFlowNode class.
"""

from builtins import range
import mdp
from .flownode import FlowNode


class OnlineFlowNode(FlowNode, mdp.OnlineNode):
    """OnlineFlowNode wraps an OnlineFlow of OnlineNodes into a single OnlineNode.

    This is handy if you want to use an OnlineFlow where an OnlineNode is required.
    Additional args and kwargs for train and execute are supported.

    An OnlineFlowNode requires that all the nodes of the input onlineflow to be either:
    (a) OnlineNodes (eg. OnlineCenteringNode(), IncSFANode(), etc.),
     or
    (b) trained Nodes (eg. a fully trained PCANode with no remaining training phases. A trained Node's
    node.is_trainable() returns True and node.is_training() returns False.),
     or
    (c) non-trainable Nodes (eg. QuadraticExpansionNode(). A non-trainable Node's node.is_trainable()
    returns False and node.is_training() returns False.).

    OnlineFlowNode does not support an onlineflow with a terminal trainable Node that is not
    an OnlineNode (see doc string of OnlineFlow for reference).

    All the read-only container slots are supported and are forwarded to the internal flow.
    """

    def __init__(self, flow, input_dim=None, output_dim=None, dtype=None, numx_rng=None):
        super(OnlineFlowNode, self).__init__(flow=flow, input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self._check_compatibility(flow)
        # numx_rng will not be set through the super call.
        # Have to set it explicitly here:
        self.numx_rng = numx_rng
        # set training type
        self._set_training_type_from_flow(flow)

    @staticmethod
    def _check_compatibility(flow):
        # valid: onlinenodes, trained or non-trainable nodes
        # invalid: trainable but not an online node
        if not isinstance(flow, mdp.OnlineFlow):
            raise TypeError("Flow must be an OnlineFlow type and not %s" % (type(flow)))
        if not isinstance(flow[-1], mdp.Node):
            raise TypeError("Flow item must be a Node instance and not %s" % (type(flow[-1])))
        elif isinstance(flow[-1], mdp.OnlineNode):
            # OnlineNode
            pass
        else:
            # Node
            if not flow[-1].is_training():
                # trained or non-trainable node
                pass
            else:
                raise TypeError("OnlineFlowNode supports either only a terminal OnlineNode, a "
                                "trained or a non-trainable Node.")

    def _set_training_type_from_flow(self, flow):
        # the training type is set to batch if there is at least one node with the batch training type.
        self._training_type = None
        for node in flow:
            if hasattr(node, 'training_type') and (node.training_type == 'batch'):
                self._training_type = 'batch'
                return

    def set_training_type(self, training_type):
        """Sets the training type"""
        if self.training_type is None:
            if training_type in self._get_supported_training_types():
                self._training_type = training_type
            else:
                raise mdp.OnlineNodeException("Unknown training type specified %s. Supported types "
                                              "%s" % (str(training_type), str(self._get_supported_training_types())))
        elif self.training_type != training_type:
            raise mdp.OnlineNodeException("Cannot change the training type to %s. It is inferred from "
                                          "the flow and is set to '%s'. " % (training_type, self.training_type))

    def _set_numx_rng(self, rng):
        # set the numx_rng for all the nodes to be the same.
        for node in self._flow:
            if hasattr(node, 'set_numx_rng'):
                node.numx_rng = rng
        self._numx_rng = rng

    def _get_train_seq(self):
        """Return a training sequence containing all training phases.

        Unlike thw FlowNode, the OnlineFlowNode requires only
        one train_seq item for each node. Each node's train function
        takes care of its multiple train phases (if any).

        """

        def get_execute_function(_i_node, _j_node):
            # This internal function is needed to channel the data through
            # the nodes i to j
            def _execute(x, *args, **kwargs):
                for i in range(_i_node, _j_node):
                    try:
                        x = self._flow[i].execute(x)
                    except Exception as e:
                        self._flow._propagate_exception(e, i)
                return x

            return _execute

        def get_empty_function():
            def _empty(x, *args, **kwargs):
                pass

            return _empty

        trainable_nodes_nrs = []  # list of trainable node ids
        trainable_nodes = []  # list of trainable nodes
        for i_node, node in enumerate(self._flow):
            if node.is_trainable():
                trainable_nodes_nrs.append(i_node)
                trainable_nodes.append(node)

        train_seq = []
        # if the first node is not trainable, channel the input data through
        # the nodes until the first trainable-node.
        if trainable_nodes_nrs[0] > 0:
            train_seq += [(get_empty_function(), get_empty_function(),
                           get_execute_function(0, trainable_nodes_nrs[0]))]

        for i in range(len(trainable_nodes_nrs)):
            if i < (len(trainable_nodes_nrs) - 1):
                # the execute function channels the data from the current node to the
                # next trainable node
                train_seq += [(trainable_nodes[i].train, trainable_nodes[i].stop_training,
                               get_execute_function(trainable_nodes_nrs[i], trainable_nodes_nrs[i + 1]))]
            else:
                # skip the execution for the last trainable node.
                train_seq += [(trainable_nodes[i].train, trainable_nodes[i].stop_training,
                               get_empty_function())]

        # try fix the dimension of the internal nodes and the FlowNode
        # after the stop training has been called.
        def _get_stop_training_wrapper(self, node, func):
            def _stop_training_wrapper(*args, **kwargs):
                func(*args, **kwargs)
                self._fix_nodes_dimensions()

            return _stop_training_wrapper

        if train_seq:
            train_seq[-1] = (train_seq[-1][0],
                             _get_stop_training_wrapper(self, self._flow[-1], train_seq[-1][1]),
                             train_seq[-1][2])

        return train_seq


class CircularOnlineFlowNode(FlowNode, mdp.OnlineNode):
    """CircularOnlineFlowNode wraps a CircularOnlineFlow of OnlineNodes into a single OnlineNode.

    This is handy if you want to use a CircularOnlineFlow where an OnlineNode is required.

    Once the node is initialized, the _flow_iterations and the _ignore_input values of a CircularOnlineFlow
    cannot be changed. However, the stored_input can be changed (or set) using 'set_stored_input' method.

    All the read-only container slots are supported and are forwarded to the internal flow.
    """

    def __init__(self, flow, input_dim=None, output_dim=None, dtype=None, numx_rng=None):
        super(CircularOnlineFlowNode, self).__init__(flow=flow, input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self._check_compatibility(flow)
        # numx_rng will not be set through the super call.
        # Have to set it explicitly here:
        self.numx_rng = numx_rng
        # set training type
        self._set_training_type_from_flow(flow)
        # get stored_input, flow_iterations and ignore_input flags from flow
        self._flow_iterations = flow._flow_iterations
        self._ignore_input = flow._ignore_input
        self._stored_input = flow._stored_input

    def set_stored_input(self, x):
        """Sets the stored input."""
        self._stored_input = x

    def get_stored_input(self):
        """Returns the stored input."""
        return self._stored_input

    @staticmethod
    def _check_compatibility(flow):
        # Only a CircularOnlineFlow is valid flow.
        if not isinstance(flow, mdp.CircularOnlineFlow):
            raise TypeError("Flow must be a CircularOnlineFlow type and not %s" % (type(flow)))

    def _set_training_type_from_flow(self, flow):
        # the training type is set to batch if there is at least one node with the batch training type.
        self._training_type = None
        for node in flow:
            if hasattr(node, 'training_type') and (node.training_type == 'batch'):
                self._training_type = 'batch'
                return

    def set_training_type(self, training_type):
        """Sets the training type"""
        if self.training_type is None:
            if training_type in self._get_supported_training_types():
                self._training_type = training_type
            else:
                raise mdp.OnlineNodeException("Unknown training type specified %s. Supported types "
                                              "%s" % (str(training_type), str(self._get_supported_training_types())))
        elif self.training_type != training_type:
            raise mdp.OnlineNodeException("Cannot change the training type to %s. It is inferred from "
                                          "the flow and is set to '%s'. " % (training_type, self.training_type))

    def _set_numx_rng(self, rng):
        # set the numx_rng for all the nodes to be the same.
        for node in self._flow:
            if hasattr(node, 'set_numx_rng'):
                node.numx_rng = rng
        self._numx_rng = rng

    def _get_train_seq(self):
        """Return a training sequence containing all training phases.

        There are three possible train_seqs depending on the values of self._ignore_input
        and self._flow_iterations.

        1) self._ignore_input = False, self._flow_iterations = 1
            This is functionally similar to the standard OnlineFlowNode.

        2) self._ignore_input = False, self._flow_iterations > 1
            For each data point, the OnlineFlowNode trains 1 loop with the
            data point and 'self._flow_iterations-1' loops with the updating stored input.

        3) self._ignore_input = True, self._flow_iterations > 1
            Input data is ignored, however, for each data point, the flow trains
            'self._flow_iterations' loops with the updating stored input.
        """

        def get_execute_function(_i_node, _j_node):
            # This internal function is needed to channel the data through
            # the nodes i to j
            def _execute(x, *args, **kwargs):
                for _i in range(_i_node, _j_node):
                    try:
                        x = self._flow[_i].execute(x)
                    except Exception as e:
                        self._flow._propagate_exception(e, _i)
                return x

            return _execute

        def get_empty_function():
            def _empty(x, *args, **kwargs):
                pass

            return _empty

        def _get_ignore_input_train_wrapper(fn):
            def _ignore_input_train_wrapper(x, *args, **kwargs):
                if self._stored_input is None:
                    raise mdp.TrainingException("No stored inputs to train on! Set using"
                                                "'set_stored_input' method")
                fn(self._stored_input, *args, **kwargs)

            return _ignore_input_train_wrapper

        def _get_ignore_input_execute_wrapper(fn):
            def _ignore_input_execute_wrapper(x, *args, **kwargs):
                return fn(self._stored_input, *args, **kwargs)

            return _ignore_input_execute_wrapper

        def _get_save_output_wrapper(fn):
            def _save_output_wrapper(*args, **kwargs):
                x = fn(*args, **kwargs)
                self._stored_input = x.copy()

            return _save_output_wrapper

        # get one iteration train sequence without stop training

        trainable_nodes_nrs = []  # list of trainable node ids
        trainable_nodes = []  # list of trainable nodes
        for i_node, node in enumerate(self._flow):
            if node.is_trainable():
                trainable_nodes_nrs.append(i_node)
                trainable_nodes.append(node)

        train_seq = []
        # if the first node is not trainable, channel the input data through
        # the nodes until the first trainable-node.
        if trainable_nodes_nrs[0] > 0:
            train_seq += [(get_empty_function(), get_empty_function(),
                           get_execute_function(0, trainable_nodes_nrs[0]))]

        for i in range(len(trainable_nodes_nrs)):
            if i < (len(trainable_nodes_nrs) - 1):
                # the execute function channels the data from the current node to the
                # next trainable node
                train_seq += [(trainable_nodes[i].train, get_empty_function(),
                               get_execute_function(trainable_nodes_nrs[i], trainable_nodes_nrs[i + 1]))]
            else:
                # the execute function channels the data through the remaining nodes
                # to generate input for the next iteration
                train_seq += [(trainable_nodes[i].train, get_empty_function(),
                               get_execute_function(trainable_nodes_nrs[i], len(self._flow)))]

        # repeat for (self._flow_iterations-1) iterations
        train_seq *= (self._flow_iterations - 1)

        # for the last iteration add stop_training calls and save output.
        if trainable_nodes_nrs[0] > 0:
            train_seq += [(get_empty_function(), get_empty_function(),
                           get_execute_function(0, trainable_nodes_nrs[0]))]

        for i in range(len(trainable_nodes_nrs)):
            if i < (len(trainable_nodes_nrs) - 1):
                # the execute function channels the data from the current node to the
                # next trainable node
                train_seq += [(trainable_nodes[i].train, trainable_nodes[i].stop_training,
                               get_execute_function(trainable_nodes_nrs[i], trainable_nodes_nrs[i + 1]))]
            else:
                # the execute function channels the data through the remaining nodes
                # to save the output for the next train call.
                train_seq += [(trainable_nodes[i].train, trainable_nodes[i].stop_training,
                               _get_save_output_wrapper(get_execute_function(trainable_nodes_nrs[i],
                                                                             len(self._flow))))]

        if self._ignore_input:
            # finally, if ignore input is set, then add the ignore input wraaper to the first train_seq.
            train_seq[0] = (_get_ignore_input_train_wrapper(train_seq[0][0]), train_seq[0][1],
                            _get_ignore_input_execute_wrapper(train_seq[0][2]))

        return train_seq
