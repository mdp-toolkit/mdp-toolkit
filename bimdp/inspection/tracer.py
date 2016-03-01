"""
Module to trace and document the training and execution of a BiFlow.

This module supports (Bi)HiNet structures. Monkey patching is used to
inject the tracing code into the Flow.

InspectionHTMLTracer is the main class. It uses TraceDecorationVisitor to add
the tracing decoration to the flow and TraceHTMLConverter to create HTML view
of the flow state (which in turn uses TraceHTMLVisitor for the flow
representation).

Note that this module does not combine the trace views into a slideshow, this
is done in the seperate slideshow module.
"""

# TODO: wrap inner methods (e.g. _train) to document effective arguments?

from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import object

import os
import pickle as pickle
import fnmatch
import copy
import traceback

import mdp
n = mdp.numx
import mdp.hinet as hinet

from bimdp import BiNode
from bimdp import BiFlow
from bimdp.hinet import BiFlowNode, CloneBiLayer

from bimdp.hinet import BiHiNetHTMLVisitor
from .utils import robust_pickle

CLICKABLE_NODE_ID = "clickable_node_%d"
# standard css filename for the complete CSS:
STANDARD_CSS_FILENAME = "mdp.css"

NODE_TRACE_METHOD_NAMES = ["execute", "train", "stop_training"]
BINODE_TRACE_METHOD_NAMES = []  # methods that are only traced in binodes
TRACING_WRAP_FLAG = "_insp_is_wrapped_for_tracing_"
ORIGINAL_METHOD_PREFIX = "_insp_original_"


class TraceDebugException(Exception):
    """Exception for return the information when debug is True."""

    def __init__(self, result):
        """Store the information necessary to finish the tracing.

        result -- The result that would otherwise be returned by the method.
        """
        super(TraceDebugException, self).__init__()
        self.result = result


class InspectionHTMLTracer(object):
    """Class for inspecting a single pass through a provided flow.

    This class is based on a visitor that decorates the flow elements with
    tracing wrappers. It also provides a callback function for the tracers
    and stores everything else needed for the inspection.

    This class is already specialized for creating HTML slides in the callback
    function.

    Note that a flow decorated for tracing is not compatible with pickling
    or parallel training and execution. Normally the decorated flow is
    only used in trace_training or trace_execution anyway.
    """

    def __init__(self, html_converter=None, css_filename=STANDARD_CSS_FILENAME):
        """Prepare for tracing and create the HTML translator.

        html_converter -- TraceHTMLConverter instance, with a convert_flow
            method to create the flow visualization for each slide.
        css_filename -- CSS file used for all the slides
            (default 'inspect.css').
        """
        if html_converter is None:
            self._html_converter = TraceHTMLConverter()
        else:
            self._html_converter = html_converter
        self._css_filename = css_filename
        self._tracing_decorator = TraceDecorationVisitor(
                            decorator=self._standard_tracer_decorate,
                            undecorator=self._standard_tracer_undecorate)
        self._trace_path = None  # path for the current trace
        self._trace_name = None  # name for the current trace
        self._flow = None  # needed for the callback HTML translation
        # step counter used in the callback, is reset automatically
        self._slide_index = None
        self._slide_filenames = None
        self._section_ids = None  # can be used during execution
        self._slide_node_ids = None  # active node for each slide index

    def _reset(self):
        """Reset the internal variables for a new tracing.

        Should be called before 'train', 'stop_training' or 'execute' is called
        on the flow.
        """
        self._slide_index = 0
        self._slide_filenames = []
        self._section_ids = []
        self._slide_node_ids = []
        self._html_converter.reset()

    def trace_training(self, path, flow, x, msg=None, stop_msg=None,
                       trace_name="training", debug=False, **kwargs):
        """Trace a single training phase and the stop_training.

        Return a tuple containing a list of the training slide filenames, the
        training node ids and the same for stop_training.

        path -- Path were the inspection files will be stored.
        trace_name -- Name prefix for this inspection (default is training).
        **kwargs -- Additional arguments for flow.train can be specified
            as keyword arguments.
        """
        self._reset()
        self._trace_path = path
        # train and stop filenames must be different
        self._trace_name = trace_name + "_t"
        self._flow = flow
        self._tracing_decorator.decorate_flow(flow)
        biflownode = BiFlowNode(BiFlow(flow.flow))
        try:
            biflownode.train(x=x, msg=msg, **kwargs)
            # reset is important for the following stop_training
            biflownode.bi_reset()
        # Note: this also catches legacy string exceptions (which are still
        #    used in numpy, e.g. np.core.multiarray.error)
        except:
            if debug:
                # insert the error slide and encapsulate the exception
                traceback.print_exc()
                self._write_error_frame()
                result = (self._slide_filenames, self._slide_node_ids,
                          None, None)
                raise TraceDebugException(result=result)
            else:
                raise
        train_filenames = self._slide_filenames
        train_node_ids = self._slide_node_ids
        self._reset()
        self._trace_name = trace_name + "_s"
        try:
            biflownode.stop_training(stop_msg)
        except:
            if debug:
                # insert the error slide and encapsulate the exception
                traceback.print_exc()
                self._write_error_frame()
                result = (train_filenames, train_node_ids,
                          self._slide_filenames, self._slide_node_ids)
                raise TraceDebugException(result=result)
            else:
                raise
        stop_filenames = self._slide_filenames
        stop_node_ids = self._slide_node_ids
        # restore undecorated flow
        self._tracing_decorator.decorate_flow(flow, undecorate_mode=True)
        return train_filenames, train_node_ids, stop_filenames, stop_node_ids

    def trace_execution(self, path, trace_name, flow, x, msg=None, target=None,
                        debug=False, **kwargs):
        """Trace a single execution.

        The return value is a tuple containing a list of the slide filenames,
        the node ids, the section_ids for a slideshow with sections
        (or None if no section_ids were used) and the execution output value.

        path -- Path were the inspection files will be stored.
        trace_name -- Name prefix for this inspection.
        **kwargs -- Additional arguments for flow.execute can be specified
            as keyword arguments.
        """
        self._reset()
        self._trace_path = path
        self._trace_name = trace_name
        self._flow = flow
        self._tracing_decorator.decorate_flow(flow)
        if (not (isinstance(flow, BiFlow) or isinstance(flow, BiNode)) and
            (msg is not None)):
            # a msg would be interpreted as nodenr by a Flow, so check this
            err = "A msg was given for a normal Flow (need BiFlow)."
            raise Exception(err)
        try:
            if msg or target:
                result = self._flow.execute(x, msg, target, **kwargs)
            # this case also works for mdp.Flow
            else:
                result = self._flow.execute(x, **kwargs)
        # Note: this also catches legacy string exceptions (which are still
        #    used in numpy, e.g. np.core.multiarray.error)
        except:
            if debug:
                # insert the error slide and encapsulate the exception
                traceback.print_exc()
                self._write_error_frame()
                if not self._section_ids:
                    self._section_ids = None
                result = (self._slide_filenames, self._slide_node_ids,
                          self._section_ids)
                raise TraceDebugException(result=result)
            else:
                raise
        self._tracing_decorator.decorate_flow(flow, undecorate_mode=True)
        if not self._section_ids:
            self._section_ids = None
        else:
            if len(self._section_ids) != len(self._slide_filenames):
                err = ("Mismatch between number of section_ids and number of "
                       "slides.")
                raise Exception(err)
        return (self._slide_filenames, self._slide_node_ids,
                self._section_ids, result)

    def _tracer_callback(self, node, method_name, method_result, method_args,
                         method_kwargs):
        """This method is called by the tracers.

        The calling tracer also provides this method with the needed state
        information and the method arguments.

        node -- The node from which the callback was initiated.
        method_name -- Name of the method from which the callback was initiated.
        result -- Return value of the method.
        args, kwargs -- The arguments of the method call.
        """
        ## write visualization to html_file
        try:
            html_file = self._begin_HTML_frame()
            section_id, node_id = self._html_converter.write_html(
                                            path=self._trace_path,
                                            html_file=html_file,
                                            flow=self._flow,
                                            node=node,
                                            method_name=method_name,
                                            method_result=method_result,
                                            method_args=method_args,
                                            method_kwargs=method_kwargs)
            self._slide_index += 1
            if section_id is not None:
                self._section_ids.append(section_id)
            self._slide_node_ids.append(node_id)
        finally:
            self._end_HTML_frame(html_file)

    ## HTML decoration ##

    def _begin_HTML_frame(self):
        """Return the HTML file for a trace frame including the header.

        The file should then be finished via _end_HTML_frame.
        """
        path = self._trace_path
        filename = self._trace_name + "_%d.html" % self._slide_index
        self._slide_filenames.append(filename)
        html_file = open(os.path.join(path, filename), "w")
        html_file = hinet.NewlineWriteFile(html_file)
        html_file.write('<html>\n<head>\n<title>Inspection Slide</title>')
        if self._css_filename:
            html_file.write('<style type="text/css" media="screen">')
            html_file.write('@import url("%s");' % self._css_filename)
            html_file.write('</style>\n</head>\n<body>')
        return html_file

    def _end_HTML_frame(self, html_file):
        """Complete and close the HTML file for a trace frame.

        The method should always be used after _begin_HTML_frame.
        """
        html_file.write('</body>\n</html>')

    def _write_error_frame(self):
        with self._begin_HTML_frame() as html_file:
            html_file.write('<div class="error">')
            html_file.write('<h3>Encountered Exception</h3>')
            traceback_html = traceback.format_exc().replace('\n', '<br>')
#        get HTML traceback, didn't work due to legacy stuff
#        TODO: retry this in the future
#        import StringIO as stringio
#        import cgitb
#        import mdp
#        exception_type, exception, tb = sys.exc_info()
#        # Problem: only the text of the original exception is stored in
#        #     mdp.FlowExceptionCR, and the text is not even correctpy displayed.
##        if exception_type is mdp.FlowExceptionCR:
##            exception.args = tuple()
##            exception.message = None
#        buffer = stringio.StringIO()
#        handler = cgitb.Hook(file=buffer)
#        handler.handle((exception_type, exception, tb))
#        traceback_html = buffer.getvalue()
            html_file.write(traceback_html)
            html_file.write('</div>')
            self._end_HTML_frame(html_file)

    ## monkey patching tracing decorator wrapper methods ##

    def _standard_tracer_decorate(self, node):
        """Adds a tracer wrapper to the node via monkey patching."""
        # add a marker to show that this node is wrapped
        setattr(node, TRACING_WRAP_FLAG, True)
        trace_method_names = list(NODE_TRACE_METHOD_NAMES)
        if isinstance(node, BiNode):
            trace_method_names += BINODE_TRACE_METHOD_NAMES
        for method_name in trace_method_names:
            new_method_name = ORIGINAL_METHOD_PREFIX + method_name
            # create a reference to the original method
            setattr(node, new_method_name, getattr(node, method_name))
            # use nested scopes  lexical closure to get proper wrapper
            def get_wrapper(_method_name, _inspector):
                _new_method_name = ORIGINAL_METHOD_PREFIX + method_name
                def wrapper(self, *args, **kwargs):
                    args_copy = copy.deepcopy(args)
                    kwargs_copy = copy.deepcopy(kwargs)
                    result = getattr(self, _new_method_name)(*args, **kwargs)
                    _inspector._tracer_callback(self, _method_name, result,
                                                args_copy, kwargs_copy)
                    return result
                return wrapper
            # hide the original method in this instance behind the wrapper
            setattr(node, method_name,
                    get_wrapper(method_name, self).__get__(node))
        # modify getstate to enable pickling (get rid of the instance methods)
        def wrapped_getstate(self):
            result = self.__dict__.copy()
            if not hasattr(node, TRACING_WRAP_FLAG):
                return result
            del result[TRACING_WRAP_FLAG]
            # delete all instance methods
            trace_method_names = list(NODE_TRACE_METHOD_NAMES)
            if isinstance(self, BiNode):
                trace_method_names += BINODE_TRACE_METHOD_NAMES
            for method_name in trace_method_names:
                del result[method_name]
                old_method_name = ORIGINAL_METHOD_PREFIX + method_name
                del result[old_method_name]
            del result["__getstate__"]
            return result
        node.__getstate__ = wrapped_getstate.__get__(node)

    def _standard_tracer_undecorate(self, node):
        """Remove a tracer wrapper from the node."""
        if not hasattr(node, TRACING_WRAP_FLAG):
            return
        delattr(node, TRACING_WRAP_FLAG)
        trace_method_names = list(NODE_TRACE_METHOD_NAMES)
        if isinstance(node, BiNode):
            trace_method_names += BINODE_TRACE_METHOD_NAMES
        for method_name in trace_method_names:
            # delete the wrapped method in the instance to unhide the original
            delattr(node, method_name)
            # delete the no longer used reference to the original method
            old_method_name = ORIGINAL_METHOD_PREFIX + method_name
            delattr(node, old_method_name)
        # restore normal getstate
        delattr(node, "__getstate__")


class TraceDecorationVisitor(object):
    """Class to add tracing wrappers to nodes in a flow."""

    def __init__(self, decorator, undecorator):
        """Initialize.

        decorator -- Callable decorator that wraps node methods.
        undecorator -- Callable decorator that removes the wrapper from a
            method.
        """
        self._decorator = decorator
        self._undecorator = undecorator
        # note that _visit_clonelayer uses the undecorate mode
        self._undecorate_mode = None

    def decorate_flow(self, flow, undecorate_mode=False):
        """Adds or removes wrappers from the nodes in the given flow."""
        self._undecorate_mode = undecorate_mode
        for node in flow:
            self._visit_node(node)

    def _visit_node(self, node):
        if hasattr(node, "flow"):
            self._visit_flownode(node)
        elif isinstance(node, mdp.hinet.CloneLayer):
            self._visit_clonelayer(node)
        elif isinstance(node, mdp.hinet.Layer):
            self._visit_layer(node)
        else:
            self._visit_standard_node(node)

    def _visit_standard_node(self, node):
        """Wrap the node."""
        if not self._undecorate_mode:
            self._decorator(node)
        else:
            self._undecorator(node)

    def _visit_flownode(self, flownode):
        for node in flownode.flow:
            self._visit_node(node)

    def _visit_layer(self, layer):
        for node in layer:
            self._visit_node(node)

    def _visit_clonelayer(self, clonelayer):
        # TODO: enable the use of a shallow copy to save memory,
        #    but this requires to implement __copy__ in Node etc. for recursive
        #    shallow copying
        if self._undecorate_mode:
            if isinstance(clonelayer, CloneBiLayer):
                # check that clonelayer is actually decorated
                if not hasattr(clonelayer, "_original_set_use_copies"):
                    return
                del clonelayer._set_use_copies
                del clonelayer._original_set_use_copies
                del clonelayer.__getstate__
                self._visit_node(clonelayer.nodes[0])
                if not clonelayer.use_copies:
                    clonelayer.nodes = ((clonelayer.node,) *
                                        len(clonelayer.nodes))
            else:
                self._visit_node(clonelayer.nodes[0])
                clonelayer.nodes = (clonelayer.node,) * len(clonelayer.nodes)
            # undecoration is complete
            return
        ## decorate clonelayer
        if ((not isinstance(clonelayer, CloneBiLayer)) or
            (not clonelayer.use_copies)):
            # use a decorated deep copy for the first node
            clonelayer.node = clonelayer.nodes[0].copy()
            clonelayer.nodes = (clonelayer.node,) + clonelayer.nodes[1:]
        # only decorate the first node
        self._visit_node(clonelayer.nodes[0])
        if isinstance(clonelayer, CloneBiLayer):
            # add a wrapper to _set_use_copies,
            # otherwise all nodes in layer would get decorated
            clonelayer._original_set_use_copies = clonelayer._set_use_copies
            flow_decorator = self
            def wrapped_use_copies(self, use_copies):
                # undecorate internal nodes to allow copy operation
                flow_decorator._undecorate_mode = True
                flow_decorator._visit_node(clonelayer.nodes[0])
                flow_decorator._undecorate_mode = False
                if use_copies and not self.use_copies:
                    # switch to node copies, no problem
                    clonelayer._original_set_use_copies(use_copies)
                elif not use_copies and self.use_copies:
                    # switch to a single node instance
                    # but use a (decorated) deep copy for first node
                    clonelayer._original_set_use_copies(use_copies)
                    clonelayer.node = clonelayer.nodes[0].copy()
                    clonelayer.nodes = ((clonelayer.node,) +
                                        clonelayer.nodes[1:])
                flow_decorator._visit_node(clonelayer.nodes[0])
            clonelayer._set_use_copies = wrapped_use_copies.__get__(clonelayer)
            # modify getstate to enable pickling
            # (get rid of the instance methods)
            def wrapped_getstate(self):
                result = self.__dict__.copy()
                # delete instance methods
                del result["_original_set_use_copies"]
                del result["_set_use_copies"]
                del result["__getstate__"]
                return result
            clonelayer.__getstate__ = wrapped_getstate.__get__(clonelayer)


_INSPECTION_CSS_FILENAME = "trace.css"

def inspection_css():
    """Return the CSS for the inspection slides."""
    css_filename = os.path.join(os.path.split(__file__)[0],
                                _INSPECTION_CSS_FILENAME)
    with open(css_filename, 'r') as css_file:
        css = css_file.read()
    return BiHiNetHTMLVisitor.hinet_css() + css


class TraceHTMLVisitor(BiHiNetHTMLVisitor):
    """Special BiHiNetHTMLVisitor to take into account runtime info.

    It highlights the currently active node.
    """

    def __init__(self, html_file, show_size=False):
        super(TraceHTMLVisitor, self).__init__(html_file,
                                               show_size=show_size)
        self._current_node = None
        self._node_id_index = None
        # this is the HTML node id, not the Node attribute
        self._current_node_id = None

    def convert_flow(self, flow, current_node=None):
        self._current_node = current_node
        self._node_id_index = 0
        self._current_node_id = None
        super(TraceHTMLVisitor, self).convert_flow(flow)

    def _open_node_env(self, node, type_id="node"):
        """Open the HTML environment for the node internals.

        This special version highlights the nodes involved in the trace.

        node -- The node itself.
        type_id -- The id string as used in the CSS.
        """
        f = self._file
        html_line = '<table class="'
        trace_class = None
        if node is self._current_node:
            trace_class = "current_node"
        elif type_id == "node" and node._train_phase_started:
            trace_class = "training_node"
        if trace_class:
            html_line += ' %s' % trace_class
        html_line += ' %s' % type_id
        # assign id only to nodes which trigger a slide creation,
        # i.e. only if the node can become active
        if hasattr(node, TRACING_WRAP_FLAG):
            node_id = CLICKABLE_NODE_ID % self._node_id_index
            if node is self._current_node:
                self._current_node_id = node_id
            self._node_id_index += 1
            html_line +=  ' clickable" id="%s">' % node_id
        else:
            html_line += '">'
        f.write(html_line)
        self._write_node_header(node, type_id)


class TraceHTMLConverter(object):
    """Class to visualize the state of a BiFlow during execution or training.

    The single snapshot is a beefed up version of the standard HTML view.
    Capturing the data to make this possible is not the responsibility of this
    class.
    """

    def __init__(self, flow_html_converter=None):
        """Initialize the internal variables."""
        if flow_html_converter is None:
            self.flow_html_converter = TraceHTMLVisitor(html_file=None)
        else:
            self.flow_html_converter= flow_html_converter
        self._html_file = None

    def reset(self):
        """Reset internal variables for a new trace.

        It is called (by TraceHTMLInspector) before calling 'train',
        'stop_training' or 'execute' on the flow.

        This method can be overridden by derived that need to keep track of the
        training or execution phase.
        """
        pass

    @staticmethod
    def _array_pretty_html(ar):
        """Return a nice HTML representation of the given numpy array."""
        ar_str = 'array with shape %s<br>\n' % str(ar.shape)
        # TODO: use np.savetxt instead?
        ar_str += (str(ar).replace(' [', '<br>\n[').
                    replace(']\n ...', ']<br>\n...'))
        return ar_str

    @classmethod
    def _dict_pretty_html(cls, dic):
        """Return a nice HTML representation of the given numpy array."""
        # TODO: use an stringio buffer for efficency
        # put array keys last, because arrays are typically rather large
        keys = [key for key, value in list(dic.items())
                if not isinstance(value, n.ndarray)]
        keys.sort()
        ar_keys = [key for key, value in list(dic.items())
                   if isinstance(value, n.ndarray)]
        ar_keys.sort()
        keys += ar_keys
        dic_strs = []
        for key in keys:
            value = dic[key]
            dic_str = '<span class="keyword">' + repr(key) + '</span>: '
            if isinstance(value, str):
                dic_str += repr(value)
            elif isinstance(value, n.ndarray):
                dic_str += cls._array_pretty_html(value)
            else:
                dic_str += str(value)
            dic_strs.append(dic_str)
        return '{' + ',<br>\n'.join(dic_strs) + '}'

    def write_html(self, path, html_file, flow, node, method_name,
                    method_result, method_args, method_kwargs):
        """Write the HTML translation of the flow into the provided file.

        Return value is the section_id and the HTML/CSS id of the active node.
        The section id is ignored during training.

        path -- Path of the slide (e.h. to store additional images).
        html_file -- File of current slide, where the translation is written.
        flow -- The overall flow.
        node -- The node that was called last.
        method_name -- The method that was called on the last node.
        method_result -- The result from the last call.
        method_args -- args that were given to the method
        method_kwargs -- kwargs that were given to the method
        """
        self._html_file = hinet.NewlineWriteFile(html_file)
        f = self._html_file
        ## create table, left side for the flow, right side for data
        f.write('<br><br>')
        f.write('<table><tr><td id="inspect_biflow_td">')
        f.write("<h3>flow state</h3>")
        self.flow_html_converter._file = f
        self.flow_html_converter.convert_flow(flow, node)
        # now the argument / result part of the table
        f.write('</td><td id="inspect_result_td">')
        section_id = self._write_data_html(
                               path=path, html_file=html_file, flow=flow,
                               node=node, method_name=method_name,
                               method_result=method_result,
                               method_args=method_args,
                               method_kwargs=method_kwargs)
        f.write('</table>')
        f.write('</td></tr>\n</table>')
        self._html_file = None
        return section_id, self.flow_html_converter._current_node_id

    def _write_data_html(self, path, html_file, flow, node, method_name,
                         method_result, method_args, method_kwargs):
        """Write the data part (right side of the slide).

        Return value can be a section_id or None. The section_id is ignored
        during training (since the slideshow sections are used for the
        training phases).

        This method can be overriden for custom visualisations. Usually this
        original method should still be called via super.

        path -- Path of the slide (e.h. to store additional images).
        html_file -- File of current slide, where the translation is written.
        flow -- The overall flow.
        node -- The node that was called last.
        method_name -- The method that was called on the last node.
        method_result -- The result from the last call.
        method_args -- args that were given to the method
        method_kwargs -- kwargs that were given to the method
        """
        f = self._html_file
        f.write('<h3>%s arguments</h3>' % method_name)
        f.write('<table class="inspect_io_data">')
        if method_name == "stop_training":
            # first argument is not x,
            # if no arguments were given method_args == (None,)
            if method_args == (None,):
                f.write('<tr><td><pre>None</pre></tr></td>')
        else:
            # deal and remove x part of arguments
            x = method_args[0]
            if isinstance(x, n.ndarray):
                f.write('<tr><td><pre>x = </pre></td>' +
                        '<td>' + self._array_pretty_html(x) + '</td></tr>')
            else:
                f.write('<tr><td><pre>x = </pre></td><td>' + str(x) +
                        '</td></tr>')
        # remaining arg is message
        method_args = method_args[1:]
        if method_args and method_args[0] is not None:
            f.write('<tr><td><pre>msg = </pre></td><td>' +
                    self._dict_pretty_html(method_args[0]) + '</td></tr>')
        # normally the kwargs should be empty
        for arg_key in method_kwargs:
            f.write('<tr><td><pre>' + arg_key + ' = </pre></td><td>' +
                    str(method_kwargs[arg_key]) + '</td></tr>')
        f.write('</table>')
        ## print results
        f.write("<h3>%s result</h3>" % method_name)
        f.write('<table class="inspect_io_data">')
        if method_result is None:
            f.write('<tr><td><pre>None</pre></tr></td>')
        elif isinstance(method_result, n.ndarray):
            f.write('<tr><td><pre>x = </pre></td><td>' +
                    self._array_pretty_html(method_result) + '</td></tr>')
        elif isinstance(method_result, tuple):
            f.write('<tr><td><pre>x = </pre></td><td>')
            if isinstance(method_result[0], n.ndarray):
                f.write(self._array_pretty_html(method_result[0]) +
                        '</td></tr>')
            else:
                f.write(str(method_result[0]) + '</td></tr>')
            # second value is msg
            f.write('<tr><td><pre>msg = </pre></td><td>')
            if isinstance(method_result[1], dict):
                f.write(self._dict_pretty_html(method_result[1]) +
                        '</td></tr>')
            else:
                f.write(str(method_result[1]) + '</td></tr>')
            # last value is target
            if len(method_result) > 2:
                f.write('<tr><td><pre>target = </pre></td><td>' +
                        str(method_result[2]) + '</td></tr>')
        else:
            f.write('<tr><td><pre>unknown result type: </pre></td><td>' +
                    str(method_result) + '</td></tr>')


## Functions to capture pickled biflow snapshots during training. ##

PICKLE_EXT = ".pckl"
PICKLE_PROTO = -1
SNAPSHOT_FILENAME = "snapshot"

def prepare_training_inspection(flow, path):
    """Use hook in the BiFlow to store a snapshot in each training phase.

    path -- Path were the snapshots are stored.

    This is done by wrapping the _stop_training_hook method of biflow.
    Some attributes are added to the biflow which store all information needed
    for the pickling (like filename). To enable pickling we use the
    __getstate__ slot, since some attributes cannot be pickled.
    """
    # add attributes to biflow which are used in wrapper_method
    flow._snapshot_counter_ = 0
    flow._snapshot_path_ = path
    flow._snapshot_name_ = SNAPSHOT_FILENAME
    flow._snapshot_instance_methods_ = []
    ### wrap _stop_training_hook to store biflow snapshots ###
    def pickle_wrap_method(_flow, _method_name):
        new_method_name = ORIGINAL_METHOD_PREFIX + _method_name
        def wrapper(self, *args, **kwargs):
            result = getattr(self, new_method_name)(*args, **kwargs)
            # pickle biflow
            filename = (self._snapshot_name_ + "_%d" % self._snapshot_counter_ +
                        PICKLE_EXT)
            robust_pickle(self._snapshot_path_, filename, self)
            self._snapshot_counter_ += 1
            return result
        # create a reference to the original method
        setattr(_flow, new_method_name, getattr(_flow, _method_name))
        # hide the original method in this instance behind the wrapper
        setattr(_flow, _method_name, wrapper.__get__(_flow))
        _flow._snapshot_instance_methods_.append(_method_name)
        _flow._snapshot_instance_methods_.append(new_method_name)
    pickle_wrap_method(flow, "_stop_training_hook")
    ### wrap __getstate__ to enable pickling ###
    # note that in the pickled flow no trace of the wrapping remains
    def wrapped_biflow_getstate(self):
        result = self.__dict__.copy()
        # delete all instancemethods
        for method_name in self._snapshot_instance_methods_:
            del result[method_name]
        # delete the special attributes which were inserted by the wrapper
        # (not really necessary)
        del result["_snapshot_counter_"]
        del result["_snapshot_path_"]
        del result["_snapshot_name_"]
        del result["_snapshot_instance_methods_"]
        # remove data attributes (generators cannot be pickled)
        # pop with default value also works when key is not present in dict
        result.pop("_train_data_iterables", None)
        result.pop("_train_data_iterator", None)
        result.pop("_train_msg_iterables", None)
        result.pop("_train_msg_iterator", None)
        result.pop("_stop_messages", None)
        result.pop("_exec_data_iterator", None)
        result.pop("_exec_msg_iterator", None)
        result.pop("_exec_target_iterator", None)
        return result
    flow.__getstate__ = wrapped_biflow_getstate.__get__(flow)
    flow._snapshot_instance_methods_.append("__getstate__")

def remove_inspection_residues(flow):
    """Remove all the changes made by prepare_training_inspection."""
    try:
        for method_name in flow._snapshot_instance_methods_:
            delattr(flow, method_name)
        del flow._snapshot_counter_
        del flow._snapshot_path_
        del flow._snapshot_name_
        del flow._snapshot_instance_methods_
    except:
        # probably the hooks were already removed, so do nothing
        pass

def _trace_biflow_training(snapshot_path, inspection_path,
                           x_samples, msg_samples=None, stop_messages=None,
                           tracer=None,
                           debug=False, show_size=False, verbose=True,
                           **kwargs):
    """Load flow snapshots and perform the inspection with the given data.

    The return value consists of the slide filenames, the slide node ids,
    and an index table (index of last slide of section indexed by node,
    phase, train and stop). If no snapshots were found the return value is
    None.

    snapshot_path -- Path were the flow training snapshots are stored.
    inspection_path -- Path were the slides are stored.
    css_filename -- Filename of the CSS file for the slides.
    x_samples, msg_samples -- Lists with the input data for the training trace.
    stop_messages -- The stop msg for the training trace.
    tracer -- Instance of InspectionHTMLTracer, can be None for
        default class.
    debug -- If True (default is False) then any exception will be
        caught and the gathered data up to that point is returned in the
        normal way. This is useful for bimdp debugging.
    show_size -- Show the approximate memory footprint of all nodes.
    verbose -- If True (default value) a status message is printed for each
        loaded snapshot.
    **kwargs -- Additional arguments for flow.train can be specified
        as keyword arguments.
    """
    if not tracer:
        tracer = InspectionHTMLTracer()
        tracer._html_converter.flow_html_converter.show_size = show_size
    i_train_node = 0  # index of the training node
    i_snapshot = 0 # snapshot counter
    index_table = [[]]  # last slide indexed by [node, phase, train 0 or stop 1]
    slide_filenames = []
    slide_node_ids = []
    try:
        # search for the snapshot files
        for file_path, dirs, files in os.walk(os.path.abspath(snapshot_path)):
            dirs.sort()
            files = fnmatch.filter(files, SNAPSHOT_FILENAME + "*" + PICKLE_EXT)
            files.sort()
            for filename in files:
                filename = os.path.join(file_path, filename)
                # load the flow snapshot
                biflow = None  # free memory
                with open(filename, "rb") as pickle_file:
                    biflow = pickle.load(pickle_file)
                # determine which node is training and set the indices
                for node in biflow[i_train_node:]:
                    if node.get_remaining_train_phase() > 0:
                        break
                    else:
                        i_train_node += 1
                        index_table.append([])
                # inspect the training
                x = x_samples[i_train_node]
                if msg_samples:
                    msg = msg_samples[i_train_node]
                else:
                    msg = None
                if stop_messages:
                    stop_msg = stop_messages[i_train_node]
                else:
                    stop_msg = None
                trace_name = "%d_%d" % (i_snapshot, i_train_node)
                train_files, train_ids, stop_files, stop_ids = \
                    tracer.trace_training(trace_name=trace_name,
                                          path=inspection_path,
                                          flow=biflow,
                                          x=x, msg=msg, stop_msg=stop_msg,
                                          debug=debug,
                                          **kwargs)
                slide_filenames += train_files
                train_index = len(slide_filenames) - 1
                slide_filenames += stop_files
                stop_index = len(slide_filenames) - 1
                index_table[i_train_node].append((train_index, stop_index))
                slide_node_ids += train_ids
                slide_node_ids += stop_ids
                if verbose:
                    print("got traces for snapshot %d" % (i_snapshot + 1))
                i_snapshot += 1
    except TraceDebugException as debug_exception:
        train_files, train_ids, stop_files, stop_ids = debug_exception.result
        slide_filenames += train_files
        train_index = len(slide_filenames) - 1
        if stop_files:
            slide_filenames += stop_files
        stop_index = len(slide_filenames) - 1
        index_table[i_train_node].append((train_index, stop_index))
        slide_node_ids += train_ids
        if stop_ids:
            slide_node_ids += stop_ids
        debug_exception.result = (slide_filenames, slide_node_ids, index_table)
        raise
    return slide_filenames, slide_node_ids, index_table
