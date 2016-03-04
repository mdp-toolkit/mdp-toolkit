"""
Module with simple functions for the complete inspection procedure.
"""

from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import str

import os
import webbrowser
import pickle as pickle
import tempfile
import traceback
import warnings

import mdp
from mdp import numx

from bimdp import BiFlow

from .tracer import (
    InspectionHTMLTracer, TraceDebugException, inspection_css,
    prepare_training_inspection, remove_inspection_residues,
    _trace_biflow_training, PICKLE_EXT, STANDARD_CSS_FILENAME
)
from .slideshow import (
    TrainHTMLSlideShow, ExecuteHTMLSlideShow,
    SectExecuteHTMLSlideShow)
from .utils import robust_write_file, robust_pickle, first_iterable_elem


def _open_custom_brower(open_browser, url):
    """Helper function to support opening a custom browser."""
    if isinstance(open_browser, str):
        try:
            custom_browser = webbrowser.get(open_browser)
            custom_browser.open(url)
        except webbrowser.Error:
            err = ("Could not open browser '%s', using default." %
                   open_browser)
            warnings.warn(err)
            webbrowser.open(url)
    else:
        webbrowser.open(url)

def standard_css():
    """Return the standard CSS for inspection."""
    return (mdp.utils.basic_css() + inspection_css() +
            TrainHTMLSlideShow.slideshow_css())


class EmptyTraceException(Exception):
    """Exception for empty traces, i.e., when no slides where generated."""
    pass


def inspect_training(snapshot_path, x_samples, msg_samples=None,
                     stop_messages=None, inspection_path=None,
                     tracer=None, debug=False,
                     slide_style=None, show_size=False,
                     verbose=True, **kwargs):
    """Return the HTML code for an inspection slideshow of the training.

    This function must be used after the training was completed. Before the
    training prepare_training_inspection must have been called to create
    snapshots. After training one should call remove_inspection_residues.

    Note that the file into which the returned slideshow HTML is inserted must
    be in the snapshot_path.

    snapshot_path -- Path were the flow training snapshots are stored.
    x_samples, msg_samples -- Lists with the input data for the training trace.
    stop_messages -- The stop msg for the training trace.
    inspection_path -- Path were the slides will be stored. If None (default
        value) then the snapshot_path is used.
    tracer -- Instance of InspectionHTMLTracer, can be None for
        default class.
    debug -- If True (default is False) then any exception will be
        caught and the gathered data up to that point is returned in the
        normal way. This is useful for bimdp debugging.
    slide_style -- CSS code for the individual slides (when they are
        viewed as single HTML files), has no effect on the slideshow appearance.
    show_size -- Show the approximate memory footprint of all nodes.
    verbose -- If True (default value) a status message is printed for each
        loaded snapshot.
    **kwargs -- Additional arguments for flow.train can be specified
        as keyword arguments.
    """
    if not inspection_path:
        inspection_path = snapshot_path
    ## create CSS file for the slides
    if not slide_style:
        slide_style = standard_css()
    robust_write_file(path=inspection_path, filename=STANDARD_CSS_FILENAME,
                      content=slide_style)
    del slide_style
    ## create slides
    try:
        slide_filenames, slide_node_ids, index_table = \
            _trace_biflow_training(snapshot_path=snapshot_path,
                                   inspection_path=inspection_path,
                                   x_samples=x_samples,
                                   msg_samples=msg_samples,
                                   stop_messages=stop_messages,
                                   tracer=tracer,
                                   debug=debug,
                                   show_size=show_size,
                                   verbose=verbose,
                                   **kwargs )
        if not slide_filenames:
            err = ("No inspection slides were generated, probably because "
                   "there are no untrained nodes in the given flow.")
            raise EmptyTraceException(err)
    except TraceDebugException as debug_exception:
        slide_filenames, slide_node_ids, index_table = debug_exception.result
    if index_table is None:
        return None  # no snapshots were found
    # create slideshow
    slideshow = TrainHTMLSlideShow(filenames=slide_filenames,
                                   node_ids=slide_node_ids,
                                   index_table=index_table,
                                   delay=500, delay_delta=100, loop=False)
    return str(slideshow)

def show_training(flow, data_iterables, msg_iterables=None, stop_messages=None,
                  path=None, tracer=None,
                  debug=False,  show_size=False, open_browser=True,
                  **kwargs):
    """Perform both the flow training and the training inspection.

    The return value is the filename of the slideshow HTML file.
    This function must be used with the untrained flow (no previous call
    of Flow.train is required, the training happens here).

    This function is more convenient than inspect_training since it includes
    all required steps, but it is also less customizable. After everything
    is complete the inspection slideshow is opened in the browser.

    flow -- The untrained Flow or BiFlow. After this function has been called
        the flow will be fully trained.
    data_iterables, msg_iterables, stop_messages -- Same as for calling train
        on a flow.
    path -- Path were both the training snapshots and the inspection slides
        will be stored. If None (default value) a temporary directory will be
        used.
    tracer -- Instance of InspectionHTMLTracer, can be None for
        default class.
    debug -- Ignore exception during training and try to complete the slideshow
        (default value is False).
    show_size -- Show the approximate memory footprint of all nodes.
    open_browser -- If True (default value) then the slideshow file is
        automatically opened in a webbrowser. One can also use string value
        with the browser name (for webbrowser.get) to request a specific
        browser.
    **kwargs -- Additional arguments for flow.train can be specified
        as keyword arguments.
    """
    if path is None:
        path = tempfile.mkdtemp(prefix='MDP_')
    # get first part of data iterators as sample data for inspection
    # if data_iterables is an array, wrap it up in a list
    if isinstance(data_iterables, numx.ndarray):
        data_iterables = [[data_iterables]] * len(flow)
    x_samples = []
    for i, data_iterable in enumerate(data_iterables):
        if data_iterable is None:
            x_sample, new_data_iterable = None, None
        else:
            x_sample, new_data_iterable = first_iterable_elem(data_iterable)
        x_samples.append(x_sample)
        data_iterables[i] = new_data_iterable
    del x_sample
    if msg_iterables:
        msg_samples = []
        for i, msg_iterable in enumerate(msg_iterables):
            if msg_iterable is None:
                msg_sample, new_msg_iterable = None, None
            else:
                msg_sample, new_msg_iterable = first_iterable_elem(msg_iterable)
            msg_samples.append(msg_sample)
            msg_iterables[i] = new_msg_iterable
        del msg_sample
    else:
        msg_samples = None
    # store the data to disk to disk to save memory and safeguard against
    # any change made to the data during the training
    robust_pickle(path, "training_data_samples.pckl",
                  (x_samples, msg_samples, stop_messages))
    del x_samples
    del msg_samples
    # perform the training and gather snapshots
    prepare_training_inspection(flow=flow, path=path)
    try:
        if isinstance(flow, BiFlow):
            flow.train(data_iterables, msg_iterables, stop_messages, **kwargs)
        else:
            flow.train(data_iterables, **kwargs)
    except Exception:
        if debug:
            traceback.print_exc()
            print ("exception during training, " +
                   "inspecting up to failure point...")
            # create the last snapshot manually
            try:
                # if a normal mdp.Flow instance was given then this fails
                flow._bi_reset()
            except Exception:
                pass
            filename = (flow._snapshot_name_ + "_%d" % flow._snapshot_counter_
                        + PICKLE_EXT)
            robust_pickle(flow._snapshot_path_, filename, flow)
        else:
            raise
    remove_inspection_residues(flow)
    # reload data samples
    with open(os.path.join(path, "training_data_samples.pckl"), "rb") as sample_file:
        x_samples, msg_samples, stop_messages = pickle.load(sample_file)
    # create slideshow
    slideshow = inspect_training(snapshot_path=path,
                                 inspection_path=path,
                                 x_samples=x_samples,
                                 msg_samples=msg_samples,
                                 stop_messages=stop_messages,
                                 tracer=tracer,
                                 debug=debug, show_size=show_size,
                                 verbose=False)
    filename = os.path.join(path, "training_inspection.html")
    title = "Training Inspection"
    with open(filename, 'w') as html_file:
        html_file.write('<html>\n<head>\n<title>%s</title>\n' % title)
        html_file.write('<style type="text/css" media="screen">')
        html_file.write(standard_css())
        html_file.write('</style>\n</head>\n<body>\n')
        html_file.write('<h3>%s</h3>\n' % title)
        html_file.write(slideshow)
        html_file.write('</body>\n</html>')
    if open_browser:
        _open_custom_brower(open_browser, os.path.abspath(filename))
    return filename

def inspect_execution(flow, x, msg=None, target=None, path=None,
                      name=None, tracer=None, debug=False,
                      slide_style=None, show_size=False,
                      **kwargs):
    """Return the HTML code for an inspection slideshow of the execution
    and the return value of the execution (in a tuple).

    Note that the file into which the slideshow HTML is inserted must be in the
    snapshot_path.

    flow -- The flow for the execution.
    x, msg, target -- Data for the execution, msg and target can only be used
        for a BiFlow (default value is None).
    path -- Path were the slideshow will be stored, if None (default value)
        a temporary directory will be used.
    name -- Name string to be used for the slide files.
    tracer -- Instance of InspectionHTMLTracer, can be None for
        default class.
    debug -- If True (default is False) then any exception will be
        caught and the gathered data up to that point is returned in the
        normal way. This is useful for bimdp debugging.
    slide_style -- CSS code for the individual slides (when they are
        viewed as single HTML files), has no effect on the slideshow appearance.
    show_size -- Show the approximate memory footprint of all nodes.
    **kwargs -- Additional arguments for flow.execute can be specified
        as keyword arguments.
    """
    if path is None:
        path = tempfile.mkdtemp(prefix='MDP_')
    if not name:
        name = "execution_inspection"
    # create CSS file for the slides
    if not slide_style:
        slide_style = standard_css()
    robust_write_file(path=path, filename=STANDARD_CSS_FILENAME,
                      content=slide_style)
    del slide_style
    if not tracer:
        tracer = InspectionHTMLTracer()
        tracer._html_converter.flow_html_converter.show_size = show_size
    # create slides
    try:
        slide_filenames, slide_node_ids, section_ids, result = \
            tracer.trace_execution(path=path,
                                   trace_name=name,
                                   flow=flow,
                                   x=x, msg=msg, target=target,
                                   debug=debug,
                                   **kwargs)
    except TraceDebugException as debug_exception:
        if not debug_exception.result:
            return None
        traceback.print_exc()
        print ("exception during excecution, " +
               "create inspection up to failure point...")
        slide_filenames, slide_node_ids, section_ids = debug_exception.result
        result = None
    # create slideshow file
    if not slide_filenames:
        err = "For some reason no execution slides were generated."
        raise EmptyTraceException(err)
    if not section_ids:
        slideshow = ExecuteHTMLSlideShow(filenames=slide_filenames,
                                         node_ids=slide_node_ids,
                                         delay=500, delay_delta=100,
                                         loop=False)
    else:
        # after an exception the last section_id entry can be missing
        if len(section_ids) < len(slide_filenames):
            section_ids.append(section_ids[-1])
        slideshow = SectExecuteHTMLSlideShow(filenames=slide_filenames,
                                             node_ids=slide_node_ids,
                                             section_ids=section_ids,
                                             delay=500, delay_delta=100,
                                             loop=False)
    return str(slideshow), result

def show_execution(flow, x, msg=None, target=None, path=None, name=None,
                   tracer=None,
                   debug=False, show_size=False, open_browser=True,
                   **kwargs):
    """Write the inspection slideshow into an HTML file and open it in the
    browser.

    The return value is a tuple with the slideshow filename and the return
    value of the execution.

    flow -- The flow for the execution.
    x, msg, target -- Data for the execution, msg and target can only be used
        for a BiFlow (default value is None).
    path -- Path were the slideshow will be stored, if None (default value)
        a temporary directory will be used.
    name -- A name for the slideshow.
    tracer -- Instance of InspectionHTMLTracer, can be None for
        default class.
    debug -- If True (default is False) then any exception will be
        caught and the gathered data up to that point is returned in the
        normal way. This is useful for bimdp debugging.
    show_size -- Show the approximate memory footprint of all nodes.
    open_browser -- If True (default value) then the slideshow file is
        automatically opened in a webbrowser. One can also use string value
        with the browser name (for webbrowser.get) to request a specific
        browser.
    **kwargs -- Additional arguments for flow.execute can be specified
        as keyword arguments.
    """
    if path is None:
        path = tempfile.mkdtemp(prefix='MDP_')
    if not name:
        name = "execution_inspection"
        title = "Execution Inspection"
    else:
        title = "Execution Inspection: " + name
    filename = os.path.join(path, name + ".html")
    slideshow, result = inspect_execution(
                        flow=flow,
                        path=path,
                        x=x, msg=msg, target=target,
                        name=name,
                        tracer=tracer,
                        debug=debug,
                        show_size=show_size,
                        **kwargs)
    # inspect execution created the path if required, so no need to check here
    with open(filename, 'w') as html_file:
        html_file = open(filename, 'w')
        html_file.write('<html>\n<head>\n<title>%s</title>\n' % title)
        html_file.write('<style type="text/css" media="screen">')
        html_file.write(standard_css())
        html_file.write('</style>\n</head>\n<body>\n')
        html_file.write('<h3>%s</h3>\n' % title)
        html_file.write(slideshow)
        html_file.write('</body>\n</html>')
    if open_browser:
        _open_custom_brower(open_browser, os.path.abspath(filename))
    return filename, result
