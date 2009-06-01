"""
Module with simple functions for the complete inspection procedure.
"""

import os
import webbrowser
import cPickle as pickle
import tempfile

import mdp.hinet as hinet
from mdp import numx

from ..biflow import BiFlow

from bihinet_translator import BINET_STYLE
from trace_inspection import (_trace_biflow_training,
                              BiNetTraceDebugException,
                              INSPECT_TRACE_STYLE, SLIDE_CSS_FILENAME,
                              TraceBiNetHTMLTranslator, HTMLTraceInspector,
                              prepare_training_inspection,
                              remove_inspection_residues)
from trace_slideshow import (INSPECT_SLIDESHOW_STYLE,
                             TrainHTMLSlideShow, ExecuteHTMLSlideShow,
                             SectExecuteHTMLSlideShow)
from utils import robust_write_file, robust_pickle, first_iterable_elem

# style for slides, used when the slides are not viewed in a slideshow
SLIDE_STYLE = (hinet.HINET_STYLE + BINET_STYLE +
               INSPECT_TRACE_STYLE)

# style for slideshow, can be used when embedding the slideshow
INSPECTION_STYLE = (hinet.HINET_STYLE + BINET_STYLE +
                    INSPECT_TRACE_STYLE + INSPECT_SLIDESHOW_STYLE)


def inspect_training(snapshot_path, x_samples, msg_samples=None,
                     stop_messages=None, inspection_path=None,
                     trace_inspector=None, debug=False,
                     slide_style=SLIDE_STYLE, show_size=False,
                     verbose=True):
    """Return the HTML code for an inspection slideshow of the training.
    
    This function must be used after the training was completed. Before the
    training prepare_training_inspection must have been called to create
    snapshots. After training one should call remove_inspection_residues. 
    
    Note that the file into which the returned slideshow HTML is inserted must
    be in the snapshot_path.
    
    snapshot_path -- Path were the flow training snapshots are stored.
    css_filename -- Filename of the CSS file for the slides.
    x_samples, msg_samples -- Lists with the input data for the training trace.
    stop_messages -- The stop msg for the training trace.
    inspection_path -- Path were the slides will be stored. If None (default
        value) then the snapshot_path is used.
    trace_inspector -- Instance of HTMLTraceInspector, can be None for
        default class.
    debug -- If True (default is False) then any exception will be
        caught and the gathered data up to that point is returned in the
        normal way. This is useful for binet debugging.
    slide_style -- CSS code for the individual slides (when they are
        viewed as single HTML files), has no effect on the slideshow appearance.
    show_size -- Show the approximate memory footprint of all nodes.
    verbose -- If True (default value) a status message is printed for each
        loaded snapshot.
    """
    if not inspection_path:
        inspection_path = snapshot_path
    # create CSS file for the slides
    css_filename = SLIDE_CSS_FILENAME
    robust_write_file(path=inspection_path, filename=css_filename,
                       content=slide_style)
    del slide_style
    # create slides
    try:
        slide_filenames, index_table = _trace_biflow_training(**vars())
    except BiNetTraceDebugException, debug_exception:
        slide_filenames, index_table = debug_exception.result
    if index_table is None:
        return None  # no snapshots were found
    # create slideshow
    slideshow = TrainHTMLSlideShow(filenames=slide_filenames,
                                   index_table=index_table,
                                   delay=500, delay_delta=100, loop=False)
    return str(slideshow)

def show_training(flow, data_iterables, msg_iterables=None, stop_messages=None,
                  path=None, trace_inspector=None, debug=False,
                  show_size=False, browser_open=True, **train_kwargs):
    """Perform both the flow training and the training inspection.
    
    The return value is the filename of the slideshow HTML file. 
    
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
    trace_inspector -- Instance of HTMLTraceInspector, can be None for
        default class.
    debug -- Ignore exception during training and try to complete the slideshow
        (default value is False).
    show_size -- Show the approximate memory footprint of all nodes.
    **train_kwargs -- Additional arguments for flow.train can be specified
        as keyword arguments.
    browser_open -- If True (default value) then the slideshow file is
        automatically opened in a webbrowser.
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
    if isinstance(flow, BiFlow):
        flow.train(data_iterables, msg_iterables, stop_messages, **train_kwargs)
    else:
        flow.train(data_iterables, **train_kwargs)
    remove_inspection_residues(flow)
    # reload data samples
    sample_file = open(os.path.join(path, "training_data_samples.pckl"), "rb")
    x_samples, msg_samples, stop_messages = pickle.load(sample_file)
    sample_file.close()
    # create slideshow
    slideshow = inspect_training(snapshot_path=path,
                                 inspection_path=path,
                                 x_samples=x_samples,
                                 msg_samples=msg_samples,
                                 stop_messages=stop_messages,
                                 debug=debug, show_size=show_size,
                                 verbose=False)
    slideshow_filename = os.path.join(path, "training_inspection.html")
    title = "Training Inspection"
    html_file = open(slideshow_filename, 'w')
    html_file.write('<html>\n<head>\n<title>%s</title>\n' % title)
    html_file.write('<style type="text/css" media="screen">')
    html_file.write(INSPECTION_STYLE)
    html_file.write(hinet.SHOW_FLOW_STYLE)
    html_file.write('</style>\n</head>\n<body>\n')
    html_file.write('<h3>%s</h3>\n' % title)
    html_file.write(slideshow)
    html_file.write('</body>\n</html>')
    html_file.close()
    if browser_open:
        webbrowser.open(slideshow_filename)
    return slideshow_filename

def inspect_execution(flow, x, msg=None, target=None, path=None, name=None,
                      trace_inspector=None, debug=False,
                      slide_style=SLIDE_STYLE, show_size=False):
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
    trace_inspector -- Optionally provide a custom HTMLTraceInspector instance.
    debug -- If True (default is False) then any exception will be
        caught and the gathered data up to that point is returned in the
        normal way. This is useful for binet debugging.
    slide_style -- CSS code for the individual slides (when they are
        viewed as single HTML files), has no effect on the slideshow appearance.
    show_size -- Show the approximate memory footprint of all nodes.
    """
    if path is None:
        path = tempfile.mkdtemp(prefix='MDP_')
    if not name:
        name = "execution_inspection"
    # create CSS file for the slides
    css_filename = SLIDE_CSS_FILENAME
    robust_write_file(path=path, filename=css_filename,
                      content=slide_style)
    del slide_style
    if not trace_inspector:
        trace_translator = TraceBiNetHTMLTranslator(show_size=show_size)
        trace_inspector = HTMLTraceInspector(
                                trace_translator=trace_translator,
                                css_filename=css_filename)
    # create slides
    try:
        slide_filenames, section_ids, result = trace_inspector.trace_execution(
                                                 path=path,
                                                 trace_name=name,
                                                 flow=flow,
                                                 x=x, msg=msg, target=target,
                                                 debug=debug)
    except BiNetTraceDebugException, debug_exception:
        if not debug_exception.result:
            return None
        slide_filenames, section_ids = debug_exception.result
        result = None
    # create slideshow file
    if not section_ids:
        slideshow = ExecuteHTMLSlideShow(filenames=slide_filenames,
                                         delay=500, delay_delta=100,
                                         loop=False)
    else:
        # after an exception the last section_id entry can be missing 
        if len(section_ids) < len(slide_filenames):
            section_ids.append(section_ids[-1])
        slideshow = SectExecuteHTMLSlideShow(filenames=slide_filenames,
                                             section_ids=section_ids,
                                             delay=500, delay_delta=100,
                                             loop=False)
    return str(slideshow), result

def show_execution(flow, x, msg=None, target=None, path=None, name=None,
                   trace_inspector=None, debug=False, show_size=False,
                   browser_open=True):
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
    trace_inspector -- Optionally provide a custom HTMLTraceInspector instance.
    debug -- If True (default is False) then any exception will be
        caught and the gathered data up to that point is returned in the
        normal way. This is useful for binet debugging.
    show_size -- Show the approximate memory footprint of all nodes.
    browser_open -- If True (default value) then the slideshow file is
        automatically opened in a webbrowser.
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
                        trace_inspector=trace_inspector,
                        debug=debug,
                        show_size=show_size)
    # inspect execution created the path if required, so no need to check here
    html_file = open(filename, 'w')
    html_file.write('<html>\n<head>\n<title>%s</title>\n' % title)
    html_file.write('<style type="text/css" media="screen">')
    html_file.write(INSPECTION_STYLE)
    html_file.write(hinet.SHOW_FLOW_STYLE)
    html_file.write('</style>\n</head>\n<body>\n')
    html_file.write('<h3>%s</h3>\n' % title)
    html_file.write(slideshow)
    html_file.write('</body>\n</html>')
    html_file.close()
    if browser_open:
        webbrowser.open(filename)
    return filename, result
