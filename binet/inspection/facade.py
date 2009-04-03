"""
Module with simple functions for the complete inspection procedure.
"""

import os
import webbrowser
import cPickle as pickle

import mdp.hinet as hinet

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

# TODO: do not loop slideshow

# style for slides, used when the slides are not viewed in a slideshow
SLIDE_STYLE = (hinet.HINET_STYLE + BINET_STYLE +
               INSPECT_TRACE_STYLE)

# style for slideshow, can be used when embedding the slideshow
INSPECTION_STYLE = (hinet.HINET_STYLE + BINET_STYLE +
                    INSPECT_TRACE_STYLE + INSPECT_SLIDESHOW_STYLE)


# used to create the CSS file for all slides and the slideshow


def inspect_training(snapshot_path, inspection_path,
                     x_samples, msg_samples=None, stop_messages=None,
                     trace_inspector=None, debug=False,
                     slide_style=SLIDE_STYLE, show_size=False):
    """Return the HTML code for an inspection slideshow of the training.
    
    This function must be used after the training was completed. Before the
    training prepare_training_inspection must have been called to create
    snapshots. After training one should call remove_inspection_residues. 
    
    Note that the file into which the returned slideshow HTML is inserted must
    be in the snapshot_path.
    
    snapshot_path -- Path were the flow training snapshots are stored.
    inspection_path -- Path were the slides are stored.
    css_filename -- Filename of the CSS file for the slides.
    x_samples, msg_samples -- Lists with the input data for the training trace.
    stop_messages -- The stop msg for the training trace.
    trace_inspector -- Instance of HTMLTraceInspector, can be None for
        default class.
    debug -- If True (default is False) then any exception will be
        caught and the gathered data up to that point is returned in the
        normal way. This is useful for binet debugging.
    slide_style -- CSS code for the individual slides (when they are
        viewed as single HTML files), has no effect on the slideshow appearance.
    show_size -- Show the approximate memory footprint of all nodes.
    """
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

def train_with_inspection(flow, snapshot_path, data_iterators,
                         msg_iterators=None, stop_messages=None,
                         debug=False, show_size=False,
                         **train_kwargs):
    """Perform both the flow training and the training inspection.
    
    This function is more convenient than inspect_training since it includes
    all required steps, but it is also less customizable. After everything
    is complete the inspection slideshow is opened in the browser.
    """
    # get first part of data iterators as sample data for inspection
    x_samples = []
    for i, data_iterator in enumerate(data_iterators):
        x_sample, new_data_iterator = first_iterable_elem(data_iterator)
        x_samples.append(x_sample)
        data_iterators[i] = new_data_iterator
    del x_sample
    if msg_iterators:
        msg_samples = []
        for i, msg_iterator in enumerate(msg_iterators):
            msg_sample, msg_sample = first_iterable_elem(msg_iterator)
            msg_samples.append(msg_sample)
            msg_iterators[i] = msg_sample
        del msg_sample
    else:
        msg_samples = None
    # store the data to disk to disk to save memory and safeguard against
    # any change made to the data during the training
    robust_pickle(snapshot_path, "training_data_samples.pckl",
                  (x_samples, msg_samples, stop_messages))
    del x_samples
    del msg_samples
    # perform the training and gather snapshots 
    prepare_training_inspection(flow=flow, path=snapshot_path)
    if isinstance(flow, BiFlow):
        flow.train(data_iterators, msg_iterators, stop_messages, **train_kwargs)
    else:
        flow.train(data_iterators, **train_kwargs)
    remove_inspection_residues(flow)
    # reload data samples
    sample_file = open(os.path.join(snapshot_path,
                                    "training_data_samples.pckl"))
    x_samples, msg_samples, stop_messages = pickle.load(sample_file)
    sample_file.close()
    # create slideshow
    slideshow = inspect_training(snapshot_path=snapshot_path, 
                                 inspection_path=snapshot_path,
                                 x_samples=x_samples,
                                 msg_samples=msg_samples,
                                 stop_messages=stop_messages,
                                 debug=debug, show_size=show_size)
    slideshow_filename = os.path.join(snapshot_path,
                                      "training_inspection.html")
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
    webbrowser.open(slideshow_filename)
    return slideshow_filename

def inspect_execution(flow, inspection_path, x, msg=None, name=None,
                      trace_inspector=None, debug=False,
                      slide_style=SLIDE_STYLE, show_size=False):
    """Return the HTML code for an inspection slideshow of the execution
    and the return value of the execution (in a tuple).
    
    Note that the file into which the slideshow HTML is inserted must be in the
    snapshot_path.
    
    x, msg -- Data for the execution.
    name -- Name string to be used for the slide files.
    trace_inspector -- Instance of HTMLTraceInspector, can be None for
        default class.
    debug -- If True (default is False) then any exception will be
        caught and the gathered data up to that point is returned in the
        normal way. This is useful for binet debugging.
    slide_style -- CSS code for the individual slides (when they are
        viewed as single HTML files), has no effect on the slideshow appearance.
    show_size -- Show the approximate memory footprint of all nodes.
    """
    if not name:
        name = "execution_inspection"
    # create CSS file for the slides
    css_filename = SLIDE_CSS_FILENAME
    robust_write_file(path=inspection_path, filename=css_filename,
                       content=slide_style)
    del slide_style
    if not trace_inspector:
        trace_translator = TraceBiNetHTMLTranslator(show_size=show_size)
        trace_inspector = HTMLTraceInspector(
                                trace_translator=trace_translator,
                                css_filename=css_filename)
    # create slides
    try:
        slide_filenames, section_ids, y = trace_inspector.trace_execution(
                                                 path=inspection_path,
                                                 trace_name=name,
                                                 flow=flow,
                                                 x=x, msg=msg, debug=debug)
    except BiNetTraceDebugException, debug_exception:
        if not debug_exception.result:
            return None
        slide_filenames, section_ids = debug_exception.result
        y = None
    # create slideshow file
    if not section_ids:
        slideshow = ExecuteHTMLSlideShow(filenames=slide_filenames,
                                         delay=500, delay_delta=100,
                                         loop=False)
    else:
        slideshow = SectExecuteHTMLSlideShow(filenames=slide_filenames,
                                             section_ids=section_ids,
                                             delay=500, delay_delta=100,
                                             loop=False)
    return str(slideshow), y

def show_execution(flow, inspection_path, x, msg=None, name=None,
                   trace_inspector=None, debug=False, show_size=False):
    """Write the inspection slideshow into an HTML file and open it in the
    browser. The return value is the return value of the execution.

    flow -- The flow to be shown.
    filename -- Filename for the HTML file to be created.
    title -- Title for the HTML file.
    show_size -- Show the approximate memory footprint of all nodes.
    """
    if not name:
        name = "execution_inspection"
        title = "Execution Inspection"
    else:
        title = "Execution Inspection: " + name
    filename = os.path.join(inspection_path, name + ".html")
    html_file = open(filename, 'w')
    html_file.write('<html>\n<head>\n<title>%s</title>\n' % title)
    html_file.write('<style type="text/css" media="screen">')
    html_file.write(INSPECTION_STYLE)
    html_file.write(hinet.SHOW_FLOW_STYLE)
    html_file.write('</style>\n</head>\n<body>\n')
    html_file.write('<h3>%s</h3>\n' % title)
    slideshow, y = inspect_execution(
                        flow=flow,
                        inspection_path=inspection_path,
                        x=x, msg=msg,
                        name=name,
                        trace_inspector=trace_inspector,
                        debug=debug,
                        show_size=show_size)
    html_file.write(slideshow)
    html_file.write('</body>\n</html>')
    html_file.close()
    webbrowser.open(filename)
    return y
