"""
Package to inspect biflow training or execution by creating an HTML slideshow.

To inspect the training of a BiFlow one must use prepare_training_inspection
before the training. Snapshots are then stored during training. After training
one should call remove_inspection_residues. Now one can create a slideshow with
training_slideshow.

Inspecting a BiFlow execution is very simple, one can either use
inspect_execution or the simpler show_execution.
"""


from bihinet_translator import (BINET_STYLE, BiNetHTMLTranslator)
from trace_inspection import (
    prepare_training_inspection, remove_inspection_residues,
    BiNetTraceDebugException, HTMLTraceInspector, TraceBiNetHTMLTranslator,
    _trace_biflow_training, INSPECT_TRACE_STYLE, NODE_TRACE_METHOD_NAMES,
    BINODE_TRACE_METHOD_NAMES
)
from trace_slideshow import (
    TrainHTMLSlideShow, SectExecuteHTMLSlideShow, ExecuteHTMLSlideShow
)
from facade import (
    INSPECTION_STYLE, SLIDE_CSS_FILENAME, inspect_training,
    show_training, inspect_execution, show_execution
)

del bihinet_translator
del trace_inspection
del trace_slideshow
del facade