"""
Package to inspect biflow training or execution by creating an HTML slideshow.
"""

from .tracer import (
    InspectionHTMLTracer, TraceHTMLConverter, TraceHTMLVisitor,
    TraceDebugException, inspection_css,
    prepare_training_inspection, remove_inspection_residues,
)
from .slideshow import (
    TrainHTMLSlideShow, SectExecuteHTMLSlideShow, ExecuteHTMLSlideShow
)
from .facade import (
    standard_css, EmptyTraceException,
    inspect_training, show_training, inspect_execution, show_execution
)

del tracer
del slideshow
del facade

from mdp.utils import fixup_namespace
fixup_namespace(__name__, None,
                ('tracer',
                 'slideshow',
                 'facade',
                 ))
del fixup_namespace
