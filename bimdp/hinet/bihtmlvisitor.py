"""
BiNet version of the htmlvisitor hinet module to convert a flow into HTML.
"""
from builtins import str

import tempfile
import os
import webbrowser

import mdp

from bimdp import BiNode
from bimdp.nodes import SenderBiNode
from bimdp.hinet import CloneBiLayer


class BiHiNetHTMLVisitor(mdp.hinet.HiNetHTMLVisitor):
    """Special version of HiNetHTMLVisitor with BiNode support.

    All bimdp attributes are highligthed via the span.bicolor css tag.
    """

    _BIHINET_STYLE = """
    span.bicolor {
        color: #6633FC;
    }
    """

    @classmethod
    def hinet_css(cls):
        """Return the standard CSS string.

        The CSS should be embedded in the final HTML file.
        """
        css = super(BiHiNetHTMLVisitor, cls).hinet_css()
        return css + cls._BIHINET_STYLE

    def _translate_clonelayer(self, clonelayer):
        """This specialized version checks for CloneBiLayer."""
        f = self._file
        self._open_node_env(clonelayer, "layer")
        f.write('<tr><td class="nodename">')
        f.write(str(clonelayer) + '<br><br>')
        f.write('%d repetitions' % len(clonelayer))
        if isinstance(clonelayer, CloneBiLayer):
            f.write('<br><br>')
            f.write('<span class="bicolor" style="font-size: x-small">')
            f.write('use copies: %s</span>' % str(clonelayer.use_copies))
        f.write('</td>')
        f.write('<td>')
        self._visit_node(clonelayer.nodes[0])
        f.write('</td></tr>')
        self._close_node_env(clonelayer)

    def _write_node_header(self, node, type_id="node"):
        """Write the header content for the node into the HTML file."""
        f = self._file
        if type_id == "flow":
            pass
        elif type_id == "flownode":
            if isinstance(node, BiNode):
                f.write('<tr><td class="dim">')
                f.write('<span class="bicolor">id: %s</span>' % node._node_id)
                f.write('</td></tr>')
        else:
            f.write('<tr><td class="dim">in-dim: %s' % str(node.input_dim))
            if isinstance(node, BiNode):
                f.write('&nbsp;&nbsp;<span class="bicolor">id: %s</span>'
                        % node._node_id)
            f.write('</td></tr>')
        f.write('<tr><td>')
        f.write('<table class="nodestruct">')


@mdp.extension_method("html", SenderBiNode)
def _html_representation(self):
    return ('<span class="bicolor">recipient id: %s </span><br>' %
            str(self._recipient_id))

#@mdp.extension_method("html_representation", BiNode)
#def _html_representation(self):
#    html_repr = super(BiNode, self).html_representation()
#    if self._stop_result:
#        html_repr = [html_repr,
#                     '<span class="bicolor">stop_msg: %s </span><br>'
#                     % str(self._stop_result)]
#    return html_repr

## Helper functions ##

def show_biflow(flow, filename=None, title="MDP flow display",
              show_size=False, browser_open=True):
    """Write a flow with BiMDP nodes into a HTML file, open it in the browser
    and return the file name. Compared the the non-bi function this provides
    special decoration for BiNode attributes.

    flow -- The flow to be shown.
    filename -- Filename for the HTML file to be created. If None
                a temporary file is created.
    title -- Title for the HTML file.
    show_size -- Show the approximate memory footprint of all nodes.
    browser_open -- If True (default value) then the slideshow file is
        automatically opened in a webbrowser.
    """
    if filename is None:
        (fd, filename) = tempfile.mkstemp(suffix=".html", prefix="MDP_")
        html_file = os.fdopen(fd, 'w')
    else:
        html_file = open(filename, 'w')
    html_file.write('<html>\n<head>\n<title>%s</title>\n' % title)
    html_file.write('<style type="text/css" media="screen">')
    html_file.write(mdp.utils.basic_css())
    html_file.write(BiHiNetHTMLVisitor.hinet_css())
    html_file.write('</style>\n</head>\n<body>\n')
    html_file.write('<h3>%s</h3>\n' % title)
    explanation = '(data flows from top to bottom)'
    html_file.write('<par class="explanation">%s</par>\n' % explanation)
    html_file.write('<br><br><br>\n')
    converter = BiHiNetHTMLVisitor(html_file, show_size=show_size)
    converter.convert_flow(flow=flow)
    html_file.write('</body>\n</html>')
    html_file.close()
    if browser_open:
        webbrowser.open(os.path.abspath(filename))
    return filename
