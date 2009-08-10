"""
Module to translate a binet node structure.

Currently we only implement the translation into HTML. This is done via the
HiNetHTML class.
"""

import mdp

from ..binode import BiNode
from ..binodes import SenderBiNode
from ..bihinet import CloneBiLayer

# TODO: use <pre>   </pre> for whitespaces?

BINET_STYLE = """
span.bicolor {
    color: #6633FC;
}
"""

@mdp.extension_method("html_representation", SenderBiNode)
def _html_representation(self):
    return '<span class="bicolor">target: %s </span><br>' % str(self._target)

#@mdp.extension_method("html_representation", BiNode)
#def _html_representation(self):
#    html_repr = super(BiNode, self).html_representation()
#    if self._stop_msg:
#        html_repr = [html_repr,
#                     '<span class="bicolor">stop_msg: %s </span><br>' 
#                     % str(self._stop_msg)]  
#    return html_repr
    

class BiNetHTMLTranslator(mdp.hinet.HiNetHTMLTranslator):
    """Special version of HiNetHTMLTranslator with BiNode support.
    
    All binet attributes are highligthed via the span.bicolor css tag.
    """
    
    def __init__(self, show_size=False):
        super(BiNetHTMLTranslator, self).__init__(show_size=show_size)
        
    def _translate_clonelayer(self, clonelayer):
        """This specialized version checks for CloneBiLayer."""
        f = self._html_file
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
        self._translate_node(clonelayer.nodes[0])
        f.write('</td></tr>')
        self._close_node_env(clonelayer)
        
    def _write_node_header(self, node, type_id="node"):
        """Write the header content for the node into the HTML file."""
        f = self._html_file
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

