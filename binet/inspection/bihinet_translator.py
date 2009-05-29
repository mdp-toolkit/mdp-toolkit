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

# Functions to define how the node parameters are represented in the
# HTML representation of a node.

def _get_html_sender(node):
    return ['<span class="bicolor">target: %s </span><br>' % str(node._target)]

def _get_html_binode(node):
    # or use node.__base__[1]?
    html_list = []
    for node_html_translator in NODE_HTML_TRANSLATORS[::-1]:
        # make sure that no circular calling happens
        if ((node_html_translator[0] is not BiNode) and
            isinstance(node, node_html_translator[0])):
            html_list += node_html_translator[1](node)
            break
    if node._stop_msg:
        html_list += ['<span class="bicolor">stop_msg: %s </span><br>' 
                      % str(node._stop_msg)]  
    return html_list
    
BINODE_HTML_TRANSLATORS = [
    (SenderBiNode, _get_html_sender),
]

NODE_HTML_TRANSLATORS = (mdp.hinet.NODE_HTML_TRANSLATORS +
                         BINODE_HTML_TRANSLATORS)


class BiNetHTMLTranslator(mdp.hinet.HiNetHTMLTranslator):
    """Special version of HiNetHTMLTranslator with BiNode support.
    
    All binet attributes are highligthed via the span.bicolor css tag.
    """
    
    def __init__(self, node_param_translators=NODE_HTML_TRANSLATORS,
                 show_size=False):
        super(BiNetHTMLTranslator, self).__init__(
                                node_param_translators=node_param_translators,
                                show_size=show_size)
        
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

