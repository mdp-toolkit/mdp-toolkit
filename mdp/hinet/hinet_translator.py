"""
Module to translate HiNet structures into other representations, like HTML.
"""
import tempfile
import os
import webbrowser
import cStringIO as StringIO

import mdp

import switchboard


class HiNetTranslator(object):
    """Generic translation class for HiNet flow.
    
    The dummy implementation in this base class turns the HiNet structure
    into nested lists of the basic nodes.
    """
    
    def __init__(self):
        """Initialize the internal variables."""
        super(HiNetTranslator, self).__init__()

    def _translate_flow(self, flow):
        """Translate the flow and return the translation."""
        flow_translation = []
        for node in flow:
            flow_translation.append(self._translate_node(node))
        return flow_translation
            
    def _translate_node(self, node):
        """Translate a node and return the translation.
        
        Depending on the type of the node this can be delegated to more
        specific methods.
        """
        if hasattr(node, "flow"):
            return self._translate_flownode(node)
        elif isinstance(node, mdp.hinet.CloneLayer):
            return self._translate_clonelayer(node)
        elif isinstance(node, mdp.hinet.SameInputLayer):
            return self._translate_sameinputlayer(node)
        elif isinstance(node, mdp.hinet.Layer):
            return self._translate_layer(node)
        else:
            return self._translate_standard_node(node)
        
    def _translate_flownode(self, flownode):
        """Translate a node containing a flow and return the translation.
        
        The internal nodes are translated recursively.
        
        Note that this method is used for translation whenever the node has
        a flow attribute. This flow attribute is then used for the iteration,
        so the node itself does not have to be an iterable.
        """
        flownode_translation = []
        for node in flownode.flow:
            flownode_translation.append(self._translate_node(node))
        return flownode_translation
    
    def _translate_layer(self, layer):
        """Translate a layer and return the translation.
        
        All the nodes in the layer are translated.
        """
        layer_translation = []
        for node in layer:
            layer_translation.append(self._translate_node(node))
        return layer_translation
    
    def _translate_clonelayer(self, layer):
        """Translate a CloneLayer and return the translation."""
        translated_node = self._translate_node(layer.node)
        return [translated_node] * len(layer)
    
    def _translate_sameinputlayer(self, layer):
        """Translate a SameInputLayer and return the translation."""
        return self._translate_layer(layer)

    def _translate_standard_node(self, node):
        """Translate a node and return the translation.
        
        This method is used when no specialized translation (like for FlowNodes
        or Layers) is required.
        """
        return node


## Specialized HTML Translator ##

# CSS for hinet representation.
#
# Warning: In nested tables the top table css overwrites the nested css if
#    they are specified like 'table.flow td' (i.e. all td's below this table).
#    So be careful about hiding/overriding nested td's.
#
# The tables "nodestruct" are used to separate the dimension values from 
# the actual node text.

HINET_STYLE = '''
table.flow {
    border-collapse: separate;
    padding: 3 3 3 3;
    border: 3px double;
    border-color: #003399;
}

table.flow table {
    width: 100%;
    margin-left: 2px;
    margin-right: 2px;
    border-color: #003399; 
}

table.flow td {
    padding: 1 5 1 5;
    border-style: none;
}

table.layer {
    border-collapse: separate;
    border: 2px dashed;
}

table.flownode {
    border-collapse: separate;
    border: 1px dotted;
}

table.nodestruct {
    border-style: none;
}

table.node {
    border-collapse: separate;
    border: 1px solid;
    border-spacing: 2px;
}

td.nodename {
    font-size: normal;
    text-align: center;
}

td.nodeparams {
    font-size: xx-small;
    text-align: left;
}

td.dim {
    font-size: xx-small;
    text-align: center;
    color: #008ADC;
}

span.memorycolor {
    color: #CCBB77;
}
'''


class  HTMLExtensionNode(mdp.ExtensionNode, mdp.Node):
    """Extension node for HTML representations of individual nodes."""
    
    extension_name = "html_representation"
    
    def html_representation(self):
        """Return an HTML representation of the node."""
        html_repr = self._html_representation()
        if type(html_repr) is str:
            return html_repr
        else:
            return " <br>\n".join(html_repr)
            
    def _html_representation(self):
        """Return either the final HTML code or a list of HTML lines."""
        return ""
    
    
@mdp.extension_method("html_representation",
                      switchboard.Rectangular2dSwitchboard,
                      "_html_representation")
def _rect2d_switchoard_html(self):
    lines = ['rec. field size (in channels): %d x %d = %d' % 
                (self.x_field_channels, self.y_field_channels,
                 self.x_field_channels * self.y_field_channels),
             '# of rec. fields (out channels): %d x %d = %d' %
                (self.x_out_channels, self.y_out_channels,
                 self.output_channels),
             'rec. field distances (in channels): (%d x %d)' %
                (self.x_field_spacing, self.y_field_spacing),
             'channel width: %d' % self.in_channel_dim]
    if self.x_unused_channels or self.y_unused_channels:
        lines.append('unused channels: (%d, %d)' %
                     (self.x_unused_channels, self.y_unused_channels))
    return lines

@mdp.extension_method("html_representation",
                      switchboard.DoubleRect2dSwitchboard,
                      "_html_representation")
def _double_rect2d_switchoard_html(self):
    lines = ['rec. field size (in channels): %d x %d = %d' % 
                (self.x_field_channels, self.y_field_channels,
                 self.x_field_channels * self.y_field_channels),
             '# of long row rec. fields (out channels): %d, %d' %
                (self.x_long_out_channels, self.y_long_out_channels),
             'total number of receptive fields: %d' % 
                self.output_channels,
             'channel width: %d' % self.in_channel_dim]
    if self.x_unused_channels or self.y_unused_channels:
        lines.append('unused channels: (%d, %d)' %
                     (self.x_unused_channels, self.y_unused_channels))
    return lines

@mdp.extension_method("html_representation",
                      switchboard.DoubleRhomb2dSwitchboard,
                      "_html_representation")
def _double_rhomb2d_switchoard_html(self):
    lines = ['rec. field size: %d' % self.diag_field_channels,
             '# of rec. fields (out channels): %d x %d = %d' %
                (self.x_out_channels, self.y_out_channels,
                 self.output_channels),
             'channel width: %d' % self.in_channel_dim]
    return lines

@mdp.extension_method("html_representation", mdp.nodes.SFA2Node,
                      "_html_representation")
def _sfa_html(self):
    return 'expansion dim: ' + str(self._expnode.output_dim)

@mdp.extension_method("html_representation", mdp.nodes.NormalNoiseNode,
                      "_html_representation")   
def _noise_html(self):
    return ['noise level: ' + str(self.noise_args[1]),
            'noise offset: ' + str(self.noise_args[0])]

@mdp.extension_method("html_representation", mdp.nodes.CutoffNode,
                      "_html_representation") 
def _cutoff_html(self):
    return ['lower bound: ' + str(self.lower_bound),
            'upper bound: ' + str(self.upper_bound)]

@mdp.extension_method("html_representation", mdp.nodes.HistogramNode,
                      "_html_representation")   
def _hist_html(self):
    return 'history data fraction: ' + str(self.hist_fraction)

@mdp.extension_method("html_representation", mdp.nodes.AdaptiveCutoffNode,
                      "_html_representation")
def _adap_html(self):
    return ['lower cutoff fraction: ' + str(self.lower_cutoff_fraction), 
            'upper cutoff fraction: ' + str(self.upper_cutoff_fraction), 
            'history data fraction: ' + str(self.hist_fraction)]
    

class NewlineWriteFile(object):
    """Decorator for file-like object.
    
    Adds a newline character to each line written with write().
    """
    
    def __init__(self, file_obj):
        """Wrap the given file-like object."""
        self.file_obj = file_obj
    
    def write(self, str_obj):
        """Write a string to the file object and append a newline character."""
        self.file_obj.write(str_obj + "\n")
        
    def __getattr__(self, attr):
        return getattr(self.file_obj, attr)
    
    
class HiNetHTMLTranslator(HiNetTranslator):
    """Specialized translator for HTML.
    
    Instead of relying on the return values the HTML lines are directly
    written to a provided file.
    """
    
    def __init__(self, show_size=False):
        """Initialize the HMTL translator.
        
        show_size -- Show the approximate memory footprint of all nodes.
        """
        self.show_size = show_size
        self._html_file = None
        
    def write_flow_to_file(self, flow, html_file):
        """Write the HTML translation of the flow into the provided file.
        
        Note that html_file file can be any file-like object with a write
        method.
        """
        self._html_file = NewlineWriteFile(html_file)
        self._translate_flow(flow)
        self._html_file = None
    
    # overwrite private methods
    
    @mdp.with_extension("html_representation")
    def _translate_flow(self, flow):
        """Translate the flow into HTML and write it into the internal file.
        
        Use write_flow_to_file instead of calling this method.
        """
        f = self._html_file
        self._open_node_env(flow, "flow")
        for node in flow:
            f.write('<tr><td>')
            self._translate_node(node)
            f.write('</td></tr>')
        self._close_node_env(flow, "flow")
        
    def _translate_flownode(self, flownode):
        f = self._html_file
        self._open_node_env(flownode, "flownode")
        for node in flownode.flow:
            f.write('<tr><td>')
            self._translate_node(node)
            f.write('</td></tr>')
        self._close_node_env(flownode, "flownode")
     
    def _translate_layer(self, layer):
        f = self._html_file
        self._open_node_env(layer, "layer")
        f.write('<tr>')
        for node in layer:
            f.write('<td>')
            self._translate_node(node)
            f.write('</td>')
        f.write('</tr>')
        self._close_node_env(layer)
        
    def _translate_clonelayer(self, layer):
        f = self._html_file
        self._open_node_env(layer, "layer")
        f.write('<tr><td class="nodename">')
        f.write(str(layer) + '<br><br>')
        f.write('%d repetitions' % len(layer))
        f.write('</td>')
        f.write('<td>')
        self._translate_node(layer.node)
        f.write('</td></tr>')
        self._close_node_env(layer)
        
    def _translate_sameinputlayer(self, layer):
        f = self._html_file
        self._open_node_env(layer, "layer")
        f.write('<tr><td colspan="%d" class="nodename">%s</td></tr>' %
                (len(layer), str(layer)))
        f.write('<tr>')
        for node in layer:
            f.write('<td>')
            self._translate_node(node)
            f.write('</td>')
        f.write('</tr>')
        self._close_node_env(layer)

    def _translate_standard_node(self, node):
        f = self._html_file
        self._open_node_env(node)
        f.write('<tr><td class="nodename">')
        f.write(str(node))
        f.write('</td></tr>')
        f.write('<tr><td class="nodeparams">')
        f.write(node.html_representation())
        f.write('</td></tr>')
        self._close_node_env(node)
        
    # helper methods for decoration
    
    def _open_node_env(self, node, type_id="node"):
        """Open the HTML environment for the node internals.
        
        node -- The node itself.
        type_id -- The id string as used in the CSS.
        """
        self._html_file.write('<table class="%s">' % type_id)
        self._write_node_header(node, type_id)
        
    def _write_node_header(self, node, type_id="node"):
        """Write the header content for the node into the HTML file."""
        f = self._html_file
        if not (type_id=="flow" or type_id=="flownode"):
            f.write('<tr><td class="dim">in-dim: %s</td></tr>' % 
                    str(node.input_dim))
        f.write('<tr><td>')
        f.write('<table class="nodestruct">')
    
    def _close_node_env(self, node, type_id="node"):
        """Close the HTML environment for the node internals.
        
        node -- The node itself.
        type_id -- The id string as used in the CSS.
        """
        f = self._html_file
        f.write('</table>')
        f.write('</td></tr>')
        if not (type_id=="flow" or type_id=="flownode"):
            f.write('<tr><td class="dim">out-dim: %s' % str(node.output_dim))
            if self.show_size:
                f.write('&nbsp;&nbsp;<span class="memorycolor">size: %s</span>' 
                        % mdp.utils.get_node_size_str(node))
            f.write('</td></tr>')
        f.write('</table>')
        

class HiNetXHTMLTranslator(HiNetHTMLTranslator):
    """Modified translator to create valid XHTML."""
    
    def write_flow_to_file(self, flow, xhtml_file):
        """Write the HTML translation of the flow into the provided file.
        
        Note that html_file file can be any file-like object with a write
        method.
        """
        # first write the normal HTML into a buffer
        html_file = StringIO.StringIO()
        self._html_file = NewlineWriteFile(html_file)
        self._translate_flow(flow)
        self._html_file = None
        # now translate it to XHTML and then write it to the file
        html_code = html_file.getvalue()
        html_code = html_code.replace('<br>', '<br/>')
        html_code = html_code.replace('&nbsp;', '&#160;')
        xhtml_file.write(html_code)
       

## Helper functions ##

def show_flow(flow, filename=None, title="MDP flow display",
              show_size=False, browser_open=True):
    """Write a flow into a HTML file, open it in the browser and
    return the file name.

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
    html_file.write(mdp.utils.BASIC_STYLE)
    html_file.write(HINET_STYLE)
    hinet_translator = mdp.hinet.HiNetHTMLTranslator(show_size=show_size)
    html_file.write('</style>\n</head>\n<body>\n')
    html_file.write('<h3>%s</h3>\n' % title)
    explanation = '(data flows from top to bottom)'
    html_file.write('<par class="explanation">%s</par>\n' % explanation)
    html_file.write('<br><br><br>\n')
    hinet_translator.write_flow_to_file(flow=flow, html_file=html_file)
    html_file.write('</body>\n</html>')
    html_file.close()
    if browser_open:
        webbrowser.open(filename)
    return filename
    
        
