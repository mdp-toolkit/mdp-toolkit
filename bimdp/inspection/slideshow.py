"""
Module for HTML trace slideshows.

The individual slides are the HTML files generated via the trace_inspection
module (the body of the HTML files is extracted and makes up a slide).
"""
from builtins import range


import os

from mdp.utils import HTMLSlideShow, SectionHTMLSlideShow


class ExecuteHTMLSlideShow(HTMLSlideShow):

    def __init__(self, filenames, node_ids, delay=500, delay_delta=100,
                  loop=False, **kwargs):
        """Return the complete HTML code for the slideshow.

        filenames -- Sequence of strings, containing the path for each slide.
        node_ids -- Sequence of the active node ids for each slide.
        """
        kwargs.update(vars())
        # create a list of the possible node ids
        unique_node_ids = list(set(node_ids))
        kwargs["unique_node_ids"] = unique_node_ids
        del kwargs["self"]
        super(ExecuteHTMLSlideShow, self).__init__(**kwargs)

    _SLIDESHOW_CSS_FILENAME = "trace_slideshow.css"

    @classmethod
    def slideshow_css(cls):
        css_filename = os.path.join(os.path.split(__file__)[0],
                                    cls._SLIDESHOW_CSS_FILENAME)
        with open(css_filename, 'r') as css_file:
            css = css_file.read()
        return css

    js_loadslide_template = r'''

    // maps slide index to active node id
    var slide_node_ids = $node_ids;
    // list of all node ids that are available
    var unique_node_ids = $unique_node_ids;

    $<js_loadhtml_template>

    that.loadSlide = function () {
        loadPage(slideselect[current_slide].value);
    }

    // is called by loadPage after the loading has happened
    function makeNodesClickable() {
        var i;
        for (i = 0; i < unique_node_ids.length; i += 1) {
            try {
                document.getElementById(unique_node_ids[i]).
                    addEventListener("click", nodeClickCallback, false);
            }
            catch (e) {
                // means that the requested node is added in a later training
                // phase and is therefore not yet in the DOM
            }
        }
    }

    function nodeClickCallback() {
        // TODO: use event.srcElement for IE (event.target for W3C)
        var node_id = this.id;
        // search for next occurance of this node id
        var i;
        for (i = current_slide + 1; i < slide_node_ids.length; i += 1) {
            if (slide_node_ids[i] === node_id) {
                current_slide = i;
                that.updateSlide();
                return;
            }
        }
        // alert("Node is not reached after this slide.");
    }

'''

    js_loadhtml_template = r'''
    /**
     * Code to load the body content from HTMl files and inject it.
     * inspired by http://www.xul.fr/ajax/responseHTML-attribute.html
     */

    // Extract body content from html content.
    function getBody(content) {
        var lowContent = content.toLowerCase();  // eliminate case sensitivity
        // deal with attributes
        var i_start = lowContent.indexOf("<body");
        if (i_start === -1) {
            return "";
        }
        i_start = lowContent.indexOf(">", i_start);
        if (i_start === -1) {
            return "";
        }
        var i_end = lowContent.lastIndexOf("</body>");
        if (i_end === -1) {
            i_end = lowContent.lastIndexOf("</html>");
        }
        // if no HTML then just grab everything till end.
        if (i_end === -1) {
            i_end = content.length;
        }
        return content.slice(i_start + 1, i_end);
    }

    // Return a XMLHttpRequest object (browser independent).
    function getXHR()
    {
        var request = false;
            try {
                request = new ActiveXObject('Msxml2.XMLHTTP');
            }
            catch (err2) {
                try {
                    request = new ActiveXObject('Microsoft.XMLHTTP');
                }
                catch (err3) {
                    try {
                        request = new XMLHttpRequest();
                    }
                    catch (err1) {
                        request = false;
                    }
                }
            }
        return request;
    }

    // Load an HTML page and inject the content.
    function loadPage(url) {
        var target = document.getElementById("html_display");
        var xhr = getXHR();
        xhr.onreadystatechange = function() {
            if(xhr.readyState == 4) {
                target.innerHTML = getBody(xhr.responseText);
                makeNodesClickable();
            }
        }
        xhr.open("GET", url, true);
        xhr.send(null);
    }
    '''

    # Note: We do not use an id prefix, since there is only one slideshow.
    html_bottom_template = r'''
<div id="html_display"></div>
'''


class SectExecuteHTMLSlideShow(SectionHTMLSlideShow, ExecuteHTMLSlideShow):
    """Execute slideshow with support for sections."""
    pass


class TrainHTMLSlideShow(SectionHTMLSlideShow, ExecuteHTMLSlideShow):

    def __init__(self, filenames, node_ids, index_table, **kwargs):
        """Return the complete HTML code for the slideshow.

        filenames -- Sequence of strings, containing the path for each slide.
        node_ids -- Sequence of the active node ids for each slide.
        index_table -- Nested lists with the index data generated by
            inspect_biflow_training (last slide indexed by node, phase, train
            and stop).
        """
        slideshow_id = self._get_random_id()
        n_nodes = len(index_table)
        n_phases = max([len(phase_indices)
                        for phase_indices in index_table])
        # create the table and mapping between slide index and phase and node
        train_id = 0  # id indexing phase, node and train or stop
        start_index = 0  # first slide index for the current phase
        end_index = 0  # last slide index for the current phase
        section_ids = []
        train_table = [[None for _ in range(n_nodes + 1)]
                       for _ in range(n_phases + 1)]
        # create labels for table
        train_table[0] = ['&nbsp;'] + ['node %d' % (i+1)
                                       for i in range(n_nodes)]
        for i_phase in range(n_phases):
            train_table[i_phase+1][0] = 'phase %d' % (i_phase + 1)
        for i_node in range(n_nodes):
            for i_phase in range(len(index_table[i_node])):
                end_index = index_table[i_node][i_phase][0]
                # train link stuff
                html_string = ('<span class="inactive_section" ' +
                        'id="%s_section_id_%d" ' % (slideshow_id, train_id) +
                        'onClick="%s.setSlide(%d);">train</span>' %
                            (slideshow_id, start_index) +
                        '&nbsp;')
                section_ids += [train_id,] * (end_index - start_index + 1)
                train_id += 1
                # stop link stuff
                start_index = end_index + 1
                end_index = index_table[i_node][i_phase][1]
                if start_index > end_index:
                    # this can happen due to an exception during training
                    start_index = end_index
                else:
                    html_string += ('<span class="inactive_section" ' +
                        'id="%s_section_id_%d" ' % (slideshow_id, train_id) +
                        'onClick="%s.setSlide(%d);">stop</span>' %
                            (slideshow_id, start_index))
                train_table[i_phase+1][i_node+1] = html_string
                section_ids += [train_id,] * (end_index - start_index + 1)
                train_id += 1
                start_index = end_index + 1
        kwargs["control_table"] = train_table
        kwargs["section_ids"] = section_ids
        kwargs["filenames"] = filenames
        kwargs.update(vars())
        del kwargs["self"]
        del kwargs["index_table"]
        super(TrainHTMLSlideShow, self).__init__(**kwargs)

    html_top_template = r'''
<table class="slideshow">
${{
for row in control_table:
    self.write('<tr>\n')
    for cell in row:
        self.write('<td> %s </td>\n' % cell)
    self.write('</tr>\n')
}}
</table>
<br>
'''

    html_controls_template = r'''
${{super(SectionHTMLSlideShow, self).html_controls_template(vars())}}
'''
