"""
Module for HTML trace slideshows.

The individual slides are the HTML files generated via the trace_inspection
module (the body of the HTML files is extracted and makes up a slide).
"""

from mdp.utils import HTMLSlideShow, SectionHTMLSlideShow

INSPECT_SLIDESHOW_STYLE = """
div.slideshow {
    text-align: center;
}

table.slideshow, table.slideshow td, table.slideshow th {
    border-collapse: collapse;
    padding: 1 2 1 2;
    font-size: small;
    border: 1px solid;
}

table.slideshow {
    border: 2px solid;
    margin: 0 auto;
}

table.slideshow td {
    text-align: center;
}

/* style for slideshow with sections (like for training) */

span.inactive_section:hover {
    color: #6666FF;
}

span.active_section {
    color: #0000EE;
    background-color: #55FF55;
    cursor: pointer;
    font-weight: bold;
}

span.active_section:hover {
    color: #6666FF;
}
"""

class ExecuteHTMLSlideShow(HTMLSlideShow):
    
    js_loadslide_template = r'''
    
    $<js_ajax_template>
    
    function loadSlide() {
        loadWholePage(document.slideform.slide[current_slide].value);
    }
    '''
    
    js_ajax_template = r'''
    /**
        Content of file responseHTML.js 
        from http://www.xul.fr/ajax/responseHTML-attribute.html
    
        responseHTML
        (c) 2007-2008 xul.fr        
        Licence Mozilla 1.1
        
        Searches for body, extracts and return the content
        New version contributed by users
    */
    
    function getBody(content) {
       var test = content.toLowerCase();  // to eliminate case sensitivity
       var x = test.indexOf("<body");
       if(x == -1) return "";
       x = test.indexOf(">", x);
       if(x == -1) return "";
       var y = test.lastIndexOf("</body>");
       if(y == -1) y = test.lastIndexOf("</html>");
       // If no HTML then just grab everything till end.
       if(y == -1) y = content.length;
       return content.slice(x + 1, y);   
    } 
    
    /**
        Loads a HTML page
        Put the content of the body tag into the current page.
        Arguments:
            url of the other HTML page to load
            id of the tag that has to hold the content
    */        
    
    function loadHTML(url, fun, storage, param) {
        var xhr = createXHR();
        xhr.onreadystatechange = function()
        { 
            if(xhr.readyState == 4)
            {
                //if(xhr.status == 200)
                {
                    storage.innerHTML = getBody(xhr.responseText);
                    fun(storage, param);
                }
            } 
        }; 
        xhr.open("GET", url, true);
        xhr.send(null); 
    } 
    
    /**
        Callback
        Assign directly a tag
    */        
    
    function processHTML(temp, target) {
        target.innerHTML = temp.innerHTML;
    }
    
    function loadWholePage(url) {
        var y = document.getElementById("storage");
        var x = document.getElementById("displayed");
        loadHTML(url, processHTML, x, y);
    }    
    
    // Create responseHTML for access by DOM's methods
    function processByDOM(responseHTML, target) {
        target.innerHTML = "Extracted by id:<br />";
        var message =
            responseHTML.getElementsByTagName("div").item(1).innerHTML;
        target.innerHTML += message;
        target.innerHTML += "<br />Extracted by name:<br />";
        message = responseHTML.getElementsByTagName("form").item(0);
        target.innerHTML += message.dyn.value;
    }
    
    function accessByDOM(url) {
        var responseHTML = document.getElementById("storage");
        var y = document.getElementById("displayed");
        loadHTML(url, processByDOM, responseHTML, y);
    }    
    
    /**
        Content of file ajax.js 
        from http://www.xul.fr/ajax/responseHTML-attribute.html
    */
    
    function createXHR() 
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
    '''
    
    html_bottom_template = r'''
    <div id="storage" style="display:none;">
    </div>
    <div id="displayed">
    </div>
    '''


class SectExecuteHTMLSlideShow(SectionHTMLSlideShow, ExecuteHTMLSlideShow):
    """Execute slideshow with support for sections."""
    pass

    
class TrainHTMLSlideShow(SectionHTMLSlideShow, ExecuteHTMLSlideShow):
    
    def __init__(self, filenames, index_table, delay=500, delay_delta=100,
                 loop=True, **kwargs):
        """Return the complete HTML code for the slideshow.
        
        filenames -- Sequence of strings, containing the path for each image.
        index_table -- Nested lists with the index data generated by 
            inspect_biflow_training (last slide indexed by node, phase, train 
            and stop).
        delay - Delay between slides in ms (default 100).
        delay_delta - Step size for increasing or decreasing the delay,
            or use the (default) value None for no delay control.
        loop -- If True continue with first slide when the last slide is
            reached during the automatic slideshow (default is False).
        """
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
        train_table[0] = ['&nbsp;'] + ['layer %d' % (i+1)
                                       for i in range(n_nodes)]
        for i_phase in range(n_phases):
            train_table[i_phase+1][0] = 'phase %d' % (i_phase + 1)
        for i_node in range(n_nodes):
            for i_phase in range(len(index_table[i_node])):
                end_index = index_table[i_node][i_phase][0]
                # train link stuff
                html_string = ('<span class="inactive_section" ' +
                        'id="section_id_%d" ' % train_id +
                        'onClick="setSlide(%d);">train</span>' % start_index +
                        '&nbsp;')
                section_ids += [train_id,] * (end_index - start_index + 1) 
                train_id += 1
                # stop link stuff
                start_index = end_index + 1
                end_index = index_table[i_node][i_phase][1]
                html_string += ('<span class="inactive_section" ' +
                        'id="section_id_%d" ' % train_id +
                        'onClick="setSlide(%d);">stop</span>' % start_index)
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
