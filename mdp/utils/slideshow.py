"""
Module for HTML slideshows.

It uses the templating library 'Templet'. 

The slideshow base class HTMLSlideShow does not display anything, but can
be used to derive custom slideshows like in BiNet. 

The JavaScript slideshow code in this module was originally inspired by a
slideshow script found at
http://javascript.internet.com/miscellaneous/image-slideshow.html
(which in turn seems to be based on something from http://www.ricocheting.com)
"""

import random
import tempfile
import os
import webbrowser

import templet

# basic default style for MDP generated HTML files
BASIC_STYLE = '''
html, body {
    font-family: sans-serif;
    font-size: normal;
    text-align: center;
}

h1, h2, h3, h4 {
    color: #003399;
}

par.explanation {
    color: #003399;
    font-size: small;
}

table.flow {
    margin-left:auto;
    margin-right:auto;
}
'''

# style for slideshow control table
SLIDESHOW_STYLE = '''
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

span.inactive_section {
    color: #0000FF;
    cursor: pointer;
}

span.inactive_section:hover {
    color: #6666FF;
}

span.active_section {
    color: #0000FF;
    background-color: #BBDDFF;
    cursor: pointer;
}

span.active_section:hover {
    color: #6666FF;
}
'''


class HTMLSlideShow(templet.Template):
    """Abstract slideshow base class.
    
    It does not display anything, but can be adapted by overriding
    some of the templating attributes. See ImageHTMLSlideShow for an example.
    """
    
    def __init__(self, title=None, delay=100, delay_delta=None,
                 loop=True, id=None, **kwargs):
        """Return the complete HTML code for the slideshow.
        
        title -- Optional slideshow title (for defualt None not title is shown).
        delay - Delay between slides in ms (default 100).
        delay_delta - Step size for increasing or decreasing the delay,
            or use the (default) value None for no delay control.
        loop -- If True continue with first slide when the last slide is
            reached during the automatic slideshow (default is False).
        id -- String with the id used for the JS closure, and this is also the
            id of the div with the slideshow (so it can be used by CSS) and it
            is used as a prefix for the HTML elements.
            If the value is None (default) then a random id is used.
        """
        # translate boolean variable into JS format
        if loop:
            loop = "true"
        else:
            loop = "false"
        if id is None:
            id = self._get_random_id()
        self.id = id
        kwargs.update(vars())
        del kwargs["self"]
        super(HTMLSlideShow, self).__init__(**kwargs)
    
    def _get_random_id(self):
        """Factory method for random slideshow id."""
        return "slideshow%d" % random.randint(10000, 99999)
        
   
    template = r'''
    <script language="JavaScript">
    <!-- Begin
    
    // use closure to create object, the id is the only global variable
    var $id = function () {

        var that = {};
        
        var current_slide = 0; // current slide index
        var show_delay = $delay; // delay in milliseconds
        var loop_slideshow = $loop; // loop in auto mode
        // shortcuts to form elements, initialized in onLoad
        var slideform;
        var slideselect;
        
        $<js_controls_template>
        $<js_loadslide_template>
        $<js_update_template>
        $<js_onload_template>
        $<js_keyboard_shortcuts_template>
        
        that.onSelectorChange = function () {
            current_slide = slideselect.selectedIndex;
            that.updateSlide();
        }
        
        that.next = function () {
            if (slideselect[current_slide+1]) {
                current_slide += 1;
                that.updateSlide();        
            }
            else {
                that.first();
            }
        }
        
        that.previous = function () {
            if (current_slide-1 >= 0) {
                current_slide -= 1;
                that.updateSlide();
            }
            else {
                that.last();
            }
        }
        
        that.first = function () {
            current_slide = 0;
            that.updateSlide();
        }
        
        that.last = function () {
            current_slide = slideselect.length-1;
            that.updateSlide();
        }
        
        // start or stop the slideshow
        that.startstop = function (text) {
            if (text === "Start") {
                slideform.startbutton.value = "Stop";
                if (!loop_slideshow &&
                    (current_slide === slideselect.length-1)) {
                    // restart slideshow
                    current_slide = 0;
                }
                that.showAuto();
            } else {
                slideform.startbutton.value = "Start";
            }
        }
        
        // continuously show the slideshow
        that.showAuto = function () {
            if (slideform.startbutton.value == "Stop") {
                if (current_slide == slideselect.length-1) {
                    if (loop_slideshow) {
                        current_slide = 0;
                        that.updateSlide();
                    } else {
                        slideform.startbutton.value = "Start";
                    }
                } else {
                    current_slide = current_slide+1;
                    that.updateSlide();
                }
                window.setTimeout("$id.showAuto()", show_delay);
            }
        }
    
        // end of closure, return created object 
        return that;
    }();
    
    //  End -->
    </script>
    
    <div class="slideshow" id="$id">
    
    $<html_top_template>
    
    <form name=${id}_slideform>
    <table class="slideshow">
    ${{
    if title:
        self.write('<tr><td><b> %s </b></td></tr>' % title)
    }}
    
    $<html_box_template>
    
    <tr><td>
    <select name="${id}_slideselect" onChange="$id.onSelectorChange();">
    <option value="${filenames[0]}" selected>${filenames[0]}
    ${{
    for filename in filenames[1:]:
        self.write('<option value="%s">%s\n' % (filename, filename))
    }}
    </select>
    </td></tr>
    <tr><td>

    $<html_buttons_template>

    </td></tr>
    
    $<html_controls_template>
    
    </table>
    </form>
    
    $<html_bottom_template>
    
    </div>
    
    <SCRIPT LANGUAGE="JavaScript">
    <!-- Begin
    $id.onLoad();
    //  End -->
    </script>
    '''
    
    js_controls_template = r'''
        // step size for in- or decreasing the delay
        var delay_delta = $delay_delta; 
        
        that.slower = function () {
            show_delay += delay_delta;
            slideform.${id}_delaytext.value = show_delay.toString();
        }
        
        that.faster = function (text) {
            show_delay -= delay_delta;
            if (show_delay < 0) {
                show_delay = 0;
            }
            slideform.${id}_delaytext.value = show_delay.toString();
        }
        
        that.changeDelay = function () {
            var new_delay = parseInt(slideform.${id}_delaytext.value, 10);
            if (new_delay < 0) {
                 new_delay = 0;
                }
            show_delay = new_delay;
            slideform.${id}_delaytext.value = new_delay.toString();
        }
    '''
    
    js_update_template = r'''
        that.updateSlide = function () {
            slideselect.selectedIndex = current_slide;
            that.loadSlide();
        }
    '''
    
    # overwrite this to implement the actual slide change
    js_loadslide_template = r'''
        that.loadSlide = function () {
        }
    '''
    
    js_onload_template = r'''
        that.onLoad = function () {
            current_slide = slideselect.selectedIndex;
            that.updateSlide();
            slideform = document.${id}_slideform;
            slideselect = slideform.${id}_slideselect;
            slideform.${id}_delaytext.value = show_delay.toString();
        }
    '''
    
    # TODO: how to deal with shortcuts for multiple slideshows?
    #    simply repeat, so that the last slideshow is accessible
    
    # define keyboard shortcuts
    js_keyboard_shortcuts_template = r'''
    document.onkeydown = function(e) {
        if (!e.ctrlKey) {  // control key must be pressed
            return;
        }
        else if (e.which == 37) { // left key
            document.getElementById("prevButton").click();
        }
        else if(e.which == 39) { // right key
            document.getElementById("nextButton").click();
        }
        else if(e.which == 38) { // up key
            document.getElementById("firstButton").click();
        }
        else if(e.which == 40) { // down key
            document.getElementById("lastButton").click();
        }
        else if(e.which == 45) { // insert key
            document.getElementById("startButton").click();
        }
    }
    '''
    
    html_buttons_template = r'''
    <input type=button onClick="$id.first();" value="|<<" title="beginning"
        id="${id}_firstButton">
    <input type=button onClick="$id.previous();" value="<" title="previous"
        id="${id}_prevButton">
    <input type=button name="startbutton" onClick="$id.startstop(this.value);"
        value="Start" title="autoplay" id="${id}_startButton">
    <input type=button onClick="$id.next();" value=">" title="next"
        id="${id}_nextButton">
    <input type=button onClick="$id.last();" value=">>|" title="end"
        id="${id}_lastButton">
    '''
    
    html_controls_template = r'''
    ${{
    if delay is not None:
        self.write('<tr><td>\n')
        self.html_delay_template(vars())
        self.write('</td></tr>\n')
    }}
    '''
    
    html_delay_template = r'''
    delay: <input type="text" name="${id}_delaytext"
        onChange="$id.changeDelay();" value="0" size="4"> ms
    <input type=button onClick="$id.faster();" value="-" title="faster">
    <input type=button onClick="$id.slower();" value="+" title="slower">
    '''
    
    html_top_template = r'''
    '''
    
    html_box_template = r'''
    '''
    
    html_bottom_template = r'''
    '''
    
    
class SectionHTMLSlideShow(HTMLSlideShow):
    """Astract slideshow with additional support for section markers."""
    
    def __init__(self, section_ids, title=None,
                 delay=100, delay_delta=None, loop=True, id=None, **kwargs):
        """Return the complete HTML code for the slideshow.
        
        section_ids -- List with the section id for each slide index. The id
            can be a string or a number.
       
        For additional keyword arguments see the super class.
        """
        # we need the id for the section names
        if id is None:
            id = self._get_random_id()
        kwargs.update(vars())
        # check if there is more than one section id,
        # otherwise some controls must be disabled to prevent infinite loop
        only_one_section = "false"
        first_section_id = section_ids[0]
        for section_id in section_ids:
            if section_id != first_section_id:
                break
        else:
            only_one_section = "true"
        kwargs["only_one_section"] = only_one_section
        # translate section_id_list into JavaScript list
        section_ids = [str(section_id) for section_id in section_ids]
        js_section_ids = "".join(['"%s_section_id_%s",\n' % (id, section_id) 
                                  for section_id in section_ids])
        js_section_ids = "\n" + js_section_ids[:-2]
        kwargs["js_section_ids"] = js_section_ids
        del kwargs["self"]
        super(SectionHTMLSlideShow, self).__init__(**kwargs)
        
        
    js_update_template = r'''
        // maps slide index to section id
        var section_ids = new Array($js_section_ids);
        // currently highlighted section id
        var current_section_id = section_ids[0];
        // store the class before
        
        that.updateSlide = function () {
            document.getElementById(current_section_id).className =
                "inactive_section";
            current_section_id = section_ids[current_slide]
            document.getElementById(current_section_id).className =
                "active_section";
            slideselect.selectedIndex = current_slide;
            that.loadSlide();
        }
        
        // use this function when a section is selected,
        // e.g. onClick="setSlide(42)"
        that.setSlide = function (index) {
            current_slide = index;
            that.updateSlide();
        }
        
        that.previousSection = function () {
            if ($only_one_section) {
                return;
            }
            while (current_section_id === section_ids[current_slide]) {
                if (current_slide > 0) {
                    current_slide -= 1;
                } else {
                    current_slide = slideselect.length-1;
                }
            }
            var new_section_id = section_ids[current_slide];
            // now go to start of this section
            while (new_section_id === section_ids[current_slide]) {
                current_slide -= 1;
                if (current_slide < 0) {
                    break;
                }
            }
            current_slide += 1;
            that.updateSlide();
        }
        
        that.nextSection = function () {
            if ($only_one_section) {
                return;
            }
            while (current_section_id === section_ids[current_slide]) {
                if (current_slide+1 < slideselect.length) {
                    current_slide += 1;
                } else {
                    current_slide = 0;
                }
            }
            that.updateSlide();
        }
        
        $<js_loadslide_template>
    '''
    
    # define keyboard shortcuts
    js_keyboard_shortcuts_template = r'''
    document.onkeydown = function(e) {
        if (!e.ctrlKey) { // control key must be pressed
            return;
        }
        else if (e.which === 37) { // left key
            document.getElementById("prevButton").click();
        }
        else if(e.which === 39) { // right key
            document.getElementById("nextButton").click();
        }
        else if(e.which === 38) { // up key
            document.getElementById("prevSectionButton").click();
        }
        else if(e.which === 40) { // down key
            document.getElementById("nextSectionButton").click();
        }
        else if(e.which === 45) { // insert key
            document.getElementById("startButton").click();
        }
    }
    '''
    
    html_buttons_template = r'''
    <input type=button onClick="$id.first();" value="|<<" title="beginning"
        id="${id}_firstButton">
    <input type=button onClick="$id.previousSection();" value="|<"
        title="previous section" id="${id}_prevSectionButton">
    <input type=button onClick="$id.previous();" value="<" title="previous"
        id="${id}_prevButton">
    <input type=button name="startbutton" onClick="$id.startstop(this.value);"
        value="Start" title="autoplay" id="${id}_startButton">
    <input type=button onClick="$id.next();" value=">" title="next"
        id="${id}_nextButton">
    <input type=button onClick="$id.nextSection();" value=">|"
        title="next section" id="${id}_nextSectionButton">
    <input type=button onClick="$id.last();" value=">>|" title="end"
        id="${id}_lastButton">
    '''
    
    html_controls_template = r'''
    ${{super(SectionHTMLSlideShow, self).html_controls_template(vars())}}
    
    <tr><td>
    ${{
    last_section_id = None
    link = ''
    for index, section_id in enumerate(section_ids):
        if section_id != last_section_id:
            if index > 0:
                self.write(link + '&nbsp;|&nbsp;')
            last_section_id = section_id
            link = ('<span class="inactive_section" ' +
                    'id="%s_section_id_%s" ' % (id, section_id) +
                    'onClick="%s.setSlide(%d);">%s</span>' %
                        (id, index, section_id))
    self.write(link + '\n')
    }}
    </td></tr>
    '''
    

class ImageHTMLSlideShow(HTMLSlideShow):
    """Slideshow for images.
    
    This also serves as an example for implementing a slideshow based on
    HTMLSlideShow.
    """
    
    def __init__(self, filenames, image_size, title=None, delay=100,
                 delay_delta=20, loop=True, id=None,
                 magnification=1, mag_control=True,
                 **kwargs):
        """Return the complete HTML code for a slideshow of the given images.
        
        filenames -- sequence of strings, containing the path for each image
        image_size -- Tuple (x,y) with the original image size, or enter
            a different size to force scaling.
        magnification -- Magnification factor for images (default 1). This
            factor is applied on top of the provided image size.
        mag_control -- Set to True (default) to display a magnification control
            element.
            
        For additional keyword arguments see the super class.
        """
        if len(filenames) == 0:
            raise Exception("Empty list was given.")
        kwargs.update(vars())
        # translate image size to width and heigh to be used in the templates
        del kwargs["image_size"]
        kwargs["width"] = image_size[0]
        kwargs["height"] = image_size[1]
        del kwargs["self"]
        super(ImageHTMLSlideShow, self).__init__(**kwargs)
        
    js_controls_template = r'''
    ${{super(ImageHTMLSlideShow, self).js_controls_template(vars())}}
    
    var magnification = $magnification; // image magnification
    var original_width = $width; // original image width
    var original_height = $height; // original image height
    
    that.smaller = function () {
        magnification = magnification / 2;
        slideform.${id}_magtext.value = magnification.toString();
        that.resizeImage();
    }
    
    that.larger = function (text) {
        magnification = magnification * 2;
        slideform.${id}_magtext.value = magnification.toString();
        that.resizeImage();
    }
    
    that.changeMag = function () {
        magnification = parseFloat(slideform.${id}_magtext.value);
        that.resizeImage();
    }
    
    that.resizeImage = function () {
        document.images.image_display.width =
            parseInt(magnification * original_width, 10);
        document.images.image_display.height =
            parseInt(magnification * original_height, 10);
    }
    '''
    
    js_loadslide_template = r'''
    that.loadSlide = function () {
        document.images.image_display.src =
            slideselect[current_slide].value;
    }
    '''
    
    js_onload_template = r'''
    that.onLoad = function () {
        slideform = document.${id}_slideform;
        slideselect = slideform.${id}_slideselect;
        current_slide = slideselect.selectedIndex;
        that.updateSlide();
        ${{
        if delay is not None:
            self.write('slideform.%s_delaytext.value = ' % id +
                       'show_delay.toString();\n')
        if mag_control:
            self.write('slideform.%s_magtext.value = ' % id +
                       'magnification.toString();\n')
        }}
        that.resizeImage();
    }
    '''
    
    html_box_template = r'''
    <tr>
    <td style="padding: 20 20 20 20">
    <img src="" name="image_display" width="$width" height="$height">
    </td>
    </tr>
    '''

    html_controls_template = r'''
    ${{
    if mag_control or (delay is not None):
        self.write('<tr><td align=center>\n')
        if mag_control:
            self.html_mag_template(vars())
            if delay is not None:
                self.write('<br>\n')
        if delay is not None:
            self.html_delay_template(vars())
        self.write('</td></tr>\n')
    }}
    '''
    
    html_mag_template = r'''
    magnification: <input type="text" name="${id}_magtext"
        onChange="$id.changeMag();" value="0" size="2">
    <input type=button onClick="$id.smaller();" value="-" title="smaller">
    <input type=button onClick="$id.larger();" value="+" title="larger">
    '''
    
    
class SectionImageHTMLSlideShow(SectionHTMLSlideShow, ImageHTMLSlideShow):
    """Image slideshow with section markers."""
    
    def __init__(self, filenames, section_ids, image_size, title=None,
                 delay=100, delay_delta=20, loop=True, id=None,
                 magnification=1, mag_control=True, **kwargs):
        """Return the HTML code for a sectioned slideshow of the given images.
        
        For keyword arguments see the super classes.
        """
        if len(section_ids) != len(filenames):
            err = ("The number of section id entries does not match the "
                   "number of slides / filenames.")
            raise Exception(err)
        kwargs.update(vars())
        del kwargs["self"]
        super(SectionImageHTMLSlideShow, self).__init__(**kwargs)
    

### helper functions ###

# TODO: extract image size automatically

def image_slideshow(filenames, image_size, title=None,
                    section_ids=None,
                    delay=100, delay_delta=20, loop=True, id=None,
                    magnification=1, mag_control=True):
    """Return a string with the JS and HTML code for an image slideshow.
    
    Note that the CSS code for the slideshow is not included, so you should
    add SLIDESHOW_STYLE or a custom style to your CSS code.
    
    filenames -- Sequence of the image filenames.
    image_size -- Tuple (x,y) with the original image size, or enter
        a different size to force scaling.
    title -- Optional slideshow title (for default None not title is shown).
    section_ids -- List with the section id for each slide index. The id
            can a string or a number. Default value None disables the
            section feature.
   
    For additional keyword arguments see the ImageHTMLSlideShow class.
    """
    if section_ids:
        slideshow = SectionImageHTMLSlideShow(**vars())
    else:
        slideshow = ImageHTMLSlideShow(**vars())
    return str(slideshow)

def show_image_slideshow(filename,
                         filenames, image_size, title=None, section_ids=None,
                         delay=100, delay_delta=20, loop=True, id=None,
                         magnification=1, mag_control=True,
                         browser_open=True):
    """Write the slideshow into a HTML file, open it in the browser and
    return the file name.
    
    filename -- Filename for the HTML file to be created. If None
            a temporary file is created.
    filenames -- Sequence of the image filenames.
    image_size -- Tuple (x,y) with the original image size, or enter
        a different size to force scaling.
    title -- Optional slideshow title (for default None not title is shown).
    section_ids -- List with the section id for each slide index. The id
            can a string or a number. Default value None disables the
            section feature.
    browser_open -- If True (default value) then the slideshow file is
        automatically opened in a webbrowser.
    
    For additional keyword arguments see the ImageHTMLSlideShow class.
    """
    if filename is None:
        (fd, filename) = tempfile.mkstemp(suffix=".html", prefix="MDP_")
        html_file = os.fdopen(fd, 'w')
    else:
        html_file = open(filename, 'w')
    html_file.write('<html>\n<head>\n<title>%s</title>\n' % title)
    html_file.write('<style type="text/css" media="screen">')
    html_file.write(BASIC_STYLE)
    html_file.write(SLIDESHOW_STYLE)
    html_file.write('</style>\n</head>\n<body>\n')
    kwargs = vars()
    del kwargs['filename']
    del kwargs['browser_open']
    del kwargs['html_file']
    html_file.write(image_slideshow(**kwargs))
    html_file.write('</body>\n</html>')
    html_file.close()
    if browser_open:
        webbrowser.open(filename)
    return filename
