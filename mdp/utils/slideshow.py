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

import templet

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
    """Slideshow base class.
    
    It does not display anything, but can be adapted by overriding
    some of the templating attributes. See ImageHTMLSlideShow for an example.
    """
    
    def __init__(self, filenames, title=None, delay=100, delay_delta=None,
                 loop=True, **kwargs):
        """Return the complete HTML code for the slideshow.
        
        filenames -- sequence of strings, containing the path for each image
        title -- Optional slideshow title (for defualt None not title is shown).
        delay - Delay between slides in ms (default 100).
        delay_delta - Step size for increasing or decreasing the delay,
            or use the (default) value None for no delay control.
        loop -- If True continue with first slide when the last slide is
            reached during the automatic slideshow (default is False).
        """
        if len(filenames) == 0:
            raise Exception("Empty list was given.")
        # translate boolean variable into JS format
        if loop:
            loop = "true"
        else:
            loop = "false"
        kwargs.update(vars())
        del kwargs["self"]
        super(HTMLSlideShow, self).__init__(**kwargs)
   
    template = r'''
    <script language="JavaScript">
    <!-- Begin
    
    var current_slide = 0; // current slide index
    var show_delay = $delay; // delay in milliseconds
    var loop_slideshow = $loop; // loop in auto mode
    
    $<js_controls_template>
    $<js_loadslide_template>
    $<js_update_template>
    $<js_onload_template>
    $<js_keyboard_shortcuts_template>
    
    function onSelectorChange() {
        current_slide = document.slideform.slide.selectedIndex;
        updateSlide();
    }
    
    function next() {
        if (document.slideform.slide[current_slide+1]) {
            ++current_slide;
            updateSlide();        
           }
        else first();
    }
    
    function previous() {
            if (current_slide-1 >= 0) {
                --current_slide;
                updateSlide();
           }
        else last();
    }
    
    function first() {
        current_slide = 0;
        updateSlide();
    }
    
    function last() {
        current_slide = document.slideform.slide.length-1;
        updateSlide();
    }
    
    function startstop(text) {
        if (text == "Start") {
            document.slideform.startbutton.value = "Stop";
            if (!loop_slideshow &&
                (current_slide == document.slideform.slide.length-1)) {
                // restart slideshow
                current_slide = 0;
            }
            showAuto();
        } else {
            document.slideform.startbutton.value = "Start";
        }
    }
    
    function showAuto() {
        if (document.slideform.startbutton.value == "Stop") {
            if (current_slide == document.slideform.slide.length-1) {
                if (loop_slideshow) {
                    current_slide = 0;
                    updateSlide();
                } else {
                    document.slideform.startbutton.value = "Start";
                }
            } else {
                current_slide = current_slide+1;
                updateSlide();
            }
            window.setTimeout("showAuto()", show_delay);
        }
    }
    
    //  End -->
    </script>
    
    <div class="slideshow">
    
    $<html_top_template>
    
    <form name=slideform>
    <table class="slideshow">
    ${{
    if title:
        self.write('<tr><td><b> %s </b></td></tr>' % title)
    }}
    
    $<html_box_template>
    
    <tr><td>
    <select name="slide" onChange="onSelectorChange();">
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
    onLoad();
    //  End -->
    </script>
    '''
    
    js_controls_template = r'''
    var delay_delta = $delay_delta; // step size for in- or decreasing the delay 
    
    function slower() {
        show_delay += delay_delta;
        document.slideform.delaytext.value = show_delay.toString();
    }
    
    function faster(text) {
        show_delay -= delay_delta;
        if (show_delay < 0) show_delay = 0;
        document.slideform.delaytext.value = show_delay.toString();
    }
    
    function changeDelay() {
        var new_delay = parseInt(document.slideform.delaytext.value);
        if (new_delay < 0) {
             new_delay = 0;
            }
        show_delay = new_delay;
        document.slideform.delaytext.value = new_delay.toString();
    }
    '''
    
    js_update_template = r'''
    function updateSlide() {
        document.slideform.slide.selectedIndex = current_slide;
        loadSlide();
    }
    '''
    
    # overwrite this to implement the actual slide change
    js_loadslide_template = r'''
    function loadSlide() {
    }
    '''
    
    js_onload_template = r'''
    function onLoad() {
        current_slide = document.slideform.slide.selectedIndex;
        updateSlide();
        document.slideform.delaytext.value = show_delay.toString();
    }
    '''
    
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
    <input type=button onClick="first();" value="|<<" title="beginning"
        id="firstButton">
    <input type=button onClick="previous();" value="<" title="previous"
        id="prevButton">
    <input type=button name="startbutton" onClick="startstop(this.value);"
        value="Start" title="autoplay" id="startButton">
    <input type=button onClick="next();" value=">" title="next"
        id="nextButton">
    <input type=button onClick="last();" value=">>|" title="end"
        id="lastButton">
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
    delay: <input type="text" name="delaytext" onChange="changeDelay();" 
        value="0" size="4"> ms
    <input type=button onClick="faster();" value="-" title="faster">
    <input type=button onClick="slower();" value="+" title="slower">
    '''
    
    html_top_template = r'''
    '''
    
    html_box_template = r'''
    '''
    
    html_bottom_template = r'''
    '''
    
    
class SectionHTMLSlideShow(HTMLSlideShow):
    """Slideshow with additional support for section markers."""
    
    def __init__(self, filenames, section_ids, title=None,
                 delay=100, delay_delta=None, loop=True, **kwargs):
        """Return the complete HTML code for the slideshow.
        
        filenames -- Sequence of strings, containing the path for each image.
        title -- Optional slideshow title (for defualt None not title is shown).
        delay - Delay between slides in ms (default 100).
        delay_delta - Step size for increasing or decreasing the delay,
            or use the (default) value None for no delay control.
        loop -- If True continue with first slide when the last slide is
            reached during the automatic slideshow (default is False).
        section_ids -- List with the section id for each slide index. The id
            can a string or a number.
        """
        kwargs.update(vars())
        # translate section_firsts into a list mapping slide id to section id
        if len(section_ids) != len(filenames):
            err = ("The number of section id entries does not match the "
                   "number of slides / filenames.")
            raise Exception(err)
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
        js_section_ids = "".join(['"section_id_%s",\n' % section_id 
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
    
    function updateSlide() {
        document.getElementById(current_section_id).className =
            "inactive_section";
        current_section_id = section_ids[current_slide]
        document.getElementById(current_section_id).className =
            "active_section";
        document.slideform.slide.selectedIndex = current_slide;
        loadSlide();
    }
    
    // use this function when a section is selected, e.g. onClick="setSlide(42)"
    function setSlide(index) {
        current_slide = index;
        updateSlide();
    }
    
    function previousSection() {
        if ($only_one_section) return;
        while (current_section_id == section_ids[current_slide]) {
            if (current_slide > 0) {
                --current_slide;
            } else {
                current_slide = document.slideform.slide.length-1;
            }
        }
        var new_section_id = section_ids[current_slide];
        // now go to start of this section
        while (new_section_id == section_ids[current_slide]) {
            --current_slide;
            if (current_slide < 0) break;
        }
        ++current_slide;
        updateSlide();
    }
    
    function nextSection() {
        if ($only_one_section) return;
        while (current_section_id == section_ids[current_slide]) {
            if (current_slide+1 < document.slideform.slide.length) {
                ++current_slide;
            } else {
                current_slide = 0;
            }
        }
        updateSlide();
    }
    
    $<js_loadslide_template>
    '''
    
    # define keyboard shortcuts
    js_keyboard_shortcuts_template = r'''
    document.onkeydown = function(e) {
        if (!e.ctrlKey) { // control key must be pressed
            return;
        }
        else if (e.which == 37) { // left key
            document.getElementById("prevButton").click();
        }
        else if(e.which == 39) { // right key
            document.getElementById("nextButton").click();
        }
        else if(e.which == 38) { // up key
            document.getElementById("prevSectionButton").click();
        }
        else if(e.which == 40) { // down key
            document.getElementById("nextSectionButton").click();
        }
        else if(e.which == 45) { // insert key
            document.getElementById("startButton").click();
        }
    }
    '''
    
    html_buttons_template = r'''
    <input type=button onClick="first();" value="|<<" title="beginning"
        id="firstButton">
    <input type=button onClick="previousSection();" value="|<"
        title="previous section" id="prevSectionButton">
    <input type=button onClick="previous();" value="<" title="previous"
        id="prevButton">
    <input type=button name="startbutton" onClick="startstop(this.value);"
        value="Start" title="autoplay" id="startButton">
    <input type=button onClick="next();" value=">" title="next" id="nextButton">
    <input type=button onClick="nextSection();" value=">|" title="next section"
        id="nextSectionButton">
    <input type=button onClick="last();" value=">>|" title="end"
        id="lastButton">
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
                    'id="section_id_%s" ' % section_id +
                    'onClick="setSlide(%d);">%s</span>' %
                    (index, section_id))
    self.write(link)
    }}
    </td></tr>
    '''
    

class ImageHTMLSlideShow(HTMLSlideShow):
    """Slideshow for images.
    
    This also serves as an example for implementing a slideshow based on
    HTMLSlideShow.
    """
    
    def __init__(self, filenames, image_size, title=None, delay=100,
                 delay_delta=20, loop=True, magnification=1, mag_control=True,
                 **kwargs):
        """Return the complete HTML code for a slideshow of the given images.
        
        filenames -- sequence of strings, containing the path for each image
        image_size -- Tuple (x,y) with the original image size, or enter
            a different size to force scaling.
        title -- Optional slideshow title (for default None not title is shown).
        delay - Delay between slides in ms (default 100).
        delay_delta - Step size for increasing or decreasing the delay,
            or use the (default) value None for no delay control.
        loop -- If True continue with first slide when the last slide is
            reached during the automatic slideshow (default is False).
        magnification -- Magnification factor for images (default 1). This
            factor is applied on top of the provided image size.
        mag_control -- Set to True (default) to display a magnification control
            element.
        """
        kwargs.update(vars())
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
    
    function smaller() {
        magnification = magnification / 2;
        document.slideform.magtext.value = magnification.toString();
        resizeImage();
    }
    
    function larger(text) {
        magnification = magnification * 2;
        document.slideform.magtext.value = magnification.toString();
        resizeImage();
    }
    
    function changeMag() {
        magnification = parseFloat(document.slideform.magtext.value);
        resizeImage();
    }
    
    function resizeImage() {
        document.images.image_display.width =
            parseInt(magnification * original_width);
        document.images.image_display.height =
            parseInt(magnification * original_height);
    }
    '''
    
    js_loadslide_template = r'''
    function loadSlide() {
        document.images.image_display.src =
            document.slideform.slide[current_slide].value;
    }
    '''
    
    js_onload_template = r'''
    function onLoad() {
        current_slide = document.slideform.slide.selectedIndex;
        updateSlide();
        ${{
        if delay is not None:
            self.write('document.slideform.delaytext.value = ' +
                       'show_delay.toString();')
        if mag_control:
            self.write('document.slideform.magtext.value = ' +
                       'magnification.toString();')
        }}
        resizeImage()
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
    magnification: <input type="text" name="magtext" onChange="changeMag();" 
        value="0" size="2">
    <input type=button onClick="smaller();" value="-" title="smaller">
    <input type=button onClick="larger();" value="+" title="larger">
    '''
    
    
class SectionImageHTMLSlideShow(SectionHTMLSlideShow, ImageHTMLSlideShow):
    """Image slideshow with section markers."""
    pass
    

def image_slideshow(filenames, image_size, title=None,
                    section_ids=None,
                    delay=100, delay_delta=20, loop=True,
                    magnification=1, mag_control=True):
    """Return a string with the JS and HTML code for an image slideshow.
    
    Note that the CSS code for the slideshow is not included, so you should
    add SLIDESHOW_STYLE or a custom style to your CSS code.
    
    filenames -- Sequence of the image filenames.
    image_size -- Tuple (x,y) with the original image size, or enter
        a different size to force scaling.
    title -- Optional slideshow title (for default None not title is shown).
    section_ids -- List with the section id for each slide index. The id
            can a string or a number. Default value None disbales the
            section feature.
    delay - Delay between slides in ms (default 100).
    delay_delta - Step size for increasing or decreasing the delay,
        or use the (default) value None for no delay control.
    loop -- If True continue with first slide when the last slide is
            reached during the automatic slideshow (default is False).
    magnification -- Magnification factor for images (default 1). This
        factor is applied on top of the provided image size.
    mag_control -- Set to True (default) to display a magnification control
        element.
    """
    if section_ids:
        slideshow = SectionImageHTMLSlideShow(**vars())
    else:
        slideshow = ImageHTMLSlideShow(**vars())
    return str(slideshow)
