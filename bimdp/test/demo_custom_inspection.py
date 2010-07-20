"""
Test the inspection of a normal (non-bi) Flow with a customized visualization.

This demonstrates how one can write custom TraceHTMLTranslator classes to
conveniently create rich visualizations with the BiMDP trace inspection.
"""

import numpy
import os
import matplotlib.pyplot as plt

import mdp
import bimdp


class CustomTraceHTMLTranslator(bimdp.TraceHTMLTranslator):
    """Custom TraceHTMLTranslator to visualize the SFA node output.

    This class also demonstrates how to use custom section_id values, and how
    to correctly reset internal variables via the reset method.
    """

    def __init__(self, show_size=False):
        self._sect_counter = None
        super(CustomTraceHTMLTranslator, self).__init__(show_size=show_size)

    def reset(self):
        """Reset the section counter."""
        super(CustomTraceHTMLTranslator, self).reset()
        self._sect_counter = 0

    def _write_right_side(self, path, html_file, flow, node, method_name,
                          method_result, method_args, method_kwargs):
        """Write the result part of the translation."""
        # check if we have reached the right node
        if isinstance(node, bimdp.BiNode) and (node.node_id == "sfa"):
            self._sect_counter += 1
            html_file.write("<h3>visualization (in section %d)<h3>" %
                            self._sect_counter)
            slide_name = os.path.split(html_file.name)[-1][:-5]
            image_filename = slide_name + "_%d.png" % self._sect_counter
            # plot the y result values
            plt.figure(figsize=(6, 4))
            ys = method_result
            for y in ys:
                plt.plot(y)
            plt.legend(["y sample %d" % (i+1) for i in range(len(ys))])
            plt.title("SFA node output")
            plt.xlabel("coordinate")
            plt.ylabel("y value")
            plt.savefig(os.path.join(path, image_filename), dpi=75)
            html_file.write('<img src="%s">' % image_filename)
        section_id = "%d" % self._sect_counter
        # now add the standard stuff
        super(CustomTraceHTMLTranslator, self)._write_right_side(
                               path=path, html_file=html_file, flow=flow,
                               node=node, method_name=method_name,
                               method_result=method_result,
                               method_args=method_args,
                               method_kwargs=method_kwargs)
        return section_id


## Create the flow.
noisenode = mdp.nodes.NormalNoiseNode(input_dim=20*20,
                                      noise_args=(0, 0.0001))
sfa_node = bimdp.nodes.SFABiNode(input_dim=20*20, output_dim=10, dtype='f',
                                 node_id="sfa")
switchboard = mdp.hinet.Rectangular2dSwitchboard(
                                          x_in_channels=100,
                                          y_in_channels=100,
                                          x_field_channels=20,
                                          y_field_channels=20,
                                          x_field_spacing=10,
                                          y_field_spacing=10)
flownode = mdp.hinet.FlowNode(mdp.Flow([noisenode, sfa_node]))
sfa_layer = mdp.hinet.CloneLayer(flownode, switchboard.output_channels)
flow = mdp.Flow([switchboard, sfa_layer])
train_data = [numpy.cast['f'](numpy.random.random((3, 100*100)))
              for _ in range(5)]
flow.train(data_iterables=[None, train_data])

## This is where the inspection happens.
trace_translator = CustomTraceHTMLTranslator()
# note that we could also specify a custom CSS file, via css_filename
trace_inspector = bimdp.TraceHTMLInspector(trace_translator=trace_translator)
filename, out = bimdp.show_execution(flow, x=train_data[0],
                                     trace_inspector=trace_inspector)

print "done."
