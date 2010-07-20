"""
This module creates the HTML rendering for the hinet tutorial example.
"""

import mdp

# create the flow
switchboard = mdp.hinet.Rectangular2dSwitchboard(x_in_channels=50,
                                                 y_in_channels=50,
                                                 x_field_channels=10,
                                                 y_field_channels=10,
                                                 x_field_spacing=5,
                                                 y_field_spacing=5,
                                                 in_channel_dim=3)
sfa_dim = 48
sfa_node = mdp.nodes.SFANode(input_dim=switchboard.out_channel_dim,
                             output_dim=sfa_dim)
sfa2_dim = 32
sfa2_node = mdp.nodes.SFA2Node(input_dim=sfa_dim,
                               output_dim=sfa2_dim)
flownode = mdp.hinet.FlowNode(mdp.Flow([sfa_node, sfa2_node]))
sfa_layer = mdp.hinet.CloneLayer(flownode,
                                 n_nodes=switchboard.output_channels)
flow = mdp.Flow([switchboard, sfa_layer])

# show the flow
mdp.hinet.show_flow(flow, filename="test.html", show_size=True)
print "opening flow HTML file in browser..."
