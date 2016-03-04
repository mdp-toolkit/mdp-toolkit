from builtins import range
from builtins import object
import mdp
import bimdp

from mdp import numx, numx_rand

class TestGradientExtension(object):

    def test_sfa_gradient(self):
        """Test gradient for combination of SFA nodes."""
        sfa_node1 = bimdp.nodes.SFABiNode(output_dim=8)
        sfa_node2 = bimdp.nodes.SFABiNode(output_dim=7)
        sfa_node3 = bimdp.nodes.SFABiNode(output_dim=5)
        flow = sfa_node1 + sfa_node2 + sfa_node3
        x = numx_rand.random((300, 10))
        flow.train(x)
        x = numx_rand.random((2, 10))
        mdp.activate_extension("gradient")
        try:
            flow.execute(x, {"method": "gradient"})
        finally:
            mdp.deactivate_extension("gradient")

    def test_gradient_product(self):
        """Test that the product of gradients is calculated correctly."""
        sfa_node1 = bimdp.nodes.SFABiNode(output_dim=5)
        sfa_node2 = bimdp.nodes.SFABiNode(output_dim=3)
        flow = sfa_node1 + sfa_node2
        x = numx_rand.random((300, 10))
        flow.train(x)
        mdp.activate_extension("gradient")
        try:
            x1 = numx_rand.random((2, 10))
            x2, msg = sfa_node1.execute(x1, {"method": "gradient"})
            grad1 = msg["grad"]
            _, msg = sfa_node2.execute(x2, {"method": "gradient"})
            grad2 = msg["grad"]
            grad12 = flow.execute(x1, {"method": "gradient"})[1]["grad"]
            # use a different way to calculate the product of the gradients,
            # this method is too memory intensive for large data
            ref_grad = numx.sum(grad2[:,:,numx.newaxis,:] *
                             numx.transpose(grad1[:,numx.newaxis,:,:], (0,1,3,2)),
                             axis=3)
            assert numx.amax(abs(ref_grad - grad12)) < 1E-9
        finally:
            mdp.deactivate_extension("gradient")

    def test_quadexpan_gradient1(self):
        """Test validity of gradient for QuadraticExpansionBiNode."""
        node = mdp.nodes.QuadraticExpansionNode()
        x = numx.array([[1, 3, 4]])
        node.execute(x)
        mdp.activate_extension("gradient")
        try:
            result = node._gradient(x)
            grad = result[1]["grad"]
            reference = numx.array(
                [[[ 1, 0, 0],   # x1
                  [ 0, 1, 0],   # x2
                  [ 0, 0, 1],   # x3
                  [ 2, 0, 0],   # x1x1
                  [ 3, 1, 0],   # x1x2
                  [ 4, 0, 1],   # x1x3
                  [ 0, 6, 0],   # x2x2
                  [ 0, 4, 3],   # x2x3
                  [ 0, 0, 8]]]) # x3x3
            assert numx.all(grad == reference)
        finally:
            mdp.deactivate_extension("gradient")

    def test_quadexpan_gradient2(self):
        """Test gradient with multiple data points."""
        node = mdp.nodes.QuadraticExpansionNode()
        x = numx_rand.random((3,5))
        node.execute(x)
        mdp.activate_extension("gradient")
        try:
            result = node._gradient(x)
            gradient = result[1]["grad"]
            assert gradient.shape == (3,20,5)
        finally:
            mdp.deactivate_extension("gradient")

    def test_sfa2_gradient(self):
        sfa2_node1 = bimdp.nodes.SFA2BiNode(output_dim=5)
        sfa2_node2 = bimdp.nodes.SFA2BiNode(output_dim=3)
        flow = sfa2_node1 + sfa2_node2
        x = numx_rand.random((300, 6))
        flow.train(x)
        x = numx_rand.random((2, 6))
        mdp.activate_extension("gradient")
        try:
            flow.execute(x, {"method": "gradient"})
        finally:
            mdp.deactivate_extension("gradient")

    def test_sfa2_gradient2(self):
        def _alt_sfa2_grad(self, x):
            """Reference grad method based on quadratic forms."""
            # note that the H and f arrays are cached in the node and remain even
            # after the extension has been deactivated
            if not hasattr(self, "__gradient_Hs"):
                quad_forms = [self.get_quadratic_form(i)
                              for i in range(self.output_dim)]
                self.__gradient_Hs = numx.vstack((quad_form.H[numx.newaxis]
                                                for quad_form in quad_forms))
                self.__gradient_fs = numx.vstack((quad_form.f[numx.newaxis]
                                                for quad_form in quad_forms))
            grad = (numx.dot(x, self.__gradient_Hs) +
                        numx.repeat(self.__gradient_fs[numx.newaxis,:,:],
                                  len(x), axis=0))
            return grad
        sfa2_node = bimdp.nodes.SFA2BiNode(output_dim=3)
        x = numx_rand.random((300, 6))
        sfa2_node.train(x)
        sfa2_node.stop_training()
        x = numx_rand.random((2, 6))
        mdp.activate_extension("gradient")
        try:
            result1 = sfa2_node.execute(x, {"method": "gradient"})
            grad1 = result1[1]["grad"]
            grad2 = _alt_sfa2_grad(sfa2_node, x)
            assert numx.amax(abs(grad1 - grad2)) < 1E-9
        finally:
            mdp.deactivate_extension("gradient")

    def test_layer_gradient(self):
        """Test gradient for a simple layer."""
        node1 = mdp.nodes.SFA2Node(input_dim=4, output_dim=3)
        node2 = mdp.nodes.SFANode(input_dim=6, output_dim=2)
        layer = mdp.hinet.Layer([node1, node2])
        x = numx_rand.random((100,10))
        layer.train(x)
        layer.stop_training()
        mdp.activate_extension("gradient")
        try:
            x = numx_rand.random((7,10))
            result = layer._gradient(x)
            grad = result[1]["grad"]
            # get reference result
            grad1 = node1._gradient(x[:, :node1.input_dim])[1]["grad"]
            grad2 = node2._gradient(x[:, node1.input_dim:])[1]["grad"]
            ref_grad = numx.zeros(((7,5,10)))
            ref_grad[:, :node1.output_dim, :node1.input_dim] = grad1
            ref_grad[:, node1.output_dim:, node1.input_dim:] = grad2
            assert numx.all(grad == ref_grad)
        finally:
            mdp.deactivate_extension("gradient")

    def test_clonebilayer_gradient(self):
        """Test gradient for a simple layer."""
        layer = bimdp.hinet.CloneBiLayer(
                            bimdp.nodes.SFA2BiNode(input_dim=5, output_dim=2),
                            n_nodes=3)
        x = numx_rand.random((100,15))
        layer.train(x)
        layer.stop_training()
        mdp.activate_extension("gradient")
        try:
            x = numx_rand.random((7,15))
            result = layer._gradient(x)
            grad = result[1]["grad"]
            assert grad.shape == (7,6,15)
        finally:
            mdp.deactivate_extension("gradient")

    def test_switchboard_gradient1(self):
        """Test that gradient is correct for a tiny switchboard."""
        sboard = mdp.hinet.Switchboard(input_dim=4, connections=[2,0])
        x = numx_rand.random((2,4))
        mdp.activate_extension("gradient")
        try:
            result = sboard._gradient(x)
            grad = result[1]["grad"]
            ref_grad = numx.array([[[0,0,1,0], [1,0,0,0]],
                                 [[0,0,1,0], [1,0,0,0]]], dtype=grad.dtype)
            assert numx.all(grad == ref_grad)
        finally:
            mdp.deactivate_extension("gradient")

    def test_switchboard_gradient2(self):
        """Test gradient for a larger switchboard."""
        dim = 100
        connections = [int(i) for i in numx.random.random((dim,)) * (dim-1)]
        sboard = mdp.hinet.Switchboard(input_dim=dim, connections=connections)
        x = numx.random.random((10, dim))
        # assume a 5-dimensional gradient at this stage
        grad = numx.random.random((10, dim, 5))
        # original reference implementation
        def _switchboard_grad(self, x):
            grad = numx.zeros((self.output_dim, self.input_dim))
            grad[list(range(self.output_dim)), self.connections] = 1
            return numx.tile(grad, (len(x), 1, 1))
        with mdp.extension("gradient"):
            result = sboard._gradient(x, grad)
            ext_grad = result[1]["grad"]
            tmp_grad = _switchboard_grad(sboard, x)
            ref_grad = numx.asarray([numx.dot(tmp_grad[i], grad[i])
                                     for i in range(len(tmp_grad))])
        assert numx.all(ext_grad == ref_grad)

    def test_network_gradient(self):
        """Test gradient for a small SFA network."""
        sfa_node = bimdp.nodes.SFABiNode(input_dim=4*4, output_dim=5)
        switchboard = bimdp.hinet.Rectangular2dBiSwitchboard(
                                                  in_channels_xy=8,
                                                  field_channels_xy=4,
                                                  field_spacing_xy=2)
        flownode = bimdp.hinet.BiFlowNode(bimdp.BiFlow([sfa_node]))
        sfa_layer = bimdp.hinet.CloneBiLayer(flownode,
                                             switchboard.output_channels)
        flow = bimdp.BiFlow([switchboard, sfa_layer])
        train_gen = [numx_rand.random((10, switchboard.input_dim))
                     for _ in range(3)]
        flow.train([None, train_gen])
        # now can test the gradient
        mdp.activate_extension("gradient")
        try:
            x = numx_rand.random((3, switchboard.input_dim))
            result = flow(x, {"method": "gradient"})
            grad = result[1]["grad"]
            assert grad.shape == (3, sfa_layer.output_dim,
                                  switchboard.input_dim)
        finally:
            mdp.deactivate_extension("gradient")
