

import unittest
import numpy as np
import mdp
import bimdp


class TestGradientExtension(unittest.TestCase):
    
    def test_sfa_gradient(self):
        """Test gradient for combination of SFA nodes."""
        sfa_node1 = bimdp.nodes.SFABiNode(output_dim=8)
        sfa_node2 = bimdp.nodes.SFABiNode(output_dim=7)
        sfa_node3 = bimdp.nodes.SFABiNode(output_dim=5)
        flow = sfa_node1 + sfa_node2 + sfa_node3
        x = np.random.random((300, 10))
        flow.train(x)
        x = np.random.random((2, 10))
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
        x = np.random.random((300, 10))
        flow.train(x)
        mdp.activate_extension("gradient")
        try:
            x1 = np.random.random((2, 10))
            x2, msg = sfa_node1.execute(x1, {"method": "gradient"})
            grad1 = msg["grad"]
            _, msg = sfa_node2.execute(x2, {"method": "gradient"})
            grad2 = msg["grad"]
            grad12 = flow.execute(x1, {"method": "gradient"})[1]["grad"]
            # use a different way to calculate the product of the gradients,
            # this method is too memory intensive for large data
            ref_grad = np.sum(grad2[:,:,np.newaxis,:] *
                             np.transpose(grad1[:,np.newaxis,:,:], (0,1,3,2)),
                             axis=3)
            self.assert_(np.max(np.abs(ref_grad - grad12)) < 1E-9)
        finally:
            mdp.deactivate_extension("gradient")
            
    def test_quadexpan_gradient1(self):
        """Test validity of gradient for QuadraticExpansionBiNode."""
        node = mdp.nodes.QuadraticExpansionNode()
        x = np.array([[1, 3, 4]])
        node.execute(x)
        mdp.activate_extension("gradient")
        try:
            result = node._gradient(x)
            grad = result[1]["grad"]
            reference = np.array(
                [[[ 1, 0, 0],   # x1
                  [ 0, 1, 0],   # x2
                  [ 0, 0, 1],   # x3
                  [ 2, 0, 0],   # x1x1
                  [ 3, 1, 0],   # x1x2
                  [ 4, 0, 1],   # x1x3
                  [ 0, 6, 0],   # x2x2
                  [ 0, 4, 3],   # x2x3
                  [ 0, 0, 8]]]) # x3x3 
            self.assert_(np.all(grad == reference))
        finally:
            mdp.deactivate_extension("gradient")
            
    def test_quadexpan_gradient2(self):
        """Test gradient with multiple data points."""
        node = mdp.nodes.QuadraticExpansionNode()
        x = np.random.random((3,5))
        node.execute(x)
        mdp.activate_extension("gradient")
        try:
            result = node._gradient(x)
            gradient = result[1]["grad"]
            self.assert_(gradient.shape == (3,20,5))
        finally:
            mdp.deactivate_extension("gradient")
            
    def test_sfa2_gradient(self):
        sfa2_node1 = bimdp.nodes.SFA2BiNode(output_dim=5)
        sfa2_node2 = bimdp.nodes.SFA2BiNode(output_dim=3)
        flow = sfa2_node1 + sfa2_node2
        x = np.random.random((300, 6))
        flow.train(x)
        x = np.random.random((2, 6))
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
                self.__gradient_Hs = np.vstack((quad_form.H[np.newaxis]
                                                for quad_form in quad_forms))
                self.__gradient_fs = np.vstack((quad_form.f[np.newaxis]
                                                for quad_form in quad_forms))
            grad = (np.dot(x, self.__gradient_Hs) +
                        np.repeat(self.__gradient_fs[np.newaxis,:,:],
                                  len(x), axis=0))
            return grad
        sfa2_node = bimdp.nodes.SFA2BiNode(output_dim=3)
        x = np.random.random((300, 6))
        sfa2_node.train(x)
        sfa2_node.stop_training()
        x = np.random.random((2, 6))
        mdp.activate_extension("gradient")
        try:
            result1 = sfa2_node.execute(x, {"method": "gradient"})
            grad1 = result1[1]["grad"]
            grad2 = _alt_sfa2_grad(sfa2_node, x)
            self.assert_(np.max(np.abs(grad1 - grad2)) < 1E-9)
        finally:
            mdp.deactivate_extension("gradient")

    def test_layer_gradient(self):
        """Test gradient for a simple layer."""
        layer = mdp.hinet.Layer([mdp.nodes.SFA2Node(input_dim=4, output_dim=3),
                                 mdp.nodes.SFANode(input_dim=6, output_dim=2)])
        x = np.random.random((100,10))
        layer.train(x)
        layer.stop_training()
        mdp.activate_extension("gradient")
        try:
            x = np.random.random((7,10))
            result = layer._gradient(x)
            gradient = result[1]["grad"]
            self.assert_(gradient.shape == (7,5,10))
        finally:
            mdp.deactivate_extension("gradient")
  
    def test_clonebilayer_gradient(self):
        """Test gradient for a simple layer."""
        layer = bimdp.hinet.CloneBiLayer(
                            bimdp.nodes.SFA2BiNode(input_dim=5, output_dim=2),
                            n_nodes=3)
        x = np.random.random((100,15))
        layer.train(x)
        layer.stop_training()
        mdp.activate_extension("gradient")
        try:
            x = np.random.random((7,15))
            result = layer._gradient(x)
            gradient = result[1]["grad"]
            self.assert_(gradient.shape == (7,6,15))
        finally:
            mdp.deactivate_extension("gradient")
            
    # TODO: add functional layer gradient tests
    
            
def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestGradientExtension))
    return suite
            
if __name__ == '__main__':
    unittest.main() 