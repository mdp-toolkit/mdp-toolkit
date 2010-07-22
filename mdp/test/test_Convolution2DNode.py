from _tools import *
import mdp
from mdp import numx, numx_rand

def testConvolution2DNodeFunctionality():
    filters = numx.empty((3,1,1))
    filters[:,0,0] = [1.,2.,3.]
    x = numx_rand.random((10,3,4))
    
    for mode in ['valid', 'same', 'full']:
        for boundary in ['fill', 'wrap', 'symm']:
            node = mdp.nodes.Convolution2DNode(filters, approach='linear', mode=mode,
                                               boundary=boundary, output_2d=False)
            y = node.execute(x)
            
            assert_equal(y.shape, (x.shape[0], 3, x.shape[1], x.shape[2]))
            for n_flt in range(3):
                assert_array_equal(x*(n_flt+1.), y[:,n_flt,:,:])

def testConvolution2DNode_2D3Dinput():
    filters = numx.empty((3,1,1))
    filters[:,0,0] = [1.,2.,3.]
    
    # 1) input 2D/3D
    x = numx_rand.random((10,12))        
    node = mdp.nodes.Convolution2DNode(filters, approach='linear',
                                       input_shape=(3,4), output_2d=False)
    y = node.execute(x)
    assert_equal(y.shape, (x.shape[0], 3, 3, 4))
    
    x = numx.random.random((10,3,4)) 
    node = mdp.nodes.Convolution2DNode(filters, output_2d=False)
    y = node.execute(x)
    assert_equal(y.shape, (x.shape[0], 3, 3, 4))
        
    # 2) output 2D/3D
    x = numx.random.random((10,12))        
    node = mdp.nodes.Convolution2DNode(filters, approach='linear',
                                       input_shape=(3,4), output_2d=True)
    y = node.execute(x)
    assert_equal(y.shape, (x.shape[0], 3*3*4))
    for i in range(3):
        assert_array_equal(x*(i+1.), y[:,i*12:(i+1)*12])
