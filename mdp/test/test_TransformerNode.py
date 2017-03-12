
from _tools import *

requires_scikits = skip_on_condition("not mdp.config.has_sklearn", "This test requires sklearn")
requires_cv_pil = skip_on_condition("not mdp.config.has_opencv or not mdp.config.has_pil",
                                    "This test requires either opencv or pil.")

def test_transformer_node():

    inp = mdp.numx_rand.randint(0,255,(1,768,1024,3)).astype('uint8')
    inp = mdp.numx.vstack((inp, inp))
    inp_shape = inp.shape[1:]
    inp_avg = inp.mean(axis=0)
    inp = inp.reshape(2, 768*1024*3).astype('float')

    # check with no transformations
    tr_node = mdp.nodes.TransformerNode(input_shape=inp_shape)
    out = tr_node(inp)
    assert_array_equal(inp, out)

    # check with a sequence of transformation
    tr_node = mdp.nodes.TransformerNode(input_shape=inp_shape, transform_seq=['set_shape', 'gray', 'img_255_1', 'center'])
    out = tr_node(inp)
    assert(out.max() <= 1)
    assert(out.min() >= 0)
    assert(out.shape == (inp.shape[0], inp.shape[1]/3))
    assert mdp.numx.all(out.mean(axis=0) == mdp.numx.zeros(out.shape[1]))

    # check transpose
    tr_node = mdp.nodes.TransformerNode(input_shape=inp_shape, transform_seq=['set_shape', 'transpose'])
    out = tr_node(inp)
    out = out.reshape(out.shape[0], *inp_shape[::-1])
    out_avg = out.mean(axis=0)
    assert mdp.numx.all(inp_avg == out_avg.T)

@requires_scikits
def test_sklearn_transform_fns():
    # Testing binarize preprocessing function of sklearn. It requires the data to be a 2d array.
    tr_node = mdp.nodes.TransformerNode(input_shape=(3, 3, 3), transform_seq=['set_shape', 'gray', 'to_2d', 'binarize'])

    inp = mdp.numx.random.randn(10, 3, 3, 3)
    inp_gray = inp.mean(axis=3)
    inp_binary = (inp_gray > 0).astype('float').reshape(10, 3*3)

    inp = inp.reshape(10, 3*3*3)
    out = tr_node(inp)
    assert mdp.numx.all(inp_binary == out)


@requires_cv_pil
def test_cv_pil_transform_fns():
    inp = mdp.numx_rand.randint(0,255,(1,768,1024,3)).astype('uint8')
    inp = mdp.numx.vstack((inp, inp))
    inp_shape = inp.shape[1:]
    inp_avg = inp.mean(axis=0)
    inp = inp.reshape(2, 768*1024*3).astype('float')

    # check with a sequence of transformation
    tr_node = mdp.nodes.TransformerNode(input_shape=inp_shape, transform_seq=['set_shape', 'resize'],
                              transform_seq_args=[None, [(384, 512)]])
    out = tr_node(inp)

    assert(out.shape == (2, 384*512*3))
