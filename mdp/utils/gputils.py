from theano import function, config, shared, sandbox
import theano.tensor as T

config.floatX = 'float32'
config.mode = 'FAST_RUN'

def theano_multiply(a,b):
    mult = function([], sandbox.cuda.basic_ops.gpu_from_host(T.dot(a,b)))
    return f()
