import mdp
from mdp import numx

random = mdp.numx_rand.random
exp, mult = numx.exp, mdp.utils.mult

# this and the other replication functions should go in mdp.utils
def rrep(x, n):
    """Replicate x n-times on a new last dimension"""
    shp = x.shape + (1,)
    return x.reshape(shp).repeat(n, axis=-1)


class RBMNode(mdp.Node):
    """Restricted Boltzmann Machine node.

    By default, the 'execute' function returns the *probability* of
    one of the hiden variables being one given the input.

    Binary <-> binary RBM class"""

    def __init__(self, hidden_dim, visible_dim = None, dtype = None):
        super(RBMNode, self).__init__(visible_dim, hidden_dim, dtype)
        self._initialized = False

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _init_weights(self):
        # weights and biases are initialized to small random values to
        # break the simmetry that might lead to degenerate solutions during
        # learning
        self._initialized = True
        
        # weights
        self.w = self._refcast(numx.random.randn(self.input_dim, self.output_dim)*0.1)
        # bias on the visibile (input) units
        self.bv = self._refcast(numx.random.randn(self.input_dim)*0.1)
        # bias on the hidden (output) units
        self.bh = self._refcast(numx.random.randn(self.output_dim)*0.1)

        # delta w, bv, bh used for momentum term
        self._delta = (0., 0., 0.)

    def _sample_h(self, v):
        # returns P(h=1|v,W,b) and a sample from it
        probs = 1./(1. + exp(-self.bh - mult(v, self.w)))
        h = (probs > random(probs.shape)).astype(self.dtype)
        return probs, h

    def _sample_v(self, h):
        # returns  P(v=1|h,W,b) and a sample from it
        probs = 1./(1. + exp(-self.bv - mult(h, self.w.T)))
        v = (probs > random(probs.shape)).astype(self.dtype)
        return probs, v

    def _train(self, v, n_updates=1, epsilon=0.1, decay=0., momentum=0.,
               verbose=False):
        """Training by Contrastive Divergence.
        
        n_updates --- number of Gibbs sampling steps
        """

        if not self._initialized:
            self._init_weights()

        # useful quantities
        N = v.shape[0]
        w, bv, bh = self.w, self.bv, self.bh

        # old gradients for momentum term
        dw, dbv, dbh = self._delta
        
        # first update of the hidden units for the data term
        ph_data, h_data = self._sample_h(v)
        # n updates of both v and h for the model term
        h_model = h_data.copy()
        for i in range(n_updates):
            pv_model, v_model = self._sample_v(h_model)
            ph_model, h_model = self._sample_h(v_model)
        
        # update w
        data_term = mult(v.T, ph_data)
        model_term = mult(v_model.T, ph_model)
        dw = momentum*dw + \
             epsilon*((data_term - model_term)/N - decay*w)
        w += dw
        
        # update bv
        data_term = v.sum(axis=0)
        model_term = v_model.sum(axis=0)
        dbv = momentum*dbv + \
              epsilon*((data_term - model_term)/N)
        bv += dbv

        # update bh
        data_term = ph_data.sum(axis=0)
        model_term = ph_model.sum(axis=0)
        dbh = momentum*dbh + \
              epsilon*((data_term - model_term)/N)
        bh += dbh

        self._delta = (dw, dbv, dbh)
        self._train_err = float(((v-v_model)**2.).sum())

        if verbose:
            print 'training error', self._train_err/v.shape[0]
            ph, h = self._sample_h(v)
            print 'energy', self._energy(v, ph).sum()

    def _stop_training(self):
        #del self._delta
        #del self._train_err
        pass

    # execution methods

    def _pre_inversion_checks(self, y):
        self._if_training_stop_training()
        # control the dimension of y
        self._check_output(y)

    def sample_h(self, v):
        self._pre_execution_checks(v)
        return self._sample_h(v)

    def sample_v(self, h):
        self._pre_inversion_checks(h)
        return self._sample_v(h)

    def _energy(self, v, h):
        return -mult(v, self.bv) - mult(h, self.bh) \
               - (mult(v, self.w)*h).sum(axis=1)

    def energy(self, v, h):
        return self._energy(v, h)

    def _execute(self, v, ignore=None, return_probs = True):
        probs, h = self._sample_h(v)
        if return_probs:
            return probs
        else:
            return h

    def is_invertible(self): return False

    def generate_input(self, h, return_probs = True):
        probs, v = self._sample_v(h)
        if return_probs:
            return probs
        else:
            return v

class RBMWithLabelsNode(RBMNode):
    """Binary <-> binary RBM class with softmax labels."""

    def __init__(self, hidden_dim, labels_dim, visible_dim = None, dtype = None):
        super(RBMNode, self).__init__(None, hidden_dim, dtype)

        self._labels_dim = labels_dim
        if visible_dim is not None:
            self.set_input_dim(visible_dim+labels_dim)
        
        self._initialized = False

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _set_input_dim(self, n):
        self._input_dim = n 
        self._visible_dim = n - self._labels_dim

    def _sample_v(self, h, sample_l=False, concatenate=True):
        # returns  P(v=1|h,W,b), a sample from it, P(l=1|h,W,b), and a sample from it

        ldim, vdim = self._labels_dim, self._visible_dim

        # activation
        a = self.bv + mult(h, self.w.T)
        av, al = a[:,:vdim], a[:,vdim:]
        
        # ## visible units: logistic activation
        probs_v = 1./(1. + exp(-av))
        v = (probs_v > random(probs_v.shape)).astype('d')
        
        # ## label units: softmax activation
        # subtract maximum to regularize exponent
        exponent = al - rrep(al.max(axis=1), ldim)
        probs_l = exp(exponent)
        probs_l /= rrep(probs_l.sum(axis=1), ldim)

        if sample_l:
            # ?? todo: I'm sure this can be optimized
            l = numx.zeros((h.shape[0], ldim))
            for t in range(h.shape[0]):
                l[t,:] = mdp.numx_rand.multinomial(1, probs_l[t,:])
        else:
            l = probs_l.copy()

        if concatenate:
            probs = numx.concatenate((probs_v, probs_l), axis=1)
            x = numx.concatenate((v, l), axis=1)
            return probs, x
        else:
            return probs_v, probs_l, v, l
    
    # execution methods

    def sample_h(self, v, l):
        x = numx.concatenate((v, l), axis=1)
        self._pre_execution_checks(x)
        return self._sample_h(x)

    def sample_v(self, h):
        self._pre_inversion_checks(h)
        
        probs_v, probs_l, v, l = self._sample_v(h, sample_l=True,
                                                concatenate=False)
        return probs_v, probs_l, v, l

    def energy(self, v, h, l):
        x = numx.concatenate((v, l), axis=1)
        return self._energy(x, h)

    def execute(self, v, l, return_probs = True):
        """?? Insert documentation"""
        x = numx.concatenate((v, l), axis=1)
        self._pre_execution_checks(x)
        
        probs, h = self._sample_h(self._refcast(x))
        if return_probs:
            return probs
        else:
            return h

    def is_invertible(self): return False

    def generate_input(self, h, return_probs = True):
        """?? Insert documentation. return visible units and labels"""
        self._pre_inversion_checks(h)
        probs_v, probs_l, v, l = self._sample_v(h, sample_l=False,
                                                concatenate=False)
        if return_probs:
            return (probs_v, probs_l)
        else:
            return (v, l)

    def _secret_train(self, v, l, *args, **kwargs):
        x = numx.concatenate((v, l), axis=1)
        self._check_input(x)
        self._train(self._refcast(x), *args, **kwargs)
   
    def train(self, v, l, *args, **kwargs):
        """?? Insert documentation."""

        if not self.is_training():
            raise TrainingFinishedException, \
                  "The training phase has already finished."

        x = numx.concatenate((v, l), axis=1)
        self._check_input(x)

        self._train_phase_started = True
        self._train_seq[self._train_phase][0](self._refcast(x), *args, **kwargs)
