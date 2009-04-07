import mdp

class XSFANode(mdp.Node):
    """Perform Non-linear Blind Source Separation using Slow Feature Analysis.

    [should we try here to explain the algorithm?]

    This node has multiple training phases. The number of training
    phases depends on the number of sources that must be extracted. The
    recommended way of training this node is through a container flow:

    >>> flow = mdp.Flow([XSFANode()])
    >>> flow.train(x)
    >>> flow.stop_training()

    doing so will automatically train all training phases. The argument
    'x' to the flow.train method can be an array or a list of iterables
    (see the section about Iterators in the MDP tutorial for more info).

    If the number of training samples is large, you may come into memory
    problems: use data iterators and chunk training to reduce memory usage.
    
    If you need to debug training and/or execution of this node, the
    suggested approach is to use the capabilities of mdp.binet. For example:

    >>> flow = mdp.Flow([XSFANode()])
    >>> tr_filename = binet.show_training(flow=flow, data_iterators=x)
    >>> ex_filename, out = binet.show_execution(flow, x=x)

    this will run training and execution with binet inspection. Snapshots
    of the internal flow state for each training phase and execution step
    will be opened in a web brower and presented as a slideshow.
    
    References:
    Sprekeler, H., Zito, T., and Wiskott, L. (2009).
    An Extension of Slow Feature Analysis for Nonlinear Blind Source Separation
    Journal of Machine Learning Research, submitted
    [pdf link follows]
    """
    def __init__(self, basic_exp=None, intern_exp=None, svd=False, verbose=False,
                 input_dim=None, output_dim=None, dtype=None):
        """
        Keyword arguments:

          basic_exp -- a tuple (node, args, kwargs) defining the node
                       used for the premilinary nonlinear expansion.
                       After this the nonlinearity is assumed to be linearly
                       invertible. The higher the degree of nonlinearity,
                       the higher are the chances of inverting an unknown
                       nonlinearity. On the other hand, the higher the
                       nonlinearity, the higher are numeric instabilities and
                       errors. The trade-off has to be evaluated carefully.
                       Default:
                       (mdp.nodes.PolynomialExpansionNode, (2, ), {})

          intern_exp -- a tuple (node, args, kwargs) defining the node
                        used for the internal nonlinear expansion of
                        the estimated sources to be removed from the input space.
                        The same trade-off as for basic_exp is valid here.
                        Default:
                        (mdp.nodes.PolynomialExpansionNode, (10, ), {})

                 svd -- enable Singular Value Decomposition for normalization
                        and regularization. Use it if the node complains about
                        singular covariance matrices.
             verbose -- show some progress during training.
        """
        
        # set up basic expansion
        if basic_exp is None:
            self.basic_exp = mdp.nodes.PolynomialExpansionNode
            self.basic_exp_args = (2, )
            self.basic_exp_kwargs = {}
        else:
            self.basic_exp = basic_exp[0]
            self.basic_exp_args = basic_exp[1]
            self.basic_exp_kwargs = basic_exp[2]

        # set up internal expansion
        if intern_exp is None:
            self.exp = mdp.nodes.PolynomialExpansionNode
            self.exp_args = (10, )
            self.exp_kwargs = {}
        else:
            self.exp = intern_exp[0]
            self.exp_args = intern_exp[1]
            self.exp_kwargs = intern_exp[2]
            
        # number of sources already extracted
        self.n_extracted_src = 0
        # internal network
        self._flow = None
        self.verbose = verbose
        self.svd = svd
        super(XSFANode, self).__init__(input_dim=input_dim,
                                       output_dim=output_dim, dtype=dtype)

        
    def _get_train_seq(self):
        #XXX: this is a  hack
        # In order to enable the output_dim to be set automatically
        # after input_dim is known, instead of forcing the user to specify
        # it by hand, we need to initialize the internal flow just before
        # starting the first training (input_dim are known at that moment).
        # Problem is that when XSFANode is trained through a container flow,
        # which is the standard way of training this kind of nodes,
        # the flow checks that the data_iterators are *not* generators
        # for multiple phases nodes. To assess if a node has multiple phases
        # it checks that len(self._train_seq) > 1. But we still
        # don't know the number of training_phases at this point, because we
        # first need to know input_dim, which we will know after we receive the
        # first chunk of data. To avoid the flow to complain we just return
        # a bogus list of training phases: it should break anything else.
        if self._flow is None:
            # we still don't know the number of training_phases yet,
            # but we can assure that we will have more than 1:
            return [(None, None), (None, None)]
        else:
            return ([(self._train, self._stop_training)] *
                    sum(self._training_phases))

    def _set_input_dim(self, n):
        self._input_dim = n
        # set output_dim if thery are still not set
        if self.output_dim is None:
            self.output_dim = n

    def _check_train_args(self, x):
        # this method will be called before starting training.
        # it is the right moment to initialize the internal flow
        if self._flow is None:
            self._initialize_internal_flow()
            if self.verbose:
                print "Extracting source 1..."

    def _initialize_internal_flow(self):
        # create the initial flow if it's not there already
        # set input_dim is needed to correctly create the first
        # network layer
        self.basic_exp_kwargs['input_dim'] = self.input_dim
        exp = self.basic_exp(*self.basic_exp_args, **self.basic_exp_kwargs)
        # first element of the flow is the basic expansion node
        # after that the first source extractor module is appended 
        self._flow = (exp + self._get_source_extractor(exp.output_dim, 0))

        # set the training phases
        # set the total number of training phases
        training_phases = [] 
        for S in range(self.output_dim):
            # get the number of training phases of every single
            # source extractor module
            mod = self._get_source_extractor(S+1, S)
            training_phases.append(len(mod._train_seq))
        
        self._training_phases = training_phases
        
        # this is a list of the training phases the correspond to
        # completed training of a source extractor module
        self._training_phases_mods = [sum(training_phases[:i+1]) for i in
                                      range(len(training_phases[:-1]))]
        
    def _get_supported_dtypes(self):
        """Return the list of dtypes supported by this node."""
        return ('float32', 'float64')
    
    def is_invertible(self):
        return False
    
    def _train(self, x):
        # train the last source extractor module in the flow
        self._flow[-1].train(self._flow[:-1](x))

    def _stop_training(self):
        # stop the current training phase
        self._flow[-1].stop_training()
        # update the current training phase
        cur_tr_ph = self.get_current_train_phase() + 1
        # if we finished to train the current source extractor module
        # and we still have to extract some sources
        # append a new source extractor module
        if (cur_tr_ph in self._training_phases_mods and
            self.n_extracted_src != (self.output_dim - 1)):
            
            self.n_extracted_src += 1
            mod = self._get_source_extractor(self._flow[-1].output_dim,
                                             self.n_extracted_src)
            self._flow.append(mod)
            if self.verbose:
                print "Extracting source %d..." % (self.n_extracted_src+1)
            
    def _execute(self, x):
        return self._flow(x)[:,:self.output_dim]

    def _get_source_extractor(self, dim, nsources):
        # returns a module to extract the next source and remove its
        # projections in the data space
        S = nsources
        L = dim-S

        # sfa - extracts the next source
        sfa = mdp.nodes.SFANode(input_dim=L, output_dim=L)
        
        # identity - copies the new sources
        idn_new1 = mdp.nodes.IdentityNode(input_dim=S+1)
        # source expansion
        self.exp_kwargs['input_dim'] = S + 1
        # N2
        src_exp = mdp.hinet.FlowNode(self.exp(*self.exp_args,
                                              **self.exp_kwargs) +
                                     NormalizeNode() + 
                                     mdp.nodes.WhiteningNode(svd=self.svd,
                                                             reduce=True))
        N2Layer = mdp.hinet.SameInputLayer((src_exp, idn_new1))
        N2ContLayer = mdp.hinet.Layer((N2Layer,
                                       mdp.nodes.IdentityNode(input_dim=L-1)))

        if S == 0:
            # don't need to copy the current sources (there are none)
            N1 = mdp.hinet.FlowNode(sfa + N2ContLayer)
        elif S == self.output_dim - 1:
            # the last source does not need to be removed
            # take care of passing the sources down along the flow
            idn_old = mdp.nodes.IdentityNode(input_dim=S)
            return mdp.hinet.Layer((idn_old,
                                    mdp.nodes.SFANode(input_dim=L,
                                                      output_dim=1)))
        else:
            # take care of passing the sources down along the flow
            idn_old = mdp.nodes.IdentityNode(input_dim=S)
            N1 = mdp.hinet.FlowNode(mdp.hinet.Layer((idn_old, sfa)) +
                                    N2ContLayer)

        # expanded sources projection
        proj = ProjectionNode(S, L-1)
        # use another identity node to copy the sources
        # we could in principle reuse the idn_new1 but using a new
        # node will make debugging much easier
        idn_new2 = mdp.nodes.IdentityNode(input_dim=S+1)
        # regularization after projection + new source copying
        reg_and_copy = mdp.hinet.Layer((idn_new2,
                                        mdp.nodes.WhiteningNode(input_dim=L-1,
                                                                svd=self.svd,
                                                                reduce=True)))
        # actual source removal flow 
        src_rem = mdp.hinet.FlowNode( proj + reg_and_copy )
        # return the actual source extraction module 
        return mdp.hinet.FlowNode(N1 + src_rem)


class ProjectionNode(mdp.Node):
    """Get expanded sources and input signals, and return
    the sources and the input signals projected into the space
    orthogonal to the expanded sources and their products."""
    def __init__(self, S, L):
        #!! IMPORTANT!!
        # this node *must* return the sources together with the
        # projected input signals
        self.proj_mtx = None
        self.L = L
        super(ProjectionNode, self).__init__(output_dim=S+1+L)
        self._cov_mtx = mdp.utils.CrossCovarianceMatrix(self.dtype)

    def _train(self, x):
        # compute covariance between expanded sources
        # and input signals
        self._cov_mtx.update(x[:,:-self.output_dim], x[:,-self.L:])

    def _stop_training(self):
        self.proj_mtx, avgx, avgy, self.tlen = self._cov_mtx.fix()
        
        
    def _execute(self, x):
        src = x[:, -self.output_dim:-self.L]
        exp = x[:, :-self.output_dim]
        inp = x[:, -self.L:]
        # result container
        result = mdp.numx.zeros((x.shape[0], self.output_dim))
        # project input on the plane orthogonal to the expanded sources
        result[:, -self.L:] = inp - mdp.utils.mult(exp, self.proj_mtx)
        # copy the sources
        result[:, :-self.L] = src
        return result

class NormalizeNode(mdp.Node):
    """Make input signal meanfree and unit variance"""
    def __init__(self, input_dim=None, output_dim=None, dtype=None):
        self._cov_mtx = mdp.utils.CovarianceMatrix(dtype)
        super(NormalizeNode, self).__init__(input_dim, output_dim, dtype)

    def _train(self, x):
        self._cov_mtx.update(x)
        
    def _stop_training(self):
        cov_mtx, avg, tlen = self._cov_mtx.fix()
        self.m = avg
        self.s = mdp.numx.sqrt(mdp.numx.diag(cov_mtx))

    def _execute(self,x):
        return (x-self.m)/self.s
