import mdp

class XSFANode(mdp.Node):
    def __init__(self, nsources, input_dim=None, dtype=None,
                 verbose=False):
        self.count = 0
        self.basic_exp = mdp.nodes.PolynomialExpansionNode
        self.basic_exp_args = (2,) # def 5
        self.basic_exp_kwargs = {}        
        self.exp = mdp.nodes.PolynomialExpansionNode
        self.exp_args = (10,) # def 10
        self.exp_kwargs = {}
        self.n_extracted_src = 0
        self.flow = None
        self.verbose = verbose
        super(XSFANode, self).__init__(input_dim=input_dim,
                                       output_dim=nsources, dtype=dtype)
        # set the number of training phases
        self._training_phases = self._get_src_extr_mod_train_phases()
        self._training_phases_mods = [sum(self._training_phases[:i+1]) for i in range(len(self._training_phases[:-1]))]

    def _set_output_dim(self, n):
        if self._output_dim is None:
            self._output_dim = n
        elif self._output_dim != n:
            msg = "Output dim can only be set at instantiation!"
            raise mdp.NodeException(msg)

    def _get_supported_dtypes(self):
        """Return the list of dtypes supported by this node."""
        return ('float32', 'float64')
    
    def is_invertible(self):
        return False

    def _get_train_seq(self):
        return [(self._train, self._stop_training)]*sum(self._training_phases)
    
    def _train(self, x):
        if self.flow is None:
            if self.verbose:
                print "Extracting source 1..."
            self.basic_exp_kwargs['input_dim']=self.input_dim
            self.basic_exp_instance = self.basic_exp(*self.basic_exp_args, **self.basic_exp_kwargs)
            input_dim_mod = self.basic_exp_instance.output_dim
            self.flow = self.basic_exp_instance + self.get_source_extractor_module(input_dim_mod, 0)
        self.flow[-1].train(self.flow[:-1](x))


    def _stop_training(self):
        self.flow[-1].stop_training()
        cur_tr_ph = self.get_current_train_phase() + 1
        if cur_tr_ph in self._training_phases_mods and self.n_extracted_src != (self.output_dim - 1):
            self.n_extracted_src += 1
            self.flow.append(self.get_source_extractor_module(self.flow[-1].output_dim, self.n_extracted_src))
            if self.verbose:
                print "Extracting source %d..."%(self.n_extracted_src+1)
##         html_file = open('hinet_test.html', 'w')
##         html_file.write('<html>\n<head>\n<title>HiNet Test</title>\n')
##         html_file.write('<style type="text/css" media="screen">')
##         html_file.write(mdp.hinet.HINET_STYLE)
##         html_file.write('</style>\n</head>\n<body>\n')
##         hinet_translator = mdp.hinet.HiNetHTMLTranslator()
##         hinet_translator.write_flow_to_file(self.flow, html_file)
##         html_file.write('</body>\n</html>')
##         html_file.close()
            
    def _execute(self, x):
        return self.flow(x)[:,:self.output_dim]

    def get_source_extractor_module(self, dim, nsources):
        S = nsources
        L = dim-S

        # sfa - extracts the next source
        sfa = mdp.nodes.SFANode(input_dim=L, output_dim=L)
        
        # identity - copies the new sources
        idn_new = mdp.nodes.IdentityNode(input_dim=S+1, output_dim=S+1)
        # source expansion
        self.exp_kwargs['input_dim'] = S + 1
        #N2
        src_exp = mdp.hinet.FlowNode(self.exp(*self.exp_args, **self.exp_kwargs) +
                           NormalizeNode() + 
                           mdp.nodes.WhiteningNode(svd=True, reduce=True))
        N2Layer = mdp.hinet.SameInputLayer((src_exp, idn_new))
        N2ContLayer = mdp.hinet.Layer((N2Layer, mdp.nodes.IdentityNode(input_dim=L-1, output_dim=L-1)))

        if S == 0:
            # don't need to copy the current sources (there are none)
            N1 = mdp.hinet.FlowNode(sfa + N2ContLayer)
        elif S == self.output_dim - 1:
            # the last source does not need to be removed
            # take care of passing the sources down along the flow
            idn_old = mdp.nodes.IdentityNode(input_dim=S, output_dim=S)
            return mdp.hinet.Layer((idn_old, mdp.nodes.SFANode(input_dim=L, output_dim=1)))
        else:
            # take care of passing the sources down along the flow
            idn_old = mdp.nodes.IdentityNode(input_dim=S, output_dim=S)
            N1 = mdp.hinet.FlowNode(mdp.hinet.Layer((idn_old, sfa)) + N2ContLayer)


        # layer
        #sw1 = mdp.hinet.Switchboard(dim, range(S, dim))
        #l1 = mdp.hinet.SameInputLayer((N1, sw1))

        # expanded sources projection
        proj = ProjectionNode(S, L-1)
        
        # regularization after projection + new source copying
        reg_and_copy = mdp.hinet.Layer((idn_new, mdp.nodes.WhiteningNode(input_dim=L-1, svd=True, reduce=True)))
        # actual source removal flow 
        src_rem = mdp.hinet.FlowNode( proj + reg_and_copy )

        # actual source extraction module 
        return mdp.hinet.FlowNode(N1 + src_rem)

    def _get_src_extr_mod_train_phases(self):
        training_phases = [] 
        for S in range(self.output_dim):
            mod = self.get_source_extractor_module(S+1, S)
            training_phases.append(len(mod._train_seq))
        return training_phases

class ProjectionNode(mdp.Node):
    def __init__(self, S, L):
        #!! IMPORTANT!!
        # this node *must* return the sources together with the
        # projected input signals
        self.proj_mtx = None
        self.L = L
        super(ProjectionNode, self).__init__(output_dim=S+1+L)
        #self.output_dim = S + 1 + L
        self._cov_mtx = mdp.utils.CrossCovarianceMatrix(self.dtype)

    def _train(self, x):
        # compute covariance between expanded sources
        # and input signals
        self._cov_mtx.update(x[:,:-self.output_dim], x[:,-self.L:])

    def _stop_training(self):
         self.proj_mtx, avgx, avgy, self.tlen = self._cov_mtx.fix()
         
        
    def _execute(self, x):
        src = x[:,-self.output_dim:-self.L]
        exp = x[:,:-self.output_dim]
        inp = x[:,-self.L:]
        # result container
        result = mdp.numx.zeros((x.shape[0],self.output_dim))
        # project input on the plane orthogonal to the expanded sources
        result[:,-self.L:] = inp - mdp.utils.mult(exp, self.proj_mtx)
        # copy the sources
        result[:,:-self.L] = src
        return result

class NormalizeNode(mdp.Node):

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
