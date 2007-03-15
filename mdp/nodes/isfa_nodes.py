import sys as _sys
import mdp
from mdp import Node, NodeException, numx, numx_rand
from mdp.nodes import WhiteningNode
from mdp.utils import DelayCovarianceMatrix, MultipleCovarianceMatrices, \
     permute, rotate, mult


# rename often used functions
sum, cos, sin, PI = numx.sum, numx.cos, numx.sin, numx.pi

def _triu(m, k=0):
    """ returns the elements on and above the k-th diagonal of m.  k=0 is the
        main diagonal, k > 0 is above and k < 0 is below the main diagonal."""
    N = m.shape[0]
    M = m.shape[1]
    x = numx.greater_equal(numx.subtract.outer(numx.arange(N),
                                               numx.arange(M)),1-k)
    out = (1-x)*m
    return out

#############
class ISFANode(Node):
    """
    Perform Independent Slow Feature Analysis on the input data.

    References:
    Blaschke, T. , Zito, T., and Wiskott, L. (2007).
    Independent Slow Feature Analysis and Nonlinear Blind Source Separation.
    Neural Computation 19(4):994-1021 (2007)
    http://itb.biologie.hu-berlin.de/~wiskott/Publications/BlasZitoWisk2007-ISFA-NeurComp.pdf
    """
    def __init__(self, lags=1, whitened=False, icaweights=None,
                 sfaweights=None, verbose=False, sfa_ica_coeff=[1.,1.],
                 eps_contrast=1e-7, max_iter=10000, RP=None,
                 input_dim=None, output_dim=None, dtype=None):
        """
        - whitened == 1 if data are already whitened.
                        Otherwise the node will whiten the data itself.

        - lags          list of the time-lags for the time-delayed covariance
                        matrices. If lags is a number, time-lags 1,...,'lags'
                        are used.

        - weights       is an array with shape (lags+1,) of weights for the
                        time-delayed covariance matrices.

        - verbose == 1  print progress information.

        - eps_contrast  rotations converge when the maximum relative contrast
                        decrease of all pairs in a sweep is smaller than
                        eps_contrast
                        
        - max_iter      maximum number of iterations
        
        - sfa_ica_coeff (further normalization implied)
        
        - output_dim    fix the number of independent components to be
                        found a-priori
        """
        # check that the "lags" argument has some meaningful value
        if isinstance(lags, (int, long)):
            lags=range(1,lags+1)
        elif isinstance(lags, (list, tuple)):
            lags = numx.array(lags, "i")
        elif type(lags) is numx.ArrayType:
            if not (lags.dtype.char in ['i', 'l']):
                err_str = "lags must be integer!"
                raise NodeException, err_str
            else:
                pass
        else:
            err_str = "Lags must be int, list or array. Found "+\
                      "%s!"%(type(lags).__name__)
            raise NodeException, err_str
        self.lags = lags

        # sanity checks for weights
        if icaweights is None:
            self.icaweights = 1.
        else:
            if (len(icaweights) != len(lags)):
                err_str = "icaweights vector length is "+str(len(icaweights))+\
                          ", should be "+ str(len(lags))
                raise NodeException, err_str
            self.icaweights = icaweights
        if sfaweights is None:
            self.sfaweights = [0]*len(lags)
            self.sfaweights[0] = 1.
        else:
            if (len(sfaweights) != len(lags)):
                err_str = "sfaweights vector length is "+str(len(sfaweights))+\
                          ", should be "+ str(len(lags))
                raise NodeException, err_str
            self.sfaweights = sfaweights        

        # store attributes
        self.sfa_ica_coeff = sfa_ica_coeff
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps_contrast = eps_contrast

        # if input is not white, insert a WhiteningNode
        self.whitened = whitened
        if not whitened:
            self.white = WhiteningNode(input_dim=input_dim,\
                                       output_dim=input_dim,\
                                       dtype=dtype)

        # initialize covariance matrices
        self.covs = [ mdp.utils.DelayCovarianceMatrix(dt, dtype=dtype) \
                      for dt in lags ]

        # initialize the global rotation-permutation matrix
        # if not set that we'll eventually be an identity matrix
        self.RP = RP
        
        # initialize verbose structure to print nice and useful progress info
        if verbose:
            info = { 'sweep' : max(len(str(self.max_iter)),5),
                     'perturbe': max(len(str(self.max_iter)),5),
                     'float' : 5+8,
                     'fmt' : "%.5e",
                     'sep' : " | "}
            f1 = "Sweep".center(info['sweep'])
            f1_2 = "Pertb". center(info['perturbe'])
            f2 = "SFA part".center(info['float'])
            f3 = "ICA part".center(info['float'])
            f4 = "Contrast".center(info['float'])
            header = info['sep'].join([f1,f1_2,f2,f3,f4])
            info['header'] = header+'\n'
            info['line'] = len(header)*"-"
            self._info = info

        # finally call base class constructor
        super(ISFANode, self).__init__(input_dim, output_dim, dtype)

    def get_supported_dtypes(self):
        """Return the list of dtypes supported by this node."""
        return ['f','d']

    def _set_dtype(self, dtype):
        # when typecode is set, we set the whitening node if needed and
        # the SFA and ICA weights
        self._dtype = dtype
        if not self.whitened and self.white.dtype is None:
            self.white.dtype = dtype
        self.icaweights = numx.array(self.icaweights, dtype)
        self.sfaweights = numx.array(self.sfaweights, dtype)
        
    def _train(self, x):
        # train the whitening node if needed
        if not self.whitened: self.white.train(x)
        # update the covariance matrices
        [self.covs[i].update(x) for i in range(len(self.lags))]

    def _execute(self, x):
        # filter through whitening node if needed
        if not self.whitened: x = self.white.execute(x)
        # rotate input
        return mult(x,self.RP)

    def _inverse(self, y):
        # counter-rotate input
        x = mult(y, self.RP.T)
        # invert whitening node if needed
        if not self.whitened:
            x = self.white.inverse(x)
        return x

    def _fmt_prog_info(self, sweep, pert, contrast, sfa = None,ica = None):
        # for internal use only!
        # format the progress information
        # don't try to understand this code: it Just Works (TM)
        fmt = self._info
        sweep_str = str(sweep).rjust(fmt['sweep'])
        pert_str = str(pert).rjust(fmt['perturbe'])
        if sfa is None:
            sfa_str = fmt['float']*' '
        else:
            sfa_str = (fmt['fmt']%(sfa)).rjust(fmt['float'])
        if ica is None:
            ica_str = fmt['float']*' '
        else:
            ica_str = (fmt['fmt']%(ica)).rjust(fmt['float'])
        contrast_str = (fmt['fmt']%(contrast)).rjust(fmt['float'])
        table_entry = fmt['sep'].join([sweep_str,
                                       pert_str,
                                       sfa_str,
                                       ica_str,
                                       contrast_str])
        return table_entry
        
    def _get_eye(self):
        # return an identity matrix with the right dimensions and type
        return numx.eye(self.input_dim, dtype=self.dtype)
    
    def _get_rnd_rotation(self,dim):
        # return a random rot matrix with the right dimensions and type
        return mdp.utils.random_rot(dim, self.dtype)
    
    def _get_rnd_permutation(self,dim):
        # return a random permut matrix with the right dimensions and type
        zero = numx.zeros((dim,dim), dtype=self.dtype)
        row = numx_rand.permutation(dim)
        for col in range(dim):
            zero[row[col],col] = 1.
        return zero
        
    def _givens_angle(self, i, j, covs, bica_bsfa=None, complete=0):
        # Return the Givens rotation angle for which the contrast function
        # is minimal
        if bica_bsfa is None: bica_bsfa = self._bica_bsfa
        if j < self.output_dim:
            return self._givens_angle_case1(i, j, covs,
                                            bica_bsfa, complete=complete)
        else:
            return self._givens_angle_case2(i, j, covs,
                                            bica_bsfa, complete=complete)
        

    def _givens_angle_case2(self, m, n, covs, bica_bsfa, complete=0):
        # This function makes use of the constants computed in the paper
        #
        # R -> R
        # m -> \mu
        # n -> \nu
        #
        # Note that the minus sign before the angle phi is there because
        # in the paper the rotation convention is the opposite of ours.

        ncovs = covs.ncovs
        covs = covs.covs
        icaweights = self.icaweights
        sfaweights = self.sfaweights
        R = self.output_dim
        bica, bsfa = bica_bsfa

        Cmm, Cmn, Cnn = covs[m,m,:], covs[m,n,:], covs[n,n,:]
        d0 =   sum(sfaweights * Cmm*Cmm)
        d1 = 4*sum(sfaweights * Cmn*Cmm)
        d2 = 2*sum(sfaweights *(2*Cmn*Cmn + Cmm*Cnn))
        d3 = 4*sum(sfaweights * Cmn*Cnn)
        d4 =   sum(sfaweights * Cnn*Cnn)
        e0 = 2*sum(icaweights *(sum(covs[:R,m,:]*covs[:R,m,:],
                                    axis=0)-Cmm*Cmm))
        e1 = 4*sum(icaweights * (sum(covs[:R,m,:]*covs[:R,n,:],
                                     axis=0)-Cmm*Cmn))
        e2 = 2*sum(icaweights * (sum(covs[:R,n,:]*covs[:R,n,:],
                                     axis=0)-Cmn*Cmn))

        s22 = 0.25 * bsfa*(d1+d3)   + 0.5* bica*(e1)
        c22 = 0.5  * bsfa*(d0-d4)   + 0.5* bica*(e0-e2)
        s24 = 0.125* bsfa*(d1-d3)
        c24 = 0.125* bsfa*(d0-d2+d4)

        # Compute the contrast function in a grid of angles to find a
        # first approximation for the minimum.  Repeat two times
        # (effectively doubling the resolution). Note that we can do
        # that because we know we have a single minimum.
        #
        # npoints should not be too large otherwise the contrast
        # funtion appears to be constant. This is because we hit the
        # maximum resolution for the cosine function (ca. 1e-15)
        npoints = 100
        left = -PI/2 - PI/(npoints+1)
        right = PI/2 + PI/(npoints+1)
        for iter in [1,2]:
            phi = numx.linspace(left, right, npoints+3)
            contrast = c22*cos(-2*phi)+s22*sin(-2*phi)+\
                       c24*cos(-4*phi)+s24*sin(-4*phi)
            minidx = numx.argmin(contrast)
            left = phi[max(minidx-1,0)]
            right = phi[min(minidx+1,len(phi)-1)]

        # The contrast is almost a parabola around the minimum.
        # To find the minimum we can therefore compute the derivative
        # (which should be a line) and calculate its root.
        # This step helps to overcome the resolution limit of the
        # cosine function and clearly improve the final result.
        der_left = 2*c22*sin(-2*left)- 2*s22*cos(-2*left)+\
                   4*c24*sin(-4*left)- 4*s24*cos(-4*left)
        der_right = 2*c22*sin(-2*right)-2*s22*cos(-2*right)+\
                    4*c24*sin(-4*right)-4*s24*cos(-4*right)
        if abs(der_left - der_right) < 1e-8:
            minimum = phi[minidx]
        else:
            minimum = right - der_right*(right-left)/(der_right-der_left)

        dc = numx.zeros((ncovs,), dtype = self.dtype)
        for t in range(ncovs):
            dc[t] = sum(numx.diag(covs[:R,:R,t])**2, axis=0)
        dc = sum((dc-Cmm*Cmm)*sfaweights)
        
        ec = numx.zeros((ncovs,), dtype = self.dtype)
        for t in range(ncovs):
            ec[t] = sum([covs[i,j,t]**2 for i in range(R-1) \
                         for j in range(i+1,R) if i!=m and j!=m])
        ec = 2*sum(ec*icaweights)
        a20 = 0.125*bsfa*(3*d0+d2+3*d4+8*dc)+0.5*bica*(e0+e2+2*ec)
        minimum_contrast = a20+c22*cos(-2*minimum)+s22*sin(-2*minimum)+\
                           c24*cos(-4*minimum)+s24*sin(-4*minimum)
        if complete:
            # Compute the contrast between -pi/2 and pi/2
            # (useful for testing purposes)
            npoints = 1000
            phi = numx.linspace(-PI/2, PI/2,npoints+1)
            contrast = a20 + c22*cos(-2*phi) + s22*sin(-2*phi) +\
                       c24*cos(-4*phi) + s24*sin(-4*phi)
            return phi, contrast, minimum, minimum_contrast
        else: 
            return minimum, minimum_contrast

    
    def _givens_angle_case1(self, m, n, covs, bica_bsfa, complete=0):
        # This function makes use of the constants computed in the paper
        #
        # R -> R
        # m -> \mu
        # n -> \nu
        #
        # Note that the minus sign before the angle phi is there because
        # in the paper the rotation convention is the opposite of ours.
        ncovs = covs.ncovs
        covs = covs.covs
        icaweights = self.icaweights
        sfaweights = self.sfaweights
        bica, bsfa = bica_bsfa
        
        Cmm, Cmn, Cnn = covs[m,m,:], covs[m,n,:], covs[n,n,:]
        d0 =   sum(sfaweights * (Cmm*Cmm+Cnn*Cnn))
        d1 = 4*sum(sfaweights * (Cmm*Cmn-Cmn*Cnn))
        d2 = 2*sum(sfaweights * (2*Cmn*Cmn+Cmm*Cnn))
        e0 = 2*sum(icaweights * Cmn*Cmn)
        e1 = 4*sum(icaweights * (Cmn*Cnn-Cmm*Cmn))
        e2 =   sum(icaweights * ((Cmm-Cnn)*(Cmm-Cnn)-2*Cmn*Cmn))

        s24 = 0.25* (bsfa * d1    + bica * e1)
        c24 = 0.25* (bsfa *(d0-d2)+ bica *(e0-e2))

        # compute the exact minimum
        # Note that 'arctan' finds always the first maximum
        # because s24sin(4p)+c24cos(4p)=const*cos(4p-arctan)
        # the minimum lies +pi/4 apart (period = pi/2).
        # In other words we want that: abs(minimum) < pi/4
        phi4 = numx.arctan2(s24, c24)
        # use if-structure until bug in numx.sign is solved
        if  phi4 >= 0:
            minimum = -0.25*(phi4-PI)
        else:
            minimum = -0.25*(phi4+PI)

        # compute all constants:
        R = self.output_dim
        dc = numx.zeros((ncovs,), dtype = self.dtype)
        for t in range(ncovs):
            dc[t] = sum(numx.diag(covs[:R,:R,t])**2, axis=0)
        dc = sum((dc-Cnn*Cnn-Cmm*Cmm)*sfaweights)
        ec = numx.zeros((ncovs,), dtype = self.dtype)
        for t in range(ncovs):
            ec[t] = (sum(numx.ravel(_triu(covs[:R,:R,t],1))**2)-
                     covs[m,n,t]**2)
        ec = 2*sum(icaweights*ec)
        a20 = 0.25*(bsfa*(4*dc+d2+3*d0)+bica*(4*ec+e2+3*e0))
        minimum_contrast = a20+c24*cos(-4*minimum)+s24*sin(-4*minimum)
        npoints = 1000
        if complete == 1:
            # Compute the contrast between -pi/2 and pi/2
            # (useful for testing purposes)
            phi = numx.linspace(-PI/2, PI/2,npoints+1)
            contrast = a20 + c24*cos(-4*phi) + s24*sin(-4*phi)            
            return phi, contrast, minimum, minimum_contrast
        elif complete == 2:
            phi = numx.linspace(-PI/4, PI/4,npoints+1)
            contrast = a20 + c24*cos(-4*phi) + s24*sin(-4*phi)
            return phi, contrast, minimum, minimum_contrast
        else:
            return minimum, minimum_contrast
    

    def _get_contrast(self, covs, bica_bsfa = None):
        
        if bica_bsfa is None:
            bica_bsfa = self._bica_bsfa
        # return current value of the contrast
        R = self.output_dim
        ncovs = covs.ncovs
        covs = covs.covs
        icaweights = self.icaweights
        sfaweights = self.sfaweights
        # unpack the bsfa and bica coefficients
        bica, bsfa = bica_bsfa 
        sfa = numx.zeros((ncovs,),dtype=self.dtype)
        ica = numx.zeros((ncovs,),dtype=self.dtype)
        for t in range(ncovs):
            sq_corr =  covs[:R,:R,t]*covs[:R,:R,t]
            sfa[t]=numx.trace(sq_corr)
            ica[t]=2*numx.sum(numx.ravel(_triu(sq_corr,1)))
        return numx.sum(bsfa*sfaweights*sfa), numx.sum(bica*icaweights*ica)
        
    def _adjust_ica_sfa_coeff(self):
        # adjust sfa/ica ratio. ica and sfa term are scaled
        # differently because sfa accounts for the diagonal terms
        # whereas ica accounts for the off-diagonal terms
        ncomp = self.output_dim
        if ncomp > 1:
            bica =  self.sfa_ica_coeff[1]/(ncomp*(ncomp-1))
            bsfa = -self.sfa_ica_coeff[0]/ncomp
        else:
            bica =  0.#self.sfa_ica_coeff[1]
            bsfa = -self.sfa_ica_coeff[0]
        self._bica_bsfa = [bica, bsfa]

    def _fix_covs(self,covs=None):
        # fiv covariance matrices
        if covs is None:
            covs = self.covs
            if not self.whitened:
                white = self.white
                white.stop_training()
                proj = white.get_projmatrix(transposed=0)
            else:
                proj = None
            # fix and whiten the covariance matrices
            for i in range(len(self.lags)):
                covs[i], avg, avg_dt, tlen = covs[i].fix(proj)
        
            # send the matrices to the container class
            covs = MultipleCovarianceMatrices(covs)
            # symmetrize the cov matrices
            covs.symmetrize()
        self.covs = covs

            
    def _optimize(self):
        # optimize contrast function

        # save initial contrast
        sfa, ica = self._get_contrast(self.covs)
        self.initial_contrast = {'SFA': sfa,
                                 'ICA': ica,
                                 'TOT': sfa + ica}
        # info headers
        if self.verbose: print self._info['header']+self._info['line']

        # initialize control variables
        # contrast
        contrast = sfa+ica
        # local rotation matrix
        Q = self._get_eye()
        # local copy of correlation matrices
        covs = self.covs.copy()
        # maximum improvement in the contrast function
        max_increase = self.eps_contrast
        # Number of sweeps
        sweep = 0
        # flag for stopping sweeping
        sweeping = True
        # flag to check if we already perturbed the outer space
        perturbed = 0
        # if there is no outer space don't perturbe
        if self.input_dim == self.output_dim:
            perturbed = -1

        # main loop
        # we'll keep on sweeping until the contrast has improved less
        # then self.eps_contrast
        while sweeping:
            # update number of sweeps
            sweep += 1
            
            # perform a single sweep
            max_increase, covs, Q, contrast = self._do_sweep(covs, Q, contrast)

            if max_increase < 0 or contrast == 0:
                # we hit numerical precision, exit!
                sweeping = False
                if perturbed == 0:
                    perturbed = -1
                else:
                    perturbed = -perturbed
            if (max_increase < self.eps_contrast) and (max_increase) >= 0 :
                # rate of change is small for all pairs in a sweep
                if perturbed == 0:
                    # perturbe the outer space one time with a random rotation
                    perturbed = 1
                    part_sweep = sweep
                elif perturbed >= 1 and part_sweep == sweep-1:
                    # after pertubation no useful step has been done. exit!
                    sweeping = False
                elif perturbed < 0:
                    # we can't perturbe anymore
                    sweeping = False

            # perform perturbation if needed
            if perturbed >= 1 and sweeping is True:
                # generate a random rotation matrix for the external subspace
                PRT = self._get_eye()
                size = self.input_dim-self.output_dim
                rot = self._get_rnd_rotation(size)
                # generate a random permutation matrix for the ext. subspace
                perm = self._get_rnd_permutation(size)
                # combine rotation and permutation
                rot_perm = mult(rot,perm)
                # apply rotation+permutation
                PRT[self.output_dim:,self.output_dim:] = rot_perm
                covs.transform(PRT)
                Q = mult(Q, PRT)
                # increment perturbation counter
                perturbed += 1

            # verbose progress information
            if self.verbose:
                table_entry = self._fmt_prog_info(sweep,
                                                  perturbed,
                                                  contrast)
                _sys.stdout.write(table_entry+len(table_entry)*'\b')
                _sys.stdout.flush()            

            # if we made too many sweeps exit with error!
            if sweep == self.max_iter:
                err_str = "Failed to converge, maximum increase= "+\
                          "%.5e"%(max_increase)
                raise NodeException, err_str

        # if we land here, we have converged
        # calculate output contrast
        
        sfa, ica =  self._get_contrast(covs)
        contrast = sfa+ica
        # print final information
        if self.verbose:
            print self._fmt_prog_info(sweep,perturbed,
                                                   contrast,sfa,ica)
            print self._info['line']

        self.final_contrast = {'SFA': sfa,
                               'ICA': ica,
                               'TOT': sfa + ica}
        
        # finally return optimal rotation matrix
        return Q

    def _do_sweep(self, covs, Q, prev_contrast):
        #perform a single sweep
        # maximal improvement in a single sweep
        max_increase = -1
        # shuffle rotation order
        numx_rand.shuffle(self.rot_axis)
        for (i,j) in self.rot_axis:
            # get the angle that minimizes the contrast
            # and the contrast value
            angle, contrast = self._givens_angle(i, j, covs)
            if contrast == 0:
                # we hit numerical precision in case when b_sfa == 0
                max_increase = -1
                break
            # relative improvement in the contrast function
            relative_diff = (prev_contrast-contrast)/abs(prev_contrast)

            if relative_diff < 0:
                # if rate of change is negative we hit numerical precision
                # or we already sit on the optimum for this pair of axis.
                # don't rotate anymore and go to the next pair
                continue

            # update the rotation matrix
            rotate(Q, angle, [i, j])
            # rotate the covariance matrices
            covs.rotate(angle, [i, j])

            # store maximum and previous rate of change
            max_increase = max(max_increase,relative_diff)
            prev_contrast = contrast
            
        return max_increase, covs, Q, contrast
                
    def _stop_training(self, covs=None, adjust=True):
        # fix, whiten, symmetrize and weight the covariance matrices
        # the functions sets also the number of input components self.ncomp
        self._fix_covs(covs)
        # if output_dim were not set, set it to be the number of input comps
        if self.output_dim is None:
            self.output_dim = self.input_dim
        # adjust b_sfa and b_ica
        self._adjust_ica_sfa_coeff()
        # maximum number of independent components in the output
        ind_comp = min(self.output_dim, self.input_dim)
        # initialize all possible rotation axes
        self.rot_axis = [(i, j) for i in range(0, ind_comp) \
                         for j in range(i+1, self.input_dim)]

        # initialize the global rotation-permutation matrix (RP):
        RP = self.RP
        if RP is None:
            RP = self._get_eye()
        else:
            # apply the global rotation matrix
            self.covs.transform(RP)

        # find optimal rotation
        Q = self._optimize()
        RP = mult(RP, Q)
        # rotate and permute the covariance matrices
        # we do it here in one step, to avoid the cumulative errors
        # of multiple rotations in _optimize
        self.covs.transform(Q)

        if adjust:
            # Reduce dimension to match output_dim#
            RP = RP[:,:self.output_dim]
            # the variance for the derivative of a whitened signal is
            # 0 <= v <= 4, therefore the diagonal elements of the delayed
            # covariance matrice with time lag = 1 (covs[:,:,0]) are
            # -1 <= v' <= +1
            # reorder the components to have them ordered by slowness
            d = numx.diag(self.covs.covs[:self.output_dim,:self.output_dim,0])
            idx = numx.argsort(d)[::-1]
            self.RP = numx.take(RP,idx,axis=1)
            #del self.covs
        else:
            self.RP = RP
