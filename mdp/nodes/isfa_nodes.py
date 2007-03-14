import sys as _sys

###################################################################
### REMEMBER TO REMOVE THE NEED FOR import random #################
###################################################################
import random

import mdp
from mdp import Node, NodeException
from mdp.nodes import WhiteningNode
from mdp.utils import DelayCovarianceMatrix, MultipleCovarianceMatrices, permute, rotate, mult

numx = mdp.numx
PI = numx.pi
tr = numx.transpose

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
    ISFANode receives a multidimensional input signal and performs
    Independent Slow Feature Analysis, using the algorithm by
    by Tobias Blaschke.

    References:
    Blaschke, T. and Wiskott, L. (2004).
    Independent Slow Feature Analysis and Nonlinear Blind Source Separation.
    5th Int. Conf. on Independent Component Analysis and Blind Signal
    Separation, ICA'04.
    http://itb.biologie.hu-berlin.de/~blaschke/publications/isfa.pdf 
    """
    def __init__(self, lags=1, whitened=False, icaweights=None, sfaweights=None,
                 verbose=False, sfa_ica_coeff=[1.,1.], eps_contrast=1e-7,
                 max_iter=10000, RP = None, shuffle = 1, first_fixed = 0,
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
        
        - threshold     ???
        - ratio         sfa/ica contrast term relative weight
        - limit1        ??? ???
        - limit2        ??? resolution to get the contrast ???
        - perturbe      ??? rotate all??? [ avoid local minima]
        - output_dim    fix the number of independent components to be
                        found a-priori
        """
        # check that the lags argument has some meaningful value
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
        self.sfa_ica_coeff = sfa_ica_coeff
        self.max_iter = max_iter
        self.verbose = verbose
        self.shuffle = shuffle
        self.first_fixed = first_fixed
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
        self.eps_contrast = eps_contrast
        self.whitened = whitened
        if not whitened:
            self.white = WhiteningNode(input_dim=input_dim,\
                                       output_dim=input_dim,\
                                       dtype=dtype)

        self.covs = [ mdp.utils.DelayCovarianceMatrix(dt, dtype=dtype) \
                      for dt in lags ]
        self.RP = RP
        super(ISFANode, self).__init__(input_dim, output_dim, dtype)

    def get_supported_dtypes(self):
        """Return the list of dtypes supported by this node."""
        return ['f','d']

    def _set_dtype(self, dtype):
        self._dtype = dtype
        if not self.whitened and self.white.dtype is None:
            self.white.dtype = dtype
        self.icaweights = numx.array(self.icaweights, dtype)
        self.sfaweights = numx.array(self.sfaweights, dtype)
        
    def _train(self, x):
        # train the whitening node if needed
        if not self.whitened: self.white.train(x)
        # update the covariance matrices
        # ??? maybe use an implicit loop ???
        for i in range(len(self.lags)):
            self.covs[i].update(x)

    def _execute(self, x):
        if not self.whitened: x = self.white.execute(x)
        return mult(x,self.RP)

    def _inverse(self, y):
        x = mult(y, self.RP.T)
        if not self.whitened:
            x = self.white.inverse(x)
        return x

    def _fmt_prog_info(self, sweep, pert, contrast, sfa = None,ica = None):
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
        """Return an identity matrix
        """
        return numx.eye(self.input_dim, dtype=self.dtype)
    
    def _get_rnd_rotation(self,dim):
        """Return random rotation matrix
        """
        return mdp.utils.random_rot(dim, self.dtype)
    
    def _get_rnd_permutation(self,dim):
        """Return a random permutation matrix
        """
        zero = numx.zeros((dim,dim), dtype=self.dtype)
        row = mdp.numx_rand.permutation(dim)
        for col in range(dim):
            zero[row[col],col] = 1.
        return zero
        
    def _givens_angle(self, i, j, covs, bica_bsfa, complete=0):
        """Return the Givens rotation angle for which the contrast function
        is minimal."""
        if j < self.output_dim:
            return self._givens_angle_case1(i, j, covs,
                                            bica_bsfa, complete=complete)
        else:
            return self._givens_angle_case2(i, j, covs,
                                            bica_bsfa, complete=complete)
        

    def _givens_angle_case2(self, m, n, covs, bica_bsfa, complete=0):
        # This functions makes use of the constants computed in
        # T. Blaschke's PhD thesis, Appendix D. All constants' names
        # are defined as in that thesis.
        # We consider here only Case 2 and compute the contrast directly
        # from Equation D.1 instead than from Equation D.4 .
        #
        # R -> R
        # m -> \mu
        # n -> \nu
        #
        # Also note that the minus sign before the angle phi is there because
        # in the paper the rotation convention is the opposite of ours.

        sum, cos, sin, out = (numx.sum, numx.cos, numx.sin,
                              numx.outer)
        ncovs = covs.ncovs
        covs = covs.covs
        icaweights = self.icaweights
        sfaweights = self.sfaweights
        R = self.output_dim
        bica, bsfa = bica_bsfa

        # compute the constants of Table D.1
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

        # compute the constants of Table D.2
        s22 =  0.25* bsfa*(d1+d3)   + 0.5* bica*(e1)
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

        # compute all constants:
        dc = numx.zeros((ncovs,), dtype = self.dtype)
        for t in range(ncovs):
            dc[t] = sum(numx.diag(covs[:R,:R,t])**2, axis=0)
        dc = sum((dc-Cmm*Cmm)*sfaweights)
        
        ec = numx.zeros((ncovs,), dtype = self.dtype)
        for t in range(ncovs):
            ec[t] = sum([covs[i,j,t]**2 for i in range(R-1) \
                         for j in range(i+1,R) if i!=m and j!=m])
            #ec[t] = (sum(numx.ravel(_triu(covs[:R,:R,t],1))**2)-
            #         covs[m,n,t]**2)
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
        # This functions makes use of the constants computed in
        # T. Blaschke's PhD thesis, Appendix D. All constants' names
        # are defined as in that thesis.
        # We consider here only Case 2 and compute the contrast directly
        # from Equation D.1 instead than from Equation D.4 .
        #
        # R -> R
        # m -> \mu
        # n -> \nu
        #
        # Also note that the minus sign before the angle phi is there because
        # in the paper the rotation convention is the opposite of ours.

        sum, cos, sin, out = (numx.sum, numx.cos, numx.sin,
                              numx.outer)
        
        ncovs = covs.ncovs
        covs = covs.covs
        icaweights = self.icaweights
        sfaweights = self.sfaweights
        bica, bsfa = bica_bsfa
        
        # compute the constants of Table D.1
        Cmm, Cmn, Cnn = covs[m,m,:], covs[m,n,:], covs[n,n,:]
        d0 =   sum(sfaweights * (Cmm*Cmm+Cnn*Cnn))
        d1 = 4*sum(sfaweights * (Cmm*Cmn-Cmn*Cnn))
        d2 = 2*sum(sfaweights * (2*Cmn*Cmn+Cmm*Cnn))
        e0 = 2*sum(icaweights * Cmn*Cmn)
        e1 = 4*sum(icaweights * (Cmn*Cnn-Cmm*Cmn))
        e2 =   sum(icaweights * ((Cmm-Cnn)*(Cmm-Cnn)-2*Cmn*Cmn))

        # compute the constants of Table D.2
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
    

    def _get_contrast(self, covs, bica_bsfa):
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
        
    def _get_ica_sfa_coeff(self):
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
        return [bica,bsfa]

    def _fix_covs(self,covs=None):
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

    def _find_best_rotation(self, comp, bica_bsfa, best_contrast):
        # comp == index of the component to be learned. 
        # start_axis == ???
        # ??? should we take as axis1 a linear combination of
        # the remaining axes instead ???
        best_QP = self._get_eye()
        start_contrast = best_contrast
        if self.verbose: print self._info['header']+self._info['line']
        # uncomment this to remove permutations
        for start_axis in range(comp, comp+1):
        #################################for start_axis in range(comp, self.input_dim):
        #for start_axis in range(self.input_dim):
            # find and perform optimal rotation  for the selected axes
            covs, QP, sweep, perturbed = self._local_rotation(comp, start_axis,
                                                   bica_bsfa, start_contrast)
            # calculate contrast
            sfa, ica =  self._get_contrast(covs, bica_bsfa)
            contrast = sfa+ica
            # keep the resulting rotation matrix only if the
            # contrast is better than before
            if contrast < best_contrast:
                best_contrast = contrast
                best_QP = QP
            # print progress information
            if self.verbose: print self._fmt_prog_info(sweep,perturbed,
                                                       contrast,sfa,ica)

        if self.verbose: print self._info['line']
        return best_QP, best_contrast
            
    def _local_rotation(self,comp,start_axis, bica_bsfa, start_contrast):
        # local rotation matrix
        Q = self._get_eye()
        # local copy of correlation matrices
        covs = self.covs.copy()
        # permute rotation matrix and covariance matrices
        permute(Q, [comp,start_axis], rows=1, cols=0)
        covs.permute([comp,start_axis])
        max_increase = self.eps_contrast
        sweep = 0
        sweeping = 1
        perturbed = 0
        if self.input_dim == self.output_dim: perturbed = -1
        while sweeping:
            sweep += 1
            # perform a single sweep
            max_increase, covs, Q,start_contrast=self._do_sweep(comp,
                                                                covs, Q,
                                                                bica_bsfa,
                                                                start_contrast)
            # if rate of change is small for all pairs in a sweep
            # then perturbe the outer space one time with a
            # random rotation
            #if max_increase == -1:
            if max_increase < 0 or start_contrast == 0:
                # we hit numerical precision, exit!
                sweeping = 0
                if perturbed == 0:
                    perturbed = -1
                else:
                    perturbed = -perturbed
            if (max_increase < self.eps_contrast) and (max_increase) >= 0 :
                if perturbed == 0:
                    # we did not perturbe yet. do it!
                    perturbed = 1
                    part_sweep = sweep
                elif perturbed >= 1 and part_sweep == sweep-1:
                    # after pertubation no useful step has been done. exit!
                    sweeping = 0
                elif perturbed < 0:
                    sweeping = 0
            if perturbed >= 1 and sweeping == 1:
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
                perturbed += 1
            # verbose progress information
            if self.verbose:
                table_entry = self._fmt_prog_info(sweep,
                                                  perturbed,
                                                  start_contrast)
                _sys.stdout.write(table_entry+len(table_entry)*'\b')
                _sys.stdout.flush()            

            # if we made too many sweeps exit with error!
###############################################################################
# RIGHT CODE
##             if sweep == self.max_iter:
##                 err_str = "Failed to converge, maximum increase= "+\
##                           "%.5e"%(max_increase)
##                 raise NodeException, err_str
##                         # perturbation matrix
###############################################################################
###############################################################################
# WRONG CODE
            if sweep == self.max_iter:
                sweeping = 0
                        # perturbation matrix
###############################################################################
            
        return covs, Q, sweep, perturbed

    def _do_sweep(self, comp, covs, Q, bica_bsfa, prev_contrast):
        """Perform a single sweep. """
        max_increase = -1
        # uncomment the following line to disable "rotate all"
        #for i in [comp]:
        if self.shuffle:
            random.shuffle(self.rot_axis)
        for (i,j) in self.rot_axis:
            # get the angle that minimizes the contrast
            # and the contrast value
            angle, contrast = self._givens_angle(i, j,
                                                 covs, bica_bsfa)
            if contrast == 0:
                max_increase = -1
                break
            relative_diff = (prev_contrast-contrast)/abs(prev_contrast)
            if relative_diff < 0:
                # if rate of change is negative we hit numerical precision
                # or we already sit on the optimum
                continue
            # update the rotation matrix
            rotate(Q, angle, [i, j])
            # rotate the covariance matrices
            covs.rotate(angle, [i, j])
            # check the rate of change
            max_increase = max(max_increase,relative_diff)
            prev_contrast = contrast

        # if rate of change is negative we hit numerical precision or
        # if contrast == 0 and bsfa = 0 : we can't do better
        # return a max_increase == -1 (to exit sweeping)
##         if (max_increase < 0) or (contrast == 0 and bica_bsfa[1] == 0):
##             print "\n|%d-%d| %.15e %.15e"%(i, j, angle, contrast)
##             covs.rotate(-angle, [i,j])
##             covs_rot = covs.copy()
##             angle_list, contrast_list, minimum, minimum_contrast = \
##                    self._givens_angle(i, j, covs, bica_bsfa, 2)
##             scipy.gplt.plot(angle_list, contrast_list)
##             angle_est = angle_list[scipy.argmin(contrast_list)]
##             contrast_est = scipy.amin(contrast_list)
##             print "|%d-%d| %.15e %.15e"%(i, j, angle_est, contrast_est)
##             covs.rotate(angle, [i, j])
##             contrast_est2 = self._get_contrast(covs, bica_bsfa)
##             print "|%d-%d| %.15e %.15e"%(i, j, angle,
##                                          scipy.sum(contrast_est2))
##             contrast_list2 = []
##             for phi in angle_list:
##                 covs_cp = covs_rot.copy()
##                 covs_cp.rotate(phi, [i,j])
##                 contrast_list2.append(scipy.sum(\
##                     self._get_contrast(covs_cp, bica_bsfa)))
##             scipy.gplt.hold('on')
##             scipy.gplt.plot(angle_list, contrast_list2)
##             scipy.gplt.hold('off')
##             raw_input('PRESS ENTER')
##             return -1, covs, Q, contrast
        return max_increase, covs, Q, contrast
                
    def _stop_training(self, covs=None, cut=1):
        # fix, whiten, symmetrize and weight the covariance matrices
        # the functions sets also the number of input components self.ncomp
        self._fix_covs(covs)
        # if output_dim were not set, set it to be the number of input comps
        if self.output_dim is None:
            self.output_dim = self.input_dim
        # initialize the global rotation-permutation matrix (RP):
        #RP = self._get_eye()
        # use eigenvectors of the first matrix: should be better!
        #dummy, RP = mdp.utils.symeig(self.covs.covs[:,:,0], overwrite=0)
        #RP = scipy.linalg.orth(scipy.rand(self.input_dim,self.input_dim)*2-1)
        RP = self.RP
        if RP is None: RP = self._get_eye()
        self.covs.transform(RP)
        # get b_ica and b_sfa coefficients
        bica_bsfa = self._get_ica_sfa_coeff()
        # initialize best_contrast
        s_sfa, s_ica = self._get_contrast(self.covs, bica_bsfa)
        best_contrast = s_sfa + s_ica
        self.initial_contrast = (s_sfa, s_ica, s_sfa + s_ica)
        if self.verbose: print "initial contrast = %.15e %.15e %.15e"%(s_sfa,
                                                                       s_ica,
                                                                       best_contrast)
        # Start the search for independent components. One after the other.
        # We can find at most min(outputdim, inputdim) components
        # ??? (the last one is fixed due to the orthogonality constraint if
        # rotate_all=0) ???
        last_comp = min(self.output_dim, self.input_dim)
        ##############################for comp in range(last_comp):
        if self.first_fixed:
            self.rot_axis = [(i, j) for i in range(1, last_comp) \
                             for j in range(i+1, self.input_dim)]
        else:
            self.rot_axis = [(i, j) for i in range(0, last_comp) \
                             for j in range(i+1, self.input_dim)]
        for comp in range(last_comp-1, last_comp):
            #if self.verbose: print "components = %d"%(comp+1)
            best_QP, best_contrast=self._find_best_rotation(comp,
                                                            bica_bsfa,
                                                            best_contrast)
            if self.verbose: print "bast contrast = %.15e"%best_contrast
            # rotate and permute the covariance matrices using the
            # best rotation-permutation found
            self.covs.transform(best_QP)
            RP = mult(RP, best_QP)
            sfa, ica = self._get_contrast(self.covs, bica_bsfa)
            if self.verbose: print "best contrast = %.15e %.15e %.15e"%(sfa,
                                                                        ica,
                                                                        sfa+ica)

            self.best_contrast = (sfa, ica, sfa+ica)

        if cut==1:
            # save the final rotation-permutation matrix
            ### ??? Reduce dimension to match output_dim ??? ###
            #self.RP = RP
            RP = RP[:,:self.output_dim]
            # the variance for the derivative of a whitened signal is
            # 0 <= v <= 4, therefore the diagonal elements of the delayed
            # covariance matrice with time lag = 1 (covs[:,:,0]) are
            # -1 <= v' <= +1
            # reorder the components to have them ordered by slowness
            d = numx.diag(self.covs.covs[:self.output_dim,:self.output_dim,0])
            idx = numx.argsort(d)[::-1]
            self.RP = numx.take(RP,idx,axis=1)
            # clean
            #del self.covs
        else:
            self.RP = RP

## def _get_analytical_solution(nsources, nmat, dim, ica_ambiguity):
##     # build a sequence of random diagonal matrices
##     matrices = [scipy.eye(dim).astype('d')]*nmat
##     # build first matrix:
##     #   - create random diagonal with elements
##     #     in [-1, 1]
##     diag = (scipy.rand(dim)-0.5)*2
##     #   - sort it in descending order (in absolute value)
##     #     [large first]
##     diag = scipy.take(diag, scipy.argsort(abs(diag)))[::-1]
##     #   - save larger elements [sfa solution] 
##     sfa_solution = diag[:nsources].copy()
##     #   - modify diagonal elements order to allow for a
##     #     different solution for isfa:
##     #     create index array
##     idx = range(0,dim)
##     #     take the second slowest element and put it at the end
##     idx = [idx[0]]+idx[2:]+[idx[1]]
##     diag = scipy.take(diag, idx)
##     #   - save isfa solution
##     isfa_solution = diag[:nsources]
##     #   - set the first matrix
##     matrices[0] = matrices[0]*diag
##     # build other matrices
##     diag_dim = nsources+ica_ambiguity 
##     for i in range(1,nmat):
##         # get a random symmetric matrix
##         matrices[i] = mdp.utils.symrand(dim)
##         # diagonalize the subspace diag_dim
##         tmp_diag = (scipy.rand(diag_dim)-0.5)*2
##         matrices[i][:diag_dim,:diag_dim] = scipy.diag(tmp_diag)
##     # put everything in MultipleCovarianceMatrices
##     matrices = MultipleCovarianceMatrices(matrices)
##     return matrices, sfa_solution, isfa_solution

## def _get_matrices_contrast(matrices, nsources, dim, sfa_ica_coeff):
##     isfa = ISFANode(matrices.ncovs, whitened=1, verbose=0,
##                     sfa_ica_coeff = sfa_ica_coeff,
##                     output_dim = nsources)
##     isfa.train(scipy.rand(100, dim))
##     bica_bsfa = isfa._get_ica_sfa_coeff()
##     return isfa._get_contrast(matrices, bica_bsfa)
    
## def _unmixing_error(nsources, goal, estimate):
##     check = mult(goal[:nsources,:], estimate[:,:nsources])
##     error = (abs(scipy.sum(scipy.sum(abs(check),axis=1)-1))+
##              abs(scipy.sum(scipy.sum(abs(check),axis=0)-1)))
##     error /= nsources*nsources
##     return error

def _test_analytical(nsources, nmat, deg, ica_ambiguity, trials, tol):
    import sys
    from testsuite_isfa import _get_analytical_solution, \
         _get_matrices_contrast, _unmixing_error 
    # sfa_ica coefficient
    sfa_ica_coeff = [1., 1.]
    print 'nsources =', nsources
    print 'matrices =', nmat
    print 'degree =', deg
    # dimensions of expanded space
    dim = mdp.nodes.expansion_nodes.expanded_dim(deg, nsources)
    print 'expanded dim =', dim
    print '[b_SFA, b_ICA] =', sfa_ica_coeff
    assert (nsources+ica_ambiguity) < dim, 'Too much ica ambiguity.'
    success = 0
    failures = 0
    sfa_list, ica_list = [],[]
    error_list = []
    for trial in range(trials):
        scipy.stats.seed(1303866164, 119970951)
        print 'Random seed 1:',scipy.stats.get_seed()
        seed = random.randint(-2**31, 2**31-1)
        random.seed(seed)
        print 'Random seed 2:', seed
        # get analytical solution:
        # prepared matrices, solution for sfa, solution for isfa
        covs, sfa_solution, isfa_solution = _get_analytical_solution(nsources,
                                                       nmat, dim,ica_ambiguity)

        # get contrast of analytical solution
        sfasrc, icasrc = _get_matrices_contrast(covs, nsources, dim,
                                                sfa_ica_coeff)
        #print "Contrast SRC = %.15e, %.15e"%(sfasrc, icasrc)
        # set up contrast and error lists
        sfa_list.append(sfasrc)
        ica_list.append(icasrc)
        # set rotation matrix
        R = mdp.utils.random_rot(dim)
        covs_rot = covs.copy()
        # rotate the analytical solution
        covs_rot.transform(R)
        # find the SFA solution to initialize ISFA
        eigval, SFARP = mdp.utils.symeig(covs_rot.covs[:,:,0])
        # order SFA solution by slowness
        SFARP = SFARP[:,-1::-1]
        # generate list of random rotation matrices to start the algorithm with
        # first matrix is the unit matrix
        #RandRot = [scipy.eye(dim)]
        #RandRot.extend(map(mdp.utils.random_rot, [dim]*(trials-1)))
        #for trial, rand_rot in enumerate(RandRot):
        print 'Trial = ', trial+1
        # run ISFA
        isfa = ISFANode(covs_rot.ncovs, whitened = 1,
                        sfa_ica_coeff = sfa_ica_coeff,
                        eps_contrast = 1e-7,
                        output_dim = nsources,
                        max_iter = 500,
                        shuffle = 1, 
                        verbose = 0,
                        RP = SFARP)
        isfa.train(scipy.rand(100,dim))
        isfa.stop_training(covs = covs_rot.copy(), cut =0)
        # check that the rotation matrix found by ISFA is R
        # up to a permutation matrix.
        # Unmixing error as in Tobias paper
        error = _unmixing_error(nsources, R, isfa.RP)
        error_list.append(error)
        if error < tol:
            success += 1
        else:
            failures += 1
        print "error = ", error
        print "perf = %dS/%dF"%(success, failures)
        print "----"
    return error_list

if __name__ == '__main__':
    # number of source signals
    nsources = 2
    # number of time lags
    nmat = 50
    # degree of polynomial expansion
    deg = 3
    # sfa_ica coefficient
    sfa_ica_coeff = [1., 1.]
    # how many independent subspaces in addition to the sources
    ica_ambiguity = 2
    # trials
    trials = 10000
    tol = 1e-5
    #error_list = _test_analytical(nsources, nmat, deg, ica_ambiguity, trials, tol)
    #print "Maximum error: %.15e"%max(error_list)
    #############################################################
    from testsuite_isfa import _get_analytical_solution, \
         _get_matrices_contrast, _unmixing_error
    nsources = 5
    #deg = 3
    nmats = range(2,51) 
    sfa_ica_coeff = [0., 1.]
    dim = 20 #mdp.nodes.expansion_nodes.expanded_dim(deg, nsources)
    trials = 100
    for nsources in range(10, 21):
        conts = []
        for nmat in nmats:
            # create random symmetric matrices
            matrices = [mdp.utils.symrand(dim) for i in range(nmat)]
            matrices = MultipleCovarianceMatrices(matrices)
            # set random starting point
            # R = mdp.utils.random_rot(dim)
            loc_mat = matrices.copy()
            # loc_mat.transform(R)
            # R = mdp.numx.eye(dim)
            isfa = ISFANode(matrices.ncovs,
                            whitened = 1,
                            sfa_ica_coeff = sfa_ica_coeff,
                            eps_contrast = 1e-7,
                            output_dim = nsources,
                            max_iter = 500,
                            shuffle = 1, 
                            verbose = 0,
                            RP = None)
            isfa.train(scipy.rand(100,dim))
            isfa.stop_training(covs = loc_mat, cut =0)
            # eig = []
            #for mat in range(nmat):
            #    eig.append(mdp.utils.symeig(isfa.covs[mat], eigenvectors=0, overwrite=0))
            #errors = []
            #for past_coll in eigs:
            #    for mat in range(nmat):
            #        errors.append(scipy.sum(abs(past_coll[mat]-eig[mat])))
            #if len(errors):
            #    print "Error = %.5e - %.5e"%(min(errors), max(errors))
            #eigs.append(eig)
            #for mat in outputs:
            #    error = _unmixing_error(nsources, mat, tr(isfa.RP))
            #    errors.append(error)
            #    #scipy.gplt.plot(errors)
            #    print mult(mat,tr(isfa.RP))
            #    raw_input()
            #outputs.append(isfa.RP)
            # get contrast
            summy, cont = _get_matrices_contrast(isfa.covs,
                                                 nsources, dim,
                                                 sfa_ica_coeff)
            # normalize contrast
            cont /= nmat
            print "S = %2d, N = %2d, Contrast = %.5e"%(nsources, nmat, cont)
            conts.append(cont)
            scipy.gplt.plot(scipy.log10(conts))
            if cont > 1e-3:
                break
            #if len(errors):
        fl = file('cont_s%d_d%d.pic'%(nsources,dim), 'w')
        import pickle
        pickle.dump(conts, fl)
        fl.close()
    
