## Automatically adapted for numpy Jun 26, 2006 by 

import mdp

numx = mdp.numx
tr = numx.transpose
epsilon = 1E-15

class QuadraticForm(object):
    """ """
    def __init__(self, H=None, f=None, c=None, typecode="d"):
        self.H = H
        self.f = f
        self.c = c
        self.typecode = typecode

    def from_SFANode(sfa_node, unit):
        """
        Return the matrix H, the vector f and the constant c of the
        quadratic form 1/2 x'Hx + f'x + c that defines the output
        of the component 'nr' of the SFA 'node'.
        Note that this implies that the SFA node follows a quadratic expansion.
        """
        typecode = sfa_node.typecode
        if self.typecode is None:
            self.typecode = typecode
        sf = sfa_node.sf[:, unit]
        c = -mdp.utils.mult(sfa_node.avg, sf)
        N = sfa_node.output_dim
        H = numx.zeros((N,N),dtype=typecode)
        k = N
        for i in range(N):
            for j in range(N):
                if j > i:
                    H[i,j] = sf[k]
                    k = k+1
                elif j == i:
                    H[i,j] = 2*sf[k]
                    k = k+1
                else:
                    H[i,j] = H[j,i]
        self.H = H
        self.f = f
        self.c = c

    def get_extrema(self, norm, tol = 1.E-4):
        """
        Find the input vectors xmax and xmin with norm 'nrm' that maximize
        resp. minimize the output of the component 'sf_nr' of the SFA2 flow

        ! The output vectors lie always in the input space.

        tol: norm error tolerance
        """
        H, f, c = self.H, self.f, self.c
        if f is None: f = numx.zeros((H.shape[0],), dtype=self.typecode)
        if c is None: c = 0    
        H_definite_positive, H_definite_negative = False, False
        E = mdp.utils.symeig(H, eigenvectors=0, overwrite=0)
        if E[0] >= 0:
            # H is positive definite
            H_definite_positive = True
        elif E[-1] <= 0:
            # H is negative definite
            H_definite_negative = True

        x0 = mdp.numx_linalg.solve(-H, f)
        if H_definite_positive and mdp.utils.norm2(x0) <= norm:
            xmin = x0
            # x0 is a minimum
        else:
            xmin = self.maximize(norm, tol, factor=-1)
        vmin = 0.5*mdp.utils.mult(mdp.utils.mult(xmin, H), xmin) + \
               mdp.utils.mult(f, xmin) + c
        if H_definite_negative and mdp.utils.norm2(x0) <= norm :
            xmax= x0
            # x0 is a maximum
        else:
            xmax = self.maximize(norm, tol, factor=None)
        vmax = 0.5*mdp.utils.mult(mdp.utils.mult(xmax, H), xmax) + \
               mdp.utils.mult(f, xmax) + c 
        self.xmax, self.xmin, self.vmax, self.vmin = xmax, xmin, vmax, vmin
        return xmax, xmin, vmax, vmin

    def maximize(self, norm, tol = 1.E-4, x0 = None, factor = None):
        H, f = self.H, self.f
        if f is None: f = numx.zeros((H.shape[0],), dtype=self.typecode)
        if factor is not None:
            H = factor*H
            f = factor*f
        if x0 is not None:
            x0 = mdp.utils.refcast(x0, self.typecode)
            f = mdp.utils.mult(H, x0)+ f
            # c = 0.5*x0'*H*x0 + f'*x0 + c -> do we need it?
        mu, V = mdp.utils.symeig(H, overwrite=0)
        alpha = numx.reshape(mdp.utils.mult(tr(V), f), (H.shape[0],) )
        # v_i = alpha_i * v_i (alpha is a raw_vector)
        V = V*alpha
        # left bound for lambda
        ll = mu[-1] # eigenvalue's maximum
        # right bound for lambda
        lr = mdp.utils.norm2(f)/norm + ll
        # search by bisection until norm(x)**2 = norm**2
        norm_2 = norm**2
        norm_x_2 = 0
        while abs(norm_x_2-norm_2) > tol and (lr-ll)/lr > epsilon:
            # bisection of the lambda-interval
            lambd = 0.5*(lr-ll)+ll
            # eigenvalues of (lambda*Id - H)^-1
            beta = (lambd-mu)**(-1)
            # solution to the second lagragian equation
            norm_x_2 = numx.sum(alpha**2*beta**2)
            #%[ll,lr]
            if norm_x_2 > norm_2:
                ll=lambd
            else:
                lr=lambd
        x = numx.sum(V*beta, axis=1)
        if x0:
            x = x + x0
        return x

    def compute_invariances(self):
        raise NotImplementedError

if __name__ == "__main__":
    # check H with negligible linear term
    noise = 1e-7
    x = mdp.numx_rand.random((10,))
    H = mdp.numx.outer(x, x)+noise*mdp.numx_rand.random((10,10))
    H = H+tr(H)
    q = QuadraticForm(H=H, f=noise*mdp.numx_rand.random((10,)))
    xmax, xmin, vmax, vmin = q.get_extrema(mdp.utils.norm2(x), tol=noise)
    print 'Should be zero:', max(abs(x-xmax))
    # check I + linear term
    f = x
    H = numx.eye(10, "d")
    q = QuadraticForm(H=H,f=f)
    xmax, xmin, vmax, vmin = q.get_extrema(mdp.utils.norm2(x), tol=noise) 
    print 'Should be zero:', max(abs(f-xmax))
