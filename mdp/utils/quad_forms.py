import mdp
from routines import refcast
numx = mdp.numx
epsilon = 1E-15

class QuadraticForm(object):
    """
    Define an inhomogeneous quadratic form as 1/2 x'Hx + f'x + c .

    WARNING: EXPERIMENTAL CODE! USE AT YOU OWN RISK!
    """

    def __init__(self, H, f, c, dtype='d'):
        """
        The quadratic form is defined as 1/2 x'Hx + f'x + c .
        'dtype' specifies the numerical type of the internal structures.
        """
        self.H = refcast(H, dtype)
        self.f = refcast(f, dtype)
        self.c = c
        self.dtype = dtype

    def apply(self, x):
        """Apply the quadratic form to the input vectors.
        Return 1/2 x'Hx + f'x + c ."""
        return 0.5*(mdp.utils.mult(x, self.H.T)*x).sum(axis=1) + \
               mdp.utils.mult(x, self.f) + self.c
        
    def get_extrema(self, norm, tol = 1.E-4):
        """
        Find the input vectors xmax and xmin with norm 'nrm' that maximize
        or minimize the quadratic form.

        tol: norm error tolerance
        """
        H, f, c = self.H, self.f, self.c
        if f is None: f = numx.zeros((H.shape[0],), dtype=self.dtype)
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
            xmin = self._maximize(norm, tol, factor=-1)
        vmin = 0.5*mdp.utils.mult(mdp.utils.mult(xmin, H), xmin) + \
               mdp.utils.mult(f, xmin) + c
        if H_definite_negative and mdp.utils.norm2(x0) <= norm :
            xmax= x0
            # x0 is a maximum
        else:
            xmax = self._maximize(norm, tol, factor=None)
        vmax = 0.5*mdp.utils.mult(mdp.utils.mult(xmax, H), xmax) + \
               mdp.utils.mult(f, xmax) + c 
        self.xmax, self.xmin, self.vmax, self.vmin = xmax, xmin, vmax, vmin
        return xmax, xmin, vmax, vmin

    def _maximize(self, norm, tol = 1.E-4, x0 = None, factor = None):
        H, f = self.H, self.f
        if f is None: f = numx.zeros((H.shape[0],), dtype=self.dtype)
        if factor is not None:
            H = factor*H
            f = factor*f
        if x0 is not None:
            x0 = mdp.utils.refcast(x0, self.dtype)
            f = mdp.utils.mult(H, x0)+ f
            # c = 0.5*x0'*H*x0 + f'*x0 + c -> do we need it?
        mu, V = mdp.utils.symeig(H, overwrite=0)
        alpha = mdp.utils.mult(V.T, f).reshape((H.shape[0],))
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
            norm_x_2 = (alpha**2*beta**2).sum()
            #%[ll,lr]
            if norm_x_2 > norm_2:
                ll=lambd
            else:
                lr=lambd
        x = (V*beta).sum(axis=1)
        if x0:
            x = x + x0
        return x

    def invariances(self):
        pass
        raise NotImplementedError
