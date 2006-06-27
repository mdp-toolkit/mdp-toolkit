## Automatically adapted for numpy Jun 26, 2006 by 

from routines import timediff, refcast, scast, rotate, random_rot, \
     ProgressBar, CrashRecoveryException, symrand, norm2, uniq, ordered_uniq, \
     cov2
from introspection import dig_node, get_node_size
from quad_forms import QuadraticForm
import mdp as _mdp
import scipy_emulation, types

#### REMEMBER: mdp.utils.inf can not be pickled with binary protocols!!!
#### Pickle bug: [ 714733 ] cPickle fails to pickle inf
#### https://sourceforge.net/tracker/?func=detail&atid=105470&aid=714733&group_id=5470

symeig = scipy_emulation._symeig_dumb
SymeigException = scipy_emulation.SymeigException
LeadingMinorException = scipy_emulation.LeadingMinorException
# matrix multiplication function
mult = _mdp.numx.dot

_inv = _mdp.numx_linalg.inv
inv = lambda x: refcast(_inv(x), x.dtype.char)
_pinv = _mdp.numx_linalg.pinv
pinv = lambda x: refcast(_pinv(x), x.dtype.char)
_solve = _mdp.numx_linalg.solve
solve = lambda x,y: refcast(_solve(x,y), x.dtype.char)

def svd(x, _mdp=_mdp):
    tc = x.dtype.char
    u,s,v = _mdp.numx_linalg.svd(x)
    return refcast(u, tc), refcast(s, tc), refcast(v, tc)

det = _mdp.numx_linalg.det
array2string = _mdp.numx.array2string
inf = _mdp.numx.inf
normal = _mdp.numx_rand.normal
squeeze = _mdp.numx.squeeze
eye = _mdp.numx.eye
diag = _mdp.numx.diag
amax = _mdp.numx.amax
amin = _mdp.numx.amin
linspace = _mdp.numx.linspace
atleast_2d = _mdp.numx.atleast_2d
mean = _mdp.numx.mean
var = _mdp.numx.var
std = _mdp.numx.std
cov = _mdp.numx.cov
iscomplexobj = _mdp.numx.iscomplexobj
assert_array_equal = _mdp.numx.testing.assert_array_equal
assert_array_almost_equal = _mdp.numx.testing.assert_array_almost_equal
assert_equal = _mdp.numx.testing.assert_equal
assert_almost_equal = _mdp.numx.testing.assert_almost_equal


# copy scipy or emulated function in mdp.utils
for name, val in scipy_emulation.__dict__.iteritems():
    if isinstance(val, types.FunctionType) and name[0] != '_':
        globals()[name] = getattr(_mdp.numx, name,
                                  getattr(_mdp.numx_linalg, name,
                                          val))

# clean up
del scipy_emulation, types, _mdp
