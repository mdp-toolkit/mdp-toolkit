from routines import timediff, refcast, scast, rotate, random_rot, \
     ProgressBar, CrashRecoveryException, symrand, norm2, uniq, ordered_uniq
from introspection import dig_node, get_node_size
import mdp as _mdp
import scipy_emulation, types

#### REMEMBER: mdp.utils.inf can not be pickled with binary protocols!!!
#### Pickle bug: [ 714733 ] cPickle fails to pickle inf
#### https://sourceforge.net/tracker/?func=detail&atid=105470&aid=714733&group_id=5470
if _mdp.numx_description == 'symeig':
    # symeig installed, we have all necessary functions
    from symeig import symeig, SymeigException, LeadingMinorException
    # redefine the superclass of SymeigException such that we can catch it
    SymeigException.__bases__ = (_mdp.MDPException,)
    # matrix multiplication function
    from routines import _matmult as mult
    from scipy_emulation import _scipy_normal as normal

    inf = _mdp.numx.inf
    det = _mdp.numx_linalg.det
    inv = _mdp.numx_linalg.inv
    solve = _mdp.numx_linalg.solve
    array2string = _mdp.numx.array2string
    svd = _mdp.numx_linalg.svd
    
elif _mdp.numx_description == 'scipy':
    # we have all necessary functions but 'symeig'
    symeig = scipy_emulation._symeig_scipy
    SymeigException = scipy_emulation.SymeigException
    LeadingMinorException = scipy_emulation.LeadingMinorException
    # matrix multiplication function
    from routines import _matmult as mult
    from scipy_emulation import _scipy_normal as normal

    inf = _mdp.numx.inf
    det = _mdp.numx_linalg.det
    inv = _mdp.numx_linalg.inv
    solve = _mdp.numx_linalg.solve
    array2string = _mdp.numx.array2string
    svd = _mdp.numx_linalg.svd
    
else:
    # Numeric or numarray, load symeig and missing scipy functions
    # symeig data
    symeig = scipy_emulation._symeig_dumb
    SymeigException = scipy_emulation.SymeigException
    LeadingMinorException = scipy_emulation.LeadingMinorException
    # matrix multiplication function
    mult = _mdp.numx.dot

    det = _mdp.numx_linalg.determinant
    _inv = _mdp.numx_linalg.inverse
    inv = lambda x: refcast(_inv(x), x.typecode())
    solve = _mdp.numx_linalg.solve_linear_equations
    if _mdp.numx_description=='Numeric':
        array2string = _mdp.numx.array2string
        inf = 1/_mdp.numx.array(0.)
    else:
        import numarray.ieeespecial
        inf = numarray.ieeespecial.inf
        array2string = _mdp.numx.arrayprint.array2string
        del numarray.ieeespecial
    normal = _mdp.numx_rand.normal
    svd = _mdp.numx_linalg.singular_value_decomposition
    
# copy scipy or emulated function in mdp.utils
for name, val in scipy_emulation.__dict__.iteritems():
    if isinstance(val, types.FunctionType) and name[0] != '_':
        globals()[name] = getattr(_mdp.numx, name,
                                  getattr(_mdp.numx_linalg, name,
                                          val))

del scipy_emulation, types, _mdp
