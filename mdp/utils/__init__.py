from routines import (timediff, refcast, scast, rotate, random_rot,
                      permute, symrand, norm2, cov2,
                      mult_diag, comb, sqrtm, get_dtypes, nongeneral_svd,
                      hermitian, cov_maxima,
                      lrep, rrep, irep, orthogonal_permutations,
                      izip_stretched,
                      weighted_choice, bool_to_sign, sign_to_bool, gabor,
                      invert_exp_funcs2)
try:
    from collections import OrderedDict
except ImportError:
    ## Getting an Ordered Dict for Python < 2.7
    from _ordered_dict import OrderedDict

try:
    from tempfile import TemporaryDirectory
except ImportError:
    from temporarydir import TemporaryDirectory

from introspection import dig_node, get_node_size, get_node_size_str
from quad_forms import QuadraticForm, QuadraticFormException
from covariance import (CovarianceMatrix, DelayCovarianceMatrix,
                        MultipleCovarianceMatrices,CrossCovarianceMatrix)
from progress_bar import progressinfo
from slideshow import (basic_css, slideshow_css, HTMLSlideShow,
                       image_slideshow_css, ImageHTMLSlideShow,
                       SectionHTMLSlideShow, SectionImageHTMLSlideShow,
                       image_slideshow, show_image_slideshow)

from _symeig import SymeigException

import mdp as _mdp
# matrix multiplication function
# we use an alias to be able to use the wrapper for the 'gemm' Lapack
# function in the future
mult = _mdp.numx.dot
matmult = mult

if _mdp.numx_description == 'scipy':
    def matmult(a,b, alpha=1.0, beta=0.0, c=None, trans_a=0, trans_b=0):
        """Return alpha*(a*b) + beta*c.
        a,b,c : matrices
        alpha, beta: scalars
        trans_a : 0 (a not transposed), 1 (a transposed),
                  2 (a conjugate transposed)
        trans_b : 0 (b not transposed), 1 (b transposed),
                  2 (b conjugate transposed)
        """
        if c:
            gemm,=_mdp.numx_linalg.get_blas_funcs(('gemm',),(a,b,c))
        else:
            gemm,=_mdp.numx_linalg.get_blas_funcs(('gemm',),(a,b))

        return gemm(alpha, a, b, beta, c, trans_a, trans_b)

# workaround to numpy issues with dtype behavior:
# 'f' is upcasted at least in the following functions
_inv = _mdp.numx_linalg.inv
inv = lambda x: refcast(_inv(x), x.dtype)
_pinv = _mdp.numx_linalg.pinv
pinv = lambda x: refcast(_pinv(x), x.dtype)
_solve = _mdp.numx_linalg.solve
solve = lambda x, y: refcast(_solve(x, y), x.dtype)

def svd(x, compute_uv = True):
    """Wrap the numx SVD routine, so that it returns arrays of the correct
    dtype and a SymeigException in case of failures."""
    tc = x.dtype
    try:
        if compute_uv:
            u, s, v = _mdp.numx_linalg.svd(x)
            return refcast(u, tc), refcast(s, tc), refcast(v, tc)
        else:
            s = _mdp.numx_linalg.svd(x, compute_uv=False)
            return refcast(s, tc)
    except _mdp.numx_linalg.LinAlgError, exc:
        raise SymeigException(str(exc))

__all__ = ['CovarianceMatrix', 'DelayCovarianceMatrix','CrossCovarianceMatrix',
           'MultipleCovarianceMatrices', 'QuadraticForm',
           'QuadraticFormException',
           'comb', 'cov2', 'dig_node', 'get_dtypes', 'get_node_size',
           'hermitian', 'inv', 'mult', 'mult_diag', 'nongeneral_svd',
           'norm2', 'permute', 'pinv', 'progressinfo',
           'random_rot', 'refcast', 'rotate', 'scast', 'solve', 'sqrtm',
           'svd', 'symrand', 'timediff', 'matmult',
           'HTMLSlideShow', 'ImageHTMLSlideShow',
           'basic_css', 'slideshow_css', 'image_slideshow_css',
           'SectionHTMLSlideShow',
           'SectionImageHTMLSlideShow', 'image_slideshow',
           'lrep', 'rrep', 'irep',
           'orthogonal_permutations', 'izip_stretched',
           'weighted_choice', 'bool_to_sign', 'sign_to_bool',
           'OrderedDict', 'TemporaryDirectory', 'gabor', 'fixup_namespace']

def without_prefix(name, prefix):
    if name.startswith(prefix):
        return name[len(prefix):]
    else:
        return None

import os
FIXUP_DEBUG = os.getenv('MDPNSDEBUG')

def fixup_namespace(mname, names, old_modules, keep_modules=()):
    import sys
    module = sys.modules[mname]
    if names is None:
        names = [name for name in dir(module) if not name.startswith('_')]
    if FIXUP_DEBUG:
        print 'NAMESPACE FIXUP: %s (%s)' % (module, mname)
    for name in names:
        item = getattr(module, name)
        if (hasattr(item, '__module__') and
            without_prefix(item.__module__, mname + '.') in old_modules):
            if FIXUP_DEBUG:
                print 'namespace fixup: {%s => %s}.%s' % (
                    item.__module__, mname, item.__name__)
            item.__module__ = mname
    # take care of removing the module filenames
    for filename in old_modules:
        # skip names in keep modules
        if filename in keep_modules:
            continue
        try:
            delattr(module, filename)
            if FIXUP_DEBUG:
                print 'NAMESPACE FIXUP: deleting %s from %s' % (filename, module)
        except AttributeError:
            # if the name is not there, we are in a reload, so do not
            # do anything
            pass

fixup_namespace(__name__, __all__,
                ('routines',
                 'introspection',
                 'quad_forms',
                 'covariance',
                 'progress_bar',
                 'slideshow',
                 '_ordered_dict',
                 'templet',
                 'temporarydir',
                 'os',
                 ))
