import mdp

# import numeric module (scipy, Numeric or numarray)
numx, numx_rand, numx_linalg  = mdp.numx, mdp.numx_rand, mdp.numx_linalg
numx_description = mdp.numx_description

def random_ortho(dim, dtype='d'):
  """Return a random orthogonal matrix, drawn from the Haar distribution
  (the only uniform distribution on SO(n)).
  The algorithm is described in the paper
  Stewart, G.W., "The efficient generation of random orthogonal
  matrices with an application to condition estimators", SIAM Journal
  on Numerical Analysis, 17(3), pp. 403-409, 1980.
  For more information see
  http://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization
  See also Mezzadri, F., "How to generate random matrices from the
  classical compact groups", arXiv:math-ph/0609050v2.
  """
  H = mdp.numx.eye(dim, dtype=dtype)
  for n in range(1, dim):
    x = mdp.numx_rand.normal(size=(dim-n+1,)).astype(dtype)
    D = mdp.numx.sign(x[0]) # random sign, 50/50, but chosen carefully to avoid roundoff error --- see Mezzadri, 2007:
    x[0] += D*mdp.numx.sqrt((x*x).sum())
    # Householder transformation
    Hx = -D*( mdp.numx.eye(dim-n+1, dtype=dtype)
              - 2.*mdp.numx.outer(x, x)/(x*x).sum() )
    mat = mdp.numx.eye(dim, dtype=dtype)
    mat[n-1:, n-1:] = Hx
    H = mdp.utils.mult(H, mat)
  return H

# ran_corr: Takes eigenvalues that sum to n,
#           and returns random correlation matrix
#           Calls o_to_corr once
def ran_corr(eigs):
  '''
  Generates a random corrrelation matrix with eigenvalues set to eigs. eigs must sum to the dimensionality.

  See Davies, Philip I; Higham, Nicholas J; "Numerically stable generation of
  correlation matrices and their factors", BIT 2000, Vol. 40, No. 4,
  pp. 640–651

  Note that they also provide an algorithm for directly generating the Cholesky decomposition.
  '''
  # if sum(eigs) != len(eigs):
  #   print("Invalid eigenvalues; must sum to dimensionality.")
  #   return

  d = len(eigs)
  m = random_ortho(d)
  m = mdp.utils.mult(mdp.utils.mult(m, mdp.numx.diag(eigs)), m.T) # Set the trace of m
  m = o_to_corr(m) # carefully rotate to unit diagonal
  return m

# o_to_corr: Rotates orthogonal matrix into correlation matrix
#            Calls givens_to_1 once, in a loop
# Now take the result of random_ortho and rotate it to have 1's on the
# diagonal with more Givens/HH rotations.
def o_to_corr(m):
  '''
  Given an orthogonal matrix m, rotate to put one's on the diagonal, turning it
  into a correlation matrix.  This also requires the trace equal the
  dimensionality.

  '''
  d = m.shape[0]
  for i in range(d-1):
    if m[i,i] == 1.:
      continue
    if m[i, i] > 1.:
      for j in range(i+1, d):
        if m[j, j] < 1.:
          break
    else:
      for j in range(i+1, d):
        if m[j, j] > 1.:
          break
    c, s = givens_to_1(m[i,i], m[j, j], m[i, j])
    g = mdp.numx.eye(d)
    g[i, i] = c
    g[j, j] = c
    g[j, i] = -s
    g[i, j] = s
    m = mdp.utils.mult(mdp.utils.mult(g.T, m), g)
    # Alternatively, just modify the ij and jj terms directly;
    #   explicitly setting ii to 1.
  return m

# givens_to_1: Computes Givens matrix to rotate a 1 onto diagonal
def givens_to_1(aii, ajj, aij, eps = 2e-13):
  '''Computes a 2x2 Givens matrix to put 1's on the diagonal for the input matrix.

  The input matrix is a 2x2 symmetric matrix M = [ aii aij ; aij ajj ]. The
  unique elements must be explicitly provided. The function will produce a
  result as long as tr(M) - det(M) >= 1, but will only produce a unit diagonal
  when tr(M) == 2 and M is not already diagonal. Note that when tr(M) == 2, the
  above condition is necessarily satisfied.

  The output matrix g is a 2x2 anti-symmetric matrix of the form [ c s ; -s c ];
  the elements c and s are returned.

  Applying the output matrix to the input matrix (as g.T M g) results in a
  matrix with ones on the diagonal when tr(M) == 2.

  '''
  # if fabs(aiid) < eps and fabs(ajjd) < eps:
  #   # Basically have ones on the diagonal already!
  #   return 1., 0. # return the identity matrix
  if mdp.numx.fabs(aij) < eps:
    # The matrix is already diagonal; we can't independently scale the diagonal entries through a rotation.
    return 1., 0

  aiid = aii - 1.
  ajjd = ajj - 1.

  dd = aij**2 - aiid*ajjd
  if dd > 0.:
    discriminant = mdp.numx.sqrt(dd)
  else:
    discriminant = 0.
    if dd < -eps:
      # Error was likely not due to round-off, so give a warning.
      # TODO: Analyze how this can happen. Only due to numerical issues?
      print("Potential problem in givens_to_1")
      print("Arguments are aii, ajj, aij: " + ','.join(map(str, (aii, ajj, aij))))
      print("aij^2 - (aii-1)*(ajj-1) should be positive but is " + str(dd))

  # The choice of t should be chosen to avoid cancellation, following Davies & Higham
  # Davies, Philip I; Higham, Nicholas J; "Numerically stable generation of
  # correlation matrices and their factors", BIT 2000, Vol. 40, No. 4, pp. 640–651
  # Make sure the numerator is small when ajjd is small.
  t = (aij + mdp.numx.sign(aij)*discriminant) / ajjd
  c = 1. / mdp.numx.sqrt(1. + t*t)
  s = c*t
#  return mat([ [ c, s ], [ -s, c ] ])
  return c, s
