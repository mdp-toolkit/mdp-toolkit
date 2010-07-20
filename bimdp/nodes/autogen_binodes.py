"""
AUTOMATICALLY GENERATED CODE, DO NOT MODIFY!

Edit and run autogen.py instead to overwrite this module.
"""

import mdp.nodes
from bimdp import BiNode

class AdaptiveCutoffBiNode(BiNode, mdp.nodes.AdaptiveCutoffNode):
    """Automatically created BiNode version of AdaptiveCutoffNode."""
    def __init__(self, lower_cutoff_fraction=None, upper_cutoff_fraction=None, hist_fraction=1.0, hist_filename=None, input_dim=None, dtype=None, node_id=None, stop_result=None):
        """Initialize the node.

        lower_cutoff_fraction -- Fraction of data that will be cut off after
            the training phase (assuming the data distribution does not
            change). If set to None (default value) no cutoff is performed.
        upper_cutoff_fraction -- Works like lower_cutoff_fraction.
        hist_fraction -- Defines the fraction of the data that is stored for the
            histogram.
        hist_filename -- Filename for the file to which the data history will
            be pickled after training. The data is pickled when stop_training
            is called and data_hist is then cleared (to free memory).
            If filename is None (default value) then data_hist is not cleared
            and can be directly used after training.
        """
        super(AdaptiveCutoffBiNode, self).__init__(lower_cutoff_fraction=lower_cutoff_fraction, upper_cutoff_fraction=upper_cutoff_fraction, hist_fraction=hist_fraction, hist_filename=hist_filename, input_dim=input_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class CuBICABiNode(BiNode, mdp.nodes.CuBICANode):
    """Automatically created BiNode version of CuBICANode."""
    def __init__(self, limit=0.001, telescope=False, verbose=False, whitened=False, white_comp=None, white_parm=None, input_dim=None, dtype=None, node_id=None, stop_result=None):
        """
        Input arguments:

        whitened -- Set whitened is True if input data are already whitened.
                    Otherwise the node will whiten the data itself.

        white_comp -- If whitened is False, you can set 'white_comp' to the
                      number of whitened components to keep during the
                      calculation (i.e., the input dimensions are reduced to
                      white_comp by keeping the components of largest variance).

        white_parm -- a dictionary with additional parameters for whitening.
                      It is passed directly to the WhiteningNode constructor.
                      Ex: white_parm = { 'svd' : True }

        limit -- convergence threshold.

        telescope -- If telescope == True, use Telescope mode: Instead of
          using all input data in a single batch try larger and larger chunks
          of the input data until convergence is achieved. This should lead to
          significantly faster convergence for stationary statistics. This mode
          has not been thoroughly tested and must be considered beta.
        """
        super(CuBICABiNode, self).__init__(limit=limit, telescope=telescope, verbose=verbose, whitened=whitened, white_comp=white_comp, white_parm=white_parm, input_dim=input_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class CutoffBiNode(BiNode, mdp.nodes.CutoffNode):
    """Automatically created BiNode version of CutoffNode."""
    def __init__(self, lower_bound=None, upper_bound=None, input_dim=None, dtype=None, node_id=None, stop_result=None):
        """Initialize node.

        lower_bound -- Data values below this are cut to the lower_bound value.
            If lower_bound is None no cutoff is performed.
        upper_bound -- Works like lower_bound.
        """
        super(CutoffBiNode, self).__init__(lower_bound=lower_bound, upper_bound=upper_bound, input_dim=input_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class EtaComputerBiNode(BiNode, mdp.nodes.EtaComputerNode):
    """Automatically created BiNode version of EtaComputerNode."""
    def __init__(self, input_dim=None, dtype=None, node_id=None, stop_result=None):
        super(EtaComputerBiNode, self).__init__(input_dim=input_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class FABiNode(BiNode, mdp.nodes.FANode):
    """Automatically created BiNode version of FANode."""
    def __init__(self, tol=0.0001, max_cycles=100, verbose=False, input_dim=None, output_dim=None, dtype=None, node_id=None, stop_result=None):
        super(FABiNode, self).__init__(tol=tol, max_cycles=max_cycles, verbose=verbose, input_dim=input_dim, output_dim=output_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class FDABiNode(BiNode, mdp.nodes.FDANode):
    """Automatically created BiNode version of FDANode."""
    def __init__(self, input_dim=None, output_dim=None, dtype=None, node_id=None, stop_result=None):
        super(FDABiNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class FastICABiNode(BiNode, mdp.nodes.FastICANode):
    """Automatically created BiNode version of FastICANode."""
    def __init__(self, approach='defl', g='pow3', guess=None, fine_g='pow3', mu=1, stabilization=False, sample_size=1, fine_tanh=1, fine_gaus=1, max_it=1000, max_it_fine=100, failures=5, limit=0.001, verbose=False, whitened=False, white_comp=None, white_parm=None, input_dim=None, dtype=None, node_id=None, stop_result=None):
        """
        Input arguments:

        General:

        whitened -- Set whitened == True if input data are already whitened.
                    Otherwise the node will whiten the data itself

        white_comp -- If whitened == False, you can set 'white_comp' to the
                      number of whitened components to keep during the
                      calculation (i.e., the input dimensions are reduced to
                      white_comp by keeping the components of largest variance).

        white_parm -- a dictionary with additional parameters for whitening.
                      It is passed directly to the WhiteningNode constructor.
                      Ex: white_parm = { 'svd' : True }

        limit -- convergence threshold.

        Specific for FastICA:

        approach  -- Approach to use. Possible values are:
                                          'defl' --> deflation
                                          'symm' --> symmetric

               g  -- Nonlinearity to use. Possible values are:
                                          'pow3' --> x^3
                                          'tanh' --> tanh(fine_tanh*x)
                                          'gaus' --> x*exp(-fine_gaus*x^2/2)
                                          'skew' --> x^2 (for skewed signals)

           fine_g -- Nonlinearity for fine tuning. Possible values
                     are the same as for 'g'. Set it to None to disable fine
                     tuning.

               mu -- Step size

    stabilization -- Enable stabilization procedure: the value of mu can
                     momentarily be halved if the algorithm is stuck between
                     two points (this is called a stroke). Also if there is no
                     convergence before half of the maximum number of
                     iterations has been reached then mu will be halved for
                     the rest of the rounds.

      sample_size -- Percentage of samples used in one iteration. If
                     sample_size < 1, samples are chosen in random order.

        fine_tanh -- parameter for 'tanh' nonlinearity
        fine_gaus -- parameter for 'gaus' nonlinearity

            guess -- initial guess for the mixing matrix (ignored if None)

           max_it -- maximum number of iterations

      max_it_fine -- maximum number of iterations for fine tuning

         failures -- maximum number of failures to allow in deflation mode

        """
        super(FastICABiNode, self).__init__(approach=approach, g=g, guess=guess, fine_g=fine_g, mu=mu, stabilization=stabilization, sample_size=sample_size, fine_tanh=fine_tanh, fine_gaus=fine_gaus, max_it=max_it, max_it_fine=max_it_fine, failures=failures, limit=limit, verbose=verbose, whitened=whitened, white_comp=white_comp, white_parm=white_parm, input_dim=input_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class GrowingNeuralGasExpansionBiNode(BiNode, mdp.nodes.GrowingNeuralGasExpansionNode):
    """Automatically created BiNode version of GrowingNeuralGasExpansionNode."""
    def __init__(self, start_poss=None, eps_b=0.20000000000000001, eps_n=0.0060000000000000001, max_age=50, lambda_=100, alpha=0.5, d=0.995, max_nodes=100, input_dim=None, dtype=None, node_id=None, stop_result=None):
        """
        For a full list of input arguments please check the documentation
        of GrowingNeuralGasNode.

        max_nodes (default 100) : maximum number of nodes in the
                                  neural gas, therefore an upper bound
                                  to the output dimension of the
                                  expansion.
        """
        super(GrowingNeuralGasExpansionBiNode, self).__init__(start_poss=start_poss, eps_b=eps_b, eps_n=eps_n, max_age=max_age, lambda_=lambda_, alpha=alpha, d=d, max_nodes=max_nodes, input_dim=input_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class GrowingNeuralGasBiNode(BiNode, mdp.nodes.GrowingNeuralGasNode):
    """Automatically created BiNode version of GrowingNeuralGasNode."""
    def __init__(self, start_poss=None, eps_b=0.20000000000000001, eps_n=0.0060000000000000001, max_age=50, lambda_=100, alpha=0.5, d=0.995, max_nodes=2147483647, input_dim=None, dtype=None, node_id=None, stop_result=None):
        """Growing Neural Gas algorithm.

        Input arguments:
        start_pos -- sequence of two arrays containing the position of the
                     first two nodes in the GNG graph. In unspecified, the
                     initial nodes are chosen with a random position generated
                     from a gaussian distribution with zero mean and unit
                     variance.

        eps_b     -- coefficient of movement of the nearest node to a new
                     data point. Typical values are 0 < eps_b << 1 .
                     Default: 0.2

        eps_n     -- coefficient of movement of the neighbours of the nearest
                     node to a new data point. Typical values are
                     0 < eps_n << eps_b .
                     Default: 0.006

        max_age   -- remove an edge after 'max_age' updates. Typical values are
                     10 < max_age < lambda .
                     Default: 50

        lambda_   -- insert a new node after 'lambda_' steps. Typical values
                     are O(100).
                     Default: 100

        alpha     -- when a new node is inserted, multiply the error of the
                     nodes from which it generated by 0<alpha<1. A typical
                     value is 0.5.
                     Default: 0.5

        d         -- each step the error of the nodes are multiplied by 0<d<1.
                     Typical values are close to 1.
                     Default: 0.995

        max_nodes -- maximal number of nodes in the graph.
                     Default: 2^31 - 1
        """
        super(GrowingNeuralGasBiNode, self).__init__(start_poss=start_poss, eps_b=eps_b, eps_n=eps_n, max_age=max_age, lambda_=lambda_, alpha=alpha, d=d, max_nodes=max_nodes, input_dim=input_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class HLLEBiNode(BiNode, mdp.nodes.HLLENode):
    """Automatically created BiNode version of HLLENode."""
    def __init__(self, k, r=0.001, svd=False, verbose=False, input_dim=None, output_dim=None, dtype=None, node_id=None, stop_result=None):
        """
        Keyword Arguments:

         k -- number of nearest neighbors to use; the node will raise
              an MDPWarning if k is smaller than
                k >= 1 + output_dim + output_dim*(output_dim+1)/2,
              because in this case a less efficient computation must be
              used, and the ablgorithm can become unstable
         r -- regularization constant; as opposed to LLENode, it is
              not possible to compute this constant automatically; it is
              only used during execution
         svd -- if True, use SVD to compute the projection matrix;
                SVD is slower but more stable
         verbose -- if True, displays information about the progress
                    of the algorithm

         output_dim -- number of dimensions to output
                       or a float between 0.0 and 1.0. In the latter case,
                       output_dim specifies the desired fraction of variance
                       to be exaplained, and the final number of output
                       dimensions is known at the end of training
                       (e.g., for 'output_dim=0.95' the algorithm will keep
                       as many dimensions as necessary in order to explain
                       95% of the input variance)
        """
        super(HLLEBiNode, self).__init__(k=k, r=r, svd=svd, verbose=verbose, input_dim=input_dim, output_dim=output_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class HistogramBiNode(BiNode, mdp.nodes.HistogramNode):
    """Automatically created BiNode version of HistogramNode."""
    def __init__(self, hist_fraction=1.0, hist_filename=None, input_dim=None, dtype=None, node_id=None, stop_result=None):
        """Initialize the node.

        hist_fraction -- Defines the fraction of the data that is stored
            randomly.
        hist_filename -- Filename for the file to which the data history will
            be pickled after training. The data is pickled when stop_training
            is called and data_hist is then cleared (to free memory).
            If filename is None (default value) then data_hist is not cleared
            and can be directly used after training.
        """
        super(HistogramBiNode, self).__init__(hist_fraction=hist_fraction, hist_filename=hist_filename, input_dim=input_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class HitParadeBiNode(BiNode, mdp.nodes.HitParadeNode):
    """Automatically created BiNode version of HitParadeNode."""
    def __init__(self, n, d=1, input_dim=None, dtype=None, node_id=None, stop_result=None):
        """
        Input arguments:
        n -- Number of maxima and minima to store
        d -- Minimum gap between two maxima or two minima
        """
        super(HitParadeBiNode, self).__init__(n=n, d=d, input_dim=input_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class ICABiNode(BiNode, mdp.nodes.ICANode):
    """Automatically created BiNode version of ICANode."""
    def __init__(self, limit=0.001, telescope=False, verbose=False, whitened=False, white_comp=None, white_parm=None, input_dim=None, dtype=None, node_id=None, stop_result=None):
        """
        Input arguments:

        whitened -- Set whitened is True if input data are already whitened.
                    Otherwise the node will whiten the data itself.

        white_comp -- If whitened is False, you can set 'white_comp' to the
                      number of whitened components to keep during the
                      calculation (i.e., the input dimensions are reduced to
                      white_comp by keeping the components of largest variance).

        white_parm -- a dictionary with additional parameters for whitening.
                      It is passed directly to the WhiteningNode constructor.
                      Ex: white_parm = { 'svd' : True }

        limit -- convergence threshold.

        telescope -- If telescope == True, use Telescope mode: Instead of
          using all input data in a single batch try larger and larger chunks
          of the input data until convergence is achieved. This should lead to
          significantly faster convergence for stationary statistics. This mode
          has not been thoroughly tested and must be considered beta.
        """
        super(ICABiNode, self).__init__(limit=limit, telescope=telescope, verbose=verbose, whitened=whitened, white_comp=white_comp, white_parm=white_parm, input_dim=input_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class ISFABiNode(BiNode, mdp.nodes.ISFANode):
    """Automatically created BiNode version of ISFANode."""
    def __init__(self, lags=1, sfa_ica_coeff=(1.0, 1.0), icaweights=None, sfaweights=None, whitened=False, white_comp=None, white_parm=None, eps_contrast=9.9999999999999995e-07, max_iter=10000, RP=None, verbose=False, input_dim=None, output_dim=None, dtype=None, node_id=None, stop_result=None):
        """
        Perform Independent Slow Feature Analysis.
        The notation is the same used in the paper by Blaschke et al. Please
        refer to the paper for more information.

        Keyword arguments:

        lags    -- list of time-lags to generate the time-delayed covariance
                   matrices (in the paper this is the set of au). If
                   lags is an integer, time-lags 1,2,...,'lags' are used.
                   Note that time-lag == 0 (instantaneous correlation) is
                   always implicitly used.

        sfa_ica_coeff -- a list of float with two entries, which defines the
                         weights of the SFA and ICA part of the
                         objective function. They are called b_{SFA} and
                         b_{ICA} in the paper.

        sfaweights -- weighting factors for the covariance matrices relative
                      to the SFA part of the objective function (called
                      \kappa_{SFA}^{au} in the paper). Default is
                      [1., 0., ..., 0.]
                      For possible values see the description of icaweights.

        icaweights -- weighting factors for the cov matrices relative
                      to the ICA part of the objective function (called
                      \kappa_{ICA}^{au} in the paper). Default is 1.
                      Possible values are:
                          an integer n: all matrices are weighted the same
                                        (note that it does not make sense to
                                         have n != 1)
                          a list or array of floats of len == len(lags): each
                                        element of the list is used for
                                        weighting the corresponding matrix
                          None: use the default values.

        whitened   -- True if input data is already white, False otherwise (the
                      data will be whitened internally).

        white_comp -- If whitened is False, you can set 'white_comp' to the
                      number of whitened components to keep during the
                      calculation (i.e., the input dimensions are reduced to
                      white_comp by keeping the components of largest variance).
        white_parm -- a dictionary with additional parameters for whitening.
                      It is passed directly to the WhiteningNode constructor.
                      Ex: white_parm = { 'svd' : True }

        eps_contrast -- Convergence is achieved when the relative
                        improvement in the contrast is below this threshold.
                        Values in the range [1E-4, 1E-10] are usually
                        reasonable.

        max_iter     -- If the algorithms does not achieve convergence within
                        max_iter iterations raise an Exception. Should be
                        larger than 100.

        RP     -- Starting rotation-permutation matrix. It is an
                  input_dim x input_dim matrix used to initially rotate the
                  input components. If not set, the identity matrix is used.
                  In the paper this is used to start the algorithm at the
                  SFA solution (which is often quite near to the optimum).

        verbose -- print progress information during convergence. This can
                   slow down the algorithm, but it's the only way to see
                   the rate of improvement and immediately spot if something
                   is going wrong.

        output_dim -- sets the number of independent components that have to
                      be extracted. Note that if this is not smaller than
                      input_dim, the problem is solved linearly and SFA
                      would give the same solution only much faster.
        """
        super(ISFABiNode, self).__init__(lags=lags, sfa_ica_coeff=sfa_ica_coeff, icaweights=icaweights, sfaweights=sfaweights, whitened=whitened, white_comp=white_comp, white_parm=white_parm, eps_contrast=eps_contrast, max_iter=max_iter, RP=RP, verbose=verbose, input_dim=input_dim, output_dim=output_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class IdentityBiNode(BiNode, mdp.nodes.IdentityNode):
    """Automatically created BiNode version of IdentityNode."""
    def __init__(self, input_dim=None, output_dim=None, dtype=None, node_id=None, stop_result=None):
        """If the input dimension and the output dimension are
        unspecified, they will be set when the 'train' or 'execute'
        method is called for the first time.
        If dtype is unspecified, it will be inherited from the data
        it receives at the first call of 'train' or 'execute'.

        Every subclass must take care of up- or down-casting the internal
        structures to match this argument (use _refcast private
        method when possible).
        """
        super(IdentityBiNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class JADEBiNode(BiNode, mdp.nodes.JADENode):
    """Automatically created BiNode version of JADENode."""
    def __init__(self, limit=0.001, max_it=1000, verbose=False, whitened=False, white_comp=None, white_parm=None, input_dim=None, dtype=None, node_id=None, stop_result=None):
        """
        Input arguments:

        General:

        whitened -- Set whitened == True if input data are already whitened.
                    Otherwise the node will whiten the data itself

        white_comp -- If whitened == False, you can set 'white_comp' to the
                      number of whitened components to keep during the
                      calculation (i.e., the input dimensions are reduced to
                      white_comp by keeping the components of largest variance).

        white_parm -- a dictionary with additional parameters for whitening.
                      It is passed directly to the WhiteningNode constructor.
                      Ex: white_parm = { 'svd' : True }

        limit -- convergence threshold.

        Specific for JADE:

        max_it -- maximum number of iterations

        """
        super(JADEBiNode, self).__init__(limit=limit, max_it=max_it, verbose=verbose, whitened=whitened, white_comp=white_comp, white_parm=white_parm, input_dim=input_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class LLEBiNode(BiNode, mdp.nodes.LLENode):
    """Automatically created BiNode version of LLENode."""
    def __init__(self, k, r=0.001, svd=False, verbose=False, input_dim=None, output_dim=None, dtype=None, node_id=None, stop_result=None):
        """
        Keyword Arguments:

         k -- number of nearest neighbors to use
         r -- regularization constant; if None, r is automatically
              computed using the method presented in deRidder and Duin;
              this method involves solving an eigenvalue problem for
              every data point, and can slow down the algorithm
              If specified, it multiplies the trace of the local covariance
              matrix of the distances, as in Saul & Roweis (faster)
         svd -- if True, use SVD to compute the projection matrix;
                SVD is slower but more stable
         verbose -- if True, displays information about the progress
                    of the algorithm

         output_dim -- number of dimensions to output
                       or a float between 0.0 and 1.0. In the latter case,
                       output_dim specifies the desired fraction of variance
                       to be exaplained, and the final number of output
                       dimensions is known at the end of training
                       (e.g., for 'output_dim=0.95' the algorithm will keep
                       as many dimensions as necessary in order to explain
                       95% of the input variance)
        """
        super(LLEBiNode, self).__init__(k=k, r=r, svd=svd, verbose=verbose, input_dim=input_dim, output_dim=output_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class LinearRegressionBiNode(BiNode, mdp.nodes.LinearRegressionNode):
    """Automatically created BiNode version of LinearRegressionNode."""
    def __init__(self, with_bias=True, use_pinv=False, input_dim=None, output_dim=None, dtype=None, node_id=None, stop_result=None):
        """
        Input arguments:

        with_bias -- If True, the linear model includes a constant term
                         True:  y_i = b_0 + b_1 x_1 + ... b_N x_N
                         False: y_i =       b_1 x_1 + ... b_N x_N
                     If present, the constant term is stored in the first
                     column of self.beta

        use_pinv -- If true, uses the pseudo-inverse function to compute
                    the linear regression coefficients, which is more robust
                    in some cases
        """
        super(LinearRegressionBiNode, self).__init__(with_bias=with_bias, use_pinv=use_pinv, input_dim=input_dim, output_dim=output_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class NIPALSBiNode(BiNode, mdp.nodes.NIPALSNode):
    """Automatically created BiNode version of NIPALSNode."""
    def __init__(self, input_dim=None, output_dim=None, dtype=None, conv=1e-08, max_it=100000, node_id=None, stop_result=None):
        """
        The number of principal components to be kept can be specified as
        'output_dim' directly (e.g. 'output_dim=10' means 10 components
        are kept) or by the fraction of variance to be explained
        (e.g. 'output_dim=0.95' means that as many components as necessary
        will be kept in order to explain 95% of the input variance).

        Other Arguments:
           conv   - convergence threshold for the residual error.
           max_it - maximum number of iterations

        """
        super(NIPALSBiNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype, conv=conv, max_it=max_it, node_id=node_id, stop_result=stop_result)

class NormalNoiseBiNode(BiNode, mdp.nodes.NormalNoiseNode):
    """Automatically created BiNode version of NormalNoiseNode."""
    def __init__(self, noise_args=(0, 1), input_dim=None, dtype=None, node_id=None, stop_result=None):
        """Set the noise parameters.

        noise_args -- Tuple of (mean, standard deviation) for the normal
            distribution, default is (0,1).
        """
        super(NormalNoiseBiNode, self).__init__(noise_args=noise_args, input_dim=input_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class PCABiNode(BiNode, mdp.nodes.PCANode):
    """Automatically created BiNode version of PCANode."""
    def __init__(self, input_dim=None, output_dim=None, dtype=None, svd=False, reduce=False, var_rel=9.9999999999999998e-13, var_abs=1.0000000000000001e-15, var_part=None, node_id=None, stop_result=None):
        """The number of principal components to be kept can be specified as
        'output_dim' directly (e.g. 'output_dim=10' means 10 components
        are kept) or by the fraction of variance to be explained
        (e.g. 'output_dim=0.95' means that as many components as necessary
        will be kept in order to explain 95% of the input variance).

        Other Keyword Arguments:

        svd -- if True use Singular Value Decomposition instead of the
               standard eigenvalue problem solver. Use it when PCANode
               complains about singular covariance matrices

        reduce -- Keep only those principal components which have a variance
                  larger than 'var_abs' and a variance relative to the
                  first principal component larger than 'var_rel' and a
                  variance relative to total variance larger than 'var_part'
                  (set var_part to None or 0 for no filtering).
                  Note: when the 'reduce' switch is enabled, the actual number
                  of principal components (self.output_dim) may be different
                  from that set when creating the instance.
        """
        super(PCABiNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype, svd=svd, reduce=reduce, var_rel=var_rel, var_abs=var_abs, var_part=var_part, node_id=node_id, stop_result=stop_result)

class PolynomialExpansionBiNode(BiNode, mdp.nodes.PolynomialExpansionNode):
    """Automatically created BiNode version of PolynomialExpansionNode."""
    def __init__(self, degree, input_dim=None, dtype=None, node_id=None, stop_result=None):
        """
        Input arguments:
        degree -- degree of the polynomial space where the input is expanded
        """
        super(PolynomialExpansionBiNode, self).__init__(degree=degree, input_dim=input_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class QuadraticExpansionBiNode(BiNode, mdp.nodes.QuadraticExpansionNode):
    """Automatically created BiNode version of QuadraticExpansionNode."""
    def __init__(self, input_dim=None, dtype=None, node_id=None, stop_result=None):
        super(QuadraticExpansionBiNode, self).__init__(input_dim=input_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class RBFExpansionBiNode(BiNode, mdp.nodes.RBFExpansionNode):
    """Automatically created BiNode version of RBFExpansionNode."""
    def __init__(self, centers, sizes, dtype=None, node_id=None, stop_result=None):
        """
        Input arguments:
        centers -- Centers of the RBFs. The dimensionality
                   of the centers determines the input dimensionality;
                   the number of centers determines the output
                   dimensionalities
        sizes -- Radius of the RBFs.
                'sizes' is a list with one element for each RBF, either
                a scalar (the variance of the RBFs for isotropic RBFs)
                or a covariance matrix (for anisotropic RBFs).
                If 'sizes' is not a list, the same variance/covariance
                is used for all RBFs.
        """
        super(RBFExpansionBiNode, self).__init__(centers=centers, sizes=sizes, dtype=dtype, node_id=node_id, stop_result=stop_result)

class RBMBiNode(BiNode, mdp.nodes.RBMNode):
    """Automatically created BiNode version of RBMNode."""
    def __init__(self, hidden_dim, visible_dim=None, dtype=None, node_id=None, stop_result=None):
        """
        Arguments:

        hidden_dim -- number of hidden variables
        visible_dim -- number of observed variables
        """
        super(RBMBiNode, self).__init__(hidden_dim=hidden_dim, visible_dim=visible_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class RBMWithLabelsBiNode(BiNode, mdp.nodes.RBMWithLabelsNode):
    """Automatically created BiNode version of RBMWithLabelsNode."""
    def __init__(self, hidden_dim, labels_dim, visible_dim=None, dtype=None, node_id=None, stop_result=None):
        super(RBMWithLabelsBiNode, self).__init__(hidden_dim=hidden_dim, labels_dim=labels_dim, visible_dim=visible_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class SFA2BiNode(BiNode, mdp.nodes.SFA2Node):
    """Automatically created BiNode version of SFA2Node."""
    def __init__(self, input_dim=None, output_dim=None, dtype=None, node_id=None, stop_result=None):
        super(SFA2BiNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class SFABiNode(BiNode, mdp.nodes.SFANode):
    """Automatically created BiNode version of SFANode."""
    def __init__(self, input_dim=None, output_dim=None, dtype=None, node_id=None, stop_result=None):
        super(SFABiNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class TDSEPBiNode(BiNode, mdp.nodes.TDSEPNode):
    """Automatically created BiNode version of TDSEPNode."""
    def __init__(self, lags=1, limit=1.0000000000000001e-05, max_iter=10000, verbose=False, whitened=False, white_comp=None, white_parm=None, input_dim=None, dtype=None, node_id=None, stop_result=None):
        """
        Input arguments:

        lags    -- list of time-lags to generate the time-delayed covariance
                   matrices. If lags is an integer, time-lags 1,2,...,'lags'
                   are used.
                   Note that time-lag == 0 (instantaneous correlation) is
                   always implicitly used.

        whitened -- Set whitened is True if input data are already whitened.
                    Otherwise the node will whiten the data itself.

        white_comp -- If whitened is False, you can set 'white_comp' to the
                      number of whitened components to keep during the
                      calculation (i.e., the input dimensions are reduced to
                      white_comp by keeping the components of largest variance).

        white_parm -- a dictionary with additional parameters for whitening.
                      It is passed directly to the WhiteningNode constructor.
                      Ex: white_parm = { 'svd' : True }

        limit -- convergence threshold.

        max_iter     -- If the algorithms does not achieve convergence within
                        max_iter iterations raise an Exception. Should be
                        larger than 100.
        """
        super(TDSEPBiNode, self).__init__(lags=lags, limit=limit, max_iter=max_iter, verbose=verbose, whitened=whitened, white_comp=white_comp, white_parm=white_parm, input_dim=input_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class TimeFramesBiNode(BiNode, mdp.nodes.TimeFramesNode):
    """Automatically created BiNode version of TimeFramesNode."""
    def __init__(self, time_frames, gap=1, input_dim=None, dtype=None, node_id=None, stop_result=None):
        """
        Input arguments:
        time_frames -- Number of delayed copies
        gap -- Time delay between the copies
        """
        super(TimeFramesBiNode, self).__init__(time_frames=time_frames, gap=gap, input_dim=input_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)

class WhiteningBiNode(BiNode, mdp.nodes.WhiteningNode):
    """Automatically created BiNode version of WhiteningNode."""
    def __init__(self, input_dim=None, output_dim=None, dtype=None, svd=False, reduce=False, var_rel=9.9999999999999998e-13, var_abs=1.0000000000000001e-15, var_part=None, node_id=None, stop_result=None):
        """The number of principal components to be kept can be specified as
        'output_dim' directly (e.g. 'output_dim=10' means 10 components
        are kept) or by the fraction of variance to be explained
        (e.g. 'output_dim=0.95' means that as many components as necessary
        will be kept in order to explain 95% of the input variance).

        Other Keyword Arguments:

        svd -- if True use Singular Value Decomposition instead of the
               standard eigenvalue problem solver. Use it when PCANode
               complains about singular covariance matrices

        reduce -- Keep only those principal components which have a variance
                  larger than 'var_abs' and a variance relative to the
                  first principal component larger than 'var_rel' and a
                  variance relative to total variance larger than 'var_part'
                  (set var_part to None or 0 for no filtering).
                  Note: when the 'reduce' switch is enabled, the actual number
                  of principal components (self.output_dim) may be different
                  from that set when creating the instance.
        """
        super(WhiteningBiNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype, svd=svd, reduce=reduce, var_rel=var_rel, var_abs=var_abs, var_part=var_part, node_id=node_id, stop_result=stop_result)

class XSFABiNode(BiNode, mdp.nodes.XSFANode):
    """Automatically created BiNode version of XSFANode."""
    def __init__(self, basic_exp=None, intern_exp=None, svd=False, verbose=False, input_dim=None, output_dim=None, dtype=None, node_id=None, stop_result=None):
        """
        Keyword arguments:

          basic_exp --  a tuple (node, args, kwargs) defining the node
                        used for the basic nonlinear expansion.  It is
                        assumed that the mixture is linearly invertible
                        after this expansion. The higher the complexity
                        of the nonlinearity, the higher are the chances
                        of inverting the unknown mixture. On the other
                        hand, high complexity of the nonlinear
                        expansion increases the danger of numeric
                        instabilities, which can cause singularities in
                        the simulation or errors in the source
                        estimation. The trade-off has to be evaluated
                        carefully.  Default:
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
        super(XSFABiNode, self).__init__(basic_exp=basic_exp, intern_exp=intern_exp, svd=svd, verbose=verbose, input_dim=input_dim, output_dim=output_dim, dtype=dtype, node_id=node_id, stop_result=stop_result)
