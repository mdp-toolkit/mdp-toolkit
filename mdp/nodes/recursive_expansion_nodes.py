"""
This module provides nodes for recursive computation of sequences of functions.

The nodes contained in this module complement the existing
*PolynomialExpansionNode* by generalization to arbitrary recursively
computed sequences of basis function and by providing implemented
polynomial and rational recursions that are numerically stable, even
for high orders. More specifically this module provides the means
for computing Legendre, Hermite or Chebyshev polynomials of first kind.

Analytically these polynomial expansions cover the same function space
if they are of same degree. However the conventional
monome-based polynomial expansion is numerically defective beyond
degrees of about 20. Input values below one are flattened to
zero and values above one explode.
Analytically this would be compensated by very small or large coefficients,
that are not properly presentable in floating point.

Thus, for the most simple use case the *RecursiveExpansionNode*
can be used. A full-featured version comes into play in the following case.

All orthogonal sequences of functions implemented require input values
in the right scope, i.e. where the respective polynomials
are designed to be orthogonal, usually values in [0, 1] or [-1, 1].
That means, one might have to scale the input appropriately.
The *NormalizingRecursiveExpansionNode* is advisable to use, if the data supplied
drops out of this interval or it corresponding cube (i.e. [0,1]^k).
For convenience, the *NormalizingRecursiveExpansionNode* can adapt the
value range of the data during a training phase and apply the corresponding
scaling on execution.

|

.. admonition:: Reference

    Example where orthogonal expansions are advisable to use:
    URL: https://arxiv.org/abs/1805.08565, section 4.2.1, Figure 11 and A.2.
"""
import mdp
from mdp import numx as np
from mdp.nodes.expansion_nodes import _ExpansionNode, expanded_dim
__docformat__ = "restructuredtext en"


def init_standard_poly(result, x, pos, cur_var):
    """Initialize the first order before starting the recursion.

    .. note::

        An init function also returns some indices relevant to further
        steps in which the recursion formula is applied. The
        first index indicates where the recursion is applied first.
        In the case of standard polynomials this is *pos+1*.
        The second index indicates the order of the element on which
        the recursion is applied first. In the case of standard polynomials
        this order is equal to 2. The third index specifies the index
        of a special member for use in the recursion. This might be a
        constant order element that is used in every recursion. In the case
        of standard polynomials this can be *pos* (as this is where *x* is) or
        None (as *x* is passed to the recursion anyway).


    Mathematically this amounts to
    P_1 = x.

    :param result: Contains the observations along the first dimension
        and the function values of expansion w.r.t. an observation
        along the second dimension.
    :type result: numpy.ndarray

    :param x: The data to be expanded.
    :type x: numpy.ndarray

    :param pos: The index of the element to be computed, along the second
        dimension of result.
    :type int: int

    :param cur_var: The index of the current variable to be considered in the
        recursion. This value will have to be lower than x.shape[1].
    :type cur_var: int

    :returns: The index of the next element that will be computed in the
        recursion, the order of the next element, and the index of a special
        element.
    :rtype: Tuple[int, int, int]
    """
    result[:, pos] = x[:, cur_var]
    # pos, n, special
    return pos+1, 2, pos


def recf_standard_poly(result, x, special, n, cur_var, pos):
    """Implementation of the recursion formula for standard polynomials.

    The recursion formula is P_n = P_{n-1} * x.

    .. note::

        We translate this recursion formula into code as follows.

        We leave out the left part of the equation, as the
        return value is automatically set to the n-th order
        expansion of the current variable, namely::

            >>> result[:, pos]

        We access all observations of the n-1-th order expansion of the current
        variable at::

            >>> result[:, pos-1]

        Using the index of the special variable we specified in our init
        function we can set the return value to::

            >>> result[:, pos-1]*result[:, special]

        Alternatively, in this case we could have used the x variable directly,
        by setting the return value to::

            >> result[:, pos-1]*x[:, cur_var]

        The recursion is natively applied to all variables contained in the data
        one after another and the results are stored in the
        one-dimensional result array.


    :param result: Contains the observations along the first dimension
        and the function values of expansion w.r.t. an observation
        along the second dimension.
    :type result: numpy.ndarray

    :param x: The data to be expanded.
    :type x: numpy.ndarray

    :param special: Index of a special element to be considered in the
        recursion formula. E.g. the first order function value.
    :type special: int

    :param n: The order of the function to be computed at this step.
    :type n: int

    :param cur_var: The index of the current variable to be considered in the
        recursion. This value will have to be lower than x.shape[1].
    :type cur_var: int

    :param pos: The index of the element to be computed, along the second
        dimension of result.
    :type int: int

    :returns: The vectorized result (along the observations) of the
        n-th recursion step of the cur_var-th variable.
    :rtype: numpy.ndarray
    """
    return result[:, pos-1]*result[:, special]


def init_legendre_poly(result, x, pos, cur_var):
    """Initialize the first and second order before starting the recursion.

    Mathematically this amounts to
    P_1 = x
    P_2 = 3/2 * P_1*P_1 - 1/2 .

    .. note::

        The procedure on how an init function is built can be found in the
        docstring of the init function for
        standard polynomials *init_standard_poly*.


    :param result: Contains the observations along the first dimension
        and the function values of expansion w.r.t. an observation
        along the second dimension.
    :type result: numpy.ndarray

    :param x: The data to be expanded.
    :type x: numpy.ndarray

    :param pos: The index of the element to be computed, along the second
        dimension of result.
    :type int: int

    :param cur_var: The index of the current variable to be considered in the
        recursion. This value will have to be lower than x.shape[1].
    :type cur_var: int

    :returns: The index of the next element that will be computed in the
        recursion, the order of the next element, and the index of a special
        element. Is this case the first order Legendre polynomial of the 
        current variable is the special element.
    :rtype: Tuple[int, int, int]
    """
    result[:, pos] = x[:, cur_var]
    # after the first variable this is needed (no zero order available)
    result[:, pos+1] = 3./2.*result[:, pos] * result[:, pos] - .5
    return pos+2, 3, pos


def recf_legendre_poly(result, x, special, n, cur_var, pos):
    """Implementation of the recursion formula for Legendre polynomials.
    The recursion formula is Bonnet's recursion formula
    P_n = (2n-1)/n * x * P_ {n-1}  - (n-1)* P_{n-2}.

        .. note::

        The procedure on how an recursion function is built can be found in the
        docstring of the recursion function for
        standard polynomials *recf_standard_poly*.

    :param result: Contains the observations along the first dimension
        and the function values of expansion w.r.t. an observation
        along the second dimension.
    :type result: numpy.ndarray

    :param x: The data to be expanded.
    :type x: numpy.ndarray

    :param special: Index of a special element to be considered in the
        recursion formula. In this case the special element is the 
        first order Legendre polynomial of the current variable.
    :type special: int

    :param n: The order of the function to be computed at this step.
    :type n: int

    :param cur_var: The index of the current variable to be considered in the
        recursion. This value will have to be lower than x.shape[1].
    :type cur_var: int

    :param pos: The index of the element to be computed, along the second
        dimension of result.
    :type int: int

    :returns: The vectorized result (along the observations) of the
        n-th recursion step of the cur_var-th variable.
    :rtype: numpy.ndarray

    .. admonition:: Reference

        https://en.wikipedia.org/wiki/Legendre_polynomials
    """
    return (2.*n-1.)/n*result[:, special] * result[:, pos-1] -\
        (n-1.)/n*result[:, pos-2]


def init_legendre_rational(result, x, pos, cur_var):
    """Initialize the first and second order before starting the recursion.

    Mathematically this amounts to
    R_1 = (x-1)/(x+1)
    R_2 = 3/2 * (x-1)/(x+1)*P_1 - 1/2.

    .. note::

        The procedure on how an init function is built can be found in the
        docstring of the init function for
        standard polynomials *init_standard_poly*.


    :param result: Contains the observations along the first dimension
        and the function values of expansion w.r.t. an observation
        along the second dimension.
    :type result: numpy.ndarray

    :param x: The data to be expanded.
    :type x: numpy.ndarray

    :param pos: The index of the element to be computed, along the second
        dimension of result.
    :type int: int

    :param cur_var: The index of the current variable to be considered in the
        recursion. This value will have to be lower than x.shape[1].
    :type cur_var: int

    :returns: The index of the next element that will be computed in the
        recursion, the order of the next element, and the index of a special
        member. As we do not need a special element in Legendre rational 
        functions, the index of the special element is set to None.
    :rtype: Tuple[int, int, int]
    """
    result[:, pos] = (x[:, cur_var]-1.)/(x[:, cur_var]+1.)
    # after the first variable the second order is needed (no zero order available)
    result[:, pos+1] = 3./2.*(x[:, cur_var]-1.) / (x[:, cur_var]+1.) \
        * result[:, pos] - 1. / 2.
    return pos+2, 3, None


def recf_legendre_rational(result, x, special, n, cur_var, pos):
    """Implementation of the recursion formula for Legendre rational functions.
    The recursion formula is
    R_n = (2n-1)/n * (x-1)/(x+1) R_{n-1} - (n-1)/n R_{n-2}.

    .. note::

        The procedure on how an recursion function is built can be found in the
        docstring of the recursion function for
        standard polynomials *recf_standard_poly*.


    :param result: Contains the observations along the first dimension
        and the function values of expansion w.r.t. an observation
        along the second dimension.
    :type result: numpy.ndarray

    :param x: The data to be expanded.
    :type x: numpy.ndarray

    :param special: Index of a special element to be considered in the
        recursion formula. In this case it is not needed and will be None.
    :type special: int

    :param n: The order of the function to be computed at this step.
    :type n: int

    :param cur_var: The index of the current variable to be considered in the
        recursion. This value will have to be lower than x.shape[1].
    :type cur_var: int

    :param pos: The index of the element to be computed, along the second
        dimension of result.
    :type int: int

    :returns: The vectorized result (along the observations) of the
        n-th recursion step of the cur_var-th variable.
    :rtype: numpy.ndarray

    .. admonition:: Reference

        https://en.wikipedia.org/wiki/Legendre_rational_functions
    """
    return (2.*n-1.)/n*(x[:, cur_var]-1.) / (x[:, cur_var]+1.) \
        * result[:, pos-1] - (n-1.) / n * result[:, pos-2]


def init_chebyshev_poly(result, x, pos, cur_var):
    """Initialize the first and second order before starting the recursion.

    Mathematically this amounts to
    T_1 = x
    T_2 = 2x T_1 -1.

    .. note::

        The procedure on how an init function is built can be found in the
        docstring of the init function for
        standard polynomials *init_standard_poly*.


    :param result: Contains the observations along the first dimension
        and the function values of expansion w.r.t. an observation
        along the second dimension.
    :type result: numpy.ndarray

    :param x: The data to be expanded.
    :type x: numpy.ndarray

    :param pos: The index of the element to be computed, along the second
        dimension of result.
    :type int: int

    :param cur_var: The index of the current variable to be considered in the
        recursion. This value will have to be lower than x.shape[1].
    :type cur_var: int

    :returns: The index of the next element that will be computed in the
        recursion, the order of the next element, and the index of a special
        member. The special element is the first Chebyshev polynomial
        of the current variable.
    :rtype: Tuple[int, int, int]
    """
    result[:, pos] = x[:, cur_var]
    # after the first variable this is needed (no zero order available)
    result[:, pos+1] = 2.*x[:, cur_var] * result[:, pos] - 1
    return pos+2, 3, pos


def recf_chebyshev_poly(result, x, special, n, cur_var, pos):
    """Implementation of the recursion formula for Chebyshev polynomials
    of the first kind.
    The recursion formula is
    T_n = 2xT_{n-1} - T_{n-2}.

    .. note::

        The procedure on how an recursion function is built can be found in the
        docstring of the recursion function for
        standard polynomials *recf_standard_poly*.


    :param result: Contains the observations along the first dimension
        and the function values of expansion w.r.t. an observation
        along the second dimension.
    :type result: numpy.ndarray

    :param x: The data to be expanded.
    :type x: numpy.ndarray

    :param special: Index of a special element to be considered in the
        recursion formula. The special element is the first
        Chebyshev polynomial of the current variable.
    :type special: int

    :param n: The order of the function to be computed at this step.
    :type n: int

    :param cur_var: The index of the current variable to be considered in the
        recursion. This value will have to be lower than x.shape[1].
    :type cur_var: int

    :param pos: The index of the element to be computed, along the second
        dimension of result.
    :type pos: int

    :returns: The vectorized result (along the observations) of the
        n-th recursion step of the cur_var-th variable.
    :rtype: numpy.ndarray

    .. admonition:: Reference

        https://en.wikipedia.org/wiki/Chebyshev_polynomials
    """
    return 2. * x[:, cur_var] * result[:, pos-1] - result[:, pos-2]


# collect the recusions and set the corresponding domains
recfs = {'standard_poly': (init_standard_poly, recf_standard_poly,
                           -float('Inf'), float('Inf')),
         'legendre_poly': (init_legendre_poly, recf_legendre_poly,
                           -1, 1),
         'legendre_rational': (init_legendre_rational,
                               recf_legendre_rational, 0, float('Inf')),
         'chebyshev_poly': (init_chebyshev_poly,
                            recf_chebyshev_poly, -1, 1)}


def process(off, leadVar, leadDeg, deg, pos, result, todoList):
    """Computes an inplace not exactly tensorproduct of the given
    basis functions (only the simplex of the single variable bases,
    as we limit the multi variable polynomials by a given maximum degree).
    """
    for v in range(leadVar):
        l = deg-leadDeg
        k = 1+deg*v
        for d in range(1, deg-leadDeg+1):
            l = deg+1-leadDeg-d
            result[:, pos:pos+l] = result[:, 1+off:1+off+l] * result[:, k:k+1]
            if l > 1 and leadVar > 0:
                todoList.append((pos-1, v, leadDeg+d))

            pos += l
            k += 1
            l -= 1

    return pos


class RecursiveExpansionNode(_ExpansionNode):
    """Recursively computable (orthogonal) expansions.

    .. attribute:: lower

        The lower bound of the domain on which the recursion function is
        defined or orthogonal.

    .. attribute:: upper

        The upper bound of the domain on which the recursion function is
        defined or orthogonal.
    """

    def __init__(self, degree=1, recf='standard_poly', check=False,
                 with0=True, input_dim=None, dtype=None):
        """Initialize a RecursiveExpansionNode.

        :param degree: The maximum order of the recursive expansion.
            The dimension of the return for single variable inputs will be
            equal to this value if with0 == False.
        :type degree: int

        :param recf: Must be in ['standard_poly', 'legendre_poly',
            'legendre_rational', 'chebyshev_poly'] or a tuple similar
            to those in the recfs dictionary in this module. The procedure
            on how an init function is built can be found in the docstring
            of the init and recursion function for standard polynomials 
            *init_standard_poly* and *recf_standard_poly*, respectively.
        :type recf: tuple or str

        :param check: Indicates whether the input data will
            be checked for compliance to the domain on which the function
            sequence selected is defined or orthogonal. The check will be made
            automatically in the execute method.
        :type check: bool

        :param with0: Parameter that specificies whether the zero-th order
            element is to be included at the beginning of the result array.
        :type with0: bool

        :param input_dim: Dimensionality of the input.
            Default is None.
        :type input_dim: int

        :param dtype: Datatype of the input.
            Default is None.
        :type dtype: numpy.dtype or str
        """
        super(RecursiveExpansionNode, self).__init__(input_dim, dtype)

        self.degree = degree
        self.check = check
        self.with0 = with0
        # if in dictionary
        if recf in recfs:
                # intialises the elements not based on recursion formula
            self.r_init = recfs[recf][0]
            # the recursion function
            self.recf = recfs[recf][1]
            # interval on which data must be
            self.lower = recfs[recf][2]
            self.upper = recfs[recf][3]
        # if supplied by user
        else:
            self.r_init = recf[0]
            self.recf = recf[1]
            self.lower = recf[2]
            self.upper = recf[3]

    def _get_supported_dtypes(self):
        """Return the list of dtypes supported by this node.

        :return: The list of dtypes supported by this node.
        :rtype: list
        """
        return mdp.utils.get_dtypes('AllFloat')

    def expanded_dim(self, num_vars):
        """Return the size of a vector of dimension 'dim' after
        an expansion of degree 'self._degree'.

        :param num_vars: The number of variables in the
            supplied data. This value is equal to x.shape[1].
        :type num_vars: int

        :retruns: The size of the result of execute along the second axis,
            that is, the dimension of the expansion.
        :rtype: int
        """
        res = expanded_dim(self.degree, num_vars)
        return res+1 if self.with0 else res

    def _execute(self, x):
        """Expansion of the data.

        :param x: The data to be expanded. Observations/samples must
            be along the first axis, variables along the second.
        :type x: numpy.ndarray

        :returns: The expansion of x with observations/samples along the
            first axis and corresponding function values (expansion)
            along the second axis.
        :rtype: numpy.ndarray
        """

        num_vars = x.shape[1]
        num_samples = x.shape[0]
        deg = self.degree

        _with0 = hasattr(self, "with0") and self.with0
        dim = self.expanded_dim(num_vars)
        dim += 1 if not self.with0 else 0
        result = np.empty(
            [num_samples, dim], dtype=self.dtype)

        if self.check:
            self.check_domain(x)

        result[:, 0] = 1.
        pos = 1

        if deg > 1:
            for cur_var in range(num_vars):
                # preset index for current variable
                pos, n, special = self.r_init(result, x, pos, cur_var)
                # single variable recursion
                while n <= deg:
                            # recursion step
                    result[:, pos] = self.recf(
                        result, x, special, n, cur_var, pos)
                    n += 1
                    pos += 1
        # in case input is unreasonable
        elif deg == 1:
            result[:, 0] = 1
            for i in range(num_vars):
                result[:, i+1] = x[:, i]
        elif self.with0:
            return np.ones((num_samples, num_vars))
        else:
            return None

        todoList = []
        for i in range(1, num_vars):
            todoList.append((i*deg, i, 1))
        # compute the rest of the "simplex" product
        while len(todoList) > 0:
            # pos = process(*todoList.pop(0), deg, pos, result, todoList)
            pos = process(*todoList.pop(0)+(deg, pos, result, todoList))
        return (result if _with0 else result[:, 1:])

    def check_domain(self, x, prec=1e-6):
        """Checks for compliance of the data x with the domain on which
            the function sequence selected is defined or orthogonal.

        :param x: The data to be expanded. Observations/samples must
            be along the first axis, variables along the second.
        :type x: numpy.ndarray

        :param prec: (Numerical) tolerance when checking validity.
        :type prec: float

        :raise mdp.NodeException: If one or more values lie outside of the function
            specific domain.
        """
        xmax = np.amax(x)-prec
        xmin = np.amin(x)+prec

        if (self.upper < xmax) or (self.lower > xmin):
            raise mdp.NodeException(
                "One or more values lie outside of the function specific domain.")


class NormalizingRecursiveExpansionNode(RecursiveExpansionNode):
    """Recursively computable (orthogonal) expansions and a
    trainable transformation to the domain of the expansions.

    .. attribute:: lower

        The lower bound of the domain on which the recursion
        function is defined or orthogonal.

    .. attribute:: upper

        The upper bound of the domain on which the recursion function
        is defined or orthogonal.
    """

    def __init__(self, degree=1, recf='standard_poly', check=True, with0=True,
                 input_dim=None, dtype=None):
        """Initialize a NormalizingRecursiveExpansionNode.

        :param degree: The maximum order of the recursive expansion.
            The dimension of the return for single variable inputs will be
            equal to this value if with0 == False.
        :type degree: int

        :param recf: Must be in ['standard_poly', 'legendre_poly',
            'legendre_rational', 'chebyshev_poly'] or a tuple similar
            to those in the recfs dictionary in this module. The procedure
            on how an init function is built can be found in the docstring
            of the init and recursion function for standard polynomials 
            *init_standard_poly* and *recf_standard_poly*, respectively.
        :type recf: tuple or str

        :param check: Indicates whether the input data will
            be checked for compliance to the domain on which the function
            sequence selected is defined or orthogonal. The check will be made
            automatically in the execute method.
        :type check: bool

        :param with0: Parameter that specificies whether the zero-th order
            element is to be included at the beginning of the result array.
        :type with0: bool

        :param input_dim: Dimensionality of the input.
            Default is None.
        :type input_dim: int

        :param dtype: Datatype of the input.
            Default is None.
        :type dtype: numpy.dtype or str
        """
        super(NormalizingRecursiveExpansionNode, self).__init__(degree,
                                                                recf=recf,
                                                                check=check,
                                                                with0=with0,
                                                                input_dim=input_dim,
                                                                dtype=dtype)

        self.amaxcolumn = None
        self.amincolumn = None
        self.amax = None
        self.amin = None

    def _execute(self, x):
        """Apply the transformation and execute RecursiveExpansionNode.

        :param x: The data to be expanded. Observations/samples must
            be along the first axis, variables along the second.
        :type x: numpy.ndarray
        """
        self.domain_transformation(x)
        return super(NormalizingRecursiveExpansionNode, self)._execute(x)

    def _train(self, x):
        """Determine coordinatewise and absolute maxima and minima.

        :param x: Chuck of data to be used for training. Observations/samples
            must be along the first axis, variables along the second.
        :type x: numpy.ndarray

        The values are used to generate a transformation to the valid domain
        for the data.
        """
        if self.amin is None:
            self.amaxcolumn = np.amax(x, axis=0)
            self.amincolumn = np.amin(x, axis=0)
            self.amax = np.amax(self.amaxcolumn)
            self.amin = np.amin(self.amincolumn)
        else:
            self.amaxcolumn = np.maximum(self.amaxcolumn, np.amax(x, axis=0))
            self.amincolumn = np.minimum(self.amincolumn, np.amin(x, axis=0))
            self.amax = np.amax(self.amaxcolumn)
            self.amin = np.amin(self.amincolumn)

    def _stop_training(self):
        """Create a transformation function, that transforms the data
        to the domain of the family of functions to evaluate.

        .. note::

            If the cube is unbounded the data is translated by the shortest
            length vector possible.

            If instead the cube is bounded the data is scaled around
            the mean of max and min, if neccessary. Then the mean of max and min
            is moved onto the cube mean by a translation.

            It is important to note, that the assumption is made, that the data on
            which the node is executed on, does not contain more "malign outliers"
            than the ones already supplied during training.
        """

        # the following conditionals go through different domain types
        if self.lower is None or self.upper is None:
            def f(x):
                return x

        elif self.lower == -float('Inf') and self.upper == float('Inf'):
            def f(x):
                return x

        elif self.lower == -float('Inf'):
            self.diff = self.amaxcolumn-self.upper
            self.diff = self.diff.clip(min=0)

            def f(x):
                x -= self.diff
                return x

        elif self.upper == float('Inf'):
            self.diff = self.amincolumn-self.lower
            self.diff = self.diff.clip(max=0)

            def f(x):
                x -= self.diff
                return x
        else:
            mean = self.lower+(self.upper-self.lower)/2.0
            dev = (self.upper-self.lower)

            # maybe swap for functions that returns both in one pass
            xmax = self.amax
            xmin = self.amin
            datamaxdev = xmax-xmin
            datamean = (xmax+xmin)/2.

            if (xmax < self.upper) and (xmin > self.lower):
                def f(x):
                    return x
            elif datamaxdev < dev:
                def f(x):
                    x += (-datamean)+mean
                    return x
            else:
                def f(x):
                    x += (-datamean)+mean*datamaxdev/dev
                    x *= dev/datamaxdev
                    return x
        self.domain_transformation = f

    @staticmethod
    def is_trainable():
        return True
