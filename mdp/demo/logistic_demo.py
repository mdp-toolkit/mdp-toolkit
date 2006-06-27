 

"""This demo shows an example of the use of MDP in real life."""


# We show an application of Slow Feature Analysis to the analysis of
# non-stationary time series. We consider a chaotic time series generated
# by the logistic map based on the logistic equation (a demographic model
# of the population biomass of species in the presence of limiting factors
# such as food supply or disease), and extract the slowly varying parameter
# that is hidden behind the time series.
#
# This example reproduces some of the results reported in:
# Laurenz Wiskott, `Estimating Driving Forces of Nonstationary Time Series
# with Slow Feature Analysis`. arXiv.org e-Print archive,
# http://arxiv.org/abs/cond-mat/0312317

import mdp

# plot things only if we have scipy and if we are not called by
# the run_all.py script
def plot(*args):
    try:
        if __name__ != 'novisual':
            mdp.numx.gplt.plot(*args)
    except:
        pass

def hold(*args):
    try:
        if __name__ != 'novisual':
            mdp.numx.gplt.hold(*args)
    except:
        pass

# Generate first the slowly varying driving force,
# a combination of three sine waves (freqs: 5, 11, 13 Hz), and define a function
# to generate the logistic map
p2 = mdp.numx.pi*2
t = mdp.numx.linspace(0,1,10000,endpoint=0) # time axis 1s, samplerate 10KHz
dforce = mdp.numx.sin(p2*5*t) + mdp.numx.sin(p2*11*t) + mdp.numx.sin(p2*13*t)
def logistic_map(x,r):
    return r*x*(1-x)

# Note that we define ``series`` to be a two-dimensional array
# Inputs to MDP must be two-dimensional arrays with variables
# on columns and observations on rows. In this case we have only
# one variable:
series = mdp.numx.zeros((10000,1),'d')

# Fix the initial condition:
series[0] = 0.6

# Generate the time-series using the logistic equation
# the driving force modifies the logistic equation parameter ``r``:
for i in range(1,10000):
    series[i] = logistic_map(series[i-1],3.6+0.13*dforce[i])

# visualize the time-series if you have gplt
plot(series,".")

# Define a flow to perform SFA in the space of polynomials of degree 3.
# We need a node that embeds the time-series in a 10 dimensional
# space, where different variables correspond to time-delayed copies
# of the original time-series: the ``TimeFramesNode(10)``.
# Then we need a node that expands the new signal in the space
# of polynomials of degree 3: the ``PolynomialExpansionNode(3)``.
# Finally we perform SFA onto the expanded signal
# and keep the slowest feature:  ``SFANode(output_dim=1)``.
# We also measure the *slowness* of the input time-series and
# of the slow feature obtained by SFA. Therefore we put at the
# beginning and at the end of the sequence an *analysis node*
# that computes the *eta-value* (a measure of slowness)
# of its input (see docs for the
# definition of eta-value): the  ``EtaComputerNode()``:
#
sequence = [mdp.nodes.EtaComputerNode(),
            mdp.nodes.TimeFramesNode(10),
            mdp.nodes.PolynomialExpansionNode(3),
            mdp.nodes.SFANode(output_dim=1),
            mdp.nodes.EtaComputerNode()]
flow = mdp.Flow(sequence, verbose=1)

# Since the time-series is short enough to be kept in memory
# we don't need to define generators and we can feed the flow
# directly with the whole signal. (see generators_demo.py for an example
# that uses generators) 
flow.train(series)

# Since the second and the third nodes are not trainable we are
# going to get two warnings ``Training Interrupted``. We can safely
# ignore them. To get rid of them you can either define a generator
# for each node:
#   >>> flow.train([[series],None,None,[series],[series]])
# or turn MDPWarning off:
#   >>> warnings.filterwarnings("ignore",'.*',mdp.MDPWarning)

# Execute the flow to get the slow feature
slow = flow.execute(series)

# The slow feautre should match the driving force
# up to a scaling factor, a constant offset and the sign.
# To allow a comparison we rescale the driving force
# to have zero mean and unit variance:
resc_dforce = (dforce - mdp.numx.mean(dforce,0))/mdp.numx.std(dforce,0)

# print covariance between the rescaled driving force and
# the slow feature. Note that embedding the time-series with
# 10 time frames leads to a time-series with 9 observations less:
cov = mdp.numx.cov(resc_dforce[:-9],slow)
print 'Covariance (driving force, slow feature) %f (1 if identical)'%cov

# print the *eta-values* of the chaotic time-series and of
# the slow feature
print 'Eta value (time-series): ', flow[0].get_eta(t=10000)
print 'Eta value (slow feature): ', flow[-1].get_eta(t=9996)

# plot them together
plot(resc_dforce)
hold("on")
plot(slow*abs(cov)/cov)

