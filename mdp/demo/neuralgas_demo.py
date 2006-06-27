 

"""
This demo shows an example of the use of the Growing Neural Gas Node."""

# We generate uniformly distributed random data points confined on different
# 2-D geometrical objects. The Growing Neural Gas Node builds a graph with the
# same topological structure.
#
# This demo requires wxPython in order to make something meaningful.
#
# Note: on our system wxPython launches a 'Deprecation Warning' and, on exit,
# a 'wxPyAssertionError'. These errors can be safely ignored. Suggestions about
# how to solve this behavior are highly appreciated.

import mdp

# plot things only if we have scipy and if we are not called by
# the run_all.py script

visual = (__name__ != 'novisual')
if visual:
    try: 
        import gui_thread
        gui_thread.start()
        plt = mdp.numx.plt
    except:
        visual = False

def setup_plot():
    if visual:
        from time import sleep
        # set up plot
        fig = plt.figure()
        sleep(0.5)
        fig.axes[0].grid_visible = 0
        fig.axes[1].grid_visible = 0
        fig.SetSize((587, 400))
        fig.update()
    else:
        fig = None
    return fig

def plot(*args):
    if visual:
        return plt.plot(*args)
    else:
        return None

def clear_all_lines(x):
    if visual:
        x.line_list.data = x.line_list.data[:1]

# Fix the random seed to obtain reproducible results:
mdp.numx_rand.seed(1266090063, 1644375755)

def plot_graph(graph, fig):
    """This function plots the edges of a graph. It communicates directly
    with scipy.plt in order to be faster."""
    if visual:
        from time import sleep
        # validate active figure
        plt.interface.validate_active()
        # each edge is represented as a line
        lines = []
        for e in graph.edges:
            x0, y0 = e.head.data.pos
            x1, y1 = e.tail.data.pos
            line = plt.plot_objects.line_object([[x0,y0],[x1,y1]])
            line.color, line.marker_type, line.line_type = ['custom']*3
            line.set_color('red')
            lines.append(line)
        # add all lines to the current figure (much faster than plotting
        # every single edge individually)
        fig.line_list.extend(lines)
        fig.update()
        # this pause is here to help slower computers to update the plot
        sleep(0.5)

## some functions to generate uniform probability distributions on
## different geometrical objects:

def uniform(min_, max_, dims):
    """Return a random number between min_ and max_ ."""
    return mdp.numx_rand.random(dims)*(max_-min_)+min_

def circumference_distr(center, radius, n):
    """Return n random points uniformly distributed on a circumference."""
    phi = uniform(0, 2*mdp.numx.pi, (n,1))
    x = radius*mdp.numx.cos(phi)+center[0]
    y = radius*mdp.numx.sin(phi)+center[1]
    return mdp.numx.concatenate((x,y), axis=1)

def circle_distr(center, radius, n):
    """Return n random points uniformly distributed on a circle."""
    phi = uniform(0, 2*mdp.numx.pi, (n,1))
    sqrt_r = mdp.numx.sqrt(uniform(0, radius*radius, (n,1)))
    x = sqrt_r*mdp.numx.cos(phi)+center[0]
    y = sqrt_r*mdp.numx.sin(phi)+center[1]
    return mdp.numx.concatenate((x,y), axis=1)

def rectangle_distr(center, w, h, n):
    """Return n random points uniformly distributed on a rectangle."""
    x = uniform(-w/2., w/2., (n,1))+center[0]
    y = uniform(-h/2., h/2., (n,1))+center[1]
    return mdp.numx.concatenate((x,y), axis=1)


N = 2000
# Explicitly collect random points from some distributions:
#
# - Circumferences:
cf1 = circumference_distr([6,-0.5], 2, N)
cf2 = circumference_distr([3,-2], 0.3, N)

# - Circles:
cl1 = circle_distr([-5,3], 0.5, N/2)
cl2 = circle_distr([3.5,2.5], 0.7, N)

# - Rectangles:
r1 = rectangle_distr([-1.5,0], 1, 4, N)
r2 = rectangle_distr([+1.5,0], 1, 4, N)
r3 = rectangle_distr([0,+1.5], 2, 1, N/2)
r4 = rectangle_distr([0,-1.5], 2, 1, N/2)

# Shuffle the points to make the statistics stationary
x = mdp.numx.concatenate([cf1, cf2, cl1, cl2, r1,r2,r3,r4], axis=0)
x = mdp.numx.take(x,mdp.numx_rand.permutation(x.shape[0]))

fig = setup_plot()

# plot a subset of the distribution
plot(x[0:x.shape[0]:25,0],x[0:x.shape[0]:25,1],'k.')

## create a ``GrowingNeuralGasNode`` and train it:
gng = mdp.nodes.GrowingNeuralGasNode(max_nodes=75)

# The training is performed in small chunks in order to visualize
# the evolution of the graph:
print "Training the Growing Neural Gas Node...",
STEP = 500
for i in range(0,x.shape[0],STEP):
    # train
    gng.train(x[i:i+STEP])
    # update graph
    # clear all lines on the plot except the distribution
    clear_all_lines(fig)
    plot_graph(gng.graph, fig)
    
gng.stop_training()
print "... done."

# Visualizing the neural gas network, we'll see that it is
# adapted to the topological structure of the data distribution:
# plot all the distribution and the graph
fig = plot(x[:,0],x[:,1],'k.')
plot_graph(gng.graph, fig)

# Calculate the number of connected components
n_obj = len(gng.graph.connected_components())
print "Number of connected components (objects): ", n_obj

if visual:
    raw_input("--- Press <ENTER> to exit ---")
    import sys
    sys.exit()

