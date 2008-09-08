from mdp import numx, numx_linalg, numx_rand
from mdp.utils import mult, symeig, nongeneral_svd
import pylab
from matplotlib import ticker, axes3d



#################################################
# Testing Functions
#################################################

def S(theta):
    """
    returns x,y
      a 2-dimensional S-shaped function
      for theta ranging from 0 to 1
    """
    t = 3*numx.pi * (theta-0.5)
    x = numx.sin(t)
    y = numx.sign(t)*(numx.cos(t)-1)
    return x,y

def rand_on_S(N,sig=0,hole=False):
    t = numx_rand.random(N)
    x,z = S(t)
    y = numx_rand.random(N)*5.0
    if sig:
        x += numx_rand.normal(scale=sig,size=N)
        y += numx_rand.normal(scale=sig,size=N)
        z += numx_rand.normal(scale=sig,size=N)
    if hole:
        indices = numx.where( ((0.3>t) | (0.7<t)) | ((1.0>y) | (4.0<y)) )
        #indices = numx.where( (0.3>t) | ((1.0>y) | (4.0<y)) )
        return x[indices],y[indices],z[indices],t[indices]
    else:
        return x,y,z,t

def scatter_2D(x,y,t=None,cmap=pylab.cm.jet):
    #fig = pylab.figure()
    pylab.subplot(212)
    if t==None:
        pylab.scatter(x,y)
    else:
        pylab.scatter(x,y,c=t,cmap=cmap)

    pylab.xlabel('x')
    pylab.ylabel('y')


def scatter_3D(x,y,z,t=None,cmap=pylab.cm.jet):
    fig = pylab.figure

    if t==None:
        ax.scatter3D(x,y,z)
    else:
        ax.scatter3D(x,y,z,c=t,cmap=cmap)

    if x.min()>-2 and x.max()<2:
        ax.set_xlim(-2,2)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # elev, az
    ax.view_init(10, -80)


def runtest1(N=1000,k=15,r=None,sig=0,output_dim=0.9,hole=False,type='LLE',svd=False):
    #generate data
    x,y,z,t = rand_on_S(N,sig,hole=hole)
    data = numx.asarray([x,y,z]).T

    #train LLE and find projection
    if type=='HLLE':
        LN = HLLENode(k=k, r=r, output_dim=output_dim, verbose=True)
    else:
        LN = LLENode(k=k, r=r, output_dim=output_dim, verbose=True, svd=True)
    LN.train(data)
    LN.stop_training()
    projection = LN.training_projection
    #projection = LN.execute(data)

    #plot input in 3D
    fig = pylab.figure(1, figsize=(6,8))
    pylab.clf()
    ax = axes3d.Axes3D(fig,rect=[0,0.5,1,0.5])
    ax.scatter3D(x,y,z,c=t,cmap=pylab.cm.jet)
    ax.set_xlim(-2,2)
    ax.view_init(10, -80)

    #plot projection in 2D
    pylab.subplot(212)
    pylab.scatter(projection[:,0],\
                  projection[:,1],\
                  c=t,cmap=pylab.cm.jet)

def runtest2(N=1000,k=15,sig=0,output_dim=0.9,hole=False,type='LLE'):
    #generate data
    x,y,z,t = rand_on_S(2*N,sig,hole=hole)
    N = len(t)/2
    data = numx.asarray([x,y,z]).T

    #train HLLE and find projection
    if type=='HLLE':
        LN = HLLENode(k=15,output_dim=0.9)
    else:
        LN = LLENode(k=15,output_dim=0.9)
    LN.train(data[:N])
    projection = LN.execute(data[N:])
    t = t[N:]
    #note that the above line could equivalently be
    #  LN.train(data)
    #  LN.stop_training(data)
    #  LN.execute(data)

    #plot input in 3D
    fig = pylab.figure(figsize=(6,8))
    ax = axes3d.Axes3D(fig,rect=[0,0.5,1,0.5])
    ax.scatter3D(x[N:],y[N:],z[N:],c=t,cmap=pylab.cm.jet)
    ax.set_xlim(-2,2)
    ax.view_init(10, -80)

    #plot projection in 2D
    pylab.subplot(212)
    pylab.scatter(projection[:,0],\
                  projection[:,1],\
                  c=t,cmap=pylab.cm.jet)


#######################################
#  Run Tests
#######################################
if __name__ == '__main__':
    runtest1(N=1000,k=15,sig=0,output_dim=0.9,hole=False,type='LLE')
    pylab.show()
