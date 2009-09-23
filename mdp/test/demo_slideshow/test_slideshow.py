
import os
import math
import matplotlib.pyplot as plt
import mdp

n_frames = 25
path = "animation"
try:
    os.makedirs(path)
except:
    pass

# create the animation images
filenames = []
points = [] 
for i in range(n_frames):
    points.append((i, math.sin(1.0*i/n_frames * 2*math.pi)))
    plt.figure()
    plt.plot(*zip(*points))
    plt.ylim(-1.2, 1.2)
    plt.xlim(0, n_frames)
    filename = "img%04d.png" % i
    plt.savefig(filename = os.path.join(path, filename))
    filenames.append(filename)

# cretate the slideshow
mdp.utils.show_image_slideshow(os.path.join(path, "animation.html"),
                               image_size=(400,300),
                               filenames=filenames, title="Test Animation")