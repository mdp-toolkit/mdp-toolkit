"""
Test demonstration for creating a single slideshow.
"""

import os
import math
import matplotlib.pyplot as plt
import mdp

n_frames = 25
path = "animation"
try:
    os.makedirs(path)
except Exception:
    pass

# create the animation images
filenames = []
section_ids = []
points = []
for i in range(n_frames):
    points.append((i, math.sin(1.0*i/n_frames * 2*math.pi)))
    if i <= n_frames/2:
        section_ids.append("positive")
    else:
        section_ids.append("negative")
    plt.figure()
    plt.plot(*zip(*points))
    plt.ylim(-1.2, 1.2)
    plt.xlim(0, n_frames)
    filename = "img%04d.png" % i
    plt.savefig(filename = os.path.join(path, filename))
    filenames.append(filename)

# cretate the slideshow
mdp.utils.show_image_slideshow(filenames=filenames, title="Test Animation",
                               image_size=(400,300),
                               filename=os.path.join(path, "animation.html"),
                               section_ids=section_ids)
