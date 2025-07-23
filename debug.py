import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

'''A tutorial on using a debugger. Code used from 
https://matplotlib.org/stable/gallery/animation/dynamic_image.html#sphx-glr-gallery-animation-dynamic-image-py'''

fig, ax = plt.subplots()


def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(60):
    x += np.pi / 15
    y += np.pi / 30
    z = f(x, y)
    im = ax.imshow(z, animated=True)
    if i == 0:
        ax.imshow(z)  # show an initial one first
    ims.append([imx])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

plt.show()