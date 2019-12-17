from scipy import ndimage, misc
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
ascent = misc.ascent()
result = ndimage.zoom(ascent, 3.0)
ax1.imshow(ascent)
ax2.imshow(result)
plt.show()

print(ascent.shape)
# (512, 512)

print(result.shape)
# (1536, 1536)
