# Adapted from https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
import numpy as np
from copy import deepcopy

# Randomly
def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

phi = np.linspace(0, np.pi, num=6)
print(phi)
theta = np.linspace(0, 2 * np.pi, num=9)
r = 1
# Outer product: https://numpy.org/doc/stable/reference/generated/numpy.outer.html
xc = r*np.outer(np.sin(theta), np.cos(phi))
yc = r*np.outer(np.sin(theta), np.sin(phi))
zc = r*np.outer(np.cos(theta), np.ones_like(phi))

# Remove the extra non-unique nodes (Ensure that all of them unique)
eps = 1e-10
# samples = list(set(zip(x,y,z)))
xcc = xc.reshape(-1)
ycc = yc.reshape(-1)
zcc = zc.reshape(-1)

samples = list(zip(xcc,ycc,zcc))
unique_samples = []
removed = []
print(len(samples))
removed_counter = 0
other_samples = deepcopy(samples)
# Simple soultion -> brute force over the samples


for i,s in enumerate(samples):
    add_flag = True
    del other_samples[0]
    # print(other_samples)
    for j,ss in enumerate(other_samples):
        dist = np.linalg.norm(np.array(s) - np.array(ss))
        # print(dist)
        if(dist < eps):
            # print(i,j)
            # print(s, ss)
            # print(dist)
            removed_counter += 1
            # print("Remove")
            add_flag = False
            break
    if(add_flag):
        unique_samples.append(s)
    else:
        removed.append(s)
    # print(s)
# print(x,y,z)
print(f"removed counter = {removed_counter}")
print(len(unique_samples))
print(unique_samples[0])
unique_samples = np.array(unique_samples)
removed = np.array(removed)
x = unique_samples[:, 0].reshape(-1,1)
y = unique_samples[:, 1].reshape(-1,1)
z = unique_samples[:, 2].reshape(-1,1)



with open("whiskers.npy", "wb") as f:
    # print(np.array(list(zip(x.reshape(-1), y.reshape(-1), z.reshape(-1)))))
    np.save(f, np.array(list(zip(x.reshape(-1), y.reshape(-1), z.reshape(-1)))))
    # np.save(f, x.reshape(-1))
    # np.save(f, y.reshape(-1))
    # np.save(f, z.reshape(-1))


ax = plt.axes(projection='3d', aspect="auto")

# fig, ax = plt.subplots(1, 1, subplot_kw={'aspect':'equal'})
ax.plot_wireframe(xc, yc, zc, color='k', rstride=1, cstride=1)
ax.scatter(x, y, z, s=100, c='b', zorder=10)
# xi, yi, zi = sample_spherical(100)
# ax.scatter(xi, yi, zi, s=100, c='r', zorder=10)
plt.show()

   
with open("whiskers.npy", "rb") as f:
    p = np.load(f)
    print(len(p))
    print(p[0])
    # x = np.load(f)
    # y = np.load(f)
    # z = np.load(f)
    # print(x)
    # print(len(x))
    
