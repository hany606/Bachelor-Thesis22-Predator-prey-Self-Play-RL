import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
 

cmap = sns.color_palette("coolwarm", as_cmap=True).reversed() #sns.cm.rocket_r

# TODO: not make it full 
# ['pop5', 'pop3', 'random']
# [[ 0.          1.486       0.60266667]
#  [ 0.          0.         -0.049     ]
#  [ 0.          0.          0.        ]]

# CRITICAL - ['pop5', 'pop3', 'random', 'latest'] (SelfPlayTesting.py:448)
# CRITICAL - 
# [[ 0.          1.486       0.60266667  0.60266667]
#  [ 0.          0.         -0.049      -0.049     ]
#  [ 0.          0.          0.         -0.31933333]
#  [ 0.          0.          0.          0.        ]] (SelfPlayTesting.py:449)


# 28.05.2022 EVO (gain1-gain2)
# pop5 > delta5 > random > pop3 > latest
# CRITICAL - ['pop5', 'pop3', 'random', 'latest', 'delta5'] (SelfPlayTesting.py:449)
# CRITICAL - 
# [[ 0.          0.23733333  0.60266667  0.81566667  0.62133333]
#  [ 0.          0.         -0.049       0.16733333 -0.36466667]
#  [ 0.          0.          0.          0.43666667 -0.00266667]
#  [ 0.          0.          0.          0.         -0.10466667]
#  [ 0.          0.          0.          0.          0.        ]] (SelfPlayTesting.py:450)

# 28.05.2022 EVO (gain1+gain2)
# CRITICAL - ['pop5', 'pop3', 'random', 'latest', 'delta5'] (SelfPlayTesting.py:449)
# CRITICAL - 
# [[ 0.          1.486       0.60266667  2.28233333  0.52666667]
#  [ 0.          0.         -0.049       1.63666667  1.48666667]
#  [ 0.          0.          0.         -0.43666667  1.88      ]
#  [ 0.          0.          0.          0.          0.52066667]
#  [ 0.          0.          0.          0.          0.        ]] (SelfPlayTesting.py:450)


# Create a dataset
N = 9
# a = np.random.rand(N, N)*8 - 4
# a = np.zeros((N,N))
# a[]
# Not correct interpretation: -1, as it was saved that the columns vs rows -> but it should be 
# a = -1*np.load("crosstest_mat_evo1.npy").T
a = np.load("crosstest_mat_pz1_try1.npy").T

print(a)
# a = np.load("crosstest_pz.npy").T
# a = np.load("crosstest_evo.npy").T
# a = np.array([
#     [0,1,2, 3],
#     [0,0,4, 5],
#     [0,0,0,6],
#     [0,0,0,0],
# ]).T
# for i in range(4):
#     a[2,i] = abs(a[2,i])
# To have lower matrix are true values and the opposite in upper
mat = -np.tril(a) + np.tril(a, -1).T
print(mat)

# To have upper matrix are true values and the opposite in lower
# mat = np.tril(a) - np.tril(a, -1).T
# To have symmetric matrix
# mat = np.tril(a) + np.tril(a, -1).T

# mat = np.random.random((4,4))-0.5
# for i in range(4):
#     mat[]

# triangular mask
# mask = np.triu(np.ones_like(mat, dtype=bool)).T
# diagonal mask
# mask = np.eye(N, dtype=bool)
# No mask
mask = np.zeros((N,N), dtype=bool)

# print(mask)
labels = ["Naive", "Random", "Cyclic", "\u03B4=0.5", "\u03B4=0.7", "\u03B4=0.9", "\u03C1=3", "\u03C1=5", "\u03C1=8"][:N]
# labels = ["Latest", "Random", "\u03C1=5"]

mat = np.flip(mat,0)
mask = np.flip(mask,0)
# print(mat)

# for i in range(mat.shape[0]):
#     for j in range(mat.shape[1]):
#         mat[i,j] /= 4

df = pd.DataFrame(mat, reversed(labels), columns=labels)
# df = pd.DataFrame(np.load("crosstest_mat_evo1.npy"), labels, columns=labels)
# plot a heatmap with annotation
ax = sns.heatmap(df, annot=True,fmt=".3f", annot_kws={"size": 10}, vmin=-1, vmax=1, center=0, cmap=cmap, mask=mask,
            #   linewidths=0.1, linecolor='black'
                cbar_kws={'label': "Normalized gain"},

                )
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 17)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 17)
plt.xticks(rotation=np.pi/2) 

ax.figure.axes[-1].yaxis.label.set_size(17)
plt.show()