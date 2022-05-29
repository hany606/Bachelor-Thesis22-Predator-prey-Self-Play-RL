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
# CRITICAL - [[ 0.          1.486       0.60266667  0.60266667]
#  [ 0.          0.         -0.049      -0.049     ]
#  [ 0.          0.          0.         -0.31933333]
#  [ 0.          0.          0.          0.        ]] (SelfPlayTesting.py:449)

# Create a dataset
text = np.array([   ['(M4,M1)', '(M4,M2)', '(M4,M3)', '(M4,M4)'],
                    ['(M3,M1)', '(M3,M2)', '(M3,M3)', '(M3,M4)'],
                    ['(M2,M1)', '(M2,M2)', '(M2,M3)', '(M2,M4)'],
                    ['(M1,M1)', '(M1,M2)', '(M1,M3)', '(M1,M4)']])

N = 4
a = np.random.rand(N, N)*2 - 1
# a = np.zeros((N,N))
# a[]
# a = np.array([
#     [0,-0.7,-1, -0.3],
#     [0,0,-0.60266667, 0.2],
#     [0,0,0,0.8],
#     [0,0,0,0],
# ]).T
# for i in range(4):
#     a[2,i] = abs(a[2,i])
# To have upper matrix are true values and the opposite in lower
# mat = -np.tril(a) + np.tril(a, -1).T
# To have lower matrix are true values and the opposite in upper
mat = np.tril(a) - np.tril(a, -1).T
# To have symmetric matrix
# mat = np.tril(a) + np.tril(a, -1).T

# mat = np.random.random((4,4))-0.5
# for i in range(4):
#     mat[]

# triangular mask
# mask = np.triu(np.ones_like(mat, dtype=bool)).T

# diagonal mask
mask = np.eye(N, dtype=bool)

# No mask
# mask = np.zeros((N,N), dtype=bool)

# print(mask)
labels = ["Method 1", "Method 2", "Method 3", "Method 4"]

mat = np.flip(mat,0)
mask = np.flip(mask,0)
# print(mat)

df = pd.DataFrame(mat, reversed(labels), columns=labels)

# plot a heatmap with annotation
ax = sns.heatmap(df, annot=text,fmt="", annot_kws={"size": 7}, vmin=-1, vmax=1, center=0, cmap=cmap, mask=mask,
                #   linewidths=0.1, linecolor='black'
                cbar_kws={'label': "Normalized gain"},
                  )
ax.figure.axes[-1].yaxis.label.set_size(12)

plt.show()