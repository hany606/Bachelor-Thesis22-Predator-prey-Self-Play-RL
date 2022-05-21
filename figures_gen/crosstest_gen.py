import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
 

cmap = sns.color_palette("coolwarm", as_cmap=True).reversed() #sns.cm.rocket_r

# ['pop5', 'random']
# [[0.         0.60266667]
#  [0.         0.        ]]
# Create a dataset
N = 4
# a = np.random.rand(N, N) - 0.5
a = np.array([
    [0,-0.7,-1, -0.3],
    [0,0,-0.60266667, 0.2],
    [0,0,0,0.8],
    [0,0,0,0],
]).T
# for i in range(4):
#     a[2,i] = abs(a[2,i])
mat = -np.tril(a) + np.tril(a, -1).T
# mat = np.random.random((4,4))-0.5
# for i in range(4):
#     mat[]

df = pd.DataFrame(mat, ["Latest", "Random", "Population", "Delta"], columns=["Latest", "Random", "Population", "Delta"])

# plot a heatmap with annotation
sns.heatmap(df, annot=True, annot_kws={"size": 7}, vmin=-1, vmax=1, center=0, cmap=cmap)
plt.show()