from bach_utils.heatmapvis import *

y_axis_reversed = True#False
N = 30
# a = np.random.rand(N, N)*2000 - 1000

mx_val = 1
mn_val = -1
center = 0
# a = np.eye(N)#*2000#-1000
# a = np.zeros(N, N)
a = np.tril(np.zeros((N,N))+mx_val,5).T+np.tril(np.zeros((N,N))+mn_val) + np.eye(N)

# a = np.tril(np.zeros((N,N))+1000,-1).T+np.tril(np.zeros((N,N))) + np.eye(N)*500
# heatmap = np.flip(a,1)

# # https://stackoverflow.com/questions/53860882/how-to-calculate-sum-of-abs-of-all-off-diagonal-elements-of-a-numpy-array#:~:text=There%20is%20a%20nice%20option,using%20any%20explicit%20for%2Dloops.
# dia = np.diag_indices(N) # indices of diagonal elements
# dia_sum = sum(a[dia]) # sum of diagonal elements
# off_dia_sum = np.sum(a) - dia_sum # subtract the diagonal sum from total array sum
# print(dia_sum)
# print (off_dia_sum)


heatmap = a

x_axis = [i for i in range(N)]

# y_axis = reversed(x_axis)
y_axis = x_axis
# x_axis = [e.replace(".", ",") for e in x_axis]
# y_axis = [e.replace(".", ",") for e in y_axis]
# print(x_axis)


HeatMapVisualizer.visSeaborn(   heatmap,
                                mn_val=mn_val,mx_val=mx_val, center=center,
                                x_axis=x_axis, y_axis=y_axis,
                                save=False, save_path=None, xticklabels=1, yticklabels=1, labels=["prey", "predator", "Episode length mean", ""], y_axis_reversed=y_axis_reversed)#,cmap="YlGnBu")

# for a in agents:
#     HeatMapVisualizer.visSeaborn(old_heatmaps[a],
#                                     mn_val=0,mx_val=1000, center=500)#, cmap="YlGnBu")#, cmap="YlGnBu")



# for i in range(len(heatmaps)):
#     HeatMapVisualizer.visSeaborn(heatmaps[i])

