from bach_utils.heatmapvis import *

y_axis_reversed = True#False
N = 30
# a = np.random.rand(N, N)*2000 - 1000

# a = np.eye(N)*2000#-1000
# a = np.zeros(N, N)
a = np.tril(np.zeros((N,N))+1000,-1).T+np.tril(np.zeros((N,N))) + np.eye(N)*500
# heatmap = np.flip(a,1)

heatmap = a

x_axis = [i for i in range(N)]

# y_axis = reversed(x_axis)
y_axis = x_axis
# x_axis = [e.replace(".", ",") for e in x_axis]
# y_axis = [e.replace(".", ",") for e in y_axis]
# print(x_axis)


HeatMapVisualizer.visSeaborn(   heatmap,
                                mn_val=0,mx_val=1000, center=500,
                                x_axis=x_axis, y_axis=y_axis,
                                save=False, save_path=None, xticklabels=1, yticklabels=1, labels=["prey", "predator", "Episode length mean", ""], y_axis_reversed=y_axis_reversed)#,cmap="YlGnBu")

# for a in agents:
#     HeatMapVisualizer.visSeaborn(old_heatmaps[a],
#                                     mn_val=0,mx_val=1000, center=500)#, cmap="YlGnBu")#, cmap="YlGnBu")



# for i in range(len(heatmaps)):
#     HeatMapVisualizer.visSeaborn(heatmaps[i])

