from bach_utils.heatmapvis import *
from os import walk
import os
from copy import deepcopy
import argparse


parser = argparse.ArgumentParser(description="Generate heatmap figures")
parser.add_argument('--path', "-p", type=str, help=help, metavar='')
# parser.add_argument('--env', "-e", type=str, help=help, metavar='')
# parser.add_argument('--method', "-m", type=str, help=help, metavar='')
parser.add_argument('--save_path', "-sp", type=str, help=help, metavar='', default="")
parser.add_argument('--save', dest='save', action='store_true')
parser.set_defaults(save=False)

args = parser.parse_args()

agents = ["pred", "prey"]

# path = os.path.join(args.env, args.method)
path = args.path
save_path = args.save_path if args.save_path != "" else os.path.join(path, "joint_heatmap.png")
print(save_path)

y_axis_reversed = True#False

heatmaps_path_agent = {a:next(walk(os.path.join(path, a)), (None, None, []))[2] for a in agents}
print(f"Generate heatmap from #{len(heatmaps_path_agent[agents[0]])} experiments")
# print(heatmaps_path_agent)
heatmaps = [[np.load(os.path.join(path, a, p)) for p in heatmaps_path_agent[a]] for a in agents]

heatmaps = {}
old_heatmaps = {}
for a in agents:
    heatmaps[a] = []
    old_heatmaps[a] = []
    for p in heatmaps_path_agent[a]:
        heatmap = np.load(os.path.join(path,a, p))
        old_heatmaps[a].append(deepcopy(heatmap))
        # if(a == "pred"):
        #     heatmap *= -1
        heatmaps[a].append(heatmap)
merged_heatmaps = []
for i in range(len(heatmaps_path_agent[agents[0]])):
    # if(y_axis_reversed):
    #     merged_heatmaps.append(np.flip(-heatmaps["pred"][i] + heatmaps["prey"][i],0))
    # else:
    # merged_heatmaps.append(heatmaps["prey"][i])

    merged_heatmaps.append((heatmaps["prey"][i] - heatmaps["pred"][i])/1000)

    # merged_heatmaps.append(-heatmaps["pred"][i] + heatmaps["prey"][i])


# print (off_dia_sum)


x_axis = np.load(os.path.join(path,"pred", "axis", "evaluation_matrix_axis_x.npy"))
y_axis = np.load(os.path.join(path,"pred", "axis", "evaluation_matrix_axis_y.npy"))

# print(x_axis)
x_axis = [int("0" if "00." in e else e.strip("0").strip(".")) for e in x_axis]
y_axis = [int("0" if "00." in e else e.strip("0").strip(".")) for e in y_axis]
# if(y_axis_reversed):
#     y_axis = reversed(y_axis)
# x_axis = [e.replace(".", ",") for e in x_axis]
# y_axis = [e.replace(".", ",") for e in y_axis]
# print(x_axis)
# HeatMapVisualizer.visSeaborn(   heatmaps["pred"],
#                             mn_val=0,mx_val=1000, center=500,
#                             x_axis=x_axis, y_axis=y_axis,
#                             save=args.save, save_path=save_path,
#                             labels=["prey", "predator", "Episode length mean", ""],
#                             y_axis_reversed=y_axis_reversed)#,cmap="YlGnBu")
# HeatMapVisualizer.visSeaborn(   heatmaps["prey"],
#                             mn_val=0,mx_val=1000, center=500,
#                             x_axis=x_axis, y_axis=y_axis,
#                             save=args.save, save_path=save_path,
#                             labels=["prey", "predator", "Episode length mean", ""],
#                             y_axis_reversed=y_axis_reversed)#,cmap="YlGnBu")



HeatMapVisualizer.visSeaborn(   merged_heatmaps,
                                mn_val=-1,mx_val=1, center=0,
                                # mn_val=0,mx_val=1000, center=500,
                                x_axis=x_axis, y_axis=y_axis,
                                save=args.save, save_path=save_path,
                                labels=["prey", "predator", "Episode length mean", ""],
                                y_axis_reversed=y_axis_reversed)#,cmap="YlGnBu")

# for a in agents:
#     HeatMapVisualizer.visSeaborn(old_heatmaps[a],
#                                     mn_val=0,mx_val=1000, center=500)#, cmap="YlGnBu")#, cmap="YlGnBu")




