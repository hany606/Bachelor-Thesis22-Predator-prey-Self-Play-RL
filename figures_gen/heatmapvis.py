from bach_utils.heatmapvis import *
from os import walk
import os

path = "pz/pop5/prey"

heatmaps_path = next(walk(path), (None, None, []))[2]
print(heatmaps_path)
heatmaps = [np.load(os.path.join(path,p)) for p in heatmaps_path]

# for i in range(len(heatmaps)):
#     HeatMapVisualizer.visSeaborn(heatmaps[i])

HeatMapVisualizer.visSeaborn(heatmaps)
