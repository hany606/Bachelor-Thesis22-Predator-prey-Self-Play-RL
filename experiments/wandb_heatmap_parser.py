import argparse
import numpy as np

from bach_utils.json_parser import WandbHeatMapParser
from bach_utils.heatmapvis import *


parser = argparse.ArgumentParser(description='Self-play experiment training script')
parser.add_argument('--json', type=str, default=None, action='append', nargs='+', help='The json filename', metavar='')
parser.add_argument('--csv', type=str, default=None, help='The json filename', metavar='')
parser.add_argument('--np', type=str, default=None, action='append', nargs='+', help='The json filename', metavar='')
parser.add_argument('--transpose', type=bool, default=False, help='Transpose matrix', metavar='')

args = parser.parse_args()

if(args.json is None and args.np is None):
    raise ValueError("json or np arguments shoudl be specified")

heatmap = None
if(args.json is not None):
    heatmap = WandbHeatMapParser.json2csv(args.json, args.csv, shapes=[300,300])

elif(args.np is not None):
    if(isinstance(args.np, list)):
        heatmap = []
        for hm in args.np[0]:
            loaded_heatmap = np.load(hm)
            if(args.transpose):
                loaded_heatmap = loaded_heatmap.T
            heatmap.append(loaded_heatmap)
    else:
        loaded_heatmap = np.load(heatmap)
        if(args.transpose):
            loaded_heatmap = loaded_heatmap.T

        heatmap = loaded_heatmap

HeatMapVisualizer.visPlotly(heatmap)
HeatMapVisualizer.visSeaborn(heatmap)

# HeatMapVisualizer.save(heatmap, filename=args.json)