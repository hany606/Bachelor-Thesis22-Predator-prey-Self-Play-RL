import argparse

from bach_utils.json_parser import WandbHeatMapParser
from bach_utils.heatmapvis import *


parser = argparse.ArgumentParser(description='Self-play experiment training script')
parser.add_argument('--json', type=str, help='The json filename', metavar='')
parser.add_argument('--csv', type=str, const=None, help='The json filename', metavar='')

args = parser.parse_args()

heatmap = WandbHeatMapParser.json2csv(args.json, args.csv, shapes=[300,300])

HeatMapVisualizer.visPlotly(heatmap)
HeatMapVisualizer.visSeaborn(heatmap)

HeatMapVisualizer.save(heatmap, filename=args.json)