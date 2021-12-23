import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

# https://plotly.com/python/heatmaps/
# http://www.heatmapper.ca/pairwise/

sns.set_theme()

class HeatMapVisualizer:
    @staticmethod
    def visSeaborn(heatmap_data,
                   mn_val=0.0,
                   mx_val=1.0,
                   center=0.5,
                   labels=["prey", "predator", "win rate", "win rate (heatmap)"],
                   annot=False,
                   linewidth=0,
                   cmap="bwr",#"YlGnBu",
                   xticklabels=5,
                   yticklabels=5,
                   cbar=True,
                   show=True):
        if(isinstance(heatmap_data, list)):
            heatmap_data = np.mean(heatmap_data, axis=0)
        ax = sns.heatmap(heatmap_data, 
                         vmin=mn_val, vmax=mx_val, center=center, 
                         cbar_kws={'label': labels[2]}, annot=annot,
                         linewidth=linewidth,
                         cmap=cmap,
                         xticklabels=xticklabels, yticklabels=yticklabels,
                         cbar=cbar
                         )
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(labels[3])
        
        if(show):
            plt.show()

        return ax
    @staticmethod
    def visPlotly(heatmap_data,
                 mn_val=0.0,
                 mx_val=1.0,
                 xrange=None,
                 yrange=None,
                 labels=["prey", "predator", "win rate", "win rate (heatmap)"],
                 cmap="YlGnBu",
                 ):
        if(isinstance(heatmap_data, list)):
            heatmap_data = np.mean(heatmap_data, axis=0)
        xrange_cfg = xrange if xrange is not None else [i for i in range(0,heatmap_data.shape[1])]
        yrange_cfg = yrange if yrange is not None else [i for i in range(0,heatmap_data.shape[0])]
        
        xaxis_cfg = dict(type="category", categoryorder="array", categoryarray=xrange_cfg) if xrange is not None else dict()
        yaxis_cfg = dict(autorange="reversed", type="category", categoryorder="array", categoryarray=yrange_cfg) if yrange is not None else dict(autorange="reversed")

        fig = go.Figure(data=go.Heatmap(
                                        z=heatmap_data, 
                                        x=xrange_cfg,
                                        y=yrange_cfg,
                                        type = 'heatmap',
                                        colorscale = cmap.lower(),
                                        # name=dict(x="Predator", y="Prey", color="Winrate"),
                                        )
                        )
        fig.update_layout(
            xaxis=xaxis_cfg,
            yaxis=yaxis_cfg,
            title=labels[3],
            xaxis_title=labels[0],
            yaxis_title=labels[1],
            legend_title=labels[2],
        )
        fig.show()
    
    @staticmethod
    def save(*args, **kwargs):
        filename = kwargs.pop('filename', "default.png")
        dpi = kwargs.pop('dpi', 400)
        filename = filename if filename.endswith(".png") else filename+".png"
        kwargs["show"] = False
        ax = HeatMapVisualizer.visSeaborn(*args, **kwargs)
        figure = ax.get_figure()
        figure.savefig(filename, dpi=dpi)

