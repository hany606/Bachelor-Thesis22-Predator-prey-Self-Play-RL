import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# https://plotly.com/python/heatmaps/
# http://www.heatmapper.ca/pairwise/

sns.set_theme()

class HeatMapVisualizer:
    @staticmethod
    def visSeaborn(heatmap_data,
                   mn_val=0.0,
                   mx_val=1.0,
                   center=0.5,
                   labels=["predator", "prey", "win rate", "win rate (heatmap)"],
                   annot=False,
                   linewidth=0,
                   cmap="YlGnBu",
                   xticklabels=5,
                   yticklabels=5,
                   cbar=True,
                   show=True):
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
                 labels=["predator", "prey", "win rate", "win rate (heatmap)"],
                 cmap="YlGnBu",
                 ):
        fig = go.Figure(data=go.Heatmap(
                                        z=heatmap_data, 
                                        x=[i for i in range(0,heatmap_data.shape[1])],
                                        y=[i for i in range(0,heatmap_data.shape[0])],
                                        type = 'heatmap',
                                        colorscale = cmap.lower(),
                                        # name=dict(x="Predator", y="Prey", color="Winrate"),
                                        )
                        )
        fig.update_layout(
            yaxis = dict(autorange="reversed"),
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

