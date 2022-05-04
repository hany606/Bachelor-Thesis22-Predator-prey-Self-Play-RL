import wandb
import numpy as np
from math import log10, ceil

wandb.init(
        project="Behavioral-Learning-Thesis",
        group="self-play",
        entity= None,
)

wandb.run.save()

num_rounds = 50
max_checkpoint_num = 3
# digits_num_rounds = ceil(log10(num_rounds))
x_axis = [f"{j:02d}.{i:01d}" for j in range(num_rounds) for i in range(max_checkpoint_num)]
# x_axis = []
# for j in range(num_rounds):
#     for i in range(max_checkpoint_num):
#         n = float(f"{j}.{i}")
#         print("%04.2f"%n)
#         # x_axis.append(f"%0{ceil(log10(num_rounds))}.{ceil(log10(max_checkpoint_num))}f"%n)
#         x_axis.append("%03.2f"%n)

# print(x_axis)
axis = [x_axis, [i for i in range(1)]]

freq_matrix = np.zeros((1, 3*50))

for i in range(50):
    for j in range(3):
        freq_matrix[0, i*3+j] += i*3+j


print(axis[0])

wandb.log({f"Testing_archive/freq_heatmap": wandb.plots.HeatMap(axis[0], axis[1], freq_matrix, show_text=True)})
wandb.log({f"Testing_archive/freq_heatmap_no_text": wandb.plots.HeatMap(axis[0], axis[1], freq_matrix, show_text=False)})
