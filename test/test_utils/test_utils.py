from bach_utils.test import test_print

test_print()

from bach_utils.os import *

file = get_latest(log_dir="save-SelfPlay1v1-Pred_Prey-v0-PPO-full-vel-07.29.2021_22.38.20/pred",
                         startswith="history")
print(file)

file = get_latest(log_dir="save-SelfPlay1v1-Pred_Prey-v0-PPO-full-vel-07.29.2021_22.38.20/pred",
                         startswith="history")
print(file)

# files = get_startswith(log_dir="save-SelfPlay1v1-Pred_Prey-v0-PPO-full-vel-07.29.2021_22.38.20/pred",
#                 startswith="history")

# print(files)