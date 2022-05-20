import os
from os import walk
import argparse
import shutil
from pathlib import Path

# shutil.copyfile(src, dst)

# # 2nd option
# shutil.copy(src, dst)  # dst can be a folder; use shutil.copy2() to preserve timestamp


parser = argparse.ArgumentParser(description="Generate heatmap figures")
parser.add_argument('--path', "-p", type=str, help=help, metavar='')
parser.add_argument('--eval_path', "-ep", type=str, help=help, metavar='')
args = parser.parse_args()

path = args.path
eval_path = args.eval_path
agents = ["pred", "prey"]
eval_mat_name = "evaluation_matrix.npy"
eval_mat_axis_names = ["evaluation_matrix_axis_x.npy", "evaluation_matrix_axis_y.npy"]

def protected_copy(src, dst):
    try:
        shutil.copy2(src, dst)
        print(f"Copy from {src} -> {dst}")
        return True
    except Exception as e:
        print(e)
        return False

def check_create_dir(p):
    path_check = Path(p)
    path_check.mkdir(parents=True, exist_ok=True)

# Check the existence and create the direcories if not exist
for a in agents:
    new_eval_mat_path = os.path.join(eval_path, a)
    check_create_dir(new_eval_mat_path)
    check_create_dir(os.path.join(new_eval_mat_path, "axis"))


# Go through all the experiments in the root directory
# print(path)
exp_paths = next(os.walk(path))[1]

# print(exp_paths)
# Copy the files
axis_copied = False
for p in exp_paths:
    for a in agents:
        eval_mat_path = os.path.join(path, p, a, eval_mat_name)
        new_eval_mat_path = os.path.join(eval_path, a, p+"-"+eval_mat_name)
        protected_copy(eval_mat_path, new_eval_mat_path)
        # Copy the axis only one time as they are the same for the same agent 
        if(not axis_copied):
            for axe in eval_mat_axis_names:
                eval_mat_axis_path = os.path.join(path, p, a, axe)
                new_eval_mat_axis_path = os.path.join(eval_path, a, "axis", axe)
                axis_copied = protected_copy(eval_mat_axis_path, new_eval_mat_axis_path) or axis_copied

# Get all the eval matrix and put them in the correct places for the predator and prey for the environment