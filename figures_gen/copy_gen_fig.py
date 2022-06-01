#!/usr/bin/python3

import subprocess
import os

envs = ["Evorobotpy2", "PZ"]
methods = ["latest", "random","cyclic", "delta20", "delta10", "delta5", "pop3", "pop5", "pop8"]
parent_path = "../experiments/selfplay-final-results"

get_eval_mat_cmd = lambda s,d: f"python3 get_eval_mat.py --p {s} -ep {d}"
heatmap_gen_cmd  = lambda p,c: f"python3 heatmap_gen.py --path {p} --save {'' if c == '' else '--save_path '+c}"



for e in envs:
    for m in methods:
        env_method = os.path.join(e, m)
        path = os.path.join(parent_path, env_method)
        print(f"###########################################################")
        print(f"Execute: {get_eval_mat_cmd(path, env_method)}")
        subprocess.call(get_eval_mat_cmd(path, env_method), shell=True)
        print(f"################## {env_method} ##################")
        cmd = heatmap_gen_cmd(env_method, e+"_"+m+".png")
        # cmd = heatmap_gen_cmd(env_method, "")
        print(f"Execute: {cmd}")
        subprocess.call(cmd, shell=True)
        print(f"###########################################################")

