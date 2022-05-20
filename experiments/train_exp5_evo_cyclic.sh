#!/bin/sh

for Var in 3 10 22 50 101
do
    python3 train_new.py --exp configs/main/baseV4_evo_cyclic_ppo.json --prefix "[Final exp, cyclic] " --seed $Var
    # python3 train_new.py --exp experiments_configs/experiment_testing.json --seed $Var
done


# python3 train_new.py --exp experiments_configs/base.json 

# python3 train_new.py --exp experiments_configs/base_v3.json 

# python3 train_new.py --exp experiments_configs/base_v3_delta.json 

# python3 train_new.py --exp experiments_configs/base_v3_delta10.json 

# python3 train_new.py --exp experiments_configs/base_v3_latest.json

# python3 train_new.py --exp experiments_configs/base_v3_cyclic.json

# python3 train_new.py --exp experiments_configs/base_v3_reverse_cyclic.json