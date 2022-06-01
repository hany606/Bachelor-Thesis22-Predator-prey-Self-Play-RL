#!/bin/sh
Var=3
python3 train_new.py --exp configs/main/baseV4_pz_random_sac.json --prefix "[Final exp, random, sac] " --seed $Var
python3 train_new.py --exp configs/main/baseV4_pz_latest_sac.json --prefix "[Final exp, latest, sac] " --seed $Var
python3 train_new.py --exp configs/main/baseV4_pz_delta5_sac.json --prefix "[Final exp, delta5, sac] " --seed $Var
python3 train_new.py --exp configs/main/baseV4_pz_cyclic_sac.json --prefix "[Final exp, cyclic, sac] " --seed $Var
python3 train_new.py --exp configs/main/baseV4_pz_pop5_random_sac.json --prefix "[Final exp, pop5, sac] " --seed $Var
python3 train_new.py --exp configs/main/baseV4_pz_delta10_sac.json --prefix "[Final exp, delta10, sac] " --seed $Var
python3 train_new.py --exp configs/main/baseV4_pz_delta20_sac.json --prefix "[Final exp, delta20, sac] " --seed $Var


# python3 train_new.py --exp experiments_configs/base.json 

# python3 train_new.py --exp experiments_configs/base_v3.json 

# python3 train_new.py --exp experiments_configs/base_v3_delta.json 

# python3 train_new.py --exp experiments_configs/base_v3_delta10.json 

# python3 train_new.py --exp experiments_configs/base_v3_latest.json

# python3 train_new.py --exp experiments_configs/base_v3_cyclic.json

# python3 train_new.py --exp experiments_configs/base_v3_reverse_cyclic.json