# Experiments

## Training

```bash

python3 train_new.py --exp <.json> 

```


## Training testing

```bash

python3 train_new.py --exp experiments_configs/experiment_testing.json 

```

This reports to the old wandb logs


## Evaluation visualization

```bash

python3 test_new.py --exp <.json>

```

For example: 

```bash

python3 test_new.py --exp experiments_configs/visualization.json

```

