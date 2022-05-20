# Generate figures:

Get the evaluation matrix from the experiment files:

```bash
python3 get_eval_mat.py --p <src> -ep <dst>
```

```bash
python3 get_eval_mat.py --p ../experiments/selfplay-final-results/Evorobotpy2/pop5 -ep Evorobotpy2/pop5

python3 get_eval_mat.py --p ../experiments/selfplay-final-results/Evorobotpy2/random -ep Evorobotpy2/random

python3 get_eval_mat.py --p ../experiments/selfplay-final-results/PZ/random -ep PZ/random

python3 get_eval_mat.py --p ../experiments/selfplay-final-results/PZ/pop5 -ep PZ/pop5
```

Generate the figures:

```bash
python3 heatmap_gen.py --path <eval-mat-parent-path> --save
```

```bash
python3 heatmap_gen.py --path PZ/random --save

python3 heatmap_gen.py --path PZ/pop5 --save

python3 heatmap_gen.py --path Evorobotpy2/random --save

python3 heatmap_gen.py --path Evorobotpy2/pop5 --save
```