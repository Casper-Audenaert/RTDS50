# Code

This folder has the code for our project on early mortality prediction in the ICU (PhysioNet 2012). We compare a fixed rule that decides at 24 h with a halting rule that can decide earlier, or abstain.

## How to run

Install the packages:

```
pip install tsdb pypots benchpots scikit-learn scipy matplotlib pandas
```

Then run the steps in order from inside this folder:

```
python step1_data.py
python step2_train.py
python step3_policies.py
python step4_figures.py
python step5_cases.py
python step6_seeds.py
python check.py
```

- `common.py` has the constants, the data loading and the two decision rules
- `step1` gives the data summary
- `step2` trains logistic regression and GRU-D for 6, 12, 24 and 48 h (this is the slow one, add `--no-grud` to skip GRU-D)
- `step3` does the recalibration, the two rules and the statistics
- `step4` makes the figures
- `step5` looks at which patients get flagged, cleared or abstained on
- `step6` compares the results over seeds (first run step 2 and 3 with `SEED=1` and `SEED=2`)
- `check.py` checks the halting rule and recomputes some numbers by hand

## Folders

- `../data` has the data we use. The first run downloads PhysioNet 2012 (this can take a while) and saves the train/val/test split there as `physionet2012_seed0.npz`, so after that nothing has to be downloaded again
- `../results` has everything the code makes (probabilities, `results.json`, tables, cases). Other seeds go to `../results_seed1` etc.
- `../figures` has the figures used in the paper

`pipeline.ipynb` has all the code in one notebook if you want to go through it step by step.
