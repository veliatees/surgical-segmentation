# Surgical scene segmentation: does a frame split hide the real error, and can the model tell?

A small study on CholecSeg8k, 8,080 annotated frames from 17 laparoscopic gallbladder operations (Cholec80). I train a U-Net that gives every pixel one of 13 classes (organs, tissue, instruments) and ask two questions:

- Frames next to each other in a video are almost the same picture. If the validation frames are drawn at random, does the score promise more than the model gives on new operations?
- Can the model's own uncertainty point to the frames where it fails?

Status: in progress. The plan, with the hypotheses and the test rules, was written before any training (`docs/plan.md`); the log is in `docs/progress.md`.

## Results

Not yet.

## Data

CholecSeg8k: Hong, W.-Y., Kao, C.-L., Kuo, Y.-H., Wang, J.-R., Chang, W.-L., Shih, C.-S. (2020). CholecSeg8k: A Semantic Segmentation Dataset for Laparoscopic Cholecystectomy Based on Cholec80. arXiv:2012.12453. Licence CC BY-NC-SA 4.0. Kaggle: https://www.kaggle.com/datasets/newslab/cholecseg8k

Cholec80: Twinanda, A. P. et al. (2017). EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos. IEEE Transactions on Medical Imaging 36(1), 86-97.

The frames are not part of this repository. The notebooks read them from the Kaggle dataset.

## Run

From your own computer with the Kaggle API (setup at the top of the script):

```
python scripts/run_on_kaggle.py 01 02 03     # data check, training of both models, validation
python scripts/run_on_kaggle.py 04           # the test, once, after looking at 03
```

Each notebook runs as a private Kaggle notebook with the dataset attached and gets the output of the notebook before it. The results are downloaded into `runs/`.

## Layout

```
notebooks/01_data_check.ipynb   frames per video, how the classes are stored, pixels per class, split by video
notebooks/02_train.ipynb        U-Net with a ResNet-34 encoder; model A (random validation frames), model B (validation videos)
notebooks/03_validation.ipynb   Dice per class, confusion, first look at the warning signal
notebooks/04_test_set.ipynb     the 3 test videos, H1-H3 with the rules from docs/plan.md
scripts/run_on_kaggle.py        runs the notebooks on Kaggle one after the other
docs/plan.md                    question, hypotheses, test rules (written before training)
docs/progress.md                log
```
