# Surgical scene segmentation: does a frame split hide the real error, and can the model tell?

A small study on CholecSeg8k, 8,080 annotated frames from 17 laparoscopic gallbladder operations (Cholec80). I train a U-Net that gives every pixel one of 13 classes (organs, tissue, instruments) and ask two questions:

- Frames next to each other in a video are almost the same picture. If the validation frames are drawn at random, does the score promise more than the model gives on new operations?
- Can the model's own uncertainty point to the frames where it fails?

Status: closed. The plan, with the hypotheses and the test rules, was written before any training (`docs/plan.md`), with one follow-up written after the validation run and before the test; the full log is in `docs/progress.md`.

## Results

Test: 3 operations (video25, video48, video55, 800 frames) that played no part in training or in any choice. Mean Dice over the classes in the ground truth, without the black border.

| | validation | test operations |
|---|---|---|
| model A, validated on random frames of its own training operations | 0.952 | 0.877 |
| model B, validated on 3 other operations | 0.798 | 0.825 |

- **A random frame split promises too much (H2: direction as expected, below the bar).** Model A's validation promised 0.952; on the new operations it reached 0.877. The drop of 0.075 is below the 0.10 I set in advance, so H2 does not hold as written. The planned comparison averages over different classes, because the validation frames hold rare classes that the test operations do not. Over the 7 classes present in both (checked after the test, not planned), the drop is 0.097. Model B's estimate from 3 other operations was off by 0.027, in the other direction.
- **The random split did not make the model worse, only its report.** Model A is the better model on the new operations (0.877 against 0.825); it was trained on 14 operations, model B on 11.
- **One score depends on which operations are held out.** Model B's Dice for the L-hook was 0.39 on its validation operations and 0.89 on the test operations; for the abdominal wall 0.85 and 0.93. With 17 operations, a score should be reported per operation, not only as one mean.
- **Rare structures cannot be tested here (H1).** None of the four small classes (blood, cystic duct, hepatic vein, liver ligament) appears in the test operations, and none in model B's validation operations. Among the classes that do appear, the gastrointestinal tract is the weakest (model B 0.48, model A 0.72).
- **Can the model warn us? (H3 and H3b: far above chance, below the bar).** The planned measure, the entropy over the predicted foreground, flags 35 of the 80 worst test frames, with 8 expected by chance (p < 1e-16, rank correlation -0.63); the whole-image entropy flags 36. The bar was 40. All 80 flagged frames come from video55, which also holds 67 of the 80 worst frames, so the signal mainly says which operation is hard for the model. Inside each operation the rank correlation is -0.57, -0.34 and -0.39.
- **The worst frames** are from one dark clip of video55 with strong glare from an instrument: the model takes the gallbladder for abdominal wall and misses the gastrointestinal tract. They are bad also when only the classes in the truth are counted (0.61 against 0.82 for the other frames), so the low scores do not come from small invented regions alone.

What I take from it: a validation on random frames of the same operations reported more than the model gave on new ones, and with 17 operations every single score also moves with the operations that are held out. The model's uncertainty is a usable warning for a whole operation, but not yet a reliable way to pick single frames for a second look.

## Data

CholecSeg8k: Hong, W.-Y., Kao, C.-L., Kuo, Y.-H., Wang, J.-R., Chang, W.-L., Shih, C.-S. (2020). CholecSeg8k: A Semantic Segmentation Dataset for Laparoscopic Cholecystectomy Based on Cholec80. arXiv:2012.12453. Licence CC BY-NC-SA 4.0. Kaggle: https://www.kaggle.com/datasets/newslab/cholecseg8k

Cholec80: Twinanda, A. P. et al. (2017). EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos. IEEE Transactions on Medical Imaging 36(1), 86-97.

The frames are not part of this repository. The notebooks read them from the Kaggle dataset. The figures in `results/` that show frames contain images from CholecSeg8k and are under its licence (CC BY-NC-SA 4.0); the code is under the MIT licence.

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
results/                        the result files of the runs (tables, figures, per-frame scores), without the model weights
docs/plan.md                    question, hypotheses, test rules (written before training)
docs/progress.md                log
```
