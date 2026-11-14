# Plan

Written on 14 November 2026, before any data was loaded or any model trained. The test videos are chosen by the rule below in notebook 01; their names are added at the end of this file once, without any other change.

## Question

In surgery, a segmentation that looks right but is wrong costs the most. Two things can hide such errors: an evaluation that is too easy, and a model that gives no sign when it is unsure.

1. A video has 25 frames per second, so frames next to each other are almost the same picture. If validation frames are drawn at random, the model has seen almost the same frames in training. How much better does such a validation score look than the score on new videos?
2. Can the model point to the frames where it fails, so that a person only checks those?

## Data

CholecSeg8k (Hong et al., 2020, arXiv:2012.12453): 8,080 frames (854 x 480) from 17 laparoscopic cholecystectomy videos of Cholec80, in clips of 80 frames, 13 classes: black background, abdominal wall, liver, gastrointestinal tract, fat, grasper, connective tissue, blood, cystic duct, L-hook electrocautery, gallbladder, hepatic vein, liver ligament. Licence CC BY-NC-SA 4.0, Kaggle dataset `newslab/cholecseg8k`.

The paper notes strong class imbalance: liver and abdominal wall cover about 30 % of the pixels each, grasper and L-hook less than 1 % each.

## Steps

1. **Data check** (`01_data_check.ipynb`). Frames per video, how the classes are stored in the masks (checked against the colour masks), pixels per class, the split.
2. **Training** (`02_train.ipynb`). A U-Net with a ResNet-34 encoder trained on ImageNet, 13 classes, frames resized to 256 x 448, cross-entropy plus Dice loss, 15 epochs, the epoch with the best validation mean Dice is kept. Two models with the same settings:
   - **model A**: validation = 15 % of the frames of the 14 non-test videos, drawn at random (`default_rng(20261115)`);
   - **model B**: 11 videos for training, 3 other videos for validation.
3. **Validation** (`03_validation.ipynb`). Dice per class, confusion between classes, a first look at the warning signal. Exploratory.
4. **Test** (`04_test_set.ipynb`). Both models once on the 3 test videos, with the rules below.

## Split

The 17 video names are sorted and shuffled with `numpy.random.default_rng(20261114)`. The first 3 are the test videos, the next 3 model B's validation videos, the other 11 model B's training videos. The test videos are not loaded in notebooks 02 and 03.

## Measures

- **Dice of a class** over a set of frames: 2 x overlap / (true pixels + predicted pixels), with the pixels of all frames added up. Only classes that occur in the ground truth of the set count.
- **Mean Dice**: the average Dice of classes 1-12. The black background (class 0) is the dark border of the endoscope picture and is left out.
- **Frame Dice**: the average Dice of the classes 1-12 that occur in that frame's ground truth or prediction.
- **Warning signal**: the mean pixel entropy of the model output over the pixels it does not call black background.
- Mask pixels whose value belongs to no class are left out of training and scoring.

## Hypotheses and test rules (written before training)

**H1: where it fails (model B).** Classes that cover less than 1 % of the pixels of the 14 non-test videos (small) score much lower than classes that cover at least 5 % (large). Both groups are fixed in notebook 01 from the non-test videos.
H1 holds if the median test Dice of the small classes is at least 0.20 below the median test Dice of the large classes.

**H2: a random frame split looks better than it is (model A).** H2 holds if model A's validation mean Dice (best epoch, as logged in notebook 02) is at least 0.10 higher than its mean Dice on the test videos. Model B's numbers are reported next to it.

**H3: can the model warn us? (model B).** All test frames are ranked by the warning signal. H3 holds if the 10 % of frames with the highest value contain at least half of the 10 % of frames with the lowest frame Dice (10 % is expected by chance). The one-sided hypergeometric p-value and the rank correlation are reported too.

All three are reported whether they hold or not.

## Known limits

- 17 videos from one hospital and one type of operation; 3 test videos are few.
- Frames of the same video are alike, so the worst test frames may come from one video. A check inside each video is reported after the test, marked as not planned.
- Frames are scored at 256 x 448, not at the full size.
- Some classes are broad (the gastrointestinal tract groups several organs), and the paper says that dark areas at the edge were hard to label.

## Time limit

- Saturday 14 November: notebooks 01-03 on Kaggle.
- Sunday 15 - Monday 16 November: look at the validation results, run notebook 04 once.
- Tuesday 17 November: README with the results.
- Wednesday 18 November: CV and cover letter updated, LMU application sent.

If there is no clean result by Tuesday 22:00, the LMU application goes out on Wednesday with the current documents, and this study continues afterwards.

## Test videos (filled in once, after notebook 01)

Chosen by the rule above in the first complete run of notebook 01 (14 November 2026): video25, video48, video55 (800 frames). Model B's validation videos: video20, video26, video35 (720 frames).

## Follow-up after the validation run (written on 14 November 2026, before notebook 04 was run)

What notebooks 02 and 03 showed on the validation frames:

- Model A, validated on random frames from its own training videos: mean Dice 0.952 (best epoch 15). Model B, validated on 3 other videos: 0.798 (best epoch 6). The same kind of model, trained the same way, looks 0.15 better when the validation frames come from the training videos. This is not yet H2: H2 compares model A's validation score with its score on the test videos.
- The largest gap is the L-hook: Dice 0.97 for model A on random frames, 0.39 for model B on new videos, where it is taken for liver (31 % of its pixels) and grasper (21 %).
- Model B's validation videos contain none of the small classes (blood, cystic duct, hepatic vein, liver ligament), so H1 could not be looked at.
- Warning signal on model B's validation frames: the planned measure (entropy over the predicted foreground) catches 19 of the 72 worst frames (7.2 expected by chance, p = 2e-5); the whole-image entropy catches 28 of 72 (p = 2e-12). Neither reaches the bar of half (36). All 72 worst frames are from one video (video35).

Fixed now, before the test:

- H1, H2 and H3 stay as they are.
- **H3b.** The whole-image mean entropy, with the same rule as H3: it holds if the 10 % of test frames with the highest value contain at least half of the 10 % of frames with the lowest frame Dice. p-value and rank correlation are reported. It is the second measure, chosen after the validation run, so it is reported as such, whether it holds or not.
- If none of the small classes occurs in the test videos, H1 cannot be tested; it is reported that way, not as "does not hold".
- Not planned, reported as such: the frame Dice counts every class that the model predicts but that is not in the truth as 0, even a few pixels. In the worst validation frames the main structures were found and small regions of other classes were added. Notebook 04 therefore also reports a frame Dice over the classes in the truth only and the number of invented classes per frame.
