# Progress log

## 2026-11-14

- Plan written (`docs/plan.md`): split by video with a fixed seed, two models that differ only in how the validation frames are chosen, three hypotheses with test rules.
- Notebooks 01-04 and `scripts/run_on_kaggle.py` written. The code was run end to end on CPU on a small synthetic copy of the dataset (random pictures, masks with the expected values): the split, the model shapes, one training epoch, the scoring and the hypothesis tests all run. No real frame has been looked at yet.
- Next step: notebook 01 on Kaggle. It checks how the classes are stored in the masks and stops if the values are not the expected ones. Then 02 and 03.
- First run of notebook 01 on Kaggle (commit 4bf49b4). 8,080 frames, 17 videos, 101 clips; between 2 clips (video18, video20) and 16 clips (video01) per video. On 300 random frames the 13 expected mask values each sit on exactly one colour of the colour mask, the colour of their class (100 %), so the class mapping is confirmed. The notebook stopped, as it should for any unknown value: value 255 covers 0.76 % of the pixels and is white in the colour mask, so it is not labelled (value 0, also white, covers almost nothing). As the plan says, pixels with no class are left out of training and scoring; the rule in 01 now treats values drawn white in the colour mask that way. Any other unknown value still stops the notebook. Nothing else changed.
- `scripts/run_on_kaggle.py` now downloads the log when a notebook fails and prints its last lines, so the error is visible in the terminal.
