# Progress log

## 2026-11-14

- Plan written (`docs/plan.md`): split by video with a fixed seed, two models that differ only in how the validation frames are chosen, three hypotheses with test rules.
- Notebooks 01-04 and `scripts/run_on_kaggle.py` written. The code was run end to end on CPU on a small synthetic copy of the dataset (random pictures, masks with the expected values): the split, the model shapes, one training epoch, the scoring and the hypothesis tests all run. No real frame has been looked at yet.
- Next step: notebook 01 on Kaggle. It checks how the classes are stored in the masks and stops if the values are not the expected ones. Then 02 and 03.
