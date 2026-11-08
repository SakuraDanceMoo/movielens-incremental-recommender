# Hybrid Rating Prediction

This experiment trains an explicit-rating Spark ALS model on the supplied hybrid dataset, which combines MovieLens 32M-derived and synthetic ratings. It writes one predicted rating for every row in `test_pairs.csv` while preserving the input row order.

`test_pairs.csv` contains `userId`, `movieId`, and `timestamp`, but no target rating. The reported RMSE is therefore computed on a temporal holdout from `ratings.csv`, not on the unlabeled test pairs. The holdout is each eligible user's latest rated movie, with at least six ratings required so the fitted model retains at least five earlier ratings for that user. The code also compares ranks 16 and 32 on this holdout, then refits the selected rank on all supplied ratings.

## Run

Run from the project root with the shared environment and the supplied data directory:

```powershell
$env:JAVA_HOME = 'D:\AgentProject2026\.venv\jdk11'
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
$dataDir = 'C:\path\to\训练数据集'
..\.venv\Scripts\python.exe -X utf8 07_rating_prediction/scripts/rating_prediction.py `
  --data-dir $dataDir
```

The prediction table is written beside the input files as `test_pairs_predictions.csv`. Validation metrics and run metadata are written under `07_rating_prediction/logs/`. Source data is read in place and is not copied into the repository.

Predictions are clipped to the supplied rating scale `[0.5, 5.0]`. Missing-factor predictions use the training global mean; the run report records how many validation and test rows needed that fallback. The hidden test labels are not present in `test_pairs.csv`, so a true test-set RMSE cannot be computed from the supplied files.
