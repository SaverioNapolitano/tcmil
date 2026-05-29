## Baseline Metrics (Ensembled per fold, threshold 0.5 implicitly or what was saved if loss-based)
Note: kfold_results.json probability is already the predicted probability.

### Fold 1
- **Threshold = 0.45**: F1 = 0.571, Precision = 0.471, Recall = 0.727, ROC-AUC = 0.836
  - Fold 2 specific: False Positives = 9, False Negatives = 3
- **Threshold = 0.50**: F1 = 0.667, Precision = 0.615, Recall = 0.727, ROC-AUC = 0.836
  - Fold 2 specific: False Positives = 5, False Negatives = 3
- **Threshold = 0.55**: F1 = 0.800, Precision = 0.889, Recall = 0.727, ROC-AUC = 0.836
  - Fold 2 specific: False Positives = 1, False Negatives = 3

### Fold 2
- **Threshold = 0.45**: F1 = 0.640, Precision = 0.571, Recall = 0.727, ROC-AUC = 0.731
- **Threshold = 0.50**: F1 = 0.476, Precision = 0.500, Recall = 0.455, ROC-AUC = 0.731
- **Threshold = 0.55**: F1 = 0.333, Precision = 0.429, Recall = 0.273, ROC-AUC = 0.731

### Fold 3
- **Threshold = 0.45**: F1 = 0.733, Precision = 0.579, Recall = 1.000, ROC-AUC = 0.951
- **Threshold = 0.50**: F1 = 0.786, Precision = 0.647, Recall = 1.000, ROC-AUC = 0.951
- **Threshold = 0.55**: F1 = 0.762, Precision = 0.800, Recall = 0.727, ROC-AUC = 0.951

### Fold 4
- **Threshold = 0.45**: F1 = 0.645, Precision = 0.500, Recall = 0.909, ROC-AUC = 0.822
- **Threshold = 0.50**: F1 = 0.600, Precision = 0.474, Recall = 0.818, ROC-AUC = 0.822
- **Threshold = 0.55**: F1 = 0.609, Precision = 0.583, Recall = 0.636, ROC-AUC = 0.822

### Fold 5
- **Threshold = 0.45**: F1 = 0.643, Precision = 0.562, Recall = 0.750, ROC-AUC = 0.863
- **Threshold = 0.50**: F1 = 0.696, Precision = 0.727, Recall = 0.667, ROC-AUC = 0.863
- **Threshold = 0.55**: F1 = 0.571, Precision = 0.667, Recall = 0.500, ROC-AUC = 0.863

