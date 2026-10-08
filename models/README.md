# Models Directory

This directory stores trained machine learning models for TAIRA.

## Contents

- `svm_model.pkl` - Trained Support Vector Machine model (auto-generated)

## Model Format

Models are saved using `joblib` with the following structure:
```python
{
    "pipeline": Pipeline object (preprocessor + model),
    "metrics": ModelMetrics object (accuracy, precision, recall, F1),
    "version": Model version string,
    "feature_columns": List of feature column names,
    "target_column": Target column name
}
```

## Model Lifecycle

1. **First Run**: If no model exists, TAIRA trains a new SVM model
2. **Subsequent Runs**: Model is loaded from disk for fast startup
3. **Retraining**: Delete `svm_model.pkl` to force retraining

## Model Performance

The saved model includes comprehensive metrics:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- Confusion matrix
- Classification report

## Version Control

⚠️ **Note**: Model files are excluded from git (see `.gitignore`)

This is intentional because:
- Models can be large (several MB)
- Models should be retrained on target deployment environment
- Prevents repository bloat

## Manual Model Management

To manually manage models:

```powershell
# View model details
python -c "import joblib; data = joblib.load('models/svm_model.pkl'); print(data['metrics'])"

# Delete model (forces retraining)
Remove-Item models\svm_model.pkl

# Backup model
Copy-Item models\svm_model.pkl models\svm_model_backup.pkl
```

---
*This directory is auto-created by TAIRA on first run*
