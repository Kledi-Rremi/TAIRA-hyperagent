# TAIRA Quick Reference Guide

## 🚀 Quick Start

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install/update dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run src/app.py
```

## 📂 File Purpose Reference

| File | Purpose |
|------|---------|
| `src/agent.py` | Orchestrates ML + rules, validates inputs |
| `src/app.py` | Streamlit web interface (1381 lines) |
| `src/config.py` | All constants and thresholds |
| `src/data_loader.py` | CSV loading, cleaning, validation |
| `src/logger.py` | Centralized logging system |
| `src/logic_core.py` | Rule-based risk assessment |
| `src/model_engine.py` | ML training, evaluation, persistence |
| `src/temporal_analysis.py` | Temporal pattern analysis |

## 🔧 Common Tasks

### Force Model Retraining
```powershell
Remove-Item models\svm_model.pkl
streamlit run src/app.py
```

### View Logs
```powershell
# Today's log
Get-Content logs\taira_$(Get-Date -Format 'yyyyMMdd').log

# Search for errors
Select-String -Path logs\*.log -Pattern "ERROR"

# Search for critical assessments
Select-String -Path logs\*.log -Pattern "CRITICAL"
```

### Check Model Info
```python
import joblib
data = joblib.load('models/svm_model.pkl')
print(f"Accuracy: {data['metrics'].accuracy:.4f}")
print(f"Version: {data['version']}")
```

### Clear Cache
```powershell
# Clear Python cache
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force

# Clear Streamlit cache (in app UI)
# Press 'c' then 'Enter' in terminal, or use menu
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | Activate venv: `.\.venv\Scripts\Activate.ps1` |
| Model won't load | Delete `models\svm_model.pkl` |
| App won't start | Check `data/traffic_accidents.csv` exists |
| No logs | Check `logs/` directory is writable |
| Slow startup | Model training (first run only) |
| Port already in use | Kill process on 8501 or use `--server.port` |

## 📊 Key Configuration

**File:** `src/config.py`

```python
# Risk Thresholds (tune these)
RISK_THRESHOLDS = {
    "dangerous_condition_critical": 4,
    "dangerous_condition_high": 3,
    "dangerous_condition_medium": 2,
    "severe_injury_rate_threshold": 0.15,
    "minimum_sample_size": 5,
    "critical_score_threshold": 3.5,
}

# Logging (enable/disable)
LOG_PREDICTIONS = True
LOG_CRITICAL_ASSESSMENTS = True

# Model Settings
MODEL_VERSION = "1.0.0"
```

## 🎯 Key Metrics

### Model Performance
- **Accuracy**: Overall correctness (aim for >80%)
- **Precision**: Correct positives per class
- **Recall**: Coverage of actual positives
- **F1-Score**: Harmonic mean of precision/recall

### Risk Levels
- **LOW** (0): Minimal risk
- **MEDIUM** (1): Moderate risk
- **HIGH** (2): Elevated risk
- **CRITICAL** (3): Immediate attention required

## 🔍 Code Patterns

### Error Handling Pattern
```python
try:
    result = do_something()
    logger.info("Success")
    return result
except ValueError as e:
    logger.error(f"Validation error: {e}")
    raise
except Exception as e:
    log_error(logger, e, {"context": data})
    raise
```

### Logging Pattern
```python
from logger import model_logger, log_error

model_logger.info("Starting operation")
model_logger.warning("Potential issue")
model_logger.error("Error occurred")
```

### Config Usage Pattern
```python
from config import RISK_THRESHOLDS

threshold = RISK_THRESHOLDS["dangerous_condition_critical"]
if count >= threshold:
    # Do something
```

## 📝 Important Notes

1. **Model Persistence**: First run trains model (~30s), subsequent runs load from disk (~2s)
2. **Logging**: All logs in `logs/taira_YYYYMMDD.log`
3. **Caching**: Streamlit caches data and models for performance
4. **Type Hints**: All functions have comprehensive type hints
5. **Error Handling**: Try-catch blocks throughout

## 🎓 For Grading/Review

Key features to highlight:
1. ✅ **Hybrid AI**: SVM (statistical) + Rules (logic)
2. ✅ **Production Code**: Error handling, logging, validation
3. ✅ **Model Persistence**: Fast restarts with saved models
4. ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1
5. ✅ **Clean Architecture**: Separated concerns, config management
6. ✅ **Documentation**: README, docstrings, type hints

## 🔗 Quick Links

- Main docs: `README.md`
- Development guide: `DEVELOPMENT.md`
- Improvements summary: `IMPROVEMENTS.md`
- Model info: `models/README.md`
- Log info: `logs/README.md`

---
*For detailed information, see DEVELOPMENT.md*
