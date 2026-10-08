# TAIRA Development Guide

## Project Structure

```
traffic_ai_project/
├── data/
│   └── traffic_accidents.csv          # Chicago Traffic Crashes dataset
├── logs/                               # Application logs (auto-generated)
│   └── taira_YYYYMMDD.log
├── models/                             # Trained ML models (auto-generated)
│   └── svm_model.pkl
├── src/
│   ├── agent.py                       # Hybrid agent orchestration
│   ├── app.py                         # Streamlit web application
│   ├── config.py                      # Configuration constants
│   ├── data_loader.py                 # Data loading and preprocessing
│   ├── logger.py                      # Centralized logging system
│   ├── logic_core.py                  # Rule-based expert system
│   ├── model_engine.py                # ML model training and evaluation
│   └── temporal_analysis.py           # Temporal pattern analysis
├── .gitignore                         # Git ignore patterns
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
└── DEVELOPMENT.md                     # This file
```

## Architecture Overview

### Hybrid AI System

TAIRA implements a two-layer hybrid architecture:

1. **Layer 1: Statistical Learning (SVM)**
   - LinearSVC with CalibratedClassifierCV for probability estimates
   - Predicts injury severity from environmental conditions
   - Trained on categorical features using OneHotEncoder
   - Balanced class weights to handle imbalanced data

2. **Layer 2: Rule-Based Expert System**
   - Evidence accumulation from multiple risk factors
   - Hard safety rules for critical combinations
   - Risk score calculation (0-3 scale)
   - Risk level classification: LOW / MEDIUM / HIGH / CRITICAL
   - Human-readable explanations

### Module Responsibilities

#### **agent.py**
- Orchestrates ML model and rule-based logic
- Validates input features
- Logs predictions and critical assessments
- Returns structured prediction results

#### **model_engine.py**
- Pipeline construction (preprocessing + model)
- Model training with comprehensive metrics
- Model persistence (save/load)
- Model comparison functionality
- Single prediction interface

#### **logic_core.py**
- Risk assessment logic
- Evidence accumulation from conditions
- Hard safety rules enforcement
- Explanation generation
- Uses configurable thresholds

#### **data_loader.py**
- CSV loading with error handling
- Data cleaning and validation
- Column standardization
- Vocabulary filtering
- Data quality logging

#### **temporal_analysis.py**
- Temporal pattern analysis (hourly, daily)
- Condition-injury correlations
- Dangerous combination identification
- Severity trend analysis

#### **logger.py**
- Centralized logging configuration
- Structured log formats
- File and console handlers
- Error tracking with context
- Prediction monitoring

#### **config.py**
- Application constants
- Risk thresholds
- Model paths and versioning
- Categorical value vocabularies
- Logging configuration

## Development Workflow

### Initial Setup

```powershell
# Navigate to project
cd "c:\Users\User\Desktop\ai project\traffic_ai_project"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```powershell
# Start Streamlit app
streamlit run src/app.py

# Or use Python module syntax
python -m streamlit run src/app.py
```

### Model Training

The application automatically handles model training:
1. Checks if `models/svm_model.pkl` exists
2. If exists, loads the model (fast startup)
3. If not, trains a new model and saves it
4. To retrain, delete the model file and restart

### Logging

Logs are automatically created in `logs/` directory:
- Format: `taira_YYYYMMDD.log`
- Includes: errors, predictions, critical assessments
- Rotates daily
- Both file and console output

## Code Quality Standards

### Error Handling

All functions implement try-catch blocks:
```python
try:
    # Main logic
    result = do_something()
    logger.info("Operation successful")
    return result
except ValueError as e:
    logger.error(f"Validation error: {e}")
    raise
except Exception as e:
    log_error(logger, e, {"context": "additional_info"})
    raise
```

### Type Hints

All functions have comprehensive type hints:
```python
def predict_one(pipeline: Pipeline, features: Dict[str, object]) -> str:
    """Predict injury severity.
    
    Args:
        pipeline: Trained sklearn pipeline
        features: Dictionary of feature values
        
    Returns:
        Predicted injury severity as string
        
    Raises:
        ValueError: If required features are missing
    """
```

### Logging Best Practices

```python
# Import logger
from logger import model_logger, log_error

# Log informational messages
model_logger.info("Starting model training")

# Log warnings
model_logger.warning("Model file not found, training new model")

# Log errors with context
try:
    model.fit(X, y)
except Exception as e:
    log_error(model_logger, e, {"data_shape": X.shape})
    raise
```

### Configuration Management

All constants in `config.py`:
```python
# Don't do this:
if dangerous_count >= 4:  # Magic number!

# Do this:
from config import RISK_THRESHOLDS
if dangerous_count >= RISK_THRESHOLDS["dangerous_condition_critical"]:
```

## Testing Workflow

### Manual Testing Checklist

1. **Model Training**
   - [ ] Delete `models/svm_model.pkl`
   - [ ] Run app, verify model trains
   - [ ] Check logs for training metrics
   - [ ] Verify model file created

2. **Prediction Flow**
   - [ ] Navigate to Interactive Lab
   - [ ] Fill in all dropdowns
   - [ ] Submit prediction
   - [ ] Verify result displays
   - [ ] Check logs for prediction entry

3. **Critical Risk Assessment**
   - [ ] Set conditions: ICE + DARKNESS + PEDESTRIAN
   - [ ] Verify CRITICAL risk alert
   - [ ] Check logs for critical assessment entry

4. **Error Handling**
   - [ ] Provide invalid inputs (if possible)
   - [ ] Verify graceful error messages
   - [ ] Check error logs

5. **Data Analysis**
   - [ ] Navigate to Crash Patterns & Story
   - [ ] Verify all charts load
   - [ ] Check for any console errors

## Performance Optimization

### Model Loading
- Cached with `@st.cache_resource`
- Loads once per session
- Persisted to disk for fast restarts

### Data Loading
- Cached with `@st.cache_data`
- Loads once per session
- Efficient DataFrame operations

### Visualization
- Wrapped in cards for clean rendering
- Matplotlib figures closed after display
- Efficient color palettes

## Deployment Considerations

### Environment Variables
For production deployment, consider:
- `DATA_PATH`: Custom data file location
- `MODEL_DIR`: Custom model storage path
- `LOG_DIR`: Custom log directory
- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

### Security
- Input validation on all user inputs
- No SQL injection risk (CSV data)
- No sensitive data stored
- Sanitized error messages to users

### Scalability
- Model persistence reduces training overhead
- Efficient caching strategy
- Minimal memory footprint
- Fast SVM inference

## Common Issues and Solutions

### Issue: Model won't load
**Solution:** Delete `models/svm_model.pkl` and restart app

### Issue: No logs appearing
**Solution:** Check `logs/` directory exists and is writable

### Issue: Import errors
**Solution:** Ensure virtual environment is activated and dependencies installed

### Issue: Streamlit not found
**Solution:** Run `pip install streamlit` or reinstall requirements

### Issue: Data file not found
**Solution:** Verify `data/traffic_accidents.csv` exists

## Metrics and Monitoring

### Key Metrics
- **Accuracy**: Overall prediction correctness
- **Precision (weighted)**: Correct positive predictions per class
- **Recall (weighted)**: Coverage of actual positives per class
- **F1-Score (weighted)**: Harmonic mean of precision and recall

### Log Monitoring
Monitor logs for:
- Training time and metrics
- Prediction frequency
- CRITICAL risk assessment frequency
- Error rates and types

## Future Enhancements

Potential improvements:
1. Cross-validation for model stability
2. SMOTE for class imbalance handling
3. Feature importance analysis
4. A/B testing framework for rule changes
5. Real-time weather API integration
6. Mobile-responsive UI improvements
7. Database integration for predictions
8. User authentication system
9. Batch prediction API
10. Model retraining scheduler

## Contributing Guidelines

When adding new features:
1. Add comprehensive error handling
2. Include type hints
3. Add logging statements
4. Update docstrings
5. Test thoroughly
6. Update this documentation

## Contact

For questions about this project:
- Check logs for error details
- Review code comments
- Consult docstrings
- Contact project team

---
*Last Updated: January 2026*
