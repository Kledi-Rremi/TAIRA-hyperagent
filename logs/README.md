# Logs Directory

This directory stores application logs for TAIRA.

## Log Files

Logs are automatically created with daily rotation:
- Format: `taira_YYYYMMDD.log`
- Example: `taira_20260113.log`

## Log Levels

TAIRA uses standard Python logging levels:
- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages (non-critical issues)
- **ERROR**: Error messages with context
- **CRITICAL**: Critical issues requiring immediate attention

## Log Contents

Each log entry includes:
- Timestamp
- Logger name (module)
- Log level
- Message

### Example Log Entries

```
2026-01-13 10:30:15 - taira.data - INFO - Loading data from data/traffic_accidents.csv
2026-01-13 10:30:16 - taira.data - INFO - Loaded 500000 records with 45 columns
2026-01-13 10:30:20 - taira.model - INFO - Starting model training: svm
2026-01-13 10:30:45 - taira.model - INFO - Model trained successfully: Accuracy=0.8234, F1=0.8156
2026-01-13 10:31:00 - taira.agent - INFO - Prediction completed: NONINCAPACITATING INJURY -> MEDIUM
2026-01-13 10:32:15 - taira.logic - WARNING - CRITICAL risk assessment | Prediction: INCAPACITATING INJURY | Score: 3.50 | Conditions: 4
```

## Monitored Events

### Predictions
- All predictions are logged (if enabled in config)
- Includes features, prediction, and risk level

### Critical Assessments
- All CRITICAL risk assessments are logged with WARNING level
- Includes conditions that triggered the assessment

### Errors
- All exceptions are logged with full traceback
- Includes contextual information for debugging

### Model Operations
- Training start/completion with metrics
- Model save/load operations
- Model validation results

## Log Management

### Viewing Logs

```powershell
# View today's log
Get-Content logs\taira_20260113.log

# View last 50 lines
Get-Content logs\taira_20260113.log -Tail 50

# Search for errors
Select-String -Path logs\taira_*.log -Pattern "ERROR"

# Search for critical assessments
Select-String -Path logs\taira_*.log -Pattern "CRITICAL"
```

### Cleaning Old Logs

Logs are not automatically deleted. To clean old logs:

```powershell
# Delete logs older than 30 days
Get-ChildItem logs\*.log | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-30)} | Remove-Item
```

## Log Configuration

Log settings are defined in `src/config.py`:
- `LOG_PREDICTIONS`: Enable/disable prediction logging
- `LOG_CRITICAL_ASSESSMENTS`: Enable/disable critical risk logging

## Version Control

⚠️ **Note**: Log files are excluded from git (see `.gitignore`)

Logs contain:
- Runtime data
- User interactions
- Potentially large files

## Troubleshooting

### No logs appearing
1. Verify this directory exists and is writable
2. Check console output for logging errors
3. Ensure logger module is imported correctly

### Logs too large
1. Implement log rotation (future enhancement)
2. Manually delete old logs
3. Reduce logging verbosity in config

---
*This directory is auto-created by TAIRA on first run*
