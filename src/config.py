"""config.py — Configuration Constants

TAIRA (Traffic Accident Intelligent Risk Advisor)
CEN 352 Term Project

This module defines system-wide configuration constants including:
- Dataset file path
- UI dropdown options for crash conditions
- Categorical values matching Chicago Traffic Crashes dataset
- Feature categories for ML model input

All categorical values are standardized to match the source dataset
for consistency and accurate predictions.
"""

import pathlib

# Paths
DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "traffic_accidents.csv"

# CHICAGO DATASET CONSTANTS
TRAFFIC_CONTROLS = [
	"TRAFFIC SIGNAL",
	"NO CONTROLS",
	"STOP SIGN/FLASHER",
	"UNKNOWN",
	"OTHER",
	"PEDESTRIAN CROSSING SIGN",
	"YIELD",
]

WEATHER_CONDITIONS = [
	"CLEAR",
	"RAIN",
	"SNOW",
	"CLOUDY/OVERCAST",
	"FOG/SMOKE/HAZE",
	"FREEZING RAIN/DRIZZLE",
	"SLEET/HAIL",
]

LIGHTING_CONDITIONS = [
	"DAYLIGHT",
	"DARKNESS, LIGHTED ROAD",
	"DUSK",
	"DARKNESS",
	"DAWN",
]

ROAD_SURFACE = [
	"DRY",
	"WET",
	"SNOW OR SLUSH",
	"ICE",
	"SAND, MUD, DIRT",
]

CRASH_TYPES = [
	"TURNING",
	"REAR END",
	"ANGLE",
	"FIXED OBJECT",
	"HEAD ON",
	"PEDESTRIAN",
	"PARKED MOTOR VEHICLE",
]

DAMAGES = [
	"$501 - $1,500",
	"OVER $1,500",
	"$500 OR LESS",
]

# TARGET LABELS
INJURY_TYPES = [
	"NO INDICATION OF INJURY",
	"REPORTED, NOT EVIDENT",
	"NONINCAPACITATING INJURY",
	"INCAPACITATING INJURY",
	"FATAL",
]

# RISK ASSESSMENT CONFIGURATION
# Thresholds for rule-based logic system
RISK_THRESHOLDS = {
	"dangerous_condition_critical": 4,  # Conditions needed for critical override
	"dangerous_condition_high": 3,      # Conditions needed for high override
	"dangerous_condition_medium": 2,    # Conditions needed for medium override
	"severe_injury_rate_threshold": 0.15,  # 15% rate for high-risk combinations
	"minimum_sample_size": 5,           # Minimum crashes to consider pattern valid
	"critical_score_threshold": 3.5,    # Score needed to reach CRITICAL risk
}

# RISK SCORE MAPPING
RISK_LEVELS = {
	0: "LOW",
	1: "MEDIUM",
	2: "HIGH",
	3: "CRITICAL",
}

# LOGGING CONFIGURATION
LOG_PREDICTIONS = True  # Enable prediction logging for monitoring
LOG_CRITICAL_ASSESSMENTS = True  # Log all CRITICAL risk assessments

# MODEL PERSISTENCE
MODEL_DIR = pathlib.Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "svm_model.pkl"
MODEL_VERSION = "1.0.0"
