"""data_loader.py — Data Loading and Preprocessing Module

TAIRA (Traffic Accident Intelligent Risk Advisor)
CEN 352 Term Project

This module handles:
- Loading the Chicago Traffic Crashes dataset
- Data cleaning and standardization
- Feature preparation for ML pipeline (Technique A)
- Data validation and quality checks

Dataset: Chicago Traffic Crashes (City of Chicago Data Portal)
File location: data/traffic_accidents.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd

from config import (
	CRASH_TYPES,
	DATA_PATH,
	INJURY_TYPES,
	LIGHTING_CONDITIONS,
	ROAD_SURFACE,
	TRAFFIC_CONTROLS,
	WEATHER_CONDITIONS,
)
from logger import data_logger, log_error


# Canonical schema for the PRE-CRASH tool (categorical-only UI)
FEATURE_COLUMNS: List[str] = [
	"TRAFFIC_CONTROL_DEVICE",
	"WEATHER_CONDITION",
	"LIGHTING_CONDITION",
	"ROADWAY_SURFACE_COND",
	"TRAFFICWAY_TYPE",
	"ALIGNMENT",
	"ROAD_DEFECT",
	"INTERSECTION_RELATED_I",
	"PRIM_CONTRIBUTORY_CAUSE",
	"FIRST_CRASH_TYPE",
]

TARGET_COLUMN = "MOST_SEVERE_INJURY"


@dataclass(frozen=True)
class DatasetSplit:
	X: pd.DataFrame
	y: pd.Series


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Upper-case column names so they match our canonical schema."""
	df = df.copy()
	df.columns = [str(c).strip().upper() for c in df.columns]
	return df


def load_data(csv_path=DATA_PATH) -> pd.DataFrame:
	"""Load the dataset from disk.
	
	Args:
		csv_path: Path to CSV file
		
	Returns:
		DataFrame with standardized columns
		
	Raises:
		FileNotFoundError: If CSV file doesn't exist
		Exception: If loading fails
	"""
	try:
		if not csv_path.exists():
			raise FileNotFoundError(f"Data file not found: {csv_path}")
		
		data_logger.info(f"Loading data from {csv_path}")
		df = pd.read_csv(csv_path)
		data_logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
		
		return _standardize_columns(df)
		
	except Exception as e:
		log_error(data_logger, e, {"csv_path": str(csv_path)})
		raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
	"""Clean raw dataset.

	Steps (per rubric):
	- Ensure required columns exist
	- Drop rows with missing values in required feature/target columns
	- Normalize text fields (strip)
	- Filter out-of-vocabulary values
	
	Args:
		df: Raw dataframe
		
	Returns:
		Cleaned dataframe
		
	Raises:
		ValueError: If required columns are missing
	"""
	try:
		original_size = len(df)
		data_logger.info(f"Starting data cleaning: {original_size} records")
		
		df = _standardize_columns(df)

		required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
		missing = sorted(required - set(df.columns))
		if missing:
			raise ValueError(f"Missing required columns: {missing}")

		df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

		# Normalize string-like categorical fields (preserve NA)
		categorical_cols: Iterable[str] = FEATURE_COLUMNS
		for col in categorical_cols:
			df[col] = df[col].astype("string").str.strip().str.upper()

		# Target normalization
		df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("string").str.strip().str.upper()

		# Ensure values match UI dropdown options (where we have explicit vocab)
		vocab_filters = {
			"TRAFFIC_CONTROL_DEVICE": set(TRAFFIC_CONTROLS),
			"WEATHER_CONDITION": set(WEATHER_CONDITIONS),
			"LIGHTING_CONDITION": set(LIGHTING_CONDITIONS),
			"ROADWAY_SURFACE_COND": set(ROAD_SURFACE),
			"FIRST_CRASH_TYPE": set(CRASH_TYPES),
			TARGET_COLUMN: set(INJURY_TYPES),
		}
		
		for col, allowed in vocab_filters.items():
			before = len(df)
			df = df[df[col].isin(allowed) | df[col].isna()]
			after = len(df)
			if before != after:
				data_logger.debug(f"Filtered {before - after} records with invalid {col} values")

		# Drop rows with any missing values
		before_dropna = len(df)
		df = df.dropna(axis=0, how="any")
		after_dropna = len(df)
		
		data_logger.info(
			f"Data cleaning complete: {after_dropna} records retained "
			f"({original_size - after_dropna} removed, "
			f"{after_dropna/original_size*100:.1f}% retention rate)"
		)
		
		return df
		
	except Exception as e:
		log_error(data_logger, e, {"df_shape": df.shape})
		raise


def get_features_and_target(df: pd.DataFrame) -> DatasetSplit:
	"""Return X/y split for modeling."""
	df = clean_data(df)
	X = df[FEATURE_COLUMNS].copy()
	y = df[TARGET_COLUMN].copy()
	return DatasetSplit(X=X, y=y)


def load_clean_split(csv_path=DATA_PATH) -> DatasetSplit:
	"""Convenience: load -> clean -> return X/y."""
	df = load_data(csv_path)
	return get_features_and_target(df)
