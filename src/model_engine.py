"""model_engine.py — Machine Learning Model Engine

TAIRA (Traffic Accident Intelligent Risk Advisor)
CEN 352 Term Project — Technique A: Statistical Learning

This module implements the ML pipeline for injury severity prediction:
- Feature preprocessing (OneHotEncoder for categorical variables)
- Model training (SVM, Random Forest)
- Model evaluation (Accuracy, F1-Score)
- Model comparison and selection

Primary Model: Support Vector Machine (LinearSVC)
- Superior handling of class imbalance with balanced class weights
- Calibrated for probability estimates
- Fast inference for real-time risk assessment
- Better performance than Random Forest baseline

Prediction Target: most_severe_injury
- NO INDICATION OF INJURY
- REPORTED, NOT EVIDENT
- NONINCAPACITATING INJURY
- INCAPACITATING INJURY
- FATAL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from config import MODEL_PATH, MODEL_DIR, MODEL_VERSION
from data_loader import FEATURE_COLUMNS, TARGET_COLUMN
from logger import model_logger, log_error

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass(frozen=True)
class ModelMetrics:
	accuracy: float
	f1_weighted: float
	precision_weighted: float
	recall_weighted: float
	confusion_matrix: Optional[List[List[int]]] = None
	classification_report: Optional[str] = None


def build_pipeline(*, random_state: int = 42, model_type: str = "svm") -> Pipeline:
	"""Create the ML pipeline for accident prediction.
	
	Args:
		model_type: 'svm' (SVM - current winner) or 'randomforest' (RF - baseline)
	"""
	categorical_features: List[str] = FEATURE_COLUMNS

	preprocessor = ColumnTransformer(
		transformers=[
			(
				"cat",
				OneHotEncoder(handle_unknown="ignore"),
				categorical_features,
			),
		],
		remainder="drop",
	)

	# Select model based on type
	if model_type == "svm":
		# Support Vector Machine - Linear SVC for efficiency
		base_svm = LinearSVC(
			C=1.0,
			class_weight="balanced",
			random_state=random_state,
			max_iter=2000,
			dual="auto",
		)
		# Wrap to get probability estimates for accident likelihood
		model = CalibratedClassifierCV(base_svm, cv=3)
	elif model_type == "randomforest":
		# Random Forest - Ensemble baseline
		model = RandomForestClassifier(
			n_estimators=250,
			random_state=random_state,
			class_weight="balanced",
			n_jobs=-1,
		)
	else:
		raise ValueError(f"Model must be 'svm' or 'randomforest', got: {model_type}")

	return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train_model(
	df: pd.DataFrame,
	*,
	test_size: float = 0.2,
	random_state: int = 42,
	model_type: str = "svm",
) -> Tuple[Pipeline, ModelMetrics]:
	"""Train the pipeline and evaluate accident prediction model.

	Expects df to contain FEATURE_COLUMNS and TARGET_COLUMN.
	Computes comprehensive evaluation metrics including precision, recall, and confusion matrix.
	
	Args:
		df: Training dataframe
		test_size: Test set proportion (default 0.2)
		random_state: Random seed for reproducibility
		model_type: 'svm' (default) or 'randomforest'
		
	Returns:
		Tuple of (trained_pipeline, metrics)
		
	Raises:
		ValueError: If model_type is invalid or required columns are missing
	"""
	try:
		if model_type not in ["svm", "randomforest"]:
			raise ValueError(f"Model must be 'svm' or 'randomforest', got: {model_type}")
			
		missing = sorted(set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns))
		if missing:
			raise ValueError(f"Training dataframe missing required columns: {missing}")

		model_logger.info(f"Starting model training: {model_type}")
		
		X = df[FEATURE_COLUMNS].copy()
		y = df[TARGET_COLUMN].copy()

		X_train, X_test, y_train, y_test = train_test_split(
			X,
			y,
			test_size=test_size,
			random_state=random_state,
			stratify=y,
		)

		pipeline = build_pipeline(random_state=random_state, model_type=model_type)
		model_logger.info(f"Training {model_type} model on {len(X_train)} samples...")
		pipeline.fit(X_train, y_train)

		model_logger.info("Generating predictions on test set...")
		y_pred = pipeline.predict(X_test)
		
		# Calculate comprehensive metrics
		accuracy = float(accuracy_score(y_test, y_pred))
		f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
		precision_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
		recall_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
		
		# Generate confusion matrix
		cm = confusion_matrix(y_test, y_pred)
		cm_list = cm.tolist()
		
		# Generate classification report
		report = classification_report(y_test, y_pred, zero_division=0)

		# Print metrics for report
		print(f"\n{'='*60}")
		print(f"Model: {model_type.upper()}")
		print(f"{'='*60}")
		print(f"Accuracy: {accuracy:.4f}")
		print(f"Precision (weighted): {precision_weighted:.4f}")
		print(f"Recall (weighted): {recall_weighted:.4f}")
		print(f"F1-Score (weighted): {f1_weighted:.4f}")
		print(f"{'='*60}")
		print("\nClassification Report:")
		print(report)
		print(f"{'='*60}\n")
		
		model_logger.info(
			f"Model trained successfully: Accuracy={accuracy:.4f}, F1={f1_weighted:.4f}"
		)

		return pipeline, ModelMetrics(
			accuracy=accuracy,
			f1_weighted=f1_weighted,
			precision_weighted=precision_weighted,
			recall_weighted=recall_weighted,
			confusion_matrix=cm_list,
			classification_report=report,
		)
		
	except Exception as e:
		log_error(model_logger, e, {"model_type": model_type, "df_shape": df.shape})
		raise


def predict_one(pipeline: Pipeline, features: Dict[str, object]) -> str:
	"""Predict MOST_SEVERE_INJURY for a single input.
	
	Args:
		pipeline: Trained sklearn pipeline
		features: Dictionary of feature values
		
	Returns:
		Predicted injury severity as string
		
	Raises:
		ValueError: If required features are missing
	"""
	try:
		# Validate that all required features are present
		missing_features = [col for col in FEATURE_COLUMNS if col not in features]
		if missing_features:
			raise ValueError(f"Missing required features: {missing_features}")
		
		X_one = pd.DataFrame([{col: features.get(col) for col in FEATURE_COLUMNS}])
		prediction = str(pipeline.predict(X_one)[0])
		model_logger.debug(f"Prediction made: {prediction}")
		return prediction
		
	except Exception as e:
		log_error(model_logger, e, {"features": features})
		raise



def save_model(pipeline: Pipeline, metrics: ModelMetrics, model_path: Optional[str] = None) -> None:
	"""Save trained model to disk with metadata.
	
	Args:
		pipeline: Trained model pipeline
		metrics: Model evaluation metrics
		model_path: Optional custom save path (defaults to config.MODEL_PATH)
	"""
	try:
		path = model_path or MODEL_PATH
		MODEL_DIR.mkdir(parents=True, exist_ok=True)
		
		model_data = {
			"pipeline": pipeline,
			"metrics": metrics,
			"version": MODEL_VERSION,
			"feature_columns": FEATURE_COLUMNS,
			"target_column": TARGET_COLUMN,
		}
		
		joblib.dump(model_data, path)
		model_logger.info(f"Model saved to {path} (version {MODEL_VERSION})")
		
	except Exception as e:
		log_error(model_logger, e, {"model_path": path})
		raise


def load_model(model_path: Optional[str] = None) -> Tuple[Pipeline, ModelMetrics]:
	"""Load trained model from disk.
	
	Args:
		model_path: Optional custom load path (defaults to config.MODEL_PATH)
		
	Returns:
		Tuple of (pipeline, metrics)
		
	Raises:
		FileNotFoundError: If model file doesn't exist
	"""
	try:
		path = model_path or MODEL_PATH
		
		if not path.exists():
			raise FileNotFoundError(f"Model file not found: {path}")
		
		model_data = joblib.load(path)
		pipeline = model_data["pipeline"]
		metrics = model_data["metrics"]
		version = model_data.get("version", "unknown")
		
		model_logger.info(f"Model loaded from {path} (version {version})")
		return pipeline, metrics
		
	except Exception as e:
		log_error(model_logger, e, {"model_path": path})
		raise


def model_exists(model_path: Optional[str] = None) -> bool:
	"""Check if a trained model exists on disk.
	
	Args:
		model_path: Optional custom path to check
		
	Returns:
		True if model file exists, False otherwise
	"""
	path = model_path or MODEL_PATH
	return path.exists()


def compare_models(
	df: pd.DataFrame,
	*,
	test_size: float = 0.2,
	random_state: int = 42,
) -> Dict[str, ModelMetrics]:
	"""Compare SVM vs Random Forest for accident prediction.
	
	Tests both models on:
	- When accidents mostly happen
	- Overall damages
	- Injury severity patterns
	
	Returns:
		Dict mapping model_type -> ModelMetrics
	"""
	models_to_test = ["svm", "randomforest"]
	results = {}
	
	print("\n" + "="*60)
	print("ACCIDENT PREDICTION MODEL COMPARISON")
	print("Testing: SVM vs Random Forest")
	print("="*60)
	
	for model_type in models_to_test:
		try:
			_, metrics = train_model(
				df,
				test_size=test_size,
				random_state=random_state,
				model_type=model_type,
			)
			results[model_type] = metrics
		except Exception as e:
			print(f"Error training {model_type}: {e}")
			continue
	
	print("\n" + "="*60)
	print("MODEL COMPARISON SUMMARY")
	print("="*60)
	for model_type, metrics in sorted(results.items(), key=lambda x: x[1].accuracy, reverse=True):
		print(f"{model_type.upper():15} | Accuracy: {metrics.accuracy:.4f} | F1: {metrics.f1_weighted:.4f}")
	print("="*60 + "\n")
	
	return results
