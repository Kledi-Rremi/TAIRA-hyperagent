"""agent.py — Intelligent Agent Orchestration Module

TAIRA (Traffic Accident Intelligent Risk Advisor)
CEN 352 Term Project

This module implements the high-level agent that integrates:
1) Statistical learning (SVM) for injury severity prediction
2) Rule-based expert system for risk level assessment (LOW/MEDIUM/HIGH/CRITICAL)

The hybrid architecture combines ML prediction power with transparent, 
policy-driven safety rules to support emergency dispatcher decision-making.
"""

from __future__ import annotations

from typing import Dict, Optional, Set

from data_loader import FEATURE_COLUMNS
from logger import agent_logger, log_prediction, log_critical_assessment, log_error


class TrafficAccidentAgent:
    """Hybrid agent used by the Streamlit Pre-Crash tool.

    This wrapper is intentionally simple:
    - Uses the ML pipeline from src/model_engine to predict MOST_SEVERE_INJURY
    - Uses src/logic_core.assess_risk to convert prediction + conditions into a risk level

    Returns a dict so the UI can render with result["explanation"], etc.
    """

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self._valid_features: Set[str] = set(FEATURE_COLUMNS)
        agent_logger.info("TrafficAccidentAgent initialized")

    def _validate_features(self, features: Dict[str, object]) -> None:
        """Validate that all required features are present and non-empty.
        
        Args:
            features: Input feature dictionary
            
        Raises:
            ValueError: If required features are missing or invalid
        """
        missing = []
        for feature in ["weather_condition", "lighting_condition", "roadway_surface_cond", 
                       "traffic_control_device", "crash_type"]:
            if feature not in features or not features[feature]:
                missing.append(feature)
        
        if missing:
            raise ValueError(f"Missing or empty required features: {missing}")

    def predict(self, features: Dict[str, object]) -> Dict[str, str]:
        """Make a prediction with comprehensive error handling and logging.
        
        Args:
            features: Dictionary of input features
            
        Returns:
            Dictionary containing:
                - predicted_injury: Injury severity prediction
                - risk_level: Risk assessment (LOW/MEDIUM/HIGH/CRITICAL)
                - explanation: Human-readable explanation
                
        Raises:
            ValueError: If features are invalid
            Exception: If prediction fails
        """
        try:
            # Validate input features
            self._validate_features(features)
            
            from logic_core import assess_risk_with_explanation
            from model_engine import predict_one

            # Map UI keys (snake_case) -> model keys (UPPERCASE)
            model_features = {
                "TRAFFIC_CONTROL_DEVICE": features.get("traffic_control_device"),
                "WEATHER_CONDITION": features.get("weather_condition"),
                "LIGHTING_CONDITION": features.get("lighting_condition"),
                "ROADWAY_SURFACE_COND": features.get("roadway_surface_cond"),
                "TRAFFICWAY_TYPE": features.get("trafficway_type"),
                "ALIGNMENT": features.get("alignment"),
                "ROAD_DEFECT": features.get("road_defect"),
                "INTERSECTION_RELATED_I": features.get("intersection_related_i"),
                "PRIM_CONTRIBUTORY_CAUSE": features.get("prim_contributory_cause"),
                # UI uses crash_type dropdown; model feature is FIRST_CRASH_TYPE
                "FIRST_CRASH_TYPE": features.get("crash_type"),
            }

            agent_logger.debug(f"Making prediction with features: {model_features}")
            predicted = predict_one(self.pipeline, model_features)

            risk_level, explanation = assess_risk_with_explanation(str(predicted), features)

            result = {
                "predicted_injury": str(predicted),
                "risk_level": str(risk_level),
                "explanation": explanation,
            }
            
            # Log prediction for monitoring
            log_prediction(agent_logger, features, predicted, risk_level)
            
            # Log critical assessments for quality assurance
            if risk_level == "CRITICAL":
                log_critical_assessment(agent_logger, features, predicted, risk_level, explanation)
            
            agent_logger.info(
                f"Prediction completed: {predicted} -> {risk_level}"
            )
            
            return result
            
        except ValueError as e:
            agent_logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            log_error(agent_logger, e, {"features": features})
            raise Exception(f"Prediction failed: {str(e)}") from e

