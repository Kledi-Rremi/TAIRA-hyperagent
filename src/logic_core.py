"""logic_core.py — Rule-Based Expert System (Technique B)

TAIRA (Traffic Accident Intelligent Risk Advisor)
CEN 352 Term Project — Technique B: Logical Reasoning

This module implements the rule-based expert system that refines ML predictions
using domain-specific safety knowledge and contextual reasoning.

Key Features:
- Evidence accumulation from multiple environmental factors
- Hard safety rules for critical condition combinations
- Risk score calculation (0-3 scale)
- Risk level classification: LOW / MEDIUM / HIGH / CRITICAL
- Human-readable explanations for all assessments

Design Philosophy:
- Weighted risk scoring accumulates evidence from conditions
- Hard rules enforce minimum risk for dangerous combinations
- Transparent logic enables audit and policy compliance
- Explainable decisions support dispatcher judgment

Risk Factors Analyzed:
- Weather conditions (precipitation, fog, extreme conditions)
- Visibility (lighting, darkness, time of day)
- Road surface (ice, snow, wet, dry)
- Crash type (head-on, pedestrian, angle collision)
- Human factors (speed, impairment, distraction)
- Road geometry (curves, grades, intersections)
- Traffic control (signals, stop signs, absence of controls)

Output: (risk_level, explanation_text)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from config import RISK_THRESHOLDS, RISK_LEVELS
from logger import logic_logger, log_error, log_critical_assessment


def assess_risk_with_explanation(
  ml_prediction: str,
  features: Dict,
) -> Tuple[str, str]:
  """Infer a risk label and explanation from model prediction + features.

  Parameters
  - ml_prediction: model prediction of MOST_SEVERE_INJURY
  - features: dict containing (ideally) keys like:
    - weather_condition
    - lighting_condition
    - roadway_surface_cond
    - traffic_control_device
    - crash_type
    - prim_contributory_cause (optional)
    - intersection_related_i (optional)
    - num_units (optional)
    - crash_hour (optional)

  Returns
  - (risk_level, explanation)
  """

  # ---- Normalize inputs ----
  pred_raw = (ml_prediction or "").strip()
  pred = pred_raw.upper()

  weather_raw = str(features.get("weather_condition", "") or "").strip()
  light_raw = str(features.get("lighting_condition", "") or "").strip()
  surface_raw = str(features.get("roadway_surface_cond", "") or "").strip()
  control_raw = str(features.get("traffic_control_device", "") or "").strip()
  crash_raw = str(features.get("crash_type", "") or "").strip()
  cause_raw = str(features.get("prim_contributory_cause", "") or "").strip()
  inter_raw = str(features.get("intersection_related_i", "") or "").strip()
  road_defect_raw = str(features.get("road_defect", "") or "").strip()
  alignment_raw = str(features.get("alignment", "") or "").strip()

  weather_u = weather_raw.upper()
  light_u = light_raw.upper()
  surface_u = surface_raw.upper()
  control_u = control_raw.upper()
  crash_u = crash_raw.upper()
  cause_u = cause_raw.upper()
  inter_u = inter_raw.upper()
  road_defect_u = road_defect_raw.upper()
  alignment_u = alignment_raw.upper()

  # numeric-ish (optional; default values keep pre-crash tool categorical-only)
  try:
    num_units = int(features.get("num_units", 1))
  except (ValueError, TypeError):
    num_units = 1

  try:
    crash_hour = int(features.get("crash_hour", 12))
  except (ValueError, TypeError):
    crash_hour = 12

  reasons: list[str] = []

  # ---- 1) Base score from predicted severity ----
  # score range: 0..3 (later clamped)
  if pred == "FATAL":
    base_severity = 3
    score = 3
    reasons.append("Predicted injury is FATAL.")
  elif "INCAPACITATING" in pred:
    base_severity = 3
    score = 3
    reasons.append("Predicted injury is incapacitating.")
  elif "NONINCAPACITATING" in pred:
    base_severity = 2
    score = 2
    reasons.append("Predicted injury is non-incapacitating.")
  elif "REPORTED" in pred:
    base_severity = 1
    score = 1
    reasons.append("Predicted injury is reported but not evident.")
  else:
    base_severity = 0
    score = 0
    reasons.append("Predicted injury is minor or no indication of injury.")

  # Check for extremely dangerous combinations that should always cause injury
  has_very_slippery = "ICE" in surface_u or "SNOW OR SLUSH" in surface_u
  has_low_visibility = (
    ("DARKNESS" in light_u and "LIGHTED" not in light_u) 
    or "FOG" in weather_u 
    or "HAZE" in weather_u
  )
  has_precipitation = any(k in weather_u for k in ["RAIN", "SNOW", "SLEET", "FREEZING", "HAIL", "DRIZZLE"])
  high_impact_crash = "HEAD ON" in crash_u or "ANGLE" in crash_u or "PEDESTRIAN" in crash_u
  human_factor = any(k in cause_u for k in ["ALCOHOL", "DUI", "DRUG", "SPEED", "SPEEDING", "RECKLESS"])
  
  # Count dangerous conditions to determine if override is needed
  dangerous_condition_count = sum([
    has_very_slippery,
    has_low_visibility,
    has_precipitation and (has_very_slippery or has_low_visibility),
    high_impact_crash,
    human_factor
  ])
  
  # Use config thresholds for dangerous condition overrides
  critical_threshold = RISK_THRESHOLDS["dangerous_condition_critical"]
  high_threshold = RISK_THRESHOLDS["dangerous_condition_high"]
  medium_threshold = RISK_THRESHOLDS["dangerous_condition_medium"]
  
  # Critical conditions should predict INCAPACITATING or higher injury
  if dangerous_condition_count >= critical_threshold and base_severity < 3:
    # Extremely critical conditions should predict at least INCAPACITATING injury
    base_severity = 3
    score = 3
    reasons.clear()
    reasons.append("Critical dangerous conditions present: likely incapacitating injury.")
  elif dangerous_condition_count >= high_threshold and base_severity == 0:
    # Multiple dangerous conditions should predict at least NONINCAPACITATING
    base_severity = 2
    score = 2
    reasons.clear()
    reasons.append("Multiple dangerous conditions present: likely non-incapacitating injury.")
  elif dangerous_condition_count >= medium_threshold and base_severity == 0:
    # Two dangerous conditions should predict at least REPORTED INJURY
    base_severity = 1
    score = 1
    reasons.clear()
    reasons.append("Dangerous conditions present: likely minor injury.")


  forced_min_risk: Optional[str] = None

  # ---- 2) Environment / context evidence bumps ----
  
  # Low visibility: darkness (only if not lighted), fog, haze
  # DARKNESS with LIGHTED ROAD is safe, so don't count it
  has_low_visibility = (
    ("DARKNESS" in light_u and "LIGHTED" not in light_u)  # Dark AND unlit = risky
    or "FOG" in weather_u 
    or "HAZE" in weather_u
  )
  if has_low_visibility:
    score += 1
    reasons.append(f"Low visibility (lighting: {light_raw}, weather: {weather_raw}).")

  # Weather conditions - precipitation adds modest risk
  has_precipitation = any(k in weather_u for k in ["RAIN", "SNOW", "SLEET", "FREEZING", "HAIL", "DRIZZLE"])
  if has_precipitation:
    score += 0.5  # Base risk from precipitation
    reasons.append(f"Precipitation ({weather_raw}) reduces traction and visibility.")
    # Additional bumps if combined with other hazards
    if has_low_visibility:
      score += 0.5  # Extra bump for visibility + precipitation combo
      reasons.append("Precipitation with low visibility increases risk further.")
    elif "ICE" in surface_u or "SNOW OR SLUSH" in surface_u or "WET" in surface_u:
      score += 0.5  # Extra bump for slippery surface + precipitation
      reasons.append(f"Precipitation ({weather_raw}) with slippery road surface.")
    # Otherwise, light precipitation alone adds modest bump only

  # Road surface conditions - differentiate by severity
  # Ice and snow/slush are most dangerous
  has_very_slippery = "ICE" in surface_u or "SNOW OR SLUSH" in surface_u
  if has_very_slippery:
    score += 1
    reasons.append(f"Icy/slush road surface ({surface_raw}) reduces traction significantly.")
  
  # Wet, sand, mud, dirt are moderate hazards - only add risk if combined with bad weather/visibility
  has_moderate_surface = surface_u == "WET" or "SAND" in surface_u or "MUD" in surface_u or "DIRT" in surface_u
  if has_moderate_surface:
    if has_precipitation or has_low_visibility:
      score += 0.5
      reasons.append(f"Road surface ({surface_raw}) with adverse weather reduces traction.")
    # Otherwise, mild surface conditions alone don't add much risk
  # Dry, normal surfaces don't contribute risk

  # Road defects degrade handling/traction
  if road_defect_u and road_defect_u not in {"NO DEFECTS", "UNKNOWN", ""}:
    score += 0.5
    reasons.append(f"Road defect reduces traction ({road_defect_raw}).")

  # Crash-type risk - inherent severity
  if "HEAD ON" in crash_u or "PEDESTRIAN" in crash_u or "ANGLE" in crash_u:
    score += 1
    reasons.append(f"High-risk crash type ({crash_raw}).")

  # Traffic control: only risky if weak/absent AND conditions are truly hazardous
  if control_u in {"NO CONTROLS", "STOP SIGN/FLASHER", "UNKNOWN"}:
    if has_low_visibility or "RAIN" in weather_u or "SNOW" in weather_u:
      score += 0.5  # Modest impact - control helps but weather/visibility is main factor
      reasons.append(
        f"Limited traffic control ({control_raw}) under challenging conditions."
      )

  # Alignment + surface interaction: curves/grades amplify risk when traction is poor
  if ("CURVE" in alignment_u or "GRADE" in alignment_u or "HILL" in alignment_u):
    if has_very_slippery:
      score += 1
      reasons.append("Curve/grade with ice/slush: higher run-off/slide risk.")
    elif has_moderate_surface:
      score += 0.5
      reasons.append("Curve/grade with reduced-traction surface: increased risk.")
    elif has_low_visibility:
      score += 0.5
      reasons.append("Curve/grade under low visibility: reduced sight distance.")

  # Multiple vehicles - moderate risk increase
  if num_units >= 3:
    score += 0.5
    reasons.append(f"Multiple units involved ({num_units} vehicles) increases complexity.")

  # Late-night / early morning - only risky if low visibility
  if crash_hour in {22, 23, 0, 1, 2, 3, 4, 5}:
    if "DARKNESS" in light_u:
      score += 0.5
      reasons.append(f"High-risk time with darkness ({crash_hour}:00).")
    # Early morning/night alone without darkness doesn't add risk

  # Aggressive / impaired driving causes - strong risk factor
  human_factor_keywords = [
    "ALCOHOL",
    "DUI",
    "DRUG",
    "SPEED",
    "SPEEDING",
    "RECKLESS",
  ]
  if any(k in cause_u for k in human_factor_keywords):
    score += 1
    reasons.append(f"Human factor likely increases severity ({cause_raw}).")

  # Intersection-related - minimal increase
  if inter_u.startswith("Y"):
    score += 0.25
    reasons.append("Intersection location (more conflict points).")

  # ---- 3) Hard rules that force minimum risk ----
  # Pedestrian crash at night or in rain should not be LOW
  if "PEDESTRIAN" in crash_u and ("DARKNESS" in light_u or "RAIN" in weather_u):
    forced_min_risk = "HIGH"
    reasons.append("Pedestrian crash in low-visibility conditions: at least HIGH risk.")

  # Multi-vehicle angle/head-on on slippery surface -> at least HIGH
  if ("HEAD ON" in crash_u or "ANGLE" in crash_u) and num_units >= 3:
    if "ICE" in surface_u or "SNOW OR SLUSH" in surface_u or surface_u == "WET":
      forced_min_risk = "HIGH"
      reasons.append(
        "Multi-vehicle head-on/angle on slippery surface: at least HIGH risk."
      )

  # Snow/sleet + darkness = CRITICAL hazard ONLY if road is also slippery
  # But NOT if the road is lighted or if road is DRY (good traction)
  is_dark_unlit = "DARKNESS" in light_u and "LIGHTED" not in light_u
  if ("SNOW" in weather_u or "SLEET" in weather_u) and is_dark_unlit:
    # Only CRITICAL if road is slippery (ice/slush), not dry
    if has_very_slippery:
      forced_min_risk = "CRITICAL"
      reasons.append("Snow/sleet with unlit darkness on icy/slush road: CRITICAL hazard.")
    # Snow + dark on DRY road is not CRITICAL, just risky
  # Also CRITICAL if snow + icy/slush road even without darkness
  elif ("SNOW" in weather_u or "SLEET" in weather_u) and has_very_slippery:
    forced_min_risk = "CRITICAL"
    reasons.append("Snow/sleet on icy/slush road: CRITICAL hazard.")

  # Severe prediction + very bad conditions -> CRITICAL
  # (Important: only allow CRITICAL if the ML prediction itself is severe.)
  if base_severity >= 3 and (
    "HEAD ON" in crash_u
    or "PEDESTRIAN" in crash_u
    or "ICE" in surface_u
    or ("RAIN" in weather_u and "DARKNESS" in light_u)
  ):
    forced_min_risk = "CRITICAL"
    reasons.append(
      "Severe injuries combined with high-risk crash type and poor conditions: CRITICAL risk."
    )

  # ---- 4) Clamp score and map to label ----
  # Allow CRITICAL to be reached through accumulated evidence, not just explicit rules
  raw_score = score
  
  # Use config threshold for CRITICAL risk
  critical_score_threshold = RISK_THRESHOLDS["critical_score_threshold"]
  
  # CRITICAL can be reached through accumulated score even without special conditions
  # Only clamp to HIGH if we haven't accumulated enough evidence
  if raw_score >= critical_score_threshold:
    max_allowed_score = 3  # Allow CRITICAL
  elif base_severity <= 1:
    max_allowed_score = 2  # Cap at HIGH for low-severity predictions
  else:
    max_allowed_score = 3
  
  score = max(0, min(int(score), max_allowed_score))

  # Map score to risk level using config
  risk = RISK_LEVELS.get(score, "LOW")

  # If forced_min_risk is set, use that if it's higher
  if forced_min_risk is not None:
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    if order[forced_min_risk] > order[risk]:
      risk = forced_min_risk

  # CRITICAL RISK OVERRIDE: When risk is CRITICAL, injury should be at least INCAPACITATING
  # Ensure that CRITICAL risk scenarios predict serious injury, not "no injury"
  if risk == "CRITICAL" and base_severity < 3:
    reasons.append("CRITICAL risk level: elevating injury prediction to at least incapacitating.")
    base_severity = 3
    # Update the ML prediction to reflect the elevated severity for downstream use
    ml_prediction = "INCAPACITATING INJURY"

  # Log CRITICAL risk assessments for monitoring and quality assurance
  if risk == "CRITICAL":
    logic_logger.warning(
      f"CRITICAL risk assessment | Prediction: {ml_prediction} | "
      f"Score: {raw_score:.2f} | Conditions: {dangerous_condition_count}"
    )

  explanation = " ".join(reasons)
  
  logic_logger.debug(f"Risk assessment: {risk} | Score: {score} | Base severity: {base_severity}")
  
  return risk, explanation


def assess_risk(
    ml_prediction: str,
    weather: str,
    light: str,
    damage: str = "",
    surface: Optional[str] = None,
) -> str:
    """Backward-compatible wrapper returning only the risk label."""
    risk, _explanation = assess_risk_with_explanation(
        ml_prediction,
        {
            "weather_condition": weather,
            "lighting_condition": light,
            "roadway_surface_cond": surface or "",
            "damage": damage,
        },
    )
    return risk


def injury_damage_stats_for_scenario(df, primary_cause, crash_type) -> dict:
    """Lookup historical injury and damage patterns for a given scenario.

    Filters to rows matching:
      - prim_contributory_cause == primary_cause
      - first_crash_type == crash_type
    """
    if df is None:
        return {"count": 0}

    # Handle either lowercase or uppercase column names
    col_map = {str(c).lower(): c for c in df.columns}
    cause_col = col_map.get("prim_contributory_cause")
    crash_col = col_map.get("first_crash_type")
    if cause_col is None or crash_col is None:
        return {"count": 0}

    filtered = df[(df[cause_col] == primary_cause) & (df[crash_col] == crash_type)]
    if filtered.empty:
        return {"count": 0}

    def _avg(name: str) -> float:
        col = col_map.get(name)
        if col is None:
            return float("nan")
        return float(filtered[col].mean())

    stats = {
        "count": int(len(filtered)),
        "avg_injuries_total": _avg("injuries_total"),
        "avg_injuries_fatal": _avg("injuries_fatal"),
        "avg_incapacitating": _avg("injuries_incapacitating"),
        "avg_non_incapacitating": _avg("injuries_non_incapacitating"),
        "avg_reported_not_evident": _avg("injuries_reported_not_evident"),
        "avg_no_indication": _avg("injuries_no_indication"),
    }

    damage_col = col_map.get("damage")
    if damage_col is not None:
        mode = filtered[damage_col].mode(dropna=True)
        if not mode.empty:
            stats["most_common_damage"] = str(mode.iloc[0])

    return stats
