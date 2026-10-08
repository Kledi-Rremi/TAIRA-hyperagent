"""temporal_analysis.py — Temporal Accident Pattern Analysis

TAIRA (Traffic Accident Intelligent Risk Advisor)
CEN 352 Term Project

Enhanced analysis utilities for Post-Crash section.
Shows WHEN accidents happen, condition-injury patterns, risk distributions, and severity trends.

This module provides temporal and pattern analysis functions:
- Injury distribution by environmental conditions
- Most dangerous condition identification
- Temporal crash patterns (hourly, daily)
- Risk pattern combinations
- Injury severity trend analysis

NOT a separate model - just analysis helpers for existing Post-Crash page.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from config import RISK_THRESHOLDS
from data_loader import TARGET_COLUMN
from logger import data_logger, log_error


def get_injury_distribution_by_condition(
    df: pd.DataFrame,
    condition_col: str,
    condition_value: str
) -> Dict[str, float]:
    """Show injury distribution for a specific condition.
    
    Example: "What injuries happen when WEATHER_CONDITION = RAIN?"
    
    Args:
        df: DataFrame containing crash data
        condition_col: Column name for the condition
        condition_value: Specific value to filter by
        
    Returns:
        Dictionary mapping injury types to proportions
    """
    try:
        if condition_col not in df.columns:
            data_logger.warning(f"Column {condition_col} not found in dataframe")
            return {}
        
        if TARGET_COLUMN not in df.columns:
            data_logger.error(f"Target column {TARGET_COLUMN} not found")
            return {}
        
        filtered = df[df[condition_col] == condition_value]
        
        if len(filtered) == 0:
            data_logger.debug(f"No records found for {condition_col}={condition_value}")
            return {}
        
        injury_dist = filtered[TARGET_COLUMN].value_counts(normalize=True).to_dict()
        return {k: round(v, 4) for k, v in injury_dist.items()}
        
    except Exception as e:
        log_error(data_logger, e, {
            "condition_col": condition_col,
            "condition_value": condition_value
        })
        return {}


def get_most_dangerous_condition_values(df: pd.DataFrame) -> Dict[str, Tuple[str, float]]:
    """Find the most dangerous value for each condition with severity rate.
    
    Returns:
        {condition_col: (most_dangerous_value, severe_injury_rate), ...}
    """
    try:
        key_features = [
            'WEATHER_CONDITION',
            'LIGHTING_CONDITION',
            'ROADWAY_SURFACE_COND',
            'TRAFFIC_CONTROL_DEVICE'
        ]
        
        if TARGET_COLUMN not in df.columns:
            data_logger.error(f"Target column {TARGET_COLUMN} not found")
            return {}
        
        results = {}
        for feature in key_features:
            if feature not in df.columns:
                data_logger.debug(f"Feature {feature} not in dataframe, skipping")
                continue
                
            severity_rates = {}
            for value in df[feature].dropna().unique():
                subset = df[df[feature] == value]
                if len(subset) == 0:
                    continue
                # Count severe injuries (INCAPACITATING or FATAL)
                severe_count = (
                    (subset[TARGET_COLUMN].str.upper() == 'INCAPACITATING INJURY') |
                    (subset[TARGET_COLUMN].str.upper() == 'FATAL')
                ).sum()
                rate = severe_count / len(subset)
                severity_rates[value] = rate
            
            if severity_rates:
                most_dangerous = max(severity_rates, key=severity_rates.get)
                results[feature] = (most_dangerous, severity_rates[most_dangerous])
        
        return results
        
    except Exception as e:
        log_error(data_logger, e, {"df_shape": df.shape})
        return {}


def get_temporal_crash_patterns(df: pd.DataFrame) -> Dict[str, any]:
    """Analyze when crashes are most severe by hour and day.
    
    Returns:
        Dictionary with hourly and daily patterns
    """
    results = {}
    
    # Try to find time columns
    hour_col = None
    day_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'hour' in col_lower or 'crash_hour' in col_lower:
            hour_col = col
        if 'day' in col_lower and 'week' in col_lower:
            day_col = col
    
    # Hourly patterns
    if hour_col:
        df_temp = df.copy()
        df_temp[hour_col] = pd.to_numeric(df_temp[hour_col], errors='coerce')
        df_temp = df_temp.dropna(subset=[hour_col])
        
        # Calculate severe injury rate by hour
        severe_mask = (
            (df_temp[TARGET_COLUMN].str.upper() == 'INCAPACITATING INJURY') |
            (df_temp[TARGET_COLUMN].str.upper() == 'FATAL')
        )
        
        hourly_data = df_temp.groupby(hour_col).agg({
            TARGET_COLUMN: 'count',
        }).rename(columns={TARGET_COLUMN: 'crash_count'})
        
        hourly_severe = df_temp[severe_mask].groupby(hour_col).size()
        hourly_data['severe_count'] = hourly_severe
        hourly_data['severe_rate'] = hourly_data['severe_count'] / hourly_data['crash_count']
        hourly_data = hourly_data.fillna(0)
        
        results['hourly_patterns'] = hourly_data.to_dict('index')
        results['most_dangerous_hour'] = int(hourly_data['severe_rate'].idxmax())
        results['most_crashes_hour'] = int(hourly_data['crash_count'].idxmax())
    
    # Daily patterns
    if day_col:
        daily_counts = df[day_col].value_counts().to_dict()
        results['daily_crash_counts'] = daily_counts
        results['most_crashes_day'] = max(daily_counts, key=daily_counts.get)
    
    return results


def get_risk_pattern_combinations(df: pd.DataFrame) -> List[Dict[str, any]]:
    """Identify dangerous condition combinations.
    
    Returns:
        List of high-risk condition combinations with injury rates
    """
    try:
        combinations = []
        
        # Define key condition columns
        weather_col = 'WEATHER_CONDITION'
        light_col = 'LIGHTING_CONDITION'
        surface_col = 'ROADWAY_SURFACE_COND'
        
        if not all(col in df.columns for col in [weather_col, light_col, surface_col]):
            data_logger.warning("Required columns for risk pattern analysis not found")
            return combinations
        
        if TARGET_COLUMN not in df.columns:
            data_logger.error(f"Target column {TARGET_COLUMN} not found")
            return combinations
        
        # Get thresholds from config
        min_sample_size = RISK_THRESHOLDS["minimum_sample_size"]
        severe_rate_threshold = RISK_THRESHOLDS["severe_injury_rate_threshold"]
        
        # Group by condition combinations
        grouped = df.groupby([weather_col, light_col, surface_col])
        
        for (weather, light, surface), group in grouped:
            if len(group) < min_sample_size:
                continue
            
            severe_count = (
                (group[TARGET_COLUMN].str.upper() == 'INCAPACITATING INJURY') |
                (group[TARGET_COLUMN].str.upper() == 'FATAL')
            ).sum()
            
            severe_rate = severe_count / len(group)
            
            if severe_rate >= severe_rate_threshold:
                combinations.append({
                    'weather': weather,
                    'lighting': light,
                    'surface': surface,
                    'crash_count': len(group),
                    'severe_count': int(severe_count),
                    'severe_rate': round(severe_rate, 4)
                })
        
        # Sort by severe rate descending
        combinations.sort(key=lambda x: x['severe_rate'], reverse=True)
        return combinations[:10]  # Top 10
        
    except Exception as e:
        log_error(data_logger, e, {"df_shape": df.shape})
        return []


def get_injury_severity_trends(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate overall injury severity distribution.
    
    Returns:
        Dictionary with percentages for each injury type
    """
    if TARGET_COLUMN not in df.columns:
        return {}
    
    injury_counts = df[TARGET_COLUMN].value_counts(normalize=True)
    return {k: round(v * 100, 2) for k, v in injury_counts.items()}
