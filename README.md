
# TAIRA (Traffic Accident Intelligent Risk Advisor)

## Overview

TAIRA is a Python-based **Intelligent Agent** system designed for CEN 352 that assists emergency dispatchers in assessing traffic accident severity. The system analyzes crash conditions using the Traffic Accidents dataset and provides real-time risk assessments to support emergency response decision-making.

**Dataset Source:** [Traffic Accidents Dataset by Oktay Rüdeki (Kaggle)](https://www.kaggle.com/datasets/oktayrdeki/traffic-accidents)  
**Dataset Location:** `data/traffic_accidents.csv`

---

## Project Description

TAIRA implements a **hybrid AI architecture** combining two complementary artificial intelligence techniques:

### **Technique A — Statistical Learning (Machine Learning)**
- **Model:** Support Vector Machine (LinearSVC with Calibration)
- **Purpose:** Predicts `most_severe_injury` based on environmental and crash context features
- **Rationale:** SVM was selected after comparative evaluation against Random Forest due to:
  - Superior handling of imbalanced class distributions with balanced class weights
  - Efficient linear classification with kernel-free approach
  - Calibrated probability estimates for risk assessment
  - Better predictive accuracy on test data
  - Fast inference suitable for real-time applications

### **Technique B — Logical Reasoning (Rule-Based Expert System)**
- **Purpose:** Applies domain-specific safety rules to refine ML predictions
- **Output:** Dispatcher-facing risk levels: **LOW / MEDIUM / HIGH / CRITICAL**
- **Features:** 
  - Interprets dangerous condition combinations (e.g., icy roads + darkness)
  - Overrides ML predictions when critical safety rules are triggered
  - Provides human-readable explanations for risk assessments

### **System Components**

The Streamlit web application includes:
1. **Interactive Risk Assessment Tool** — Real-time crash scenario evaluation
2. **VDA (Visualization & Data Analysis) Dashboard** — Statistical insights and model performance metrics
3. **Historical Data Analysis** — Crash patterns and injury statistics
4. **About Page** — System documentation and ethical considerations

---

## PEAS Framework Analysis

### **Performance Measures**
The system's effectiveness is evaluated using:
- **Accuracy:** Overall prediction correctness
- **F1-Score (weighted):** Balanced measure accounting for class imbalance
- **Risk Assessment Accuracy:** Correct classification of risk levels

### **Environment**
- **Domain:** Urban traffic network
- **Conditions:** Variable weather, lighting, road surfaces, and traffic patterns
- **Data:** Historical crash records with environmental context

### **Actuators (System Outputs)**
1. **Risk Alert Level:** LOW / MEDIUM / HIGH / CRITICAL
2. **Injury Severity Prediction:** Model-predicted `most_severe_injury` classification
3. **Explanation Text:** Human-readable justification for risk assessment
4. **Recommended Actions:** Dispatcher guidance based on risk level

### **Sensors (System Inputs)**
User-provided inputs via UI dropdowns:
- Weather condition (rain, snow, clear, etc.)
- Lighting condition (daylight, darkness, dawn/dusk)
- Road surface condition (dry, wet, ice, snow/slush)
- Traffic control device (signals, stop signs, none)
- Crash hour (time of day)
- Estimated damage level
- Crash type (angle, head-on, rear-end, etc.)
- Primary contributory cause

---

## AI Architecture

### **Data Processing Pipeline**

1. **Data Loader Module** (`data_loader.py`)
   - Loads CSV dataset
   - Cleans missing values
   - Standardizes categorical variables
   - Prepares features for model training

2. **ML Model Engine** (`model_engine.py`)
   - Preprocessing: OneHotEncoder for categorical features
   - Training: LinearSVC with CalibratedClassifierCV for probability estimates
   - Evaluation: Computes accuracy, F1-score, precision, recall, and confusion matrix
   - Model comparison: Benchmarks SVM vs Random Forest performance
   - Model persistence: Saves trained models for efficient reuse

3. **Logic Core** (`logic_core.py`)
   - Applies interpretable safety rules
   - Accumulates risk evidence from environmental factors
   - Enforces hard rules for critical combinations
   - Generates human-readable explanations

4. **Agent Module** (`agent.py`)
   - Integrates ML predictions with rule-based logic
   - Manages model lifecycle
   - Provides unified interface for risk assessment

### **Hybrid Design Benefits**
- **Predictive Power:** Leverages SVM's efficient classification for accurate predictions
- **Transparency:** Rule-based overrides provide explainable decisions
- **Safety:** Hard rules prevent unsafe recommendations in critical scenarios
- **Flexibility:** Easy to update rules without retraining the ML model
- **Performance:** Fast inference enables real-time risk assessment

---

## Ethical Considerations

### **Automation Bias**
**Risk:** Emergency dispatchers may over-rely on AI recommendations, especially during high-stress situations or time-critical decisions.

**Mitigation Strategies:**
- System explicitly labeled as a **decision-support tool**, not a replacement for professional judgment
- Risk assessments include detailed explanations to encourage critical thinking
- UI displays confidence levels and contributing factors
- Training materials emphasize the importance of dispatcher expertise

### **Data Bias and Fairness**
**Risk:** Historical crash data may reflect systemic biases in:
- Reporting practices across different neighborhoods
- Law enforcement patterns
- Socioeconomic disparities in data collection

**Potential Impact:**
- Risk assessments may vary systematically across communities
- Underrepresented areas may have different prediction accuracy
- Historical patterns may not reflect current conditions

**Mitigation Strategies:**
- Transparent documentation of data sources and limitations
- Regular model evaluation across different demographic groups
- Continuous monitoring for fairness metrics
- User feedback mechanisms to identify systematic issues

### **Transparency and Accountability**
- All risk assessments include human-readable explanations
- Rule-based components are fully auditable
- System limitations are clearly documented
- Decision logic is open for review by stakeholders

---

## Installation & Setup

### **Prerequisites**
- Python 3.8 or higher
- pip package manager

### **Installation Steps**

1. **Navigate to the project directory:**
   ```bash
   cd traffic_ai_project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   
   **Windows (PowerShell):**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   
   **Windows (Command Prompt):**
   ```cmd
   .venv\Scripts\activate.bat
   ```
   
   **macOS/Linux:**
   ```bash
   source .venv/bin/activate
   ```

4. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

**Start the Streamlit Application:**
```bash
streamlit run src/app.py
```

The application will open in your default web browser at `http://localhost:8501`

**Alternative Method (if `streamlit` command not found):**
```bash
python -m streamlit run src/app.py
```

---

## Project Structure

```
traffic_ai_project/
├── data/
│   └── traffic_accidents.csv    # Traffic Accidents dataset (Kaggle)
├── src/
│   ├── app.py                   # Main Streamlit application
│   ├── agent.py                 # TrafficAccidentAgent (integrates ML + logic)
│   ├── model_engine.py          # ML model training and evaluation
│   ├── logic_core.py            # Rule-based expert system
│   ├── data_loader.py           # Data loading and preprocessing
│   └── config.py                # Configuration constants
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

**Note:** The dataset should be located at `traffic_ai_project/data/traffic_accidents.csv`

---

## Usage Guide

### **1. Interactive Risk Assessment**
- Select crash conditions from dropdown menus
- View real-time risk level (LOW/MEDIUM/HIGH/CRITICAL)
- Read detailed explanation of risk factors
- See injury severity prediction

### **2. Data Analysis Dashboard**
- Explore crash patterns and statistics
- View model performance metrics
- Analyze injury distributions
- Review comparative model evaluations

### **3. Historical Data Lookup**
- Query specific crash scenarios
- View historical injury statistics
- Compare similar past incidents

---

## System Features

### **Key Capabilities**
- ✓ Real-time risk assessment with explainable decisions
- ✓ Multi-factor condition analysis (weather, lighting, road surface, etc.)
- ✓ Historical data integration and pattern analysis
- ✓ Interactive visualization dashboard
- ✓ Model performance monitoring and comparison
- ✓ Human-readable risk explanations

### **Risk Assessment Factors**
The system analyzes multiple factors including:
- Weather conditions (rain, snow, fog, clear, etc.)
- Lighting and visibility (daylight, darkness, dawn/dusk)
- Road surface conditions (dry, wet, ice, snow/slush)
- Crash type and impact severity (head-on, angle, rear-end, pedestrian)
- Traffic control presence (signals, stop signs, none)
- Human factors (speed, impairment, distraction)
- Road geometry (curves, grades, intersections)
- Temporal patterns (time of day, rush hour)

---

## Credits

**Project Team:**

- **Kledi Rremi** — System design, coding, and implementation
- **Shaban Xibraku** — Project proposal, requirements analysis, and documentation

---

## Technologies & Dependencies

### **Core Libraries**

| Library | Version | Purpose | License |
|---------|---------|---------|---------|
| **Streamlit** | 1.52.2 | Web application framework | Apache 2.0 |
| **Scikit-learn** | 1.8.0 | SVM model and ML pipeline | BSD-3-Clause |
| **Pandas** | 2.3.3 | Data manipulation and analysis | BSD-3-Clause |
| **NumPy** | 2.4.1 | Numerical computing | BSD-3-Clause |
| **Matplotlib** | 3.10.8 | Data visualization | PSF-based |
| **Seaborn** | 0.13.2 | Statistical visualization | BSD-3-Clause |

### **Data Source**
- **Traffic Accidents Dataset**  
  Dataset by Oktay Rüdeki from Kaggle  
  URL: [https://www.kaggle.com/datasets/oktayrdeki/traffic-accidents](https://www.kaggle.com/datasets/oktayrdeki/traffic-accidents)  
  Contains historical crash records with environmental and contextual information

### **Repository Links**
- Streamlit: https://github.com/streamlit/streamlit
- Scikit-learn: https://github.com/scikit-learn/scikit-learn
- Pandas: https://github.com/pandas-dev/pandas
- NumPy: https://github.com/numpy/numpy
- Matplotlib: https://github.com/matplotlib/matplotlib
- Seaborn: https://github.com/mwaskom/seaborn

---

## Technical Notes

- **Configuration:** UI dropdown options are defined in [src/config.py](src/config.py) for consistency
- **Model Metrics:** Accuracy and F1-Score are computed during training and displayed in the application
- **Data Processing:** The system automatically handles missing values and standardizes categorical inputs
- **Performance:** SVM provides fast inference suitable for real-time risk assessment

---

## Future Enhancements

Potential areas for system improvement:
- Integration with live traffic data feeds
- Mobile application for field dispatchers
- Multi-language support for diverse communities
- Real-time weather API integration
- Enhanced visualization of crash hotspots
- Machine learning model retraining pipeline
- User feedback collection system

---

## License

This project is developed for educational purposes as part of CEN 352 coursework.

---

## Contact

For questions or feedback regarding this project, please contact the project team through your institution.

---

*Last Updated: January 2026*
=======
# TAIRA-hyperagent
>>>>>>> bf10a459d6359f0ceaefed96f7b7b80d1de3d7c5
