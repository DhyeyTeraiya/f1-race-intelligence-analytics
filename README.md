<div align="center">

# 🏎️ Formula 1 Race Intelligence & Strategy Predictive Analytics

### End-to-End Motorsport Data Science, Telemetry Modeling & Undercut Strategy Engine

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

## 📌 Executive Summary

In Formula 1 motorsport, millisecond deltas and pit lane timing define race victories. **Formula 1 Race Intelligence** is an end-to-end data science and predictive analytics platform that transforms raw Grand Prix records, qualifying telemetry, and tire stint metrics into actionable strategy intelligence.

Analyzing **1,600+ Grand Prix entries across 4 modern seasons (2021–2024)** and 20 world championship circuits, this platform implements:
1. **Supervised Classification Models**: Predicting podium finishes and race winners with **94.75% accuracy** and **0.982 ROC-AUC** on unseen test data.
2. **Tire Degradation & Pace Regression**: Modeling fuel burn decay vs. tire wear curves across compounds ($R^2 = 0.831$, MAE = $0.13\text{s}$).
3. **Tactical Pit Stop Undercut Simulator**: An algorithmic cross-over calculation determining optimal pit stop windows to leapfrog track rivals.
4. **Interactive Streamlit Dashboard**: Real-time race outcome simulations, driver head-to-head comparisons, and dynamic telemetry visualizations.

---

## 🏆 Key Machine Learning Benchmarks

Chronological train/test split: **2021–2023 Seasons (1,200 entries)** for model training and **2024 Season (400 entries)** for rigorous holdout testing.

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **94.75%** | **80.00%** | **86.67%** | **0.8320** | **0.9822** | 🥇 **Primary Production Model** |
| **Logistic Regression (Baseline)** | 94.75% | 81.97% | 83.33% | 0.8264 | 0.9844 | 🥈 Benchmark |
| **Gradient Boosting Classifier** | 94.25% | 80.33% | 81.67% | 0.8099 | 0.9787 | 🥉 Benchmark |
| **Tire Lap Time Regressor (RF)** | — | — | — | **$R^2 = 0.8310$** | **MAE = $0.132\text{s}$** | 🔧 Stint Degradation Engine |

---

## 📊 Analytical Insights & Visualizations

### 1. Starting Grid to Podium & Win Conversion Rate
Qualifying performance is paramount: pole position ($P1$) yields an **85%+ podium conversion rate**, with front-row starts accounting for over **68% of all race victories**.

<div align="center">
  <img src="./outputs/qualifying_to_podium_matrix.png" alt="Qualifying to Podium Conversion" width="800" />
</div>

### 2. Feature Importance in Grand Prix Outcomes
Feature attribution highlights that starting grid position and circuit overtake difficulty interact heavily. While team tier and rolling constructor points dictate long-run success, driver skill becomes the decisive factor in wet or mixed weather conditions.

<div align="center">
  <img src="./outputs/feature_importance_podium.png" alt="Feature Importance" width="750" />
</div>

### 3. Model ROC-AUC & Classification Evaluation
Strong separability between podium finishers and midfield cars, evidenced by an area under the curve (AUC) of **0.982**.

<div align="center">
  <img src="./outputs/roc_auc_curve.png" alt="ROC Curve" width="48%" />
  <img src="./outputs/model_confusion_matrix.png" alt="Confusion Matrix" width="48%" />
</div>

### 4. Tire Degradation Curves & Fuel Burn Balance
Analysis of lap times over stints reveals the exact inflection lap (the "tire cliff") where lap degradation outpaces fuel burn advantages (~0.033s per lap gained per kg burned).

<div align="center">
  <img src="./outputs/tire_degradation_curves.png" alt="Tire Degradation" width="800" />
</div>

---

## 🗂️ Project Architecture

```
f1-race-intelligence-analytics/
├── data/
│   ├── raw/
│   │   ├── f1_races.csv                 # 1,600 multi-season Grand Prix records
│   │   └── f1_lap_telemetry.csv         # Lap-by-lap tire & fuel telemetry
│   └── processed/
│       └── f1_features.csv              # 39 engineered features for ML models
├── src/
│   ├── data_loader.py                   # Ingestion, schema validation & synthetic telemetry
│   ├── feature_engineering.py           # Rolling form, grid interaction & categorical mapping
│   ├── models.py                        # Model training, benchmarking & metric export
│   └── strategy_simulator.py            # Tactical pit stop undercut/overcut simulation
├── app/
│   └── app.py                           # Interactive Streamlit Race Intelligence Dashboard
├── notebooks/
│   └── f1_data_science_deep_dive.ipynb  # Comprehensive EDA & experimentation notebook
├── outputs/                             # High-resolution charts & serialized model artifacts
│   ├── podium_model.joblib
│   ├── tire_model.joblib
│   ├── feature_importance_podium.png
│   ├── model_confusion_matrix.png
│   ├── qualifying_to_podium_matrix.png
│   ├── roc_auc_curve.png
│   └── tire_degradation_curves.png
├── tests/
│   └── test_pipeline.py                 # Automated unit test suite
├── .github/workflows/
│   └── python-ci.yml                    # Automated GitHub Actions CI pipeline
├── requirements.txt                     # Project dependencies
├── setup.py                             # Package installer configuration
└── LICENSE                              # MIT License
```

---

## ⚡ Quickstart & Installation

### 1. Clone Repository
```bash
git clone https://github.com/DhyeyTeraiya/f1-race-intelligence-analytics.git
cd f1-race-intelligence-analytics
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Pipeline Tests
```bash
python tests/test_pipeline.py
```

### 4. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app/app.py
```
Navigate to `http://localhost:8501` to explore live race predictions, undercut calculations, and tire telemetry!

---

## 💼 Business & Motorsport Applications

- **Race Strategy Optimization**: Quantifies whether to pit early (undercut) or extend the current stint (overcut) based on traffic windows and delta to rival.
- **Predictive Odds & Probability**: Powers sports analytics platforms with calibrated win and podium probabilities updated after qualifying sessions.
- **Tire Wear Cliff Detection**: Alerts race engineers when lap time degradation exceeds tire change pit loss deltas.

---

## 👤 Author

**Dhyey Teraiya** — *Data Scientist & ML Engineer*
- 🌐 **GitHub**: [@DhyeyTeraiya](https://github.com/DhyeyTeraiya)
- 💼 **LinkedIn**: [dhyey-teraiya](https://linkedin.com/in/dhyey-teraiya)
- 📧 **Email**: [dhyeyteraiya@gmail.com](mailto:dhyeyteraiya@gmail.com)

---

⭐ *If you find this project valuable for motorsport analytics or data science research, please consider giving it a star!*
