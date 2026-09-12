"""
Formula 1 Machine Learning & Predictive Modeling Engine
Trains, evaluates, and exports classification models for race podium / winner prediction
and regression models for tire degradation & stint lap time projection.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from feature_engineering import get_modeling_data, load_raw_data

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Set visual styling for charts
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.sans-serif": "Arial", "font.family": "sans-serif"})


def train_and_evaluate_classification():
    data = get_modeling_data()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train_podium"], data["y_test_podium"]
    feature_cols = data["feature_cols"]

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("clf", GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42))
        ])
    }

    results = {}
    best_f1 = -1
    best_model_name = None
    best_pipeline = None

    print("\n" + "="*60)
    print(" FORMULA 1 PODIUM PREDICTION MODEL BENCHMARK ")
    print("="*60)

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "pipeline": pipeline
        }

        print(f"[{name}]")
        print(f"  • Accuracy : {acc:.4f} | Precision: {prec:.4f}")
        print(f"  • Recall   : {rec:.4f} | F1-Score : {f1:.4f} | ROC-AUC: {auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = pipeline

    # Save Best Model
    model_save_path = os.path.join(OUTPUTS_DIR, "podium_model.joblib")
    joblib.dump({"pipeline": best_pipeline, "features": feature_cols, "name": best_model_name}, model_save_path)
    print(f"\nBest Classifier ({best_model_name}) saved to {model_save_path}")

    # Generate Visualizations
    generate_roc_curve(results, y_test)
    generate_confusion_matrix(results[best_model_name]["y_pred"], y_test, best_model_name)
    generate_feature_importance(best_pipeline, feature_cols, best_model_name)

    return results, best_model_name


def generate_roc_curve(results, y_test):
    plt.figure(figsize=(8, 6), dpi=150)
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['ROC-AUC']:.3f})", lw=2)

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Random Guess")
    plt.title("Formula 1 Podium Prediction — ROC-AUC Comparison", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "roc_auc_curve.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Exported ROC curve to {out_path}")


def generate_confusion_matrix(y_pred, y_test, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5), dpi=150)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Podium", "Podium (P1-P3)"],
                yticklabels=["No Podium", "Podium (P1-P3)"])
    plt.title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Outcome", fontsize=11)
    plt.ylabel("Actual Race Result", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "model_confusion_matrix.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Exported Confusion Matrix to {out_path}")


def generate_feature_importance(pipeline, feature_cols, model_name):
    clf = pipeline.named_steps.get("clf")
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return

    feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    feat_df.sort_values(by="Importance", ascending=True, inplace=True)

    plt.figure(figsize=(9, 6), dpi=150)
    plt.barh(feat_df["Feature"], feat_df["Importance"], color="#E10600", alpha=0.85, edgecolor="#8B0000")
    plt.title(f"Key Driver Performance Drivers ({model_name})", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Normalized Relative Importance", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "feature_importance_podium.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Exported Feature Importance chart to {out_path}")


def train_tire_degradation_model():
    _, df_telemetry = load_raw_data()
    print("\n" + "="*60)
    print(" TIRE DEGRADATION & LAP TIME REGRESSION ENGINE ")
    print("="*60)

    # Convert categorical compound to one-hot
    df = pd.get_dummies(df_telemetry, columns=["compound"], drop_first=False)
    compound_cols = [c for c in df.columns if c.startswith("compound_")]

    feature_cols = ["tire_age_laps", "fuel_load_kg"] + compound_cols
    target = "lap_time_sec"

    X = df[feature_cols]
    y = df[target]

    reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    reg.fit(X, y)
    y_pred = reg.predict(X)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"Tire Lap Time Model:")
    print(f"  • R² Score : {r2:.4f}")
    print(f"  • MAE      : {mae:.4f} seconds")

    # Save Regressor
    reg_save_path = os.path.join(OUTPUTS_DIR, "tire_model.joblib")
    joblib.dump({"model": reg, "features": feature_cols, "compounds": compound_cols}, reg_save_path)
    print(f"Tire Regressor saved to {reg_save_path}")

    # Plot Tire Degradation Curves
    plot_tire_degradation(df_telemetry)
    return reg


def plot_tire_degradation(df_telemetry):
    plt.figure(figsize=(10, 6), dpi=150)
    colors = {"SOFT": "#FF1801", "MEDIUM": "#FFD700", "HARD": "#F0F0F0"}

    for compound in ["MEDIUM", "HARD"]:
        subset = df_telemetry[df_telemetry["compound"] == compound]
        avg_deg = subset.groupby("tire_age_laps")["lap_time_sec"].mean().reset_index()
        plt.plot(avg_deg["tire_age_laps"], avg_deg["lap_time_sec"], label=f"{compound} Compound",
                 color=colors.get(compound, "#333333"), lw=2.5, marker="o", markersize=4)

    plt.title("Formula 1 Pace Evolution: Fuel Burn vs. Tire Degradation", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Tire Age (Laps Completed in Stint)", fontsize=11)
    plt.ylabel("Lap Time (seconds)", fontsize=11)
    plt.legend(frameon=True, fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "tire_degradation_curves.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Exported Tire Degradation curve to {out_path}")


def generate_qualifying_podium_matrix():
    data = get_modeling_data()
    df = data["df_full"]

    # Calculate probability of podium and win from each starting grid position (1 to 10)
    grid_analysis = df[df["grid_position"] <= 10].groupby("grid_position").agg(
        Total_Starts=("race_id", "count"),
        Podium_Count=("podium_finish", "sum"),
        Win_Count=("race_winner", "sum")
    ).reset_index()

    grid_analysis["Podium_Conversion_Pct"] = (grid_analysis["Podium_Count"] / grid_analysis["Total_Starts"]) * 100
    grid_analysis["Win_Conversion_Pct"] = (grid_analysis["Win_Count"] / grid_analysis["Total_Starts"]) * 100

    plt.figure(figsize=(10, 5), dpi=150)
    bar_width = 0.38
    x = np.arange(len(grid_analysis["grid_position"]))

    plt.bar(x - bar_width/2, grid_analysis["Podium_Conversion_Pct"], width=bar_width,
            label="Podium Conversion % (P1-P3)", color="#E10600", alpha=0.85)
    plt.bar(x + bar_width/2, grid_analysis["Win_Conversion_Pct"], width=bar_width,
            label="Win Conversion % (P1)", color="#1E41FF", alpha=0.85)

    plt.xticks(x, [f"P{pos}" for pos in grid_analysis["grid_position"]], fontsize=10)
    plt.title("Grid Position vs. Race Success Conversion Rate", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Starting Grid Position", fontsize=11)
    plt.ylabel("Historical Conversion Rate (%)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()

    out_path = os.path.join(OUTPUTS_DIR, "qualifying_to_podium_matrix.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Exported Qualifying-to-Podium matrix to {out_path}")


if __name__ == "__main__":
    train_and_evaluate_classification()
    train_tire_degradation_model()
    generate_qualifying_podium_matrix()
