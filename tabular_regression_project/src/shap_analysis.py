"""# SHAP analysis - TODO

SHAP Analysis Script
Generates SHAP plots and feature importance analysis for trained XGBoost models.

This script:
1. Loads trained XGBoost models (house and energy)
2. Computes SHAP values using TreeExplainer
3. Generates various SHAP visualizations:
   - Feature importance bar plots
   - Beeswarm plots
   - Waterfall plots (for individual predictions)
   - Dependence plots (top 3 features)
4. Saves feature importance CSVs and summary report
"""

from pathlib import Path
from typing import Tuple
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# ============================================================
# Paths (Main Repo Structure)
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

# Ensure plots directory exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
HOUSE_MODEL_PATH = MODELS_DIR / "xgb_house_tuned.pkl"
ENERGY_MODEL_PATH = MODELS_DIR / "xgb_energy_tuned.pkl"

# Test set paths (saved by train_xgboost.py)
HOUSE_TEST_X = METRICS_DIR / "house_test_X.csv"
HOUSE_TEST_Y = METRICS_DIR / "house_test_y.csv"
ENERGY_TEST_X = METRICS_DIR / "energy_test_X.csv"
ENERGY_TEST_Y = METRICS_DIR / "energy_test_y.csv"

# Full dataset paths (for background sampling)
HOUSE_FULL = DATA_DIR / "house_prices" / "house_prices_cleaned_v2.csv"
ENERGY_FULL = DATA_DIR / "appliances_energy" / "appliances_energy_cleaned_v2.csv"

# Set plot style
plt.style.use("seaborn-v0_8-darkgrid")


# ============================================================
# Data Loading Functions
# ============================================================
def load_artifacts() -> Tuple:
    """
    Load trained models and test datasets.
    
    Returns:
        Tuple of (house_model, energy_model, X_test_house, X_test_energy, X_house_full, X_energy_full)
    """
    print("Loading models and data...")
    
    # Load models
    with open(HOUSE_MODEL_PATH, "rb") as f:
        house_model = pickle.load(f)
    print(f"  Loaded: {HOUSE_MODEL_PATH.name}")
    
    with open(ENERGY_MODEL_PATH, "rb") as f:
        energy_model = pickle.load(f)
    print(f"  Loaded: {ENERGY_MODEL_PATH.name}")
    
    # Load test sets
    X_test_house = pd.read_csv(HOUSE_TEST_X)
    X_test_energy = pd.read_csv(ENERGY_TEST_X)
    print(f"  Loaded test sets: House ({len(X_test_house)}), Energy ({len(X_test_energy)})")
    
    # Load full datasets for background sampling
    X_house_full = pd.read_csv(HOUSE_FULL).drop(columns=["SalePrice"])
    X_energy_full = pd.read_csv(ENERGY_FULL).drop(columns=["Appliances"])
    print(f"  Loaded full datasets for background sampling")
    
    return house_model, energy_model, X_test_house, X_test_energy, X_house_full, X_energy_full


# ============================================================
# SHAP Computation Functions
# ============================================================
def shap_explainer(model, background: pd.DataFrame, X_test: pd.DataFrame):
    """
    Create SHAP explainer and compute SHAP values.
    
    Uses the model's predict function with a masker for compatibility
    with newer XGBoost versions.
    
    Args:
        model: Trained XGBoost model
        background: Background dataset for SHAP (sampled from training data)
        X_test: Test set to explain
    
    Returns:
        SHAP Explanation object
    """
    # Use Explainer with model's predict function for XGBoost 2.0+ compatibility
    explainer = shap.Explainer(model.predict, background)
    return explainer(X_test)


# ============================================================
# Plotting Functions
# ============================================================
def summary_plots(name: str, shap_values, X_test: pd.DataFrame):
    """
    Generate SHAP summary plots (bar and beeswarm).
    
    Args:
        name: Dataset name ('House' or 'Energy')
        shap_values: SHAP Explanation object
        X_test: Test features DataFrame
    """
    # Bar plot (feature importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance - {name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = PLOTS_DIR / f"shap_{name.lower()}_importance_bar.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")

    # Beeswarm plot (feature impact distribution)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP Feature Impact - {name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = PLOTS_DIR / f"shap_{name.lower()}_beeswarm.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


def waterfall_plot(name: str, shap_values):
    """
    Generate SHAP waterfall plot for the first prediction.
    
    Args:
        name: Dataset name ('House' or 'Energy')
        shap_values: SHAP Explanation object
    """
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title(f"SHAP Waterfall - {name} (First Prediction)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = PLOTS_DIR / f"shap_{name.lower()}_waterfall.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


def dependence_plots(name: str, shap_values, X_test: pd.DataFrame) -> np.ndarray:
    """
    Generate SHAP dependence plots for top 3 features.
    
    Args:
        name: Dataset name ('House' or 'Energy')
        shap_values: SHAP Explanation object
        X_test: Test features DataFrame
    
    Returns:
        Array of feature importance values
    """
    # Get top 3 features by importance
    feature_importance = np.abs(shap_values.values).mean(0)
    top_idx = np.argsort(feature_importance)[-3:][::-1]
    top_features = X_test.columns[top_idx]

    # Create subplots for top 3 features
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, feature in enumerate(top_features):
        plt.sca(axes[idx])
        shap.dependence_plot(feature, shap_values.values, X_test, show=False, ax=axes[idx])
        axes[idx].set_title(f"Dependence: {feature}", fontsize=12, fontweight="bold")
    
    plt.suptitle(f"SHAP Dependence Plots - {name} (Top 3 Features)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = PLOTS_DIR / f"shap_{name.lower()}_dependence.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")

    return feature_importance


def importance_tables(shap_values, X_test: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Create feature importance table from SHAP values.
    
    Args:
        shap_values: SHAP Explanation object
        X_test: Test features DataFrame
        top_n: Number of top features to include
    
    Returns:
        DataFrame with Feature and Importance columns
    """
    return (
        pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": np.abs(shap_values.values).mean(0)
        })
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )


def comparison_plot(house_df: pd.DataFrame, energy_df: pd.DataFrame):
    """
    Generate side-by-side comparison of feature importance across datasets.
    
    Args:
        house_df: House prices feature importance DataFrame
        energy_df: Energy feature importance DataFrame
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # House prices
    axes[0].barh(range(len(house_df)), house_df["Importance"], color="steelblue")
    axes[0].set_yticks(range(len(house_df)))
    axes[0].set_yticklabels(house_df["Feature"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean |SHAP Value|")
    axes[0].set_title("House Prices - Top Features", fontweight="bold")
    axes[0].grid(axis="x", alpha=0.3)

    # Energy
    axes[1].barh(range(len(energy_df)), energy_df["Importance"], color="coral")
    axes[1].set_yticks(range(len(energy_df)))
    axes[1].set_yticklabels(energy_df["Feature"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean |SHAP Value|")
    axes[1].set_title("Energy Consumption - Top Features", fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3)

    plt.suptitle("Feature Importance Comparison Across Datasets", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = PLOTS_DIR / "shap_comparison_datasets.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# Summary Report Functions
# ============================================================
def write_summary(house_df: pd.DataFrame, energy_df: pd.DataFrame):
    """
    Write a text summary report of SHAP analysis.
    
    Args:
        house_df: House prices feature importance DataFrame
        energy_df: Energy feature importance DataFrame
    """
    summary_path = METRICS_DIR / "shap_analysis_summary.txt"
    
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("SHAP ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("HOUSE PRICES DATASET\n")
        f.write("-" * 60 + "\n")
        f.write("Top 10 Most Important Features:\n\n")
        f.write(house_df.to_string(index=False))
        f.write("\n\n")

        f.write("ENERGY CONSUMPTION DATASET\n")
        f.write("-" * 60 + "\n")
        f.write("Top 10 Most Important Features:\n\n")
        f.write(energy_df.to_string(index=False))
        f.write("\n\n")

        f.write("KEY INSIGHTS\n")
        f.write("-" * 60 + "\n")
        f.write("1. Feature importance differs across domains; overlap suggests shared drivers.\n")
        f.write("2. SHAP magnitudes show contribution strength; sign shows direction.\n")
        f.write("3. Dependence plots surface non-linear effects worth testing in ablations.\n")
        f.write("4. Waterfall plots reveal individual prediction breakdowns for error analysis.\n")
    
    print(f"  Saved: {summary_path.name}")


# ============================================================
# Main Analysis Function
# ============================================================
def run_shap_for_dataset(name: str, model, background: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Run complete SHAP analysis for one dataset.
    
    Args:
        name: Dataset name ('House' or 'Energy')
        model: Trained XGBoost model
        background: Background samples for SHAP
        X_test: Test set features
    
    Returns:
        Tuple of (importance_df, feature_importance_array)
    """
    print(f"\n  Computing SHAP values for {name}...")
    shap_values = shap_explainer(model, background, X_test)
    
    print(f"  Generating plots for {name}...")
    summary_plots(name, shap_values, X_test)
    waterfall_plot(name, shap_values)
    feature_importance = dependence_plots(name, shap_values, X_test)
    
    # Create and save importance table
    importance_df = importance_tables(shap_values, X_test)
    csv_path = METRICS_DIR / f"shap_{name.lower()}_top_features.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path.name}")
    
    return importance_df, feature_importance


def main():
    """Main function to run SHAP analysis on both datasets."""
    print("=" * 60)
    print("SHAP Analysis Script")
    print("=" * 60)
    
    # Load all artifacts
    (
        house_model,
        energy_model,
        X_test_house,
        X_test_energy,
        X_house_full,
        X_energy_full,
    ) = load_artifacts()

    # Sample background data for SHAP (100 samples from full dataset)
    print("\nSampling background data...")
    background_house = shap.sample(X_house_full, 100, random_state=42)
    background_energy = shap.sample(X_energy_full, 100, random_state=42)
    print(f"  House background: {len(background_house)} samples")
    print(f"  Energy background: {len(background_energy)} samples")

    # Run SHAP analysis for each dataset
    print("\n" + "-" * 60)
    print("Analyzing House Prices Dataset")
    print("-" * 60)
    house_importance_df, _ = run_shap_for_dataset(
        "House", house_model, background_house, X_test_house
    )

    print("\n" + "-" * 60)
    print("Analyzing Energy Consumption Dataset")
    print("-" * 60)
    energy_importance_df, _ = run_shap_for_dataset(
        "Energy", energy_model, background_energy, X_test_energy
    )

    # Generate comparison plot and summary
    print("\n" + "-" * 60)
    print("Generating Comparison and Summary")
    print("-" * 60)
    comparison_plot(house_importance_df, energy_importance_df)
    write_summary(house_importance_df, energy_importance_df)

    # Print summary to console
    print("\n" + "=" * 60)
    print("SHAP Analysis Complete!")
    print("=" * 60)
    
    print("\nHouse Prices - Top 5 Features:")
    print(house_importance_df.head().to_string(index=False))
    
    print("\nEnergy - Top 5 Features:")
    print(energy_importance_df.head().to_string(index=False))
    
    print(f"\nPlots saved to: {PLOTS_DIR}")
    print(f"Metrics saved to: {METRICS_DIR}")


if __name__ == "__main__":
    main()
