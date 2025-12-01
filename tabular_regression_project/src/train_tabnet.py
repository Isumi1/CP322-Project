"""# TabNet training - TODO

TabNet Training Script
Trains TabNet on House Prices and Appliances Energy datasets.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetRegressor


# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def train_tabnet_model(name, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train a single TabNetRegressor on one dataset and return metrics.
    `name` is a short string like 'house' or 'energy'.
    
    Uses validation set for early stopping, test set only for final evaluation.
    """
    # TabNet expects numpy.float32 and y as 2D
    X_train_np = np.array(X_train, dtype=np.float32)
    X_val_np = np.array(X_val, dtype=np.float32)
    X_test_np = np.array(X_test, dtype=np.float32)
    y_train_np = np.array(y_train, dtype=np.float32).reshape(-1, 1)
    y_val_np = np.array(y_val, dtype=np.float32).reshape(-1, 1)
    y_test_np = np.array(y_test, dtype=np.float32)

    model = TabNetRegressor(
        n_d=16,
        n_a=16,
        n_steps=4,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        seed=42,
        verbose=1,
    )

    # Use VALIDATION set for early stopping (not test set!)
    model.fit(
        X_train_np,
        y_train_np,
        eval_set=[(X_val_np, y_val_np)],
        eval_name=["valid"],
        eval_metric=["rmse"],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )

    # Final evaluation on HELD-OUT TEST set (never seen during training)
    y_pred = model.predict(X_test_np).reshape(-1)

    metrics = {
        "Model": "TabNet",
        "Dataset": name,
        "RMSE": rmse(y_test_np, y_pred),
        "MAE": mean_absolute_error(y_test_np, y_pred),
        "R2": r2_score(y_test_np, y_pred),
    }

    # Save model
    model_path = MODELS_DIR / f"tabnet_{name}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    return metrics


def load_house_data():
    """
    Load and split the cleaned house prices dataset.
    Target: SalePrice
    
    IMPORTANT: Uses SAME split as train_xgboost.py:
    - First split: 70% train, 30% temp (test_size=0.3)
    - Second split: 50% val, 50% test from temp (test_size=0.5)
    """
    df = pd.read_csv(DATA_DIR / "house_prices" / "house_prices_cleaned_v2.csv")

    target_col = "SalePrice"
    assert target_col in df.columns, f"{target_col} not in house_prices_cleaned.csv"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Match train_xgboost.py split exactly
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_energy_data():
    """
    Load and split the cleaned appliances energy dataset.
    Target: Appliances
    
    IMPORTANT: Uses SAME split as train_xgboost.py:
    - First split: 70% train, 30% temp (test_size=0.3)
    - Second split: 50% val, 50% test from temp (test_size=0.5)
    """
    df = pd.read_csv(
        DATA_DIR / "appliances_energy" / "appliances_energy_cleaned_v2.csv"
    )

    target_col = "Appliances"
    assert target_col in df.columns, f"{target_col} not in appliances_energy_cleaned.csv"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Match train_xgboost.py split exactly
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    print("=" * 60)
    print("TabNet Training Script")
    print("=" * 60)
    
    metrics_list = []

    # House prices
    print("\n[1/2] Training TabNet on House Prices dataset...")
    X_train_h, X_val_h, X_test_h, y_train_h, y_val_h, y_test_h = load_house_data()
    print(f"  Train: {len(X_train_h)}, Val: {len(X_val_h)}, Test: {len(X_test_h)}")
    house_metrics = train_tabnet_model(
        name="house", 
        X_train=X_train_h, 
        y_train=y_train_h,
        X_val=X_val_h,
        y_val=y_val_h,
        X_test=X_test_h, 
        y_test=y_test_h
    )
    metrics_list.append(house_metrics)
    print(f"  House RMSE: {house_metrics['RMSE']:.2f}, R2: {house_metrics['R2']:.4f}")

    # Energy
    print("\n[2/2] Training TabNet on Appliances Energy dataset...")
    X_train_e, X_val_e, X_test_e, y_train_e, y_val_e, y_test_e = load_energy_data()
    print(f"  Train: {len(X_train_e)}, Val: {len(X_val_e)}, Test: {len(X_test_e)}")
    energy_metrics = train_tabnet_model(
        name="energy", 
        X_train=X_train_e, 
        y_train=y_train_e,
        X_val=X_val_e,
        y_val=y_val_e,
        X_test=X_test_e, 
        y_test=y_test_e
    )
    metrics_list.append(energy_metrics)
    print(f"  Energy RMSE: {energy_metrics['RMSE']:.2f}, R2: {energy_metrics['R2']:.4f}")

    # Save metrics table
    df_metrics = pd.DataFrame(metrics_list)
    out_path = METRICS_DIR / "tabnet_results.csv"
    df_metrics.to_csv(out_path, index=False)
    
    print("\n" + "=" * 60)
    print("TabNet Training Complete!")
    print("=" * 60)
    print("\nResults:")
    print(df_metrics.to_string(index=False))
    print(f"\nMetrics saved to: {out_path}")


if __name__ == "__main__":
    main()
