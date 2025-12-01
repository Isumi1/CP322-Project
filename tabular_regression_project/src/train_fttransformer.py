"""# FT-Transformer training - TODO

FT-Transformer Training Script
Trains FT-Transformer (Feature Tokenizer + Transformer) on House Prices and Appliances Energy datasets.
Implemented from scratch using PyTorch.
"""

import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Reproducibility
# ============================================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper Functions
# ============================================================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ============================================================
# Data Loading Functions
# ============================================================
def load_house_data():
    """
    Load and split the cleaned house prices dataset.
    Target: SalePrice
    
    Split: 70% train, 15% val, 15% test (random_state=42)
    """
    df = pd.read_csv(DATA_DIR / "house_prices" / "house_prices_cleaned_v2.csv")

    target_col = "SalePrice"
    assert target_col in df.columns, f"{target_col} not in house_prices_cleaned.csv"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Same split as train_xgboost.py and train_tabnet.py
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
    
    Split: 70% train, 15% val, 15% test (random_state=42)
    """
    df = pd.read_csv(DATA_DIR / "appliances_energy" / "appliances_energy_cleaned_v2.csv")

    target_col = "Appliances"
    assert target_col in df.columns, f"{target_col} not in appliances_energy_cleaned.csv"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Same split as train_xgboost.py and train_tabnet.py
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# FT-Transformer Model
# ============================================================
class FTTransformerRegressor(nn.Module):
    """
    FT-Transformer style regressor for tabular data.
    
    Architecture:
    1. Feature projection: Linear layer to embed all features into d_model
    2. Transformer encoder: Self-attention layers
    3. Regression head: Linear layer to predict target
    """
    
    def __init__(self, num_features, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Feature tokenizer: project all features to d_model dimensions
        self.feature_proj = nn.Linear(num_features, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Regression head
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x: (batch, num_features)
        x = self.feature_proj(x)        # (batch, d_model)
        x = x.unsqueeze(1)              # (batch, 1, d_model) - sequence of length 1
        x = self.encoder(x)             # (batch, 1, d_model)
        x = x.mean(dim=1)               # (batch, d_model)
        out = self.head(x)              # (batch, 1)
        return out.squeeze(-1)          # (batch,)


# ============================================================
# Training Function
# ============================================================
def train_fttransformer_model(name, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train FT-Transformer on one dataset and return metrics dict.
    
    Args:
        name: 'house' or 'energy'
        X_train, y_train: Training data
        X_val, y_val: Validation data (for early stopping)
        X_test, y_test: Test data (for final evaluation only)
    
    Returns:
        dict with Model, Dataset, RMSE, MAE, R2
    """
    # Convert pandas to torch tensors
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    # Create DataLoaders
    batch_size = 256
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    num_features = X_train.shape[1]
    model = FTTransformerRegressor(num_features=num_features)
    model.to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training hyperparameters
    max_epochs = 200
    patience = 20
    
    # Early stopping tracking
    best_val_rmse = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    print(f"  Training for up to {max_epochs} epochs (patience={patience})...")
    
    for epoch in range(max_epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # ---- Validation ----
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_pred = model(X_batch)
                val_preds.append(y_pred.cpu().numpy())
                val_targets.append(y_batch.numpy())
        
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_rmse = rmse(val_targets, val_preds)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.2f}")
        
        # Early stopping check
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"    Early stopping at epoch {epoch+1} (best epoch: {epoch+1-patience})")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"    Best validation RMSE: {best_val_rmse:.2f}")
    
    # ---- Final Evaluation on TEST set ----
    model.eval()
    with torch.no_grad():
        X_test_t = X_test_t.to(device)
        y_pred_test = model(X_test_t).cpu().numpy()
    
    y_test_np = y_test.values
    
    test_rmse = rmse(y_test_np, y_pred_test)
    test_mae = mean_absolute_error(y_test_np, y_pred_test)
    test_r2 = r2_score(y_test_np, y_pred_test)
    
    # Save model
    model_path = MODELS_DIR / f"fttransformer_{name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved to: {model_path}")
    
    # Also save with joblib for consistency with other scripts
    joblib_path = MODELS_DIR / f"fttransformer_{name}.pkl"
    joblib.dump({
        'state_dict': model.state_dict(),
        'num_features': num_features,
    }, joblib_path)
    
    metrics = {
        "Model": "FT-Transformer",
        "Dataset": name,
        "RMSE": test_rmse,
        "MAE": test_mae,
        "R2": test_r2,
    }
    
    return metrics


# ============================================================
# Main Function
# ============================================================
def main():
    print("=" * 60)
    print("FT-Transformer Training Script")
    print("=" * 60)
    
    metrics_list = []
    
    # 1) House dataset
    print("\n[1/2] Training FT-Transformer on House Prices dataset...")
    X_train_h, X_val_h, X_test_h, y_train_h, y_val_h, y_test_h = load_house_data()
    print(f"  Train: {len(X_train_h)}, Val: {len(X_val_h)}, Test: {len(X_test_h)}")
    print(f"  Features: {X_train_h.shape[1]}")
    
    house_metrics = train_fttransformer_model(
        name="house",
        X_train=X_train_h,
        y_train=y_train_h,
        X_val=X_val_h,
        y_val=y_val_h,
        X_test=X_test_h,
        y_test=y_test_h,
    )
    metrics_list.append(house_metrics)
    print(f"  House RMSE: {house_metrics['RMSE']:.2f}, R2: {house_metrics['R2']:.4f}")
    
    # 2) Energy dataset
    print("\n[2/2] Training FT-Transformer on Appliances Energy dataset...")
    X_train_e, X_val_e, X_test_e, y_train_e, y_val_e, y_test_e = load_energy_data()
    print(f"  Train: {len(X_train_e)}, Val: {len(X_val_e)}, Test: {len(X_test_e)}")
    print(f"  Features: {X_train_e.shape[1]}")
    
    energy_metrics = train_fttransformer_model(
        name="energy",
        X_train=X_train_e,
        y_train=y_train_e,
        X_val=X_val_e,
        y_val=y_val_e,
        X_test=X_test_e,
        y_test=y_test_e,
    )
    metrics_list.append(energy_metrics)
    print(f"  Energy RMSE: {energy_metrics['RMSE']:.2f}, R2: {energy_metrics['R2']:.4f}")
    
    # Save metrics as CSV
    df_metrics = pd.DataFrame(metrics_list)
    out_path = METRICS_DIR / "fttransformer_results.csv"
    df_metrics.to_csv(out_path, index=False)
    
    print("\n" + "=" * 60)
    print("FT-Transformer Training Complete!")
    print("=" * 60)
    print("\nResults:")
    print(df_metrics.to_string(index=False))
    print(f"\nMetrics saved to: {out_path}")


if __name__ == "__main__":
    main()
