import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import optuna
import pickle

df_house = pd.read_csv('../data/house_prices/house_prices_cleaned_v2.csv')
df_energy = pd.read_csv('../data/appliances_energy/appliances_energy_cleaned_v2.csv')

X_house = df_house.drop('SalePrice', axis=1)
y_house = df_house['SalePrice']
X_energy = df_energy.drop('Appliances', axis=1)
y_energy = df_energy['Appliances']

X_train_h, X_temp_h, y_train_h, y_temp_h = train_test_split(X_house, y_house, test_size=0.3, random_state=42)
X_val_h, X_test_h, y_val_h, y_test_h = train_test_split(X_temp_h, y_temp_h, test_size=0.5, random_state=42)

X_train_e, X_temp_e, y_train_e, y_temp_e = train_test_split(X_energy, y_energy, test_size=0.3, random_state=42)
X_val_e, X_test_e, y_val_e, y_test_e = train_test_split(X_temp_e, y_temp_e, test_size=0.5, random_state=42)

# Linear Regression - House
lr_house = LinearRegression()
lr_house.fit(X_train_h, y_train_h)
y_pred_lr_h = lr_house.predict(X_test_h)
rmse_lr_h = np.sqrt(mean_squared_error(y_test_h, y_pred_lr_h))
r2_lr_h = r2_score(y_test_h, y_pred_lr_h)

# Linear Regression - Energy
lr_energy = LinearRegression()
lr_energy.fit(X_train_e, y_train_e)
y_pred_lr_e = lr_energy.predict(X_test_e)
rmse_lr_e = np.sqrt(mean_squared_error(y_test_e, y_pred_lr_e))
r2_lr_e = r2_score(y_test_e, y_pred_lr_e)

# XGBoost Default - House
xgb_house_default = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_house_default.fit(X_train_h, y_train_h)
y_pred_xgb_h = xgb_house_default.predict(X_test_h)
rmse_xgb_h = np.sqrt(mean_squared_error(y_test_h, y_pred_xgb_h))
r2_xgb_h = r2_score(y_test_h, y_pred_xgb_h)

# XGBoost Default - Energy
xgb_energy_default = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_energy_default.fit(X_train_e, y_train_e)
y_pred_xgb_e = xgb_energy_default.predict(X_test_e)
rmse_xgb_e = np.sqrt(mean_squared_error(y_test_e, y_pred_xgb_e))
r2_xgb_e = r2_score(y_test_e, y_pred_xgb_e)

# Optuna - House
def objective_house(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42, 'n_jobs': -1
    }
    model = XGBRegressor(**params)
    model.fit(X_train_h, y_train_h)
    y_pred = model.predict(X_val_h)
    return np.sqrt(mean_squared_error(y_val_h, y_pred))

study_house = optuna.create_study(direction='minimize')
study_house.optimize(objective_house, n_trials=50, show_progress_bar=True)

xgb_house_tuned = XGBRegressor(**study_house.best_params, random_state=42, n_jobs=-1)
xgb_house_tuned.fit(X_train_h, y_train_h)
y_pred_tuned_h = xgb_house_tuned.predict(X_test_h)
rmse_tuned_h = np.sqrt(mean_squared_error(y_test_h, y_pred_tuned_h))
r2_tuned_h = r2_score(y_test_h, y_pred_tuned_h)

# Optuna - Energy
def objective_energy(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42, 'n_jobs': -1
    }
    model = XGBRegressor(**params)
    model.fit(X_train_e, y_train_e)
    y_pred = model.predict(X_val_e)
    return np.sqrt(mean_squared_error(y_val_e, y_pred))

study_energy = optuna.create_study(direction='minimize')
study_energy.optimize(objective_energy, n_trials=50, show_progress_bar=True)

xgb_energy_tuned = XGBRegressor(**study_energy.best_params, random_state=42, n_jobs=-1)
xgb_energy_tuned.fit(X_train_e, y_train_e)
y_pred_tuned_e = xgb_energy_tuned.predict(X_test_e)
rmse_tuned_e = np.sqrt(mean_squared_error(y_test_e, y_pred_tuned_e))
r2_tuned_e = r2_score(y_test_e, y_pred_tuned_e)

# Results
results = pd.DataFrame({
    'Model': ['Linear Regression', 'XGBoost (Default)', 'XGBoost (Tuned)'],
    'House_RMSE': [rmse_lr_h, rmse_xgb_h, rmse_tuned_h],
    'House_R2': [r2_lr_h, r2_xgb_h, r2_tuned_h],
    'Energy_RMSE': [rmse_lr_e, rmse_xgb_e, rmse_tuned_e],
    'Energy_R2': [r2_lr_e, r2_xgb_e, r2_tuned_e]
})

print("\n" + results.to_string(index=False))
results.to_csv('../artifacts/metrics/xgboost_results.csv', index=False)

# Save models
with open('../artifacts/models/xgb_house_tuned.pkl', 'wb') as f:
    pickle.dump(xgb_house_tuned, f)
with open('../artifacts/models/xgb_energy_tuned.pkl', 'wb') as f:
    pickle.dump(xgb_energy_tuned, f)

X_test_h.to_csv('../artifacts/metrics/house_test_X.csv', index=False)
y_test_h.to_csv('../artifacts/metrics/house_test_y.csv', index=False)
X_test_e.to_csv('../artifacts/metrics/energy_test_X.csv', index=False)
y_test_e.to_csv('../artifacts/metrics/energy_test_y.csv', index=False)
