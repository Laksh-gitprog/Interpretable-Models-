import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = r'C:\Users\Laksh\Downloads\ISBSG_ALL.csv'
RESULTS_DIR = Path('results')
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga'],
    'max_iter': [1000, 5000, 10000],
    'tol': [1e-4, 1e-3, 1e-2]
}

ridge_regressor = Ridge(random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=ridge_regressor,
    param_grid=param_grid,
    cv=CV_FOLDS,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=0,
    return_train_score=True
)

grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
best_cv_rmse = np.sqrt(-grid_search.best_score_)

y_pred = best_model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mre = np.abs((y_test - y_pred) / y_test)
mmre = np.mean(mre) * 100
mdmre = np.median(mre) * 100
sa = np.sum(mre <= 0.25) / len(mre) * 100

print("Performance Metrics:")
print(f"MAE:    {mae:.2f} hours")
print(f"RMSE:   {rmse:.2f} hours")
print(f"R²:     {r2:.4f}")
print(f"MMRE:   {mmre:.2f}%")
print(f"MdMRE:  {mdmre:.2f}%")
print(f"SA:     {sa:.2f}% (PRED(25))")

feature_coefficients = pd.DataFrame({
    'feature': X.columns,
    'coefficient': best_model.coef_,
    'abs_coefficient': np.abs(best_model.coef_)
}).sort_values('abs_coefficient', ascending=False)

model_path = RESULTS_DIR / 'best_ridge_model.pkl'
scaler_path = RESULTS_DIR / 'feature_scaler.pkl'
joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)

metrics = {
    'test_metrics': {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mmre': float(mmre),
        'mdmre': float(mdmre),
        'sa': float(sa)
    },
    'cv_metrics': {
        'best_cv_rmse': float(best_cv_rmse),
        'cv_folds': CV_FOLDS
    },
    'best_params': grid_search.best_params_,
    'data_info': {
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'n_features': int(X.shape[1]),
        'target_mean': float(y.mean()),
        'target_std': float(y.std())
    },
    'model_intercept': float(best_model.intercept_)
}

metrics_path = RESULTS_DIR / 'ridge_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

feature_coefficients_path = RESULTS_DIR / 'ridge_feature_coefficients.csv'
feature_coefficients.to_csv(feature_coefficients_path, index=False)

predictions_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred,
    'error': y_test.values - y_pred,
    'abs_error': np.abs(y_test.values - y_pred),
    'pct_error': np.abs((y_test.values - y_pred) / y_test.values) * 100
})
predictions_path = RESULTS_DIR / 'ridge_test_predictions.csv'
predictions_df.to_csv(predictions_path, index=False)
