import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    mre = np.abs((y_true - y_pred) / y_true)
    mmre = np.mean(mre) * 100
    mdmre = np.median(mre) * 100
    
    pred_25 = np.sum(mre <= 0.25) / len(mre) * 100
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mmre': float(mmre),
        'mdmre': float(mdmre),
        'sa': float(pred_25)
    }

print("Loading data...")
df = pd.read_csv(r"C:\Users\Laksh\Downloads\Desharnais.csv")
print(f"Dataset shape: {df.shape}")

X = df.drop('Effort', axis=1)
y = df['Effort']
print(f"Features: {X.shape[1]}")
print(f"Target variable: Effort")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nStarting Grid Search for MLP...")

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (100, 50, 25), (100, 100)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'lbfgs', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [500, 1000]
}

mlp = MLPRegressor(random_state=42, early_stopping=True)
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_root_mean_squared_error', 
                          n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

print("\nBest parameters found:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

print("\nEvaluating on test set...")
y_pred = best_model.predict(X_test_scaled)

test_metrics = calculate_metrics(y_test.values, y_pred)

print("\n" + "="*60)
print("MLP (Multi-Layer Perceptron) - Performance Metrics")
print("="*60)
print(f"MAE:    {test_metrics['mae']:.2f} hours")
print(f"RMSE:   {test_metrics['rmse']:.2f} hours")
print(f"R²:     {test_metrics['r2']:.4f}")
print(f"MMRE:   {test_metrics['mmre']:.2f}%")
print(f"MdMRE:  {test_metrics['mdmre']:.2f}%")
print(f"SA:     {test_metrics['sa']:.2f}% (PRED(25))")
print(f"Best CV RMSE: {-grid_search.best_score_:.2f}")
print("="*60)

output_dir = Path('results/mlp')
output_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, output_dir / 'model.pkl')
joblib.dump(scaler, output_dir / 'scaler.pkl')
print(f"\nResults saved to: {output_dir}")
print("Model saved")
print("Scaler saved")

metrics = {
    'model': 'MLP',
    'best_params': grid_search.best_params_,
    'test_metrics': test_metrics,
    'cv_metrics': {
        'best_cv_rmse': float(-grid_search.best_score_)
    }
}

with open(output_dir / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved")

predictions_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred
})
predictions_df.to_csv(output_dir / 'test_predictions.csv', index=False)
print("Predictions saved")
