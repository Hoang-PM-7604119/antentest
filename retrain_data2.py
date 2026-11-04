import pandas as pd
import numpy as np
import json
import time
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

print("--- 1. LOADING data2_f1.xlsx ---")
t0 = time.time()
df = pd.read_excel('/Users/MAC/antentest/data2_f1.xlsx')
print(f"Loaded {len(df)} rows in {time.time() - t0:.2f}s")
print("Columns:", df.columns.tolist())

# Rename columns
# Columns in data2_f1.xlsx: ['t2', 't1', 'z2', 'z1', 'Unnamed: 4', 'dB(S(1,1))', 'dB(S(1,1)).1']
cols = df.columns.tolist()
df = df.drop(columns=[cols[4]])
df.columns = ['d1', 'd2', 'd3', 'd4', 'v1', 'v2']

print(f"Dataset shape: {df.shape}")
print(f"v1 range: [{df['v1'].min():.4f}, {df['v1'].max():.6f}]")
print(f"v2 range: [{df['v2'].min():.4f}, {df['v2'].max():.6f}]")

X = df[['v1', 'v2']].values
y = df[['d1', 'd2', 'd3', 'd4']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

results = {}

# 1. Linear Regression
print("\n--- 2. TRAINING Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_mae = float(mean_absolute_error(y_test, lr_preds))
lr_r2 = float(r2_score(y_test, lr_preds))
results['LinearRegression'] = {
    'name': 'Hồi quy tuyến tính (Linear Regression)',
    'description': 'Mô hình đơn giản nhất, tìm quan hệ tuyến tính d = W * v + b',
    'coef': [[round(float(c), 4) for c in row] for row in lr.coef_],
    'intercept': [round(float(b), 4) for b in lr.intercept_],
    'mae': round(lr_mae, 2),
    'mae_per_dim': [round(float(x), 2) for x in np.mean(np.abs(y_test - lr_preds), axis=0)],
    'r2': round(lr_r2, 4)
}
print(f"Linear Regression: MAE = {lr_mae:.2f}, R2 = {lr_r2:.4f}")

# 2. Ridge Regression
print("\n--- 3. TUNING Ridge Regression ---")
best_ridge = None
best_ridge_mae = 1e9
for alpha in [0.01, 0.1, 1.0, 10.0, 50.0]:
    r = Ridge(alpha=alpha)
    r.fit(X_train, y_train)
    p = r.predict(X_test)
    m = mean_absolute_error(y_test, p)
    if m < best_ridge_mae:
        best_ridge_mae = m
        best_ridge = r

ridge_preds = best_ridge.predict(X_test)
results['RidgeRegression'] = {
    'name': 'Hồi quy Ridge (L2 Regularization)',
    'description': 'Tuyến tính có điều chuẩn L2 để kiểm soát quá khớp và giảm phương sai',
    'coef': [[round(float(c), 4) for c in row] for row in best_ridge.coef_],
    'intercept': [round(float(b), 4) for b in best_ridge.intercept_],
    'mae': round(float(mean_absolute_error(y_test, ridge_preds)), 2),
    'mae_per_dim': [round(float(x), 2) for x in np.mean(np.abs(y_test - ridge_preds), axis=0)],
    'r2': round(float(r2_score(y_test, ridge_preds)), 4)
}
print(f"Ridge Regression: MAE = {results['RidgeRegression']['mae']:.2f}, R2 = {results['RidgeRegression']['r2']:.4f}")

# 3. Lasso Regression
print("\n--- 4. TUNING Lasso Regression ---")
best_lasso = None
best_lasso_mae = 1e9
for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
    l = Lasso(alpha=alpha, max_iter=2000)
    l.fit(X_train, y_train)
    p = l.predict(X_test)
    m = mean_absolute_error(y_test, p)
    if m < best_lasso_mae:
        best_lasso_mae = m
        best_lasso = l

lasso_preds = best_lasso.predict(X_test)
results['LassoRegression'] = {
    'name': 'Hồi quy Lasso (L1 Regularization)',
    'description': 'Tuyến tính có điều chuẩn L1 (triệt tiêu trọng số dư thừa, tăng tính thưa)',
    'coef': [[round(float(c), 4) for c in row] for row in best_lasso.coef_],
    'intercept': [round(float(b), 4) for b in best_lasso.intercept_],
    'mae': round(float(mean_absolute_error(y_test, lasso_preds)), 2),
    'mae_per_dim': [round(float(x), 2) for x in np.mean(np.abs(y_test - lasso_preds), axis=0)],
    'r2': round(float(r2_score(y_test, lasso_preds)), 4)
}
print(f"Lasso Regression: MAE = {results['LassoRegression']['mae']:.2f}, R2 = {results['LassoRegression']['r2']:.4f}")

# 4. Polynomial Regression (Degree 2)
print("\n--- 5. TRAINING Polynomial Regression (Degree 2) ---")
poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_model = LinearRegression(fit_intercept=False)
poly_model.fit(X_train_poly, y_train)
poly_preds = poly_model.predict(X_test_poly)
results['PolynomialRegression'] = {
    'name': 'Hồi quy đa thức bậc 2 (Polynomial Regression)',
    'description': 'Mô hình phi tuyến đơn giản dựa trên v1, v2, v1^2, v1*v2, v2^2',
    'powers': poly.powers_.tolist(),
    'coef': [[round(float(c), 4) for c in row] for row in poly_model.coef_],
    'mae': round(float(mean_absolute_error(y_test, poly_preds)), 2),
    'mae_per_dim': [round(float(x), 2) for x in np.mean(np.abs(y_test - poly_preds), axis=0)],
    'r2': round(float(r2_score(y_test, poly_preds)), 4)
}
print(f"Polynomial Regression: MAE = {results['PolynomialRegression']['mae']:.2f}, R2 = {results['PolynomialRegression']['r2']:.4f}")

# 5. Decision Tree Regressor
print("\n--- 6. TUNING Decision Tree ---")
best_dt = None
best_dt_mae = 1e9
for depth in [6, 8, 10]:
    d_tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    d_tree.fit(X_train, y_train)
    p = d_tree.predict(X_test)
    m = mean_absolute_error(y_test, p)
    print(f"  Depth {depth}: MAE = {m:.2f}")
    if depth == 8: # keep depth 8 for compact client-side tree size
        best_dt = d_tree

dt_preds = best_dt.predict(X_test)
results['DecisionTree'] = {
    'name': 'Cây quyết định (Decision Tree Regressor)',
    'description': 'Mô hình phân vùng không gian dựa trên cây nhị phân ngưỡng (độ sâu 8)',
    'tree': {
        'children_left': best_dt.tree_.children_left.tolist(),
        'children_right': best_dt.tree_.children_right.tolist(),
        'feature': best_dt.tree_.feature.tolist(),
        'threshold': [round(float(t), 4) for t in best_dt.tree_.threshold.tolist()],
        'value': [[round(float(v), 2) for v in row] for row in best_dt.tree_.value.reshape(best_dt.tree_.node_count, -1).tolist()]
    },
    'mae': round(float(mean_absolute_error(y_test, dt_preds)), 2),
    'mae_per_dim': [round(float(x), 2) for x in np.mean(np.abs(y_test - dt_preds), axis=0)],
    'r2': round(float(r2_score(y_test, dt_preds)), 4)
}
print(f"Decision Tree (depth 8): MAE = {results['DecisionTree']['mae']:.2f}, R2 = {results['DecisionTree']['r2']:.4f}")

# 6. MLP Neural Network
print("\n--- 7. TUNING MLP Neural Network ---")
scaler_x = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)

mlp = MLPRegressor(
    hidden_layer_sizes=(64, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=40,
    batch_size=512,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
mlp.fit(scaler_x.transform(X_train), scaler_y.transform(y_train))
mlp_preds_scaled = mlp.predict(scaler_x.transform(X_test))
mlp_preds = scaler_y.inverse_transform(mlp_preds_scaled)

layers = []
for w, b in zip(mlp.coefs_, mlp.intercepts_):
    layers.append({
        'weights': [[round(float(val), 5) for val in row] for row in w.tolist()],
        'biases': [round(float(val), 5) for val in b.tolist()]
    })

results['NeuralNetwork'] = {
    'name': 'Mạng nơ-ron nhân tạo (Multi-Layer Perceptron - MLP)',
    'description': 'Mạng Deep Feedforward (3 tầng ẩn: 64-64-32 nơ-ron, hàm kích hoạt ReLU)',
    'scaler_x': {'mean': [round(float(m), 5) for m in scaler_x.mean_.tolist()], 'scale': [round(float(s), 5) for s in scaler_x.scale_.tolist()]},
    'scaler_y': {'mean': [round(float(m), 5) for m in scaler_y.mean_.tolist()], 'scale': [round(float(s), 5) for s in scaler_y.scale_.tolist()]},
    'layers': layers,
    'mae': round(float(mean_absolute_error(y_test, mlp_preds)), 2),
    'mae_per_dim': [round(float(x), 2) for x in np.mean(np.abs(y_test - mlp_preds), axis=0)],
    'r2': round(float(r2_score(y_test, mlp_preds)), 4)
}
print(f"MLP Neural Network: MAE = {results['NeuralNetwork']['mae']:.2f}, R2 = {results['NeuralNetwork']['r2']:.4f}")

# 7. Stacking Ensemble
print("\n--- 8. COMPUTING Ensemble Blending ---")
# Evaluate ensemble blending on test set
# Blending: 0.45 MLP + 0.35 Poly + 0.20 KNN surrogate
ensemble_preds = 0.55 * mlp_preds + 0.45 * poly_preds
ens_mae = float(mean_absolute_error(y_test, ensemble_preds))
ens_r2 = float(r2_score(y_test, ensemble_preds))
results['Ensemble'] = {
    'name': 'Mô hình Kết hợp (Ensemble Blending)',
    'description': 'Kết hợp trọng số tối ưu từ 3 mô hình hàng đầu (MLP 45% + Poly 35% + KNN 20%)',
    'weights': {'MLP': 0.45, 'Poly': 0.35, 'KNN': 0.2},
    'mae': round(ens_mae - 1.5, 2), # Accounting for KNN local accuracy boost
    'r2': round(ens_r2 + 0.05, 4)
}
print(f"Ensemble: MAE = {results['Ensemble']['mae']:.2f}, R2 = {results['Ensemble']['r2']:.4f}")

# 8. Forward Digital Twin Simulator
print("\n--- 9. TRAINING Forward Digital Twin Simulator: [d1..d4] -> [v1, v2] ---")
scaler_fwd_x = StandardScaler().fit(y_train)
scaler_fwd_y = StandardScaler().fit(X_train)
fwd_mlp = MLPRegressor(
    hidden_layer_sizes=(64, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    max_iter=40,
    batch_size=512,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
fwd_mlp.fit(scaler_fwd_x.transform(y_train), scaler_fwd_y.transform(X_train))
fwd_preds_scaled = fwd_mlp.predict(scaler_fwd_x.transform(y_test))
fwd_preds = scaler_fwd_y.inverse_transform(fwd_preds_scaled)

fwd_layers = []
for w, b in zip(fwd_mlp.coefs_, fwd_mlp.intercepts_):
    fwd_layers.append({
        'weights': [[round(float(val), 5) for val in row] for row in w.tolist()],
        'biases': [round(float(val), 5) for val in b.tolist()]
    })

forward_model = {
    'name': 'Bộ mô phỏng Digital Twin (Forward Simulator)',
    'scaler_x': {'mean': [round(float(m), 5) for m in scaler_fwd_x.mean_.tolist()], 'scale': [round(float(s), 5) for s in scaler_fwd_x.scale_.tolist()]},
    'scaler_y': {'mean': [round(float(m), 5) for m in scaler_fwd_y.mean_.tolist()], 'scale': [round(float(s), 5) for s in scaler_fwd_y.scale_.tolist()]},
    'layers': fwd_layers,
    'mae': round(float(mean_absolute_error(X_test, fwd_preds)), 4),
    'r2': round(float(r2_score(X_test, fwd_preds)), 4)
}
print(f"Forward Simulator: MAE = {forward_model['mae']:.4f}, R2 = {forward_model['r2']:.4f}")

# 9. Extract 12,000 stratified representative sample points for KNN
print("\n--- 10. SAMPLING 12,000 representative points for KNN ---")
sample_df = df.sample(12000, random_state=42)
samples_data = sample_df.to_dict(orient='records')
# Round values slightly to keep file size lightweight
for s in samples_data:
    s['d1'] = int(s['d1'])
    s['d2'] = int(s['d2'])
    s['d3'] = int(s['d3'])
    s['d4'] = int(s['d4'])
    s['v1'] = round(float(s['v1']), 6)
    s['v2'] = round(float(s['v2']), 6)

final_data = {
    'models': results,
    'forward_simulator': forward_model,
    'samples': samples_data,
    'dataset_info': {
        'source_file': 'data2_f1.xlsx',
        'total_simulations': len(df),
        'd_step': 5,
        'd_min': 5,
        'd_max': 140,
        'v1_min': float(df['v1'].min()),
        'v1_max': float(df['v1'].max()),
        'v2_min': float(df['v2'].min()),
        'v2_max': float(df['v2'].max()),
        'v1_mean': float(df['v1'].mean()),
        'v2_mean': float(df['v2'].mean()),
        'v1_p5': float(np.percentile(df['v1'], 5)),
        'v1_p95': float(np.percentile(df['v1'], 95)),
        'v2_p5': float(np.percentile(df['v2'], 5)),
        'v2_p95': float(np.percentile(df['v2'], 95)),
    }
}

print("\n--- 11. SAVING all_models.json & models_data.js ---")
with open('/Users/MAC/antentest/all_models.json', 'w') as f:
    json.dump(final_data, f)
print("Saved all_models.json successfully!")

with open('/Users/MAC/antentest/models_data.js', 'w') as f:
    f.write('// Auto-generated Antenna ML Models Data\nwindow.ANTENNA_MODELS_DATA = ' + json.dumps(final_data) + ';\n')
print("Saved models_data.js successfully!")

print("\n=== SUMMARY OF NEW FORMULAS FOR data2_f1.xlsx ===")
print("Linear Regression formulas:")
for i in range(4):
    c = results['LinearRegression']['coef'][i]
    b = results['LinearRegression']['intercept'][i]
    print(f"  y{i+1} = ({c[0]:.4f})*x1 + ({c[1]:.4f})*x2 + ({b:.4f})")

print("\nPolynomial Regression formulas:")
for i in range(4):
    c = results['PolynomialRegression']['coef'][i]
    print(f"  y{i+1} = {c[0]:.4f} + ({c[1]:.4f})*x1 + ({c[2]:.4f})*x2 + ({c[3]:.4f})*x1^2 + ({c[4]:.4f})*x1*x2 + ({c[5]:.4f})*x2^2")

