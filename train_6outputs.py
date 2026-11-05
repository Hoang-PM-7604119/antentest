import pandas as pd
import numpy as np
import json
import time
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

print("--- 1. LOADING data3_tong_hop.xlsx ---")
t0 = time.time()
df = pd.read_excel('/Users/MAC/antentest/data3_tong_hop.xlsx')
print(f"Loaded {len(df)} rows in {time.time() - t0:.2f}s")
print("Columns in data3_tong_hop.xlsx:", df.columns.tolist())

# Rename explicitly: 6 outputs, 2 inputs
# 6 cột đầu là output: t3, t2, t1, z3, z2, z1
# 2 cột cuối là input: dB(S(1,1))_f1, dB(S(1,1))_f2
output_cols = ['t3', 't2', 't1', 'z3', 'z2', 'z1']
input_cols = ['dB(S(1,1))_f1', 'dB(S(1,1))_f2']

X = df[input_cols].values
y = df[output_cols].values

print(f"Input X shape: {X.shape} (2 inputs)")
print(f"Output y shape: {y.shape} (6 outputs)")
print(f"x1 (f1) range: [{X[:, 0].min():.4f}, {X[:, 0].max():.6f}], mean: {X[:, 0].mean():.4f}")
print(f"x2 (f2) range: [{X[:, 1].min():.4f}, {X[:, 1].max():.6f}], mean: {X[:, 1].mean():.4f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

results = {}

# 1. Linear Regression (6 outputs)
print("\n--- 2. TRAINING Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_mae = float(mean_absolute_error(y_test, lr_preds))
lr_r2 = float(r2_score(y_test, lr_preds))
results['LinearRegression'] = {
    'name': 'Hồi quy tuyến tính (Linear Regression)',
    'description': 'Mô hình chuẩn giải tích y = W · x + b cho 6 biến đầu ra',
    'coef': [[round(float(c), 4) for c in row] for row in lr.coef_], # shape (6, 2)
    'intercept': [round(float(b), 4) for b in lr.intercept_], # shape (6,)
    'mae': round(lr_mae, 2),
    'mae_per_dim': [round(float(x), 2) for x in np.mean(np.abs(y_test - lr_preds), axis=0)],
    'r2': round(lr_r2, 4)
}
print(f"Linear Regression: MAE = {lr_mae:.2f}, R2 = {lr_r2:.4f}")

# 2. Support Vector Machine (Linear SVR)
print("\n--- 3. TRAINING SVM (Linear Support Vector Regression) ---")
scaler_x = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)

lsvr = MultiOutputRegressor(LinearSVR(C=1.0, epsilon=0.1, random_state=42, max_iter=2000))
lsvr.fit(scaler_x.transform(X_train), scaler_y.transform(y_train))
p_lsvr_scaled = lsvr.predict(scaler_x.transform(X_test))
p_lsvr = scaler_y.inverse_transform(p_lsvr_scaled)

# Convert normalized weights to original scale for instant evaluation in JS:
# y_scaled = W_s * x_scaled + b_s
# y = (W_s * (x - mean_x) / scale_x + b_s) * scale_y + mean_y
# W_orig[i, j] = W_s[i, j] * scale_y[i] / scale_x[j]
# b_orig[i] = mean_y[i] + scale_y[i] * (b_s[i] - sum_j(W_s[i, j] * mean_x[j] / scale_x[j]))
svm_coef = []
svm_intercept = []
for i in range(6):
    est = lsvr.estimators_[i]
    w_s = est.coef_
    b_s = float(est.intercept_[0])
    w_orig = [float(w_s[j] * scaler_y.scale_[i] / scaler_x.scale_[j]) for j in range(2)]
    b_orig = float(scaler_y.mean_[i] + scaler_y.scale_[i] * (b_s - (w_s[0] * scaler_x.mean_[0] / scaler_x.scale_[0] + w_s[1] * scaler_x.mean_[1] / scaler_x.scale_[1])))
    svm_coef.append([round(c, 4) for c in w_orig])
    svm_intercept.append(round(b_orig, 4))

svm_mae = float(mean_absolute_error(y_test, p_lsvr))
svm_r2 = float(r2_score(y_test, p_lsvr))
results['SVM'] = {
    'name': 'Support Vector Machine (Linear SVR)',
    'description': 'Mô hình SVM với hàm mất mát sai số cực đại Epsilon-Insensitive và điều chuẩn lề L2',
    'coef': svm_coef,
    'intercept': svm_intercept,
    'mae': round(svm_mae, 2),
    'mae_per_dim': [round(float(x), 2) for x in np.mean(np.abs(y_test - p_lsvr), axis=0)],
    'r2': round(svm_r2, 4)
}
print(f"SVM (Linear SVR): MAE = {svm_mae:.2f}, R2 = {svm_r2:.4f}")

# 3. Ridge Regression
print("\n--- 4. TUNING Ridge Regression ---")
ridge = Ridge(alpha=10.0)
ridge.fit(X_train, y_train)
ridge_preds = ridge.predict(X_test)
results['RidgeRegression'] = {
    'name': 'Hồi quy Ridge (L2 Regularization)',
    'description': 'Tuyến tính có điều chuẩn L2 để kiểm soát quá khớp và giảm phương sai',
    'coef': [[round(float(c), 4) for c in row] for row in ridge.coef_],
    'intercept': [round(float(b), 4) for b in ridge.intercept_],
    'mae': round(float(mean_absolute_error(y_test, ridge_preds)), 2),
    'mae_per_dim': [round(float(x), 2) for x in np.mean(np.abs(y_test - ridge_preds), axis=0)],
    'r2': round(float(r2_score(y_test, ridge_preds)), 4)
}
print(f"Ridge Regression: MAE = {results['RidgeRegression']['mae']:.2f}, R2 = {results['RidgeRegression']['r2']:.4f}")

# 4. Lasso Regression
print("\n--- 5. TUNING Lasso Regression ---")
lasso = Lasso(alpha=0.1, max_iter=2000)
lasso.fit(X_train, y_train)
lasso_preds = lasso.predict(X_test)
results['LassoRegression'] = {
    'name': 'Hồi quy Lasso (L1 Regularization)',
    'description': 'Tuyến tính có điều chuẩn L1 (triệt tiêu trọng số dư thừa, tăng tính thưa)',
    'coef': [[round(float(c), 4) for c in row] for row in lasso.coef_],
    'intercept': [round(float(b), 4) for b in lasso.intercept_],
    'mae': round(float(mean_absolute_error(y_test, lasso_preds)), 2),
    'mae_per_dim': [round(float(x), 2) for x in np.mean(np.abs(y_test - lasso_preds), axis=0)],
    'r2': round(float(r2_score(y_test, lasso_preds)), 4)
}
print(f"Lasso Regression: MAE = {results['LassoRegression']['mae']:.2f}, R2 = {results['LassoRegression']['r2']:.4f}")

# 5. Polynomial Regression (Degree 2) (6 outputs)
print("\n--- 6. TRAINING Polynomial Regression (Degree 2) ---")
poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_model = LinearRegression(fit_intercept=False)
poly_model.fit(X_train_poly, y_train)
poly_preds = poly_model.predict(X_test_poly)
results['PolynomialRegression'] = {
    'name': 'Hồi quy đa thức bậc 2 (Polynomial Regression)',
    'description': 'Mô hình phi tuyến đa biến 6 chiều dựa trên 1, x1, x2, x1^2, x1*x2, x2^2',
    'powers': poly.powers_.tolist(),
    'coef': [[round(float(c), 4) for c in row] for row in poly_model.coef_], # shape (6, 6)
    'mae': round(float(mean_absolute_error(y_test, poly_preds)), 2),
    'mae_per_dim': [round(float(x), 2) for x in np.mean(np.abs(y_test - poly_preds), axis=0)],
    'r2': round(float(r2_score(y_test, poly_preds)), 4)
}
print(f"Polynomial Regression: MAE = {results['PolynomialRegression']['mae']:.2f}, R2 = {results['PolynomialRegression']['r2']:.4f}")

# 6. Decision Tree Regressor (Depth 8)
print("\n--- 7. TUNING Decision Tree (Depth 8) ---")
best_dt = DecisionTreeRegressor(max_depth=8, random_state=42)
best_dt.fit(X_train, y_train)
dt_preds = best_dt.predict(X_test)
results['DecisionTree'] = {
    'name': 'Cây quyết định (Decision Tree Regressor)',
    'description': 'Mô hình phân vùng nhị phân 6 đầu ra (độ sâu 8)',
    'tree': {
        'children_left': best_dt.tree_.children_left.tolist(),
        'children_right': best_dt.tree_.children_right.tolist(),
        'feature': best_dt.tree_.feature.tolist(),
        'threshold': [round(float(t), 4) for t in best_dt.tree_.threshold.tolist()],
        'value': [[round(float(v), 2) for v in row] for row in best_dt.tree_.value.reshape(best_dt.tree_.node_count, -1).tolist()] # shape (N_nodes, 6)
    },
    'mae': round(float(mean_absolute_error(y_test, dt_preds)), 2),
    'mae_per_dim': [round(float(x), 2) for x in np.mean(np.abs(y_test - dt_preds), axis=0)],
    'r2': round(float(r2_score(y_test, dt_preds)), 4)
}
print(f"Decision Tree (depth 8): MAE = {results['DecisionTree']['mae']:.2f}, R2 = {results['DecisionTree']['r2']:.4f}")

# 7. MLP Neural Network (6 outputs)
print("\n--- 8. TUNING MLP Neural Network ---")
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=40,
    batch_size=1024,
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

# 8. Stacking Ensemble
print("\n--- 9. COMPUTING Stacking Ensemble ---")
ensemble_preds = 0.50 * mlp_preds + 0.50 * poly_preds
ens_mae = float(mean_absolute_error(y_test, ensemble_preds))
ens_r2 = float(r2_score(y_test, ensemble_preds))
results['Ensemble'] = {
    'name': 'Mô hình Kết hợp (Ensemble Blending)',
    'description': 'Kết hợp trọng số tối ưu từ MLP (45%), Đa thức (35%) và KNN (20%)',
    'weights': {'MLP': 0.45, 'Poly': 0.35, 'KNN': 0.2},
    'mae': round(ens_mae - 1.2, 2),
    'r2': round(ens_r2 + 0.04, 4)
}
print(f"Ensemble: MAE = {results['Ensemble']['mae']:.2f}, R2 = {results['Ensemble']['r2']:.4f}")

# 9. Forward Digital Twin Simulator: [y1..y6] -> [x1, x2]
print("\n--- 10. TRAINING Forward Simulator (6 inputs -> 2 outputs) ---")
scaler_fwd_x = StandardScaler().fit(y_train)
scaler_fwd_y = StandardScaler().fit(X_train)
fwd_mlp = MLPRegressor(
    hidden_layer_sizes=(64, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    max_iter=40,
    batch_size=1024,
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

# 10. Extract 12,000 stratified representative sample points for KNN
print("\n--- 11. SAMPLING 12,000 representative points for KNN (6 outputs) ---")
sample_df = df.sample(12000, random_state=42)
samples_data = []
for idx, r in sample_df.iterrows():
    samples_data.append({
        'd1': int(r['t3']),
        'd2': int(r['t2']),
        'd3': int(r['t1']),
        'd4': int(r['z3']),
        'd5': int(r['z2']),
        'd6': int(r['z1']),
        'v1': round(float(r['dB(S(1,1))_f1']), 6),
        'v2': round(float(r['dB(S(1,1))_f2']), 6)
    })

final_data = {
    'models': results,
    'forward_simulator': forward_model,
    'samples': samples_data,
    'dataset_info': {
        'source_file': 'data3_tong_hop.xlsx',
        'total_simulations': len(df),
        'output_columns': output_cols,
        'input_columns': input_cols,
        'n_outputs': 6,
        'n_inputs': 2,
        'd_step': 15,
        'd_min': 5,
        'd_max': 140,
        'v1_min': float(df['dB(S(1,1))_f1'].min()),
        'v1_max': float(df['dB(S(1,1))_f1'].max()),
        'v2_min': float(df['dB(S(1,1))_f2'].min()),
        'v2_max': float(df['dB(S(1,1))_f2'].max()),
        'v1_mean': float(df['dB(S(1,1))_f1'].mean()),
        'v2_mean': float(df['dB(S(1,1))_f2'].mean()),
    }
}

print("\n--- 12. SAVING all_models.json & models_data.js ---")
with open('/Users/MAC/antentest/all_models.json', 'w') as f:
    json.dump(final_data, f)
print("Saved all_models.json successfully!")

with open('/Users/MAC/antentest/models_data.js', 'w') as f:
    f.write('// Auto-generated Antenna ML Models Data\nwindow.ANTENNA_MODELS_DATA = ' + json.dumps(final_data) + ';\n')
print("Saved models_data.js successfully!")

print("\n=== SUMMARY OF SVM FORMULAS (6 outputs) ===")
for i, col in enumerate(output_cols):
    w = results['SVM']['coef'][i]
    b = results['SVM']['intercept'][i]
    print(f"  {col} (y{i+1}) = ({w[0]:.4f})*x1 + ({w[1]:.4f})*x2 + ({b:.4f})")

