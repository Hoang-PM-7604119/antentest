// Pure Multi-Output ML Predictor Logic (2 Inputs -> 4 Outputs)
document.addEventListener('DOMContentLoaded', () => {
    if (!window.ANTENNA_MODELS_DATA) {
        console.error('Không tìm thấy dữ liệu mô hình window.ANTENNA_MODELS_DATA');
        return;
    }

    const { models, forward_simulator, samples } = window.ANTENNA_MODELS_DATA;

    // State
    const state = {
        x1: -15.0,
        x2: -5.0,
        currentModel: 'Ensemble',
        snapToGrid: true,
        predictions: { y1: 75.0, y2: 70.0, y3: 85.0, y4: 60.0 },
        verification: { simX1: -14.8, simX2: -4.9, err1: 0.2, err2: 0.1 }
    };

    // DOM Elements
    const sliderX1 = document.getElementById('sliderX1');
    const inputX1 = document.getElementById('inputX1');
    const x1Badge = document.getElementById('x1Badge');

    const sliderX2 = document.getElementById('sliderX2');
    const inputX2 = document.getElementById('inputX2');
    const x2Badge = document.getElementById('x2Badge');

    const modelSelect = document.getElementById('modelSelect');
    const snapStep = document.getElementById('snapStep');

    const valY1 = document.getElementById('valY1');
    const valY2 = document.getElementById('valY2');
    const valY3 = document.getElementById('valY3');
    const valY4 = document.getElementById('valY4');

    const barY1 = document.getElementById('barY1');
    const barY2 = document.getElementById('barY2');
    const barY3 = document.getElementById('barY3');
    const barY4 = document.getElementById('barY4');

    const simX1 = document.getElementById('simX1');
    const simX2 = document.getElementById('simX2');
    const deltaX1 = document.getElementById('deltaX1');
    const deltaX2 = document.getElementById('deltaX2');
    const statusBadge = document.getElementById('statusBadge');

    const formulaCode = document.getElementById('formulaCode');
    const copyBtn = document.getElementById('copyBtn');
    const exportCsvBtn = document.getElementById('exportCsvBtn');

    // Clamp & Round helper
    function clampOutput(val) {
        let v = Math.max(5.0, Math.min(140.0, val));
        if (state.snapToGrid) {
            v = Math.round(v / 5.0) * 5.0;
        } else {
            v = Math.round(v * 10) / 10;
        }
        return v;
    }

    // 1. Linear Regression
    function predictLinear(x1, x2) {
        const m = models.LinearRegression;
        return [0, 1, 2, 3].map(i => clampOutput(m.coef[i][0] * x1 + m.coef[i][1] * x2 + m.intercept[i]));
    }

    // 2. Ridge Regression
    function predictRidge(x1, x2) {
        const m = models.RidgeRegression;
        return [0, 1, 2, 3].map(i => clampOutput(m.coef[i][0] * x1 + m.coef[i][1] * x2 + m.intercept[i]));
    }

    // 3. Lasso Regression
    function predictLasso(x1, x2) {
        const m = models.LassoRegression;
        return [0, 1, 2, 3].map(i => clampOutput(m.coef[i][0] * x1 + m.coef[i][1] * x2 + m.intercept[i]));
    }

    // 4. Polynomial Regression (Degree 2)
    function predictPolynomial(x1, x2) {
        const m = models.PolynomialRegression;
        const feats = [1.0, x1, x2, x1 * x1, x1 * x2, x2 * x2];
        const res = [];
        for (let i = 0; i < 4; i++) {
            let sum = 0;
            for (let f = 0; f < feats.length; f++) {
                sum += m.coef[i][f] * feats[f];
            }
            res.push(clampOutput(sum));
        }
        return res;
    }

    // 5. Decision Tree Regressor
    function predictDecisionTree(x1, x2) {
        const m = models.DecisionTree.tree;
        let node = 0;
        const inputs = [x1, x2];
        while (m.children_left[node] !== -1) {
            const feat = m.feature[node];
            const thresh = m.threshold[node];
            if (inputs[feat] <= thresh) {
                node = m.children_left[node];
            } else {
                node = m.children_right[node];
            }
        }
        return m.value[node].map(v => clampOutput(v));
    }

    // 6. Neural Network MLP
    function predictMLP(x1, x2) {
        const m = models.NeuralNetwork;
        let cur = [
            (x1 - m.scaler_x.mean[0]) / m.scaler_x.scale[0],
            (x2 - m.scaler_x.mean[1]) / m.scaler_x.scale[1]
        ];

        for (let l = 0; l < m.layers.length; l++) {
            const layer = m.layers[l];
            const isLast = (l === m.layers.length - 1);
            const next = [];
            for (let j = 0; j < layer.biases.length; j++) {
                let sum = layer.biases[j];
                for (let i = 0; i < cur.length; i++) {
                    sum += cur[i] * layer.weights[i][j];
                }
                next.push(isLast ? sum : Math.max(0, sum));
            }
            cur = next;
        }

        return cur.map((val, i) => clampOutput(val * m.scaler_y.scale[i] + m.scaler_y.mean[i]));
    }

    // 7. KNN / Database Matching
    function predictKNN(x1, x2, k = 3) {
        if (!samples || samples.length === 0) return predictMLP(x1, x2);
        const dists = samples.map(s => {
            const d1 = s.v1 - x1;
            const d2 = s.v2 - x2;
            return {
                dist: Math.sqrt(d1 * d1 + d2 * d2),
                y: [s.d1, s.d2, s.d3, s.d4]
            };
        });
        dists.sort((a, b) => a.dist - b.dist);

        let sumW = 0;
        let sumY = [0, 0, 0, 0];
        for (let i = 0; i < Math.min(k, dists.length); i++) {
            const w = 1.0 / (dists[i].dist + 1e-6);
            sumW += w;
            for (let j = 0; j < 4; j++) sumY[j] += dists[i].y[j] * w;
        }
        return sumY.map(v => clampOutput(v / sumW));
    }

    // 8. Ensemble Blending
    function predictEnsemble(x1, x2) {
        const mlp = predictMLP(x1, x2);
        const poly = predictPolynomial(x1, x2);
        const knn = predictKNN(x1, x2, 3);
        return [0, 1, 2, 3].map(i => clampOutput(0.45 * mlp[i] + 0.35 * poly[i] + 0.20 * knn[i]));
    }

    // Forward Verification: [y1, y2, y3, y4] -> [x1_sim, x2_sim]
    function simulateForward(y1, y2, y3, y4) {
        const fwd = forward_simulator;
        const dims = [y1, y2, y3, y4];
        let cur = dims.map((val, i) => (val - fwd.scaler_x.mean[i]) / fwd.scaler_x.scale[i]);

        for (let l = 0; l < fwd.layers.length; l++) {
            const layer = fwd.layers[l];
            const isLast = (l === fwd.layers.length - 1);
            const next = [];
            for (let j = 0; j < layer.biases.length; j++) {
                let sum = layer.biases[j];
                for (let i = 0; i < cur.length; i++) {
                    sum += cur[i] * layer.weights[i][j];
                }
                next.push(isLast ? sum : Math.max(0, sum));
            }
            cur = next;
        }
        return [
            cur[0] * fwd.scaler_y.scale[0] + fwd.scaler_y.mean[0],
            cur[1] * fwd.scaler_y.scale[1] + fwd.scaler_y.mean[1]
        ];
    }

    // Run selected model
    function runPrediction(modelName, x1, x2) {
        switch (modelName) {
            case 'LinearRegression': return predictLinear(x1, x2);
            case 'RidgeRegression': return predictRidge(x1, x2);
            case 'LassoRegression': return predictLasso(x1, x2);
            case 'PolynomialRegression': return predictPolynomial(x1, x2);
            case 'DecisionTree': return predictDecisionTree(x1, x2);
            case 'KNN': return predictKNN(x1, x2, 3);
            case 'NeuralNetwork': return predictMLP(x1, x2);
            case 'Ensemble':
            default:
                return predictEnsemble(x1, x2);
        }
    }

    // Update everything
    function updateAll() {
        const x1 = state.x1;
        const x2 = state.x2;

        const y = runPrediction(state.currentModel, x1, x2);
        state.predictions = { y1: y[0], y2: y[1], y3: y[2], y4: y[3] };

        const [simX1_val, simX2_val] = simulateForward(y[0], y[1], y[2], y[3]);
        const err1 = Math.abs(simX1_val - x1);
        const err2 = Math.abs(simX2_val - x2);
        state.verification = { simX1: simX1_val, simX2: simX2_val, err1, err2 };

        // Update UI
        x1Badge.textContent = x1.toFixed(2);
        x2Badge.textContent = x2.toFixed(2);

        valY1.textContent = y[0].toFixed(state.snapToGrid ? 0 : 1);
        valY2.textContent = y[1].toFixed(state.snapToGrid ? 0 : 1);
        valY3.textContent = y[2].toFixed(state.snapToGrid ? 0 : 1);
        valY4.textContent = y[3].toFixed(state.snapToGrid ? 0 : 1);

        barY1.style.width = `${((y[0] - 5) / 135) * 100}%`;
        barY2.style.width = `${((y[1] - 5) / 135) * 100}%`;
        barY3.style.width = `${((y[2] - 5) / 135) * 100}%`;
        barY4.style.width = `${((y[3] - 5) / 135) * 100}%`;

        simX1.textContent = simX1_val.toFixed(2);
        simX2.textContent = simX2_val.toFixed(2);
        deltaX1.textContent = err1.toFixed(2);
        deltaX2.textContent = err2.toFixed(2);

        const avgErr = (err1 + err2) / 2;
        if (avgErr < 0.5) {
            statusBadge.textContent = 'Rất sát (Lệch < 0.5)';
            statusBadge.style.color = '#059669';
        } else if (avgErr < 1.5) {
            statusBadge.textContent = 'Khớp tốt (Lệch < 1.5)';
            statusBadge.style.color = '#0284c7';
        } else {
            statusBadge.textContent = `Lệch trung bình: ${avgErr.toFixed(2)}`;
            statusBadge.style.color = '#d97706';
        }

        renderFormula();
    }

    function renderFormula() {
        const x1 = state.x1;
        const x2 = state.x2;
        const { y1, y2, y3, y4 } = state.predictions;
        let txt = '';

        if (state.currentModel === 'LinearRegression') {
            const m = models.LinearRegression;
            txt = `Hồi quy Tuyến tính: y = W · x + b (Thay số với x₁ = ${x1.toFixed(2)}, x₂ = ${x2.toFixed(2)}):
y₁ = ${m.coef[0][0].toFixed(4)}·(${x1}) + (${m.coef[0][1].toFixed(4)})·(${x2}) + ${m.intercept[0].toFixed(4)} = ${y1.toFixed(1)}
y₂ = ${m.coef[1][0].toFixed(4)}·(${x1}) + (${m.coef[1][1].toFixed(4)})·(${x2}) + ${m.intercept[1].toFixed(4)} = ${y2.toFixed(1)}
y₃ = ${m.coef[2][0].toFixed(4)}·(${x1}) + (${m.coef[2][1].toFixed(4)})·(${x2}) + ${m.intercept[2].toFixed(4)} = ${y3.toFixed(1)}
y₄ = ${m.coef[3][0].toFixed(4)}·(${x1}) + (${m.coef[3][1].toFixed(4)})·(${x2}) + ${m.intercept[3].toFixed(4)} = ${y4.toFixed(1)}`;
        } else if (state.currentModel === 'LassoRegression') {
            const m = models.LassoRegression;
            txt = `Hồi quy Lasso (L1 Regularization):
y₁ = ${m.coef[0][0].toFixed(4)}·(${x1}) + (${m.coef[0][1].toFixed(4)})·(${x2}) + ${m.intercept[0].toFixed(4)} = ${y1.toFixed(1)}
y₂ = ${m.coef[1][0].toFixed(4)}·(${x1}) + (${m.coef[1][1].toFixed(4)})·(${x2}) + ${m.intercept[1].toFixed(4)} = ${y2.toFixed(1)}
y₃ = ${m.coef[2][0].toFixed(4)}·(${x1}) + (${m.coef[2][1].toFixed(4)})·(${x2}) + ${m.intercept[2].toFixed(4)} = ${y3.toFixed(1)}
y₄ = ${m.coef[3][0].toFixed(4)}·(${x1}) + (${m.coef[3][1].toFixed(4)})·(${x2}) + ${m.intercept[3].toFixed(4)} = ${y4.toFixed(1)}`;
        } else if (state.currentModel === 'PolynomialRegression') {
            txt = `Hồi quy Đa thức bậc 2: y_i = c₀ + c₁·x₁ + c₂·x₂ + c₃·x₁² + c₄·(x₁·x₂) + c₅·x₂²
y₁ = 75.40 + 3.64·x₁ - 3.78·x₂ + 0.16·x₁² - 0.33·x₁x₂ - 0.17·x₂² = ${y1.toFixed(1)}
y₂ = 73.81 - 1.97·x₁ + 6.78·x₂ - 0.10·x₁² + 0.47·x₁x₂ + 0.25·x₂² = ${y2.toFixed(1)}
y₃ = 62.13 - 5.20·x₁ - 5.33·x₂ - 0.21·x₁² - 0.05·x₁x₂ - 0.38·x₂² = ${y3.toFixed(1)}
y₄ = 53.77 - 6.97·x₁ - 12.88·x₂ - 0.20·x₁² - 0.46·x₁x₂ - 0.51·x₂² = ${y4.toFixed(1)}`;
        } else if (state.currentModel === 'Ensemble') {
            txt = `Mô hình Tổ Hợp Tối Ưu (Stacking Ensemble):
y = 0.45 · y(MLP) + 0.35 · y(Poly) + 0.20 · y(KNN)
--> Kết quả: y₁ = ${y1.toFixed(1)},  y₂ = ${y2.toFixed(1)},  y₃ = ${y3.toFixed(1)},  y₄ = ${y4.toFixed(1)}`;
        } else if (state.currentModel === 'NeuralNetwork') {
            txt = `Mạng Nơ-ron Nhân Tạo Feedforward (3 tầng ẩn 64-64-32 nơ-ron, ReLU):
h⁽¹⁾ = ReLU(W⁽¹⁾·x + b⁽¹⁾)
h⁽²⁾ = ReLU(W⁽²⁾·h⁽¹⁾ + b⁽²⁾)
h⁽³⁾ = ReLU(W⁽³⁾·h⁽²⁾ + b⁽³⁾)
y = W⁽⁴⁾·h⁽³⁾ + b⁽⁴⁾  -->  y₁ = ${y1.toFixed(1)},  y₂ = ${y2.toFixed(1)},  y₃ = ${y3.toFixed(1)},  y₄ = ${y4.toFixed(1)}`;
        } else if (state.currentModel === 'KNN') {
            txt = `Khớp mẫu thực tế (Nearest Neighbors trên 614,656 mẫu):
D(x, x_k) = √[(x₁ - x₁,k)² + (x₂ - x₂,k)²]
Đã tìm thấy 3 điểm dữ liệu thực tế gần nhất: y₁ = ${y1.toFixed(1)}, y₂ = ${y2.toFixed(1)}, y₃ = ${y3.toFixed(1)}, y₄ = ${y4.toFixed(1)}`;
        } else {
            txt = `Mô hình: ${state.currentModel}
y₁ = ${y1.toFixed(1)}, y₂ = ${y2.toFixed(1)}, y₃ = ${y3.toFixed(1)}, y₄ = ${y4.toFixed(1)}`;
        }
        formulaCode.textContent = txt;
    }

    // Event listeners
    sliderX1.addEventListener('input', (e) => {
        state.x1 = parseFloat(e.target.value);
        inputX1.value = state.x1.toFixed(1);
        updateAll();
    });

    inputX1.addEventListener('change', (e) => {
        let v = parseFloat(e.target.value);
        if (isNaN(v)) v = -15.0;
        v = Math.max(-55.8, Math.min(0.0, v));
        state.x1 = v;
        sliderX1.value = v;
        e.target.value = v.toFixed(1);
        updateAll();
    });

    sliderX2.addEventListener('input', (e) => {
        state.x2 = parseFloat(e.target.value);
        inputX2.value = state.x2.toFixed(1);
        updateAll();
    });

    inputX2.addEventListener('change', (e) => {
        let v = parseFloat(e.target.value);
        if (isNaN(v)) v = -5.0;
        v = Math.max(-47.8, Math.min(0.0, v));
        state.x2 = v;
        sliderX2.value = v;
        e.target.value = v.toFixed(1);
        updateAll();
    });

    modelSelect.addEventListener('change', (e) => {
        state.currentModel = e.target.value;
        updateAll();
    });

    snapStep.addEventListener('change', (e) => {
        state.snapToGrid = e.target.checked;
        updateAll();
    });

    copyBtn.addEventListener('click', () => {
        const text = `y1 = ${state.predictions.y1}\ny2 = ${state.predictions.y2}\ny3 = ${state.predictions.y3}\ny4 = ${state.predictions.y4}\n(Input: x1=${state.x1}, x2=${state.x2}, Model: ${state.currentModel})`;
        navigator.clipboard.writeText(text).then(() => {
            const orig = copyBtn.innerHTML;
            copyBtn.innerHTML = `✓ Đã sao chép!`;
            setTimeout(() => { copyBtn.innerHTML = orig; }, 1800);
        });
    });

    exportCsvBtn.addEventListener('click', () => {
        const rows = [
            ['Name', 'Value', 'Column'],
            ['Input_1 (x1)', state.x1, 'Cột 5'],
            ['Input_2 (x2)', state.x2, 'Cột 6'],
            ['Model', state.currentModel, 'Thuật toán'],
            ['Output_1 (y1)', state.predictions.y1, 'Cột 1'],
            ['Output_2 (y2)', state.predictions.y2, 'Cột 2'],
            ['Output_3 (y3)', state.predictions.y3, 'Cột 3'],
            ['Output_4 (y4)', state.predictions.y4, 'Cột 4'],
            ['Simulated_x1', state.verification.simX1.toFixed(2), 'Kiểm tra x1'],
            ['Simulated_x2', state.verification.simX2.toFixed(2), 'Kiểm tra x2'],
            ['Delta_x1', state.verification.err1.toFixed(2), 'Độ lệch x1'],
            ['Delta_x2', state.verification.err2.toFixed(2), 'Độ lệch x2']
        ];
        const csv = 'data:text/csv;charset=utf-8,' + rows.map(r => r.join(',')).join('\n');
        const link = document.createElement('a');
        link.setAttribute('href', encodeURI(csv));
        link.setAttribute('download', `prediction_x1_${state.x1}_x2_${state.x2}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    updateAll();
});
