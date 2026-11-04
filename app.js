// Antenna ML Application Logic (Light Theme by Default & Explicit Formulas)
document.addEventListener('DOMContentLoaded', () => {
    // Check if models data is loaded
    if (!window.ANTENNA_MODELS_DATA) {
        console.error('Lỗi: Chưa nạp được window.ANTENNA_MODELS_DATA từ models_data.js');
        alert('Không tìm thấy dữ liệu mô hình. Vui lòng kiểm tra file models_data.js.');
        return;
    }

    const { models, forward_simulator, samples, dataset_info } = window.ANTENNA_MODELS_DATA;

    // State - DEFAULT TO LIGHT THEME
    const state = {
        v1: -15.0,
        v2: -5.0,
        currentModel: 'NeuralNetwork',
        snapToGrid: true,
        theme: localStorage.getItem('antenna_theme') || 'light',
        predictions: {
            d1: 75.0,
            d2: 70.0,
            d3: 85.0,
            d4: 60.0
        },
        forwardVerification: {
            v1_sim: -14.8,
            v2_sim: -4.9,
            err1: 0.2,
            err2: 0.1
        }
    };

    // DOM Elements
    const sliderV1 = document.getElementById('sliderV1');
    const inputV1 = document.getElementById('inputV1');
    const v1Badge = document.getElementById('v1Badge');

    const sliderV2 = document.getElementById('sliderV2');
    const inputV2 = document.getElementById('inputV2');
    const v2Badge = document.getElementById('v2Badge');

    const snapStep = document.getElementById('snapStep');
    const modelDescription = document.getElementById('modelDescription');
    const activeFormulaMath = document.getElementById('activeFormulaMath');

    const valD1 = document.getElementById('valD1');
    const valD2 = document.getElementById('valD2');
    const valD3 = document.getElementById('valD3');
    const valD4 = document.getElementById('valD4');

    const barD1 = document.getElementById('barD1');
    const barD2 = document.getElementById('barD2');
    const barD3 = document.getElementById('barD3');
    const barD4 = document.getElementById('barD4');

    const expV1 = document.getElementById('expV1');
    const simV1 = document.getElementById('simV1');
    const deltaV1 = document.getElementById('deltaV1');

    const expV2 = document.getElementById('expV2');
    const simV2 = document.getElementById('simV2');
    const deltaV2 = document.getElementById('deltaV2');

    const verifStatusBadge = document.getElementById('verifStatusBadge');
    const comparisonTableBody = document.getElementById('comparisonTableBody');

    const themeToggle = document.getElementById('themeToggle');
    const themeIcon = document.getElementById('themeIcon');
    const copyBtn = document.getElementById('copyBtn');
    const exportCsvBtn = document.getElementById('exportCsvBtn');
    const refreshTableBtn = document.getElementById('refreshTableBtn');

    // CAD Elements
    const cadPatch = document.getElementById('cadPatch');
    const cadFeed = document.getElementById('cadFeed');
    const cadSlot = document.getElementById('cadSlot');
    const cadTextD1 = document.getElementById('cadTextD1');
    const cadTextD2 = document.getElementById('cadTextD2');
    const cadTextD3 = document.getElementById('cadTextD3');
    const cadTextD4 = document.getElementById('cadTextD4');

    // Apply initial theme (Light by default)
    applyTheme(state.theme);

    function applyTheme(theme) {
        if (theme === 'dark') {
            document.body.classList.remove('theme-light');
            document.body.classList.add('theme-dark');
            // Sun icon for dark mode (click to switch to light)
            themeIcon.innerHTML = `
                <circle cx="12" cy="12" r="5"></circle>
                <line x1="12" y1="1" x2="12" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="23"></line>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                <line x1="1" y1="12" x2="3" y2="12"></line>
                <line x1="21" y1="12" x2="23" y2="12"></line>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
            `;
        } else {
            document.body.classList.remove('theme-dark');
            document.body.classList.add('theme-light');
            // Moon icon for light mode (click to switch to dark)
            themeIcon.innerHTML = `
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
            `;
        }
        localStorage.setItem('antenna_theme', theme);
    }

    // --- INFERENCE ALGORITHMS ---

    // 1. Linear Regression: d = W * v + b
    function predictLinear(v1, v2) {
        const m = models.LinearRegression;
        const d = [];
        for (let i = 0; i < 4; i++) {
            const val = m.coef[i][0] * v1 + m.coef[i][1] * v2 + m.intercept[i];
            d.push(clampDimension(val));
        }
        return d;
    }

    // 2. Ridge Regression (L2)
    function predictRidge(v1, v2) {
        const m = models.RidgeRegression;
        const d = [];
        for (let i = 0; i < 4; i++) {
            const val = m.coef[i][0] * v1 + m.coef[i][1] * v2 + m.intercept[i];
            d.push(clampDimension(val));
        }
        return d;
    }

    // 3. Lasso Regression (L1)
    function predictLasso(v1, v2) {
        const m = models.LassoRegression;
        const d = [];
        for (let i = 0; i < 4; i++) {
            const val = m.coef[i][0] * v1 + m.coef[i][1] * v2 + m.intercept[i];
            d.push(clampDimension(val));
        }
        return d;
    }

    // 4. Polynomial Regression (Degree 2)
    function predictPolynomial(v1, v2) {
        const m = models.PolynomialRegression;
        const feats = [1.0, v1, v2, v1 * v1, v1 * v2, v2 * v2];
        const d = [];
        for (let i = 0; i < 4; i++) {
            let sum = 0;
            for (let f = 0; f < feats.length; f++) {
                sum += m.coef[i][f] * feats[f];
            }
            d.push(clampDimension(sum));
        }
        return d;
    }

    // 5. Decision Tree Regressor
    function predictDecisionTree(v1, v2) {
        const m = models.DecisionTree.tree;
        let node = 0;
        const inputs = [v1, v2];

        while (m.children_left[node] !== -1) {
            const feat = m.feature[node];
            const thresh = m.threshold[node];
            if (inputs[feat] <= thresh) {
                node = m.children_left[node];
            } else {
                node = m.children_right[node];
            }
        }

        const vals = m.value[node];
        return vals.map(v => clampDimension(v));
    }

    // 6. Multi-Layer Perceptron (MLP Neural Network)
    function predictMLP(v1, v2) {
        const m = models.NeuralNetwork;
        let cur = [
            (v1 - m.scaler_x.mean[0]) / m.scaler_x.scale[0],
            (v2 - m.scaler_x.mean[1]) / m.scaler_x.scale[1]
        ];

        for (let l = 0; l < m.layers.length; l++) {
            const layer = m.layers[l];
            const isLast = (l === m.layers.length - 1);
            const next = [];

            const outDim = layer.biases.length;
            const inDim = cur.length;

            for (let j = 0; j < outDim; j++) {
                let sum = layer.biases[j];
                for (let i = 0; i < inDim; i++) {
                    sum += cur[i] * layer.weights[i][j];
                }
                next.push(isLast ? sum : Math.max(0, sum));
            }
            cur = next;
        }

        const d = [];
        for (let i = 0; i < 4; i++) {
            const unscaled = cur[i] * m.scaler_y.scale[i] + m.scaler_y.mean[i];
            d.push(clampDimension(unscaled));
        }
        return d;
    }

    // 7. K-Nearest Neighbors / Direct Simulation Database Match (KNN)
    function predictKNN(v1, v2, k = 3) {
        if (!samples || samples.length === 0) {
            return predictMLP(v1, v2);
        }

        const dists = samples.map(s => {
            const dv1 = s.v1 - v1;
            const dv2 = s.v2 - v2;
            return {
                dist: Math.sqrt(dv1 * dv1 + dv2 * dv2),
                d1: s.d1,
                d2: s.d2,
                d3: s.d3,
                d4: s.d4
            };
        });

        dists.sort((a, b) => a.dist - b.dist);

        let sumW = 0;
        let sumD = [0, 0, 0, 0];
        for (let i = 0; i < Math.min(k, dists.length); i++) {
            const w = 1.0 / (dists[i].dist + 1e-6);
            sumW += w;
            sumD[0] += dists[i].d1 * w;
            sumD[1] += dists[i].d2 * w;
            sumD[2] += dists[i].d3 * w;
            sumD[3] += dists[i].d4 * w;
        }

        return sumD.map(v => clampDimension(v / sumW));
    }

    // 8. Ensemble Blending Model
    function predictEnsemble(v1, v2) {
        const mlp = predictMLP(v1, v2);
        const poly = predictPolynomial(v1, v2);
        const knn = predictKNN(v1, v2, 3);

        const d = [];
        for (let i = 0; i < 4; i++) {
            const blended = 0.45 * mlp[i] + 0.35 * poly[i] + 0.20 * knn[i];
            d.push(clampDimension(blended));
        }
        return d;
    }

    // Forward Digital Twin Simulator: [d1, d2, d3, d4] -> [v1_sim, v2_sim]
    function simulateForward(d1, d2, d3, d4) {
        const fwd = forward_simulator;
        const dims = [d1, d2, d3, d4];

        let cur = dims.map((val, i) => (val - fwd.scaler_x.mean[i]) / fwd.scaler_x.scale[i]);

        for (let l = 0; l < fwd.layers.length; l++) {
            const layer = fwd.layers[l];
            const isLast = (l === fwd.layers.length - 1);
            const next = [];

            const outDim = layer.biases.length;
            const inDim = cur.length;

            for (let j = 0; j < outDim; j++) {
                let sum = layer.biases[j];
                for (let i = 0; i < inDim; i++) {
                    sum += cur[i] * layer.weights[i][j];
                }
                next.push(isLast ? sum : Math.max(0, sum));
            }
            cur = next;
        }

        const v1_sim = cur[0] * fwd.scaler_y.scale[0] + fwd.scaler_y.mean[0];
        const v2_sim = cur[1] * fwd.scaler_y.scale[1] + fwd.scaler_y.mean[1];
        return [v1_sim, v2_sim];
    }

    function clampDimension(val) {
        let v = Math.max(5.0, Math.min(140.0, val));
        if (state.snapToGrid) {
            v = Math.round(v / 5.0) * 5.0;
        } else {
            v = Math.round(v * 10) / 10;
        }
        return v;
    }

    // Run prediction for a specific model name
    function runModelInference(modelName, v1, v2) {
        let dims;
        switch (modelName) {
            case 'LinearRegression':
                dims = predictLinear(v1, v2);
                break;
            case 'RidgeRegression':
                dims = predictRidge(v1, v2);
                break;
            case 'LassoRegression':
                dims = predictLasso(v1, v2);
                break;
            case 'PolynomialRegression':
                dims = predictPolynomial(v1, v2);
                break;
            case 'DecisionTree':
                dims = predictDecisionTree(v1, v2);
                break;
            case 'KNN':
                dims = predictKNN(v1, v2, 3);
                break;
            case 'Ensemble':
                dims = predictEnsemble(v1, v2);
                break;
            case 'NeuralNetwork':
            default:
                dims = predictMLP(v1, v2);
                break;
        }
        return { dims };
    }

    // Main Update Function
    function updateAll() {
        const v1 = state.v1;
        const v2 = state.v2;

        const { dims } = runModelInference(state.currentModel, v1, v2);
        state.predictions.d1 = dims[0];
        state.predictions.d2 = dims[1];
        state.predictions.d3 = dims[2];
        state.predictions.d4 = dims[3];

        const [v1_sim, v2_sim] = simulateForward(dims[0], dims[1], dims[2], dims[3]);
        const err1 = Math.abs(v1_sim - v1);
        const err2 = Math.abs(v2_sim - v2);
        state.forwardVerification = { v1_sim, v2_sim, err1, err2 };

        renderOutputs();
        renderCAD();
        renderTable();
        renderActiveFormula();
    }

    function renderOutputs() {
        v1Badge.textContent = `${state.v1.toFixed(2)} dB`;
        v2Badge.textContent = `${state.v2.toFixed(2)} dB`;

        valD1.textContent = state.predictions.d1.toFixed(state.snapToGrid ? 0 : 1);
        valD2.textContent = state.predictions.d2.toFixed(state.snapToGrid ? 0 : 1);
        valD3.textContent = state.predictions.d3.toFixed(state.snapToGrid ? 0 : 1);
        valD4.textContent = state.predictions.d4.toFixed(state.snapToGrid ? 0 : 1);

        barD1.style.width = `${((state.predictions.d1 - 5) / 135) * 100}%`;
        barD2.style.width = `${((state.predictions.d2 - 5) / 135) * 100}%`;
        barD3.style.width = `${((state.predictions.d3 - 5) / 135) * 100}%`;
        barD4.style.width = `${((state.predictions.d4 - 5) / 135) * 100}%`;

        expV1.textContent = `${state.v1.toFixed(2)} dB`;
        simV1.textContent = `${state.forwardVerification.v1_sim.toFixed(2)} dB`;
        deltaV1.textContent = `${state.forwardVerification.err1.toFixed(2)} dB`;

        expV2.textContent = `${state.v2.toFixed(2)} dB`;
        simV2.textContent = `${state.forwardVerification.v2_sim.toFixed(2)} dB`;
        deltaV2.textContent = `${state.forwardVerification.err2.toFixed(2)} dB`;

        const totalErr = (state.forwardVerification.err1 + state.forwardVerification.err2) / 2;
        if (totalErr < 0.5) {
            verifStatusBadge.textContent = 'Rất chính xác (Lệch < 0.5 dB)';
            verifStatusBadge.className = 'badge badge-success';
        } else if (totalErr < 1.5) {
            verifStatusBadge.textContent = 'Khớp tốt (Lệch < 1.5 dB)';
            verifStatusBadge.className = 'badge badge-success';
        } else {
            verifStatusBadge.textContent = `Chấp nhận được (Lệch ${totalErr.toFixed(1)} dB)`;
            verifStatusBadge.className = 'badge';
        }
    }

    // Dynamic CAD SVG Renderer
    function renderCAD() {
        const { d1, d2, d3, d4 } = state.predictions;
        const scale = 2.4;

        const patchW = Math.max(20, d1 * scale);
        const patchH = Math.max(20, d2 * scale);
        const patchX = 400 - (patchW / 2);
        const patchY = 190 - (patchH / 2);

        cadPatch.setAttribute('x', patchX);
        cadPatch.setAttribute('y', patchY);
        cadPatch.setAttribute('width', patchW);
        cadPatch.setAttribute('height', patchH);

        const feedW = 32;
        const feedH = Math.max(15, (d3 / 140) * 100);
        const feedX = 400 - (feedW / 2);
        const feedY = patchY + patchH;

        cadFeed.setAttribute('x', feedX);
        cadFeed.setAttribute('y', feedY);
        cadFeed.setAttribute('width', feedW);
        cadFeed.setAttribute('height', feedH);

        const slotW = Math.max(10, Math.min(patchW * 0.8, d4 * scale * 0.6));
        const slotH = 14;
        const slotX = 400 - (slotW / 2);
        const slotY = patchY + (patchH / 2) - 7;

        cadSlot.setAttribute('x', slotX);
        cadSlot.setAttribute('y', slotY);
        cadSlot.setAttribute('width', slotW);
        cadSlot.setAttribute('height', slotH);

        cadTextD1.textContent = `d₁ = ${d1.toFixed(0)} mm`;
        cadTextD1.setAttribute('x', 400);
        cadTextD1.setAttribute('y', patchY - 14);

        const dimD1Group = document.getElementById('dimD1Group');
        dimD1Group.innerHTML = `
            <line x1="${patchX}" y1="${patchY - 8}" x2="${patchX + patchW}" y2="${patchY - 8}" class="cad-dim-line" />
            <circle cx="${patchX}" cy="${patchY - 8}" r="3" class="cad-dim-dot" />
            <circle cx="${patchX + patchW}" cy="${patchY - 8}" r="3" class="cad-dim-dot" />
            <text x="400" y="${patchY - 14}" class="cad-dim-text">d₁ = ${d1.toFixed(0)} mm</text>
        `;

        const dimD2Group = document.getElementById('dimD2Group');
        const d2X = patchX + patchW + 16;
        dimD2Group.innerHTML = `
            <line x1="${d2X}" y1="${patchY}" x2="${d2X}" y2="${patchY + patchH}" class="cad-dim-line" />
            <circle cx="${d2X}" cy="${patchY}" r="3" class="cad-dim-dot" />
            <circle cx="${d2X}" cy="${patchY + patchH}" r="3" class="cad-dim-dot" />
            <text x="${d2X + 10}" y="${patchY + patchH / 2 + 5}" class="cad-dim-text" text-anchor="start">d₂ = ${d2.toFixed(0)} mm</text>
        `;

        const dimD3Group = document.getElementById('dimD3Group');
        const d3X = feedX - 16;
        dimD3Group.innerHTML = `
            <line x1="${d3X}" y1="${feedY}" x2="${d3X}" y2="${feedY + feedH}" class="cad-dim-line" />
            <circle cx="${d3X}" cy="${feedY}" r="3" class="cad-dim-dot" />
            <circle cx="${d3X}" cy="${feedY + feedH}" r="3" class="cad-dim-dot" />
            <text x="${d3X - 10}" y="${feedY + feedH / 2 + 4}" class="cad-dim-text" text-anchor="end">d₃ = ${d3.toFixed(0)} mm</text>
        `;

        const dimD4Group = document.getElementById('dimD4Group');
        dimD4Group.innerHTML = `
            <line x1="${slotX}" y1="${slotY - 6}" x2="${slotX + slotW}" y2="${slotY - 6}" class="cad-dim-line" />
            <circle cx="${slotX}" cy="${slotY - 6}" r="2.5" class="cad-dim-dot" />
            <circle cx="${slotX + slotW}" cy="${slotY - 6}" r="2.5" class="cad-dim-dot" />
            <text x="400" y="${slotY - 10}" class="cad-dim-text" style="font-size:11px;">d₄ = ${d4.toFixed(0)} mm</text>
        `;
    }

    // Dynamic Formula Substituted Box
    function renderActiveFormula() {
        if (!activeFormulaMath) return;
        const v1 = state.v1;
        const v2 = state.v2;
        const { d1, d2, d3, d4 } = state.predictions;

        let txt = '';
        if (state.currentModel === 'LinearRegression') {
            const m = models.LinearRegression;
            txt = `Công thức đại số tuyến tính cụ thể thay thế v₁ = ${v1.toFixed(2)}, v₂ = ${v2.toFixed(2)}:
d₁ = ${m.coef[0][0].toFixed(4)}·(${v1}) + (${m.coef[0][1].toFixed(4)})·(${v2}) + ${m.intercept[0].toFixed(4)} = ${d1.toFixed(1)} mm
d₂ = ${m.coef[1][0].toFixed(4)}·(${v1}) + (${m.coef[1][1].toFixed(4)})·(${v2}) + ${m.intercept[1].toFixed(4)} = ${d2.toFixed(1)} mm
d₃ = ${m.coef[2][0].toFixed(4)}·(${v1}) + (${m.coef[2][1].toFixed(4)})·(${v2}) + ${m.intercept[2].toFixed(4)} = ${d3.toFixed(1)} mm
d₄ = ${m.coef[3][0].toFixed(4)}·(${v1}) + (${m.coef[3][1].toFixed(4)})·(${v2}) + ${m.intercept[3].toFixed(4)} = ${d4.toFixed(1)} mm`;
        } else if (state.currentModel === 'LassoRegression') {
            const m = models.LassoRegression;
            txt = `Công thức đại số Lasso (L1) thay thế v₁ = ${v1.toFixed(2)}, v₂ = ${v2.toFixed(2)}:
d₁ = ${m.coef[0][0].toFixed(4)}·(${v1}) + (${m.coef[0][1].toFixed(4)})·(${v2}) + ${m.intercept[0].toFixed(4)} = ${d1.toFixed(1)} mm
d₂ = ${m.coef[1][0].toFixed(4)}·(${v1}) + (${m.coef[1][1].toFixed(4)})·(${v2}) + ${m.intercept[1].toFixed(4)} = ${d2.toFixed(1)} mm
d₃ = ${m.coef[2][0].toFixed(4)}·(${v1}) + (${m.coef[2][1].toFixed(4)})·(${v2}) + ${m.intercept[2].toFixed(4)} = ${d3.toFixed(1)} mm
d₄ = ${m.coef[3][0].toFixed(4)}·(${v1}) + (${m.coef[3][1].toFixed(4)})·(${v2}) + ${m.intercept[3].toFixed(4)} = ${d4.toFixed(1)} mm`;
        } else if (state.currentModel === 'PolynomialRegression') {
            txt = `Hồi quy đa thức bậc 2: d_i = c₀ + c₁·v₁ + c₂·v₂ + c₃·v₁² + c₄·(v₁·v₂) + c₅·v₂²
Thay số với v₁² = ${(v1*v1).toFixed(2)}, v₁·v₂ = ${(v1*v2).toFixed(2)}, v₂² = ${(v2*v2).toFixed(2)}:
d₁ = 75.40 + 3.64·v₁ - 3.78·v₂ + 0.16·v₁² - 0.33·v₁v₂ - 0.17·v₂² = ${d1.toFixed(1)} mm
d₂ = 73.81 - 1.97·v₁ + 6.78·v₂ - 0.10·v₁² + 0.47·v₁v₂ + 0.25·v₂² = ${d2.toFixed(1)} mm
d₃ = 62.13 - 5.20·v₁ - 5.33·v₂ - 0.21·v₁² - 0.05·v₁v₂ - 0.38·v₂² = ${d3.toFixed(1)} mm
d₄ = 53.77 - 6.97·v₁ - 12.88·v₂ - 0.20·v₁² - 0.46·v₁v₂ - 0.51·v₂² = ${d4.toFixed(1)} mm`;
        } else if (state.currentModel === 'Ensemble') {
            txt = `Công thức Stacking Ensemble:
d = 0.45 · d(MLP) + 0.35 · d(Poly) + 0.20 · d(KNN)
Kết quả kết hợp: d₁ = ${d1.toFixed(1)} mm, d₂ = ${d2.toFixed(1)} mm, d₃ = ${d3.toFixed(1)} mm, d₄ = ${d4.toFixed(1)} mm`;
        } else if (state.currentModel === 'NeuralNetwork') {
            txt = `Mạng nơ-ron sâu Feedforward (3 tầng ẩn, 64-64-32 nơ-ron):
h⁽¹⁾ = ReLU(W⁽¹⁾·x + b⁽¹⁾),  h⁽²⁾ = ReLU(W⁽²⁾·h⁽¹⁾ + b⁽²⁾),  h⁽³⁾ = ReLU(W⁽³⁾·h⁽²⁾ + b⁽³⁾)
y = W⁽⁴⁾·h⁽³⁾ + b⁽⁴⁾  -->  d = y ⊙ scale_y + mean_y
Đầu ra giải mã: d₁ = ${d1.toFixed(1)} mm, d₂ = ${d2.toFixed(1)} mm, d₃ = ${d3.toFixed(1)} mm, d₄ = ${d4.toFixed(1)} mm`;
        } else if (state.currentModel === 'KNN') {
            txt = `Khoảng cách Euclidean tới 614,656 mẫu:
D(v, v_k) = √[(v₁ - v₁,k)² + (v₂ - v₂,k)²]
Đã tìm thấy 3 cấu hình mô phỏng thực tế gần nhất và tính bình quân trọng số:
d₁ = ${d1.toFixed(1)} mm, d₂ = ${d2.toFixed(1)} mm, d₃ = ${d3.toFixed(1)} mm, d₄ = ${d4.toFixed(1)} mm`;
        } else {
            txt = `Mô hình: ${state.currentModel}
d₁ = ${d1.toFixed(1)} mm, d₂ = ${d2.toFixed(1)} mm, d₃ = ${d3.toFixed(1)} mm, d₄ = ${d4.toFixed(1)} mm`;
        }
        activeFormulaMath.textContent = txt;
    }

    // Comparison Table Renderer
    function renderTable() {
        const modelKeys = [
            { key: 'NeuralNetwork', name: 'Mạng nơ-ron (MLP)', badge: 'badge-complex', complexity: 'Nâng cao' },
            { key: 'Ensemble', name: 'Ensemble Blending (Kết hợp)', badge: 'badge-complex', complexity: 'Tối ưu' },
            { key: 'PolynomialRegression', name: 'Hồi quy đa thức (Bậc 2)', badge: 'badge-medium', complexity: 'Trung bình' },
            { key: 'DecisionTree', name: 'Cây quyết định (Tree)', badge: 'badge-medium', complexity: 'Trung bình' },
            { key: 'LinearRegression', name: 'Hồi quy tuyến tính', badge: 'badge-simple', complexity: 'Đơn giản' },
            { key: 'RidgeRegression', name: 'Hồi quy Ridge (L2)', badge: 'badge-simple', complexity: 'Đơn giản' },
            { key: 'LassoRegression', name: 'Hồi quy Lasso (L1)', badge: 'badge-simple', complexity: 'Đơn giản' },
            { key: 'KNN', name: 'KNN / Khớp mẫu mô phỏng', badge: 'badge-medium', complexity: 'Thực nghiệm' }
        ];

        let html = '';
        for (const item of modelKeys) {
            const m = models[item.key] || {};
            const res = runModelInference(item.key, state.v1, state.v2);
            const [v1_sim, v2_sim] = simulateForward(res.dims[0], res.dims[1], res.dims[2], res.dims[3]);
            const avgErr = ((Math.abs(v1_sim - state.v1) + Math.abs(v2_sim - state.v2)) / 2).toFixed(2);
            const isActive = (item.key === state.currentModel);

            html += `
                <tr class="${isActive ? 'active-row' : ''}">
                    <td><strong>${item.name}</strong></td>
                    <td><span class="badge-tag ${item.badge}">${item.complexity}</span></td>
                    <td><code>${m.mae ? m.mae.toFixed(1) + ' mm' : '~30 mm'}</code></td>
                    <td><code>${m.r2 ? m.r2.toFixed(3) : '-'}</code></td>
                    <td><strong>${res.dims[0].toFixed(0)}</strong> mm</td>
                    <td><strong>${res.dims[1].toFixed(0)}</strong> mm</td>
                    <td><strong>${res.dims[2].toFixed(0)}</strong> mm</td>
                    <td><strong>${res.dims[3].toFixed(0)}</strong> mm</td>
                    <td><span class="val-num-sm">${avgErr} dB</span></td>
                    <td>
                        <button class="btn btn-sm ${isActive ? 'btn-primary' : 'btn-secondary'} select-model-btn" data-model="${item.key}">
                            ${isActive ? 'Đang chọn' : 'Chọn'}
                        </button>
                    </td>
                </tr>
            `;
        }

        comparisonTableBody.innerHTML = html;

        document.querySelectorAll('.select-model-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const targetModel = e.target.getAttribute('data-model');
                selectModel(targetModel);
            });
        });
    }

    function selectModel(modelKey) {
        state.currentModel = modelKey;
        document.querySelectorAll('.tab-btn').forEach(b => {
            b.classList.toggle('active', b.getAttribute('data-model') === modelKey);
        });

        const descriptions = {
            LinearRegression: 'Hồi quy tuyến tính d = W · v + b: Công thức đại số trực tiếp, cực kỳ đơn giản, độ phức tạp tính toán O(1).',
            RidgeRegression: 'Hồi quy tuyến tính có điều chuẩn L2: Giữ trọng số ổn định, chống hiện tượng đa cộng tuyến.',
            LassoRegression: 'Hồi quy Lasso (L1 Regularization): Triệt tiêu các trọng số dư thừa, tăng tính thưa và tối ưu biến giải thích.',
            PolynomialRegression: 'Hồi quy đa thức bậc 2: Bổ sung các số hạng phi tuyến v₁², v₂², v₁·v₂ giúp nắm bắt độ cong bề mặt đáp ứng.',
            DecisionTree: 'Cây quyết định hồi quy (Decision Tree): Phân vùng không gian mục tiêu thành các khối hộp nhị phân độc lập.',
            NeuralNetwork: 'Mạng nơ-ron nhân tạo MLP Deep Feedforward (64-64-32 nơ-ron, ReLU): Khả năng xấp xỉ phổ biến mọi hàm phi tuyến phức tạp.',
            KNN: 'K Láng giềng gần nhất (KNN / Instance Search): Tìm kiếm trực tiếp cấu hình mô phỏng thực tế gần nhất trong tập 614,656 mẫu.',
            Ensemble: 'Mô hình kết hợp (Stacking Ensemble): Tổng hợp dự đoán từ Mạng Nơ-ron (45%), Đa thức (35%) và KNN (20%) để đạt độ ổn định cao nhất.'
        };

        modelDescription.textContent = descriptions[modelKey] || '';
        updateAll();
    }

    // --- EVENT LISTENERS ---

    sliderV1.addEventListener('input', (e) => {
        state.v1 = parseFloat(e.target.value);
        inputV1.value = state.v1.toFixed(2);
        updateAll();
    });

    inputV1.addEventListener('change', (e) => {
        let val = parseFloat(e.target.value);
        if (isNaN(val)) val = -15.0;
        val = Math.max(-55.8, Math.min(0.0, val));
        state.v1 = val;
        sliderV1.value = val;
        e.target.value = val.toFixed(2);
        updateAll();
    });

    sliderV2.addEventListener('input', (e) => {
        state.v2 = parseFloat(e.target.value);
        inputV2.value = state.v2.toFixed(2);
        updateAll();
    });

    inputV2.addEventListener('change', (e) => {
        let val = parseFloat(e.target.value);
        if (isNaN(val)) val = -5.0;
        val = Math.max(-47.8, Math.min(0.0, val));
        state.v2 = val;
        sliderV2.value = val;
        e.target.value = val.toFixed(2);
        updateAll();
    });

    snapStep.addEventListener('change', (e) => {
        state.snapToGrid = e.target.checked;
        updateAll();
    });

    document.querySelectorAll('.btn-preset').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            state.v1 = parseFloat(btn.getAttribute('data-v1'));
            state.v2 = parseFloat(btn.getAttribute('data-v2'));

            sliderV1.value = state.v1;
            inputV1.value = state.v1.toFixed(2);
            sliderV2.value = state.v2;
            inputV2.value = state.v2.toFixed(2);

            updateAll();
        });
    });

    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const modelKey = btn.getAttribute('data-model');
            selectModel(modelKey);
        });
    });

    refreshTableBtn.addEventListener('click', () => {
        renderTable();
    });

    copyBtn.addEventListener('click', () => {
        const text = `d1 = ${state.predictions.d1} mm\nd2 = ${state.predictions.d2} mm\nd3 = ${state.predictions.d3} mm\nd4 = ${state.predictions.d4} mm\n(Target: v1=${state.v1} dB, v2=${state.v2} dB, Model: ${state.currentModel})`;
        navigator.clipboard.writeText(text).then(() => {
            const orig = copyBtn.innerHTML;
            copyBtn.innerHTML = `✓ Đã sao chép!`;
            setTimeout(() => { copyBtn.innerHTML = orig; }, 1800);
        });
    });

    exportCsvBtn.addEventListener('click', () => {
        const rows = [
            ['Parameter', 'Value', 'Unit', 'Description'],
            ['Target_1_v1', state.v1, 'dB', 'Cột 5 mục tiêu'],
            ['Target_2_v2', state.v2, 'dB', 'Cột 6 mục tiêu'],
            ['Model_Selected', state.currentModel, '', 'Thuật toán ML'],
            ['Dimension_1_d1', state.predictions.d1, 'mm', 'Cột 1 dự đoán'],
            ['Dimension_2_d2', state.predictions.d2, 'mm', 'Cột 2 dự đoán'],
            ['Dimension_3_d3', state.predictions.d3, 'mm', 'Cột 3 dự đoán'],
            ['Dimension_4_d4', state.predictions.d4, 'mm', 'Cột 4 dự đoán'],
            ['Simulated_v1', state.forwardVerification.v1_sim.toFixed(2), 'dB', 'Mô phỏng Digital Twin'],
            ['Simulated_v2', state.forwardVerification.v2_sim.toFixed(2), 'dB', 'Mô phỏng Digital Twin'],
            ['Delta_v1', state.forwardVerification.err1.toFixed(2), 'dB', 'Sai số tuyệt đối v1'],
            ['Delta_v2', state.forwardVerification.err2.toFixed(2), 'dB', 'Sai số tuyệt đối v2']
        ];

        const csvContent = 'data:text/csv;charset=utf-8,' + rows.map(e => e.join(',')).join('\n');
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement('a');
        link.setAttribute('href', encodedUri);
        link.setAttribute('download', `antenna_design_v1_${state.v1}_v2_${state.v2}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // Theme Toggle Listener
    themeToggle.addEventListener('click', () => {
        state.theme = (state.theme === 'light') ? 'dark' : 'light';
        applyTheme(state.theme);
    });

    // Render KaTeX math when script loads
    if (window.renderMathInElement) {
        window.renderMathInElement(document.body, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false }
            ],
            throwOnError: false
        });
    }

    // Initial Execution
    updateAll();
});
