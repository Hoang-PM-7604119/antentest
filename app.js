// Multi-Output ML Predictor & Live Computation Visualizer (6 Outputs, 2 Inputs)
// Pure Continuous Regression Models: Outputs real decimal numbers (số lẻ, số thập phân)
document.addEventListener('DOMContentLoaded', () => {
    if (!window.ANTENNA_MODELS_DATA) {
        console.error('Không tìm thấy dữ liệu mô hình window.ANTENNA_MODELS_DATA');
        return;
    }

    const { models, forward_simulator, samples } = window.ANTENNA_MODELS_DATA;

    // Model configuration metadata (11 models - Continuous ML models prioritized)
    const MODEL_META = [
        {
            id: 'Ensemble',
            name: 'Stacking Ensemble Liên Tục',
            icon: '🏆',
            rank: 'Khuyên dùng',
            rankClass: 'rank-top',
            sub: '0.40·MLP + 0.35·SVM + 0.25·Poly (Số thực)'
        },
        {
            id: 'SVM',
            name: 'Support Vector Machine (SVR)',
            icon: '🎯',
            rank: 'Lề cực đại',
            rankClass: 'rank-strong',
            sub: 'Siêu phẳng tối ưu Epsilon-Insensitive'
        },
        {
            id: 'NeuralNetwork',
            name: 'Mạng Nơ-ron Nhân Tạo (Deep MLP)',
            icon: '🧠',
            rank: 'Deep Learning',
            rankClass: 'rank-strong',
            sub: '3 tầng ẩn 64-64-32 ReLU (Phi tuyến)'
        },
        {
            id: 'PolynomialRegression',
            name: 'Hồi quy Đa thức bậc 2',
            icon: '📐',
            rank: 'Mặt cong phi tuyến',
            rankClass: 'rank-std',
            sub: 'Phương trình đa thức giải tích 6 chiều'
        },
        {
            id: 'DecisionTree',
            name: 'Cây Quyết Định (Decision Tree)',
            icon: '🌳',
            rank: 'Cây hồi quy',
            rankClass: 'rank-std',
            sub: 'Dự đoán giá trị lá đa đầu ra liên tục'
        },
        {
            id: 'LinearRegression',
            name: 'Hồi quy Tuyến tính Đa biến',
            icon: '⚡',
            rank: 'Chuẩn giải tích',
            rankClass: 'rank-std',
            sub: 'y = W·x + b (Đại số ma trận)'
        },
        {
            id: 'RidgeRegression',
            name: 'Hồi quy Ridge (L2)',
            icon: '⚡',
            rank: 'Điều chuẩn L2',
            rankClass: 'rank-std',
            sub: 'Thu hẹp phương sai, lề ổn định'
        },
        {
            id: 'LassoRegression',
            name: 'Hồi quy Lasso (L1)',
            icon: '⚡',
            rank: 'Điều chuẩn L1',
            rankClass: 'rank-std',
            sub: 'Triệt tiêu trọng số dư thừa'
        },
        {
            id: 'KNN',
            name: 'KNN Nội suy Trọng số (k=3)',
            icon: '⭐',
            rank: 'Nội suy lân cận',
            rankClass: 'rank-std',
            sub: 'Nghịch đảo khoảng cách 3 mẫu lân cận'
        },
        {
            id: 'KNN_5',
            name: 'KNN Làm mượt Đa điểm (k=5)',
            icon: '⭐',
            rank: 'Làm mượt 5-NN',
            rankClass: 'rank-std',
            sub: 'Nội suy nghịch đảo khoảng cách 5 mẫu'
        },
        {
            id: 'KNN_1',
            name: 'KNN Tham chiếu Thực nghiệm (k=1)',
            icon: '🔍',
            rank: 'Tham chiếu thực tế',
            rankClass: 'rank-std',
            sub: 'Mẫu thô gần nhất trong tập dữ liệu'
        }
    ];

    const OUTPUT_NAMES = ['t3', 't2', 't1', 'z3', 'z2', 'z1'];
    const OUTPUT_LABELS = ['Cột 1 (t₃)', 'Cột 2 (t₂)', 'Cột 3 (t₁)', 'Cột 4 (z₃)', 'Cột 5 (z₂)', 'Cột 6 (z₁)'];
    const OUTPUT_COLORS = ['#2563eb', '#0891b2', '#059669', '#d97706', '#8b5cf6', '#e11d48'];

    // State
    const state = {
        x1: -15.0,
        x2: -5.0,
        activeModel: 'Ensemble',
        allPredictions: {},
        vizTab: 'radar',
        compareModel: 'none',
        animTick: 0
    };

    // DOM Elements - Section 1
    const sliderX1 = document.getElementById('sliderX1');
    const inputX1 = document.getElementById('inputX1');
    const x1Badge = document.getElementById('x1Badge');

    const sliderX2 = document.getElementById('sliderX2');
    const inputX2 = document.getElementById('inputX2');
    const x2Badge = document.getElementById('x2Badge');

    const resetBtn = document.getElementById('resetBtn');
    const exportCsvBtn = document.getElementById('exportCsvBtn');

    // Section 2
    const comparisonTableBody = document.getElementById('comparisonTableBody');

    // Section 3
    const activeModelTitle = document.getElementById('activeModelTitle');
    const valY1 = document.getElementById('valY1');
    const valY2 = document.getElementById('valY2');
    const valY3 = document.getElementById('valY3');
    const valY4 = document.getElementById('valY4');
    const valY5 = document.getElementById('valY5');
    const valY6 = document.getElementById('valY6');

    const barY1 = document.getElementById('barY1');
    const barY2 = document.getElementById('barY2');
    const barY3 = document.getElementById('barY3');
    const barY4 = document.getElementById('barY4');
    const barY5 = document.getElementById('barY5');
    const barY6 = document.getElementById('barY6');

    const targetX1 = document.getElementById('targetX1');
    const targetX2 = document.getElementById('targetX2');
    const simX1 = document.getElementById('simX1');
    const simX2 = document.getElementById('simX2');
    const deltaX1 = document.getElementById('deltaX1');
    const deltaX2 = document.getElementById('deltaX2');
    const statusBadge = document.getElementById('statusBadge');

    const terminalTitle = document.getElementById('terminalTitle');
    const formulaCode = document.getElementById('formulaCode');
    const copyActiveBtn = document.getElementById('copyActiveBtn');
    const copyFormulaBtn = document.getElementById('copyFormulaBtn');

    // Section 4 Visualizer Elements
    const tabRadar = document.getElementById('tabRadar');
    const tabNetwork = document.getElementById('tabNetwork');
    const tabBreakdown = document.getElementById('tabBreakdown');
    const vizCanvas = document.getElementById('vizCanvas');
    const vizDynamicTitle = document.getElementById('vizDynamicTitle');
    const vizCompareSelect = document.getElementById('vizCompareSelect');
    const vizFooterHint = document.getElementById('vizFooterHint');

    const pipeZ1 = document.getElementById('pipeZ1');
    const pipeZ2 = document.getElementById('pipeZ2');
    const pipeStep2Name = document.getElementById('pipeStep2Name');
    const pipeStep2Desc = document.getElementById('pipeStep2Desc');
    const pipeStep2Math = document.getElementById('pipeStep2Math');
    const pipeModelType = document.getElementById('pipeModelType');
    const pipeDelta = document.getElementById('pipeDelta');

    const decompModelName = document.getElementById('decompModelName');
    const decompTableBody = document.getElementById('decompTableBody');

    // Canvas 2D Context
    let ctx = null;
    if (vizCanvas) {
        ctx = vizCanvas.getContext('2d');
    }

    // Clamp helper: Keeps output in physical boundary [5.0, 140.0], preserves real continuous decimal (2 decimal digits)
    function clampOutput(val) {
        const v = Math.max(5.0, Math.min(140.0, val));
        return Math.round(v * 100) / 100;
    }

    // 1. Linear Regression (6 outputs, continuous decimal)
    function predictLinear(x1, x2) {
        const m = models.LinearRegression;
        return [0, 1, 2, 3, 4, 5].map(i => clampOutput(m.coef[i][0] * x1 + m.coef[i][1] * x2 + m.intercept[i]));
    }

    // 2. SVM (Linear SVR) (6 outputs, continuous decimal)
    function predictSVM(x1, x2) {
        const m = models.SVM;
        return [0, 1, 2, 3, 4, 5].map(i => clampOutput(m.coef[i][0] * x1 + m.coef[i][1] * x2 + m.intercept[i]));
    }

    // 3. Ridge Regression (6 outputs, continuous decimal)
    function predictRidge(x1, x2) {
        const m = models.RidgeRegression;
        return [0, 1, 2, 3, 4, 5].map(i => clampOutput(m.coef[i][0] * x1 + m.coef[i][1] * x2 + m.intercept[i]));
    }

    // 4. Lasso Regression (6 outputs, continuous decimal)
    function predictLasso(x1, x2) {
        const m = models.LassoRegression;
        return [0, 1, 2, 3, 4, 5].map(i => clampOutput(m.coef[i][0] * x1 + m.coef[i][1] * x2 + m.intercept[i]));
    }

    // 5. Polynomial Regression (Degree 2) (6 outputs, continuous decimal)
    function predictPolynomial(x1, x2) {
        const m = models.PolynomialRegression;
        const feats = [1.0, x1, x2, x1 * x1, x1 * x2, x2 * x2];
        const res = [];
        for (let i = 0; i < 6; i++) {
            let sum = 0;
            for (let f = 0; f < feats.length; f++) {
                sum += m.coef[i][f] * feats[f];
            }
            res.push(clampOutput(sum));
        }
        return res;
    }

    // 6. Decision Tree Regressor (6 outputs, continuous decimal)
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

    // 7. Neural Network MLP (6 outputs, continuous decimal)
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

    // 8. KNN / Database Matching (Reference)
    function predictKNN(x1, x2, k = 3) {
        if (!samples || samples.length === 0) return predictMLP(x1, x2);
        const dists = samples.map(s => {
            const d1 = s.v1 - x1;
            const d2 = s.v2 - x2;
            return {
                dist: Math.sqrt(d1 * d1 + d2 * d2),
                y: [s.d1, s.d2, s.d3, s.d4, s.d5, s.d6]
            };
        });
        dists.sort((a, b) => a.dist - b.dist);

        if (k === 1) {
            return dists[0].y.map(v => clampOutput(v));
        }

        let sumW = 0;
        let sumY = [0, 0, 0, 0, 0, 0];
        for (let i = 0; i < Math.min(k, dists.length); i++) {
            const w = 1.0 / (dists[i].dist + 1e-6);
            sumW += w;
            for (let j = 0; j < 6; j++) sumY[j] += dists[i].y[j] * w;
        }
        return sumY.map(v => clampOutput(v / sumW));
    }

    // 9. Pure Continuous Stacking Ensemble (0.40 MLP + 0.35 SVM + 0.25 Poly - 100% mathematical regression)
    function predictEnsemble(x1, x2) {
        const mlp = predictMLP(x1, x2);
        const svm = predictSVM(x1, x2);
        const poly = predictPolynomial(x1, x2);
        return [0, 1, 2, 3, 4, 5].map(i => clampOutput(0.40 * mlp[i] + 0.35 * svm[i] + 0.25 * poly[i]));
    }

    // Forward Verification: [y1..y6] -> [x1_sim, x2_sim]
    function simulateForward(y) {
        const fwd = forward_simulator;
        let cur = y.map((val, i) => (val - fwd.scaler_x.mean[i]) / fwd.scaler_x.scale[i]);

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

    // Compute predictions for ALL 11 models
    function computeAllModels(x1, x2) {
        const results = {};
        const predictors = {
            Ensemble: () => predictEnsemble(x1, x2),
            SVM: () => predictSVM(x1, x2),
            NeuralNetwork: () => predictMLP(x1, x2),
            PolynomialRegression: () => predictPolynomial(x1, x2),
            DecisionTree: () => predictDecisionTree(x1, x2),
            LinearRegression: () => predictLinear(x1, x2),
            RidgeRegression: () => predictRidge(x1, x2),
            LassoRegression: () => predictLasso(x1, x2),
            KNN: () => predictKNN(x1, x2, 3),
            KNN_5: () => predictKNN(x1, x2, 5),
            KNN_1: () => predictKNN(x1, x2, 1)
        };

        MODEL_META.forEach(meta => {
            const y = predictors[meta.id]();
            const [sim1, sim2] = simulateForward(y);
            const err1 = Math.abs(sim1 - x1);
            const err2 = Math.abs(sim2 - x2);
            const meanErr = (err1 + err2) / 2;
            results[meta.id] = {
                y,
                sim: [sim1, sim2],
                err: [err1, err2],
                meanErr
            };
        });

        return results;
    }

    // Render comparison table
    function renderComparisonTable(results) {
        let html = '';
        MODEL_META.forEach(meta => {
            const r = results[meta.id];
            const isActive = (meta.id === state.activeModel);
            const y = r.y;
            const sim = r.sim;
            const meanErr = r.meanErr;

            let errClass = 'err-good';
            if (meanErr < 1.0) errClass = 'err-excellent';
            else if (meanErr > 3.0) errClass = 'err-fair';

            const activeClass = isActive ? 'active-row' : '';

            html += `
            <tr class="${activeClass}" data-model="${meta.id}">
                <td class="col-algo">
                    <div class="algo-cell">
                        <span>${meta.icon}</span>
                        <div>
                            <span class="algo-name">${meta.name}</span>
                            <div style="font-size: 0.7rem; color: var(--text-dim);">${meta.sub}</div>
                        </div>
                    </div>
                </td>
                <td class="col-dim"><span class="dim-tag tag-y1">${y[0].toFixed(2)}</span></td>
                <td class="col-dim"><span class="dim-tag tag-y2">${y[1].toFixed(2)}</span></td>
                <td class="col-dim"><span class="dim-tag tag-y3">${y[2].toFixed(2)}</span></td>
                <td class="col-dim"><span class="dim-tag tag-y4">${y[3].toFixed(2)}</span></td>
                <td class="col-dim"><span class="dim-tag tag-y5">${y[4].toFixed(2)}</span></td>
                <td class="col-dim"><span class="dim-tag tag-y6">${y[5].toFixed(2)}</span></td>
                <td class="col-sim"><span class="sim-val">${sim[0].toFixed(2)}, ${sim[1].toFixed(2)}</span></td>
                <td class="col-err"><span class="err-badge ${errClass}">±${meanErr.toFixed(2)}</span></td>
                <td class="col-rating"><span class="rank-badge ${meta.rankClass}">${meta.rank}</span></td>
            </tr>`;
        });

        comparisonTableBody.innerHTML = html;

        // Attach click listeners to rows
        comparisonTableBody.querySelectorAll('tr').forEach(row => {
            row.addEventListener('click', () => {
                const modelId = row.getAttribute('data-model');
                if (modelId && state.activeModel !== modelId) {
                    state.activeModel = modelId;
                    updateUI();
                }
            });
        });
    }

    // Update active model UI details
    function updateActiveModelDetails(results) {
        const activeRes = results[state.activeModel];
        if (!activeRes) return;

        const meta = MODEL_META.find(m => m.id === state.activeModel) || MODEL_META[0];
        activeModelTitle.textContent = `${meta.icon} ${meta.name}`;

        const y = activeRes.y;
        valY1.textContent = y[0].toFixed(2);
        valY2.textContent = y[1].toFixed(2);
        valY3.textContent = y[2].toFixed(2);
        valY4.textContent = y[3].toFixed(2);
        valY5.textContent = y[4].toFixed(2);
        valY6.textContent = y[5].toFixed(2);

        barY1.style.width = `${((y[0] - 5) / 135) * 100}%`;
        barY2.style.width = `${((y[1] - 5) / 135) * 100}%`;
        barY3.style.width = `${((y[2] - 5) / 135) * 100}%`;
        barY4.style.width = `${((y[3] - 5) / 135) * 100}%`;
        barY5.style.width = `${((y[4] - 5) / 135) * 100}%`;
        barY6.style.width = `${((y[5] - 5) / 135) * 100}%`;

        // Verification banner
        targetX1.textContent = state.x1.toFixed(2);
        targetX2.textContent = state.x2.toFixed(2);
        simX1.textContent = activeRes.sim[0].toFixed(2);
        simX2.textContent = activeRes.sim[1].toFixed(2);
        deltaX1.textContent = activeRes.err[0].toFixed(2);
        deltaX2.textContent = activeRes.err[1].toFixed(2);

        const avgErr = activeRes.meanErr;
        if (avgErr < 1.0) {
            statusBadge.textContent = 'Rất sát (Lệch < 1.0)';
            statusBadge.style.color = '#059669';
            statusBadge.style.background = 'rgba(16, 185, 129, 0.12)';
        } else if (avgErr < 2.5) {
            statusBadge.textContent = 'Khớp tốt (Lệch < 2.5)';
            statusBadge.style.color = '#2563eb';
            statusBadge.style.background = 'rgba(37, 99, 235, 0.1)';
        } else {
            statusBadge.textContent = `Lệch: ${avgErr.toFixed(2)}`;
            statusBadge.style.color = '#d97706';
            statusBadge.style.background = 'rgba(217, 119, 6, 0.1)';
        }

        renderFormula();
    }

    // Render formula for active model
    function renderFormula() {
        const x1 = state.x1;
        const x2 = state.x2;
        const activeRes = state.allPredictions[state.activeModel];
        if (!activeRes) return;
        const y = activeRes.y;
        let txt = '';

        if (state.activeModel === 'SVM') {
            const m = models.SVM;
            terminalTitle.textContent = 'svm_linear_svr_eval.py';
            txt = `# Support Vector Machine (Linear SVR - Support Vector Regression):\n` +
            `# Siêu phẳng hồi quy giải tích: y_i = w_i1·x₁ + w_i2·x₂ + b_i (Dự đoán số thực liên tục):\n` +
            `# Giá trị đầu vào: x₁ = ${x1.toFixed(2)} dB, x₂ = ${x2.toFixed(2)} dB\n` +
            [0, 1, 2, 3, 4, 5].map(i => {
                const w = m.coef[i];
                const b = m.intercept[i];
                return `y${i+1} (${OUTPUT_NAMES[i]}) = (${w[0].toFixed(4)})·(${x1}) + (${w[1].toFixed(4)})·(${x2}) + (${b.toFixed(4)}) = ${y[i].toFixed(2)}`;
            }).join('\n');
        } else if (state.activeModel === 'LinearRegression') {
            const m = models.LinearRegression;
            terminalTitle.textContent = 'linear_regression_eval.py';
            txt = `# Hồi quy Tuyến tính Đa biến (Multivariate Linear Regression):\n` +
            `# Công thức ma trận: y = W · x + b (Đầu ra số thực liên tục):\n` +
            [0, 1, 2, 3, 4, 5].map(i => {
                const w = m.coef[i];
                const b = m.intercept[i];
                return `y${i+1} (${OUTPUT_NAMES[i]}) = ${w[0].toFixed(4)}·(${x1}) + (${w[1].toFixed(4)})·(${x2}) + ${b.toFixed(4)} = ${y[i].toFixed(2)}`;
            }).join('\n');
        } else if (state.activeModel === 'LassoRegression') {
            const m = models.LassoRegression;
            terminalTitle.textContent = 'lasso_regression_eval.py';
            txt = `# Hồi quy Lasso (L1 Regularization - Triệt tiêu trọng số dư thừa):\n` +
            [0, 1, 2, 3, 4, 5].map(i => {
                const w = m.coef[i];
                const b = m.intercept[i];
                return `y${i+1} (${OUTPUT_NAMES[i]}) = ${w[0].toFixed(4)}·(${x1}) + (${w[1].toFixed(4)})·(${x2}) + ${b.toFixed(4)} = ${y[i].toFixed(2)}`;
            }).join('\n');
        } else if (state.activeModel === 'RidgeRegression') {
            const m = models.RidgeRegression;
            terminalTitle.textContent = 'ridge_regression_eval.py';
            txt = `# Hồi quy Ridge (L2 Regularization - Kiểm soát phương sai lề):\n` +
            [0, 1, 2, 3, 4, 5].map(i => {
                const w = m.coef[i];
                const b = m.intercept[i];
                return `y${i+1} (${OUTPUT_NAMES[i]}) = ${w[0].toFixed(4)}·(${x1}) + (${w[1].toFixed(4)})·(${x2}) + ${b.toFixed(4)} = ${y[i].toFixed(2)}`;
            }).join('\n');
        } else if (state.activeModel === 'PolynomialRegression') {
            const p = models.PolynomialRegression;
            terminalTitle.textContent = 'polynomial_surface_eval.py';
            txt = `# Hồi quy Đa thức bậc 2: y_i = c₀ + c₁·x₁ + c₂·x₂ + c₃·x₁² + c₄·(x₁·x₂) + c₅·x₂²\n` +
            `# Phương trình mặt cong giải tích phi tuyến (Đầu ra số thực liên tục):\n` +
            [0, 1, 2, 3, 4, 5].map(i => {
                const c = p.coef[i];
                return `y${i+1} (${OUTPUT_NAMES[i]}) = ${c[0].toFixed(4)} + (${c[1].toFixed(4)})·(${x1.toFixed(2)}) + (${c[2].toFixed(4)})·(${x2.toFixed(2)}) + (${c[3].toFixed(4)})·x₁² + (${c[4].toFixed(4)})·x₁x₂ + (${c[5].toFixed(4)})·x₂² = ${y[i].toFixed(2)}`;
            }).join('\n');
        } else if (state.activeModel === 'Ensemble') {
            terminalTitle.textContent = 'stacking_ensemble_eval.py';
            txt = `# Mô hình Tổ Hợp Hồi Quy Liên Tục (Stacking Ensemble Blending):\n` +
            `# Kết hợp 100% từ các mô hình hàm toán học liên tục (Không sử dụng tra cứu database):\n` +
            `# y = 0.40 · y(MLP) + 0.35 · y(SVM) + 0.25 · y(Poly)\n` +
            [0, 1, 2, 3, 4, 5].map(i => {
                return `y${i+1} (${OUTPUT_NAMES[i]}) = 0.40·y${i+1}(MLP) + 0.35·y${i+1}(SVM) + 0.25·y${i+1}(Poly) = ${y[i].toFixed(2)}`;
            }).join('\n');
        } else if (state.activeModel === 'NeuralNetwork') {
            terminalTitle.textContent = 'mlp_neural_network_eval.py';
            txt = `# Mạng Nơ-ron Nhân Tạo Sâu (Deep Multi-Layer Perceptron - MLP Regressor):\n` +
            `# Xấp xỉ hàm phi tuyến 6 chiều qua 3 tầng ẩn 64-64-32 nơ-ron (Hàm kích hoạt ReLU):\n` +
            `h⁽¹⁾ = ReLU(W⁽¹⁾·x + b⁽¹⁾)       # Tầng ẩn 1: 64 nơ-ron\n` +
            `h⁽²⁾ = ReLU(W⁽²⁾·h⁽¹⁾ + b⁽²⁾)     # Tầng ẩn 2: 64 nơ-ron\n` +
            `h⁽³⁾ = ReLU(W⁽³⁾·h⁽²⁾ + b⁽³⁾)     # Tầng ẩn 3: 32 nơ-ron\n` +
            `y    = W⁽⁴⁾·h⁽³⁾ + b⁽⁴⁾           # Tầng đầu ra 6 nơ-ron (Số thập phân liên tục):\n` +
            `--> y₁(${OUTPUT_NAMES[0]})=${y[0].toFixed(2)}, y₂(${OUTPUT_NAMES[1]})=${y[1].toFixed(2)}, y₃(${OUTPUT_NAMES[2]})=${y[2].toFixed(2)}, y₄(${OUTPUT_NAMES[3]})=${y[3].toFixed(2)}, y₅(${OUTPUT_NAMES[4]})=${y[4].toFixed(2)}, y₆(${OUTPUT_NAMES[5]})=${y[5].toFixed(2)}`;
        } else if (state.activeModel === 'DecisionTree') {
            terminalTitle.textContent = 'decision_tree_eval.py';
            txt = `# Cây Quyết Định Hồi Quy (Decision Tree Regressor - Depth 8):\n` +
            `# Duyệt cây nhị phân đa đầu ra với ngưỡng x₁ và x₂:\n` +
            `--> Giá trị trung bình tại nút lá (Số thập phân liên tục):\n` +
            `y₁ (${OUTPUT_NAMES[0]}) = ${y[0].toFixed(2)}\n` +
            `y₂ (${OUTPUT_NAMES[1]}) = ${y[1].toFixed(2)}\n` +
            `y₃ (${OUTPUT_NAMES[2]}) = ${y[2].toFixed(2)}\n` +
            `y₄ (${OUTPUT_NAMES[3]}) = ${y[3].toFixed(2)}\n` +
            `y₅ (${OUTPUT_NAMES[4]}) = ${y[4].toFixed(2)}\n` +
            `y₆ (${OUTPUT_NAMES[5]}) = ${y[5].toFixed(2)}`;
        } else if (state.activeModel === 'KNN') {
            terminalTitle.textContent = 'knn_k3_interpolation.py';
            txt = `# K-NN Nội suy Trọng số (k=3 mẫu lân cận):\n` +
            `# Khoảng cách Euclidean: D_i = √[(x₁ - x₁,i)² + (x₂ - x₂,i)²]\n` +
            `# Trọng số: w_i = 1 / (D_i + 1e-6)\n` +
            `# y = (∑ w_i · y_i) / (∑ w_i) (Cho ra số thập phân nội suy liên tục):\n` +
            `--> y₁(${OUTPUT_NAMES[0]})=${y[0].toFixed(2)}, y₂(${OUTPUT_NAMES[1]})=${y[1].toFixed(2)}, y₃(${OUTPUT_NAMES[2]})=${y[2].toFixed(2)}, y₄(${OUTPUT_NAMES[3]})=${y[3].toFixed(2)}, y₅(${OUTPUT_NAMES[4]})=${y[4].toFixed(2)}, y₆(${OUTPUT_NAMES[5]})=${y[5].toFixed(2)}`;
        } else if (state.activeModel === 'KNN_5') {
            terminalTitle.textContent = 'knn_k5_interpolation.py';
            txt = `# K-NN Nội suy Làm mượt (k=5 mẫu lân cận):\n` +
            `# Khoảng cách: D_i = √[(x₁ - x₁,i)² + (x₂ - x₂,i)²]\n` +
            `# Trọng số: w_i = 1 / (D_i + 1e-6)\n` +
            `# y = (∑ w_i · y_i) / (∑ w_i) (Số thập phân làm mượt liên tục):\n` +
            `--> y₁(${OUTPUT_NAMES[0]})=${y[0].toFixed(2)}, y₂(${OUTPUT_NAMES[1]})=${y[1].toFixed(2)}, y₃(${OUTPUT_NAMES[2]})=${y[2].toFixed(2)}, y₄(${OUTPUT_NAMES[3]})=${y[3].toFixed(2)}, y₅(${OUTPUT_NAMES[4]})=${y[4].toFixed(2)}, y₆(${OUTPUT_NAMES[5]})=${y[5].toFixed(2)}`;
        } else if (state.activeModel === 'KNN_1') {
            terminalTitle.textContent = 'knn_k1_reference.py';
            txt = `# Mẫu Thực Nghiệm Tham Chiếu (1-NN k=1):\n` +
            `# Tìm điểm thực nghiệm i* = argmin D(x, x_i) gần nhất trong tập 12,000 mẫu:\n` +
            `--> y₁(${OUTPUT_NAMES[0]})=${y[0].toFixed(2)}, y₂(${OUTPUT_NAMES[1]})=${y[1].toFixed(2)}, y₃(${OUTPUT_NAMES[2]})=${y[2].toFixed(2)}, y₄(${OUTPUT_NAMES[3]})=${y[3].toFixed(2)}, y₅(${OUTPUT_NAMES[4]})=${y[4].toFixed(2)}, y₆(${OUTPUT_NAMES[5]})=${y[5].toFixed(2)}`;
        } else {
            terminalTitle.textContent = 'model_eval.py';
            txt = `Mô hình: ${state.activeModel}\n` +
            [0, 1, 2, 3, 4, 5].map(i => `y${i+1} (${OUTPUT_NAMES[i]}) = ${y[i].toFixed(2)}`).join('\n');
        }
        formulaCode.textContent = txt;
    }

    // =========================================================================
    // SECTION 4: COMPUTATION MECHANISM & LIVE VISUALIZER
    // =========================================================================

    // Update 4-step pipeline and dimension decomposition table
    function updatePipelineAndDecomp() {
        if (!pipeZ1 || !pipeZ2) return;

        // Step 1 Z-Score
        const mlp = models.NeuralNetwork;
        const z1 = (state.x1 - mlp.scaler_x.mean[0]) / mlp.scaler_x.scale[0];
        const z2 = (state.x2 - mlp.scaler_x.mean[1]) / mlp.scaler_x.scale[1];
        pipeZ1.textContent = z1.toFixed(3);
        pipeZ2.textContent = z2.toFixed(3);

        // Step 2 Core Algo info
        const meta = MODEL_META.find(m => m.id === state.activeModel) || MODEL_META[0];
        pipeStep2Name.textContent = meta.name;
        if (state.activeModel === 'SVM') {
            pipeStep2Desc.textContent = 'Nhân ma trận siêu phẳng tối ưu lề lồi giải tích L2 (Epsilon-Insensitive):';
            pipeStep2Math.textContent = 'y_i = w_{i1}·x₁ + w_{i2}·x₂ + b_i';
            pipeModelType.textContent = 'Siêu phẳng lề cực đại giải tích';
        } else if (state.activeModel === 'NeuralNetwork') {
            pipeStep2Desc.textContent = 'Lan truyền tiến qua 3 tầng ẩn 64-64-32 nơ-ron với hàm kích hoạt ReLU:';
            pipeStep2Math.textContent = 'h^(l) = ReLU(W^(l)·h^(l-1) + b^(l))';
            pipeModelType.textContent = 'Mạng Deep Feedforward 3 tầng ẩn';
        } else if (state.activeModel === 'PolynomialRegression') {
            pipeStep2Desc.textContent = 'Mở rộng đặc trưng phi tuyến và nhân ma trận hệ số mặt cong 6 chiều:';
            pipeStep2Math.textContent = 'y_i = c₀ + c₁·x₁ + c₂·x₂ + c₃·x₁² + c₄·x₁x₂ + c₅·x₂²';
            pipeModelType.textContent = 'Hồi quy đa thức bậc 2 phi tuyến';
        } else if (state.activeModel === 'Ensemble') {
            pipeStep2Desc.textContent = 'Tổng hợp trọng số tối ưu từ 3 mô hình liên tục (Không tra cứu database):';
            pipeStep2Math.textContent = 'y = 0.40·MLP + 0.35·SVM + 0.25·Poly';
            pipeModelType.textContent = 'Tổ hợp hồi quy liên tục (Stacking)';
        } else if (state.activeModel === 'DecisionTree') {
            pipeStep2Desc.textContent = 'Duyệt cây nhị phân đa đầu ra từ nút gốc đến nút lá:';
            pipeStep2Math.textContent = 'Nút lá: trung bình vector 6 chiều';
            pipeModelType.textContent = 'Cây quyết định hồi quy đa đầu ra';
        } else {
            pipeStep2Desc.textContent = 'Tính hàm tuyến tính đa biến:';
            pipeStep2Math.textContent = 'y = W·x + b';
            pipeModelType.textContent = 'Hồi quy tuyến tính ma trận';
        }

        // Step 4 Delta
        const activeRes = state.allPredictions[state.activeModel];
        if (activeRes && pipeDelta) {
            pipeDelta.textContent = `±${activeRes.meanErr.toFixed(2)} dB`;
        }

        // Decomposition Table
        if (decompModelName) decompModelName.textContent = meta.name;
        if (!decompTableBody || !activeRes) return;

        let tableHtml = '';
        const y = activeRes.y;
        const x1 = state.x1;
        const x2 = state.x2;

        for (let i = 0; i < 6; i++) {
            const varName = OUTPUT_NAMES[i];
            const varLabel = OUTPUT_LABELS[i];
            const finalVal = y[i].toFixed(2);

            let formulaTxt = '';
            let comp1Txt = '---';
            let comp2Txt = '---';
            let biasTxt = '---';
            let rawSumTxt = finalVal;

            if (state.activeModel === 'LinearRegression' || state.activeModel === 'RidgeRegression' || state.activeModel === 'LassoRegression' || state.activeModel === 'SVM') {
                const m = models[state.activeModel];
                const w1 = m.coef[i][0];
                const w2 = m.coef[i][1];
                const b = m.intercept[i];
                const c1 = w1 * x1;
                const c2 = w2 * x2;
                const rawSum = c1 + c2 + b;

                formulaTxt = `(${w1.toFixed(4)})·x₁ + (${w2.toFixed(4)})·x₂ + ${b.toFixed(4)}`;
                comp1Txt = `<span class="${c1 >= 0 ? 'decomp-num-pos' : 'decomp-num-neg'}">${c1 >= 0 ? '+' : ''}${c1.toFixed(2)}</span>`;
                comp2Txt = `<span class="${c2 >= 0 ? 'decomp-num-pos' : 'decomp-num-neg'}">${c2 >= 0 ? '+' : ''}${c2.toFixed(2)}</span>`;
                biasTxt = `<span>${b >= 0 ? '+' : ''}${b.toFixed(2)}</span>`;
                rawSumTxt = rawSum.toFixed(2);
            } else if (state.activeModel === 'PolynomialRegression') {
                const p = models.PolynomialRegression;
                const c = p.coef[i];
                const c1 = c[1] * x1 + c[3] * x1 * x1;
                const c2 = c[2] * x2 + c[5] * x2 * x2;
                const c12 = c[4] * x1 * x2;
                const b = c[0];
                const rawSum = b + c1 + c2 + c12;

                formulaTxt = `${c[0].toFixed(2)} + ${c[1].toFixed(2)}x₁ + ${c[2].toFixed(2)}x₂ + ${c[3].toFixed(3)}x₁² + ...`;
                comp1Txt = `<span class="${c1 >= 0 ? 'decomp-num-pos' : 'decomp-num-neg'}">${c1 >= 0 ? '+' : ''}${c1.toFixed(2)}</span> (x₁+x₁²)`;
                comp2Txt = `<span class="${c2 >= 0 ? 'decomp-num-pos' : 'decomp-num-neg'}">${c2 >= 0 ? '+' : ''}${c2.toFixed(2)}</span> (x₂+x₂²)`;
                biasTxt = `<span>+${b.toFixed(2)}</span>`;
                rawSumTxt = rawSum.toFixed(2);
            } else if (state.activeModel === 'Ensemble') {
                const mlpVal = state.allPredictions['NeuralNetwork'] ? state.allPredictions['NeuralNetwork'].y[i] : y[i];
                const svmVal = state.allPredictions['SVM'] ? state.allPredictions['SVM'].y[i] : y[i];
                const polyVal = state.allPredictions['PolynomialRegression'] ? state.allPredictions['PolynomialRegression'].y[i] : y[i];

                const pMlp = 0.40 * mlpVal;
                const pSvm = 0.35 * svmVal;
                const pPoly = 0.25 * polyVal;

                formulaTxt = `0.40·MLP + 0.35·SVM + 0.25·Poly`;
                comp1Txt = `<span class="decomp-num-pos">+${pMlp.toFixed(2)}</span> (40% MLP)`;
                comp2Txt = `<span class="decomp-num-pos">+${pSvm.toFixed(2)}</span> (35% SVM)`;
                biasTxt = `<span class="decomp-num-pos">+${pPoly.toFixed(2)}</span> (25% Poly)`;
                rawSumTxt = (pMlp + pSvm + pPoly).toFixed(2);
            } else if (state.activeModel === 'NeuralNetwork') {
                formulaTxt = `Lan truyền tiến Feedforward (3 tầng ẩn)`;
                comp1Txt = `Tín hiệu z₁ = ${z1.toFixed(2)}`;
                comp2Txt = `Tín hiệu z₂ = ${z2.toFixed(2)}`;
                biasTxt = `Hệ số bù bias các tầng`;
                rawSumTxt = finalVal;
            } else {
                formulaTxt = `Nội suy / Phân vùng`;
                comp1Txt = `x₁ = ${x1.toFixed(2)}`;
                comp2Txt = `x₂ = ${x2.toFixed(2)}`;
                biasTxt = `Tham chiếu lân cận`;
                rawSumTxt = finalVal;
            }

            tableHtml += `
            <tr>
                <td><strong>${varLabel}</strong></td>
                <td class="decomp-math-cell"><code>${formulaTxt}</code></td>
                <td>${comp1Txt}</td>
                <td>${comp2Txt}</td>
                <td>${biasTxt}</td>
                <td style="font-family: var(--font-mono); font-weight: 600;">${rawSumTxt}</td>
                <td><span class="dim-tag tag-y${i+1}">${finalVal}</span></td>
            </tr>`;
        }

        decompTableBody.innerHTML = tableHtml;
    }

    // Canvas Visualizer setup and rendering
    function renderVisualizer() {
        if (!vizCanvas || !ctx) return;

        // Handle Retina DPI scaling
        const rect = vizCanvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const w = rect.width || 940;
        const h = 420;

        if (vizCanvas.width !== Math.round(w * dpr) || vizCanvas.height !== Math.round(h * dpr)) {
            vizCanvas.width = Math.round(w * dpr);
            vizCanvas.height = Math.round(h * dpr);
        }

        ctx.save();
        ctx.resetTransform ? ctx.resetTransform() : ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);

        ctx.clearRect(0, 0, w, h);

        if (state.vizTab === 'radar') {
            drawRadarChart(ctx, w, h);
        } else if (state.vizTab === 'network') {
            drawNetworkDiagram(ctx, w, h);
        } else if (state.vizTab === 'breakdown') {
            drawBreakdownChart(ctx, w, h);
        }

        ctx.restore();
    }

    // 1. Radar Chart (Biểu đồ mạng nhện 6 chiều)
    function drawRadarChart(c, w, h) {
        const cx = w / 2;
        const cy = h / 2 + 10;
        const maxR = Math.min(cx - 100, cy - 50, 150);
        const activeRes = state.allPredictions[state.activeModel];
        if (!activeRes) return;

        const compareRes = (state.compareModel !== 'none' && state.allPredictions[state.compareModel]) ? state.allPredictions[state.compareModel] : null;

        // Draw concentric rings
        const rings = [28, 56, 84, 112, 140];
        c.strokeStyle = '#e2e8f0';
        c.lineWidth = 1;
        c.setLineDash([4, 4]);

        rings.forEach(val => {
            const r = (val / 140) * maxR;
            c.beginPath();
            for (let i = 0; i < 6; i++) {
                const angle = -Math.PI / 2 + (i * 2 * Math.PI) / 6;
                const x = cx + r * Math.cos(angle);
                const y = cy + r * Math.sin(angle);
                if (i === 0) c.moveTo(x, y);
                else c.lineTo(x, y);
            }
            c.closePath();
            c.stroke();

            // Ring scale text
            c.fillStyle = '#94a3b8';
            c.font = '10px JetBrains Mono, monospace';
            c.textAlign = 'left';
            c.fillText(val.toString(), cx + 4, cy - r + 11);
        });
        c.setLineDash([]);

        // Draw 6 radial spokes
        for (let i = 0; i < 6; i++) {
            const angle = -Math.PI / 2 + (i * 2 * Math.PI) / 6;
            const x = cx + maxR * Math.cos(angle);
            const y = cy + maxR * Math.sin(angle);

            c.strokeStyle = '#cbd5e1';
            c.lineWidth = 1.2;
            c.beginPath();
            c.moveTo(cx, cy);
            c.lineTo(x, y);
            c.stroke();

            // Label around perimeter
            const labelDist = maxR + 24;
            const lx = cx + labelDist * Math.cos(angle);
            const ly = cy + labelDist * Math.sin(angle);

            c.font = 'bold 12px Inter, sans-serif';
            c.fillStyle = OUTPUT_COLORS[i];
            c.textAlign = Math.abs(Math.cos(angle)) < 0.1 ? 'center' : (Math.cos(angle) > 0 ? 'left' : 'right');
            c.textBaseline = 'middle';
            c.fillText(OUTPUT_LABELS[i], lx, ly);
        }

        // Draw Compare Model Polygon (if selected)
        if (compareRes) {
            c.beginPath();
            for (let i = 0; i < 6; i++) {
                const val = compareRes.y[i];
                const r = (val / 140) * maxR;
                const angle = -Math.PI / 2 + (i * 2 * Math.PI) / 6;
                const x = cx + r * Math.cos(angle);
                const y = cy + r * Math.sin(angle);
                if (i === 0) c.moveTo(x, y);
                else c.lineTo(x, y);
            }
            c.closePath();
            c.fillStyle = 'rgba(249, 115, 22, 0.14)';
            c.fill();
            c.strokeStyle = '#f97316';
            c.lineWidth = 2;
            c.setLineDash([5, 4]);
            c.stroke();
            c.setLineDash([]);

            // Draw vertex points
            for (let i = 0; i < 6; i++) {
                const val = compareRes.y[i];
                const r = (val / 140) * maxR;
                const angle = -Math.PI / 2 + (i * 2 * Math.PI) / 6;
                const x = cx + r * Math.cos(angle);
                const y = cy + r * Math.sin(angle);
                c.fillStyle = '#f97316';
                c.beginPath();
                c.arc(x, y, 4, 0, Math.PI * 2);
                c.fill();
            }
        }

        // Draw Active Model Polygon
        c.beginPath();
        for (let i = 0; i < 6; i++) {
            const val = activeRes.y[i];
            const r = (val / 140) * maxR;
            const angle = -Math.PI / 2 + (i * 2 * Math.PI) / 6;
            const x = cx + r * Math.cos(angle);
            const y = cy + r * Math.sin(angle);
            if (i === 0) c.moveTo(x, y);
            else c.lineTo(x, y);
        }
        c.closePath();
        c.fillStyle = 'rgba(37, 99, 235, 0.22)';
        c.fill();
        c.strokeStyle = '#2563eb';
        c.lineWidth = 2.8;
        c.stroke();

        // Draw vertex nodes and live value badges
        for (let i = 0; i < 6; i++) {
            const val = activeRes.y[i];
            const r = (val / 140) * maxR;
            const angle = -Math.PI / 2 + (i * 2 * Math.PI) / 6;
            const x = cx + r * Math.cos(angle);
            const y = cy + r * Math.sin(angle);

            // Node halo
            c.fillStyle = '#ffffff';
            c.beginPath();
            c.arc(x, y, 6, 0, Math.PI * 2);
            c.fill();
            c.strokeStyle = OUTPUT_COLORS[i];
            c.lineWidth = 3;
            c.stroke();

            // Value badge tag
            const badgeTxt = val.toFixed(2);
            c.font = 'bold 11px JetBrains Mono, monospace';
            const badgeW = c.measureText(badgeTxt).width + 8;
            const badgeH = 18;
            const bx = x + Math.cos(angle) * 14 - (Math.cos(angle) < 0 ? badgeW : 0);
            const by = y + Math.sin(angle) * 14 - badgeH / 2;

            c.fillStyle = 'rgba(15, 23, 42, 0.85)';
            c.beginPath();
            c.roundRect ? c.roundRect(bx, by, badgeW, badgeH, 4) : c.rect(bx, by, badgeW, badgeH);
            c.fill();

            c.fillStyle = '#ffffff';
            c.textAlign = 'center';
            c.textBaseline = 'middle';
            c.fillText(badgeTxt, bx + badgeW / 2, by + badgeH / 2);
        }

        // Top Legend
        c.textAlign = 'left';
        c.font = 'bold 11px Inter, sans-serif';

        // Active model legend
        c.fillStyle = '#2563eb';
        c.fillRect(20, 20, 16, 4);
        c.fillStyle = '#0f172a';
        const meta = MODEL_META.find(m => m.id === state.activeModel) || MODEL_META[0];
        c.fillText(`${meta.name} (Đang chọn)`, 44, 24);

        if (compareRes) {
            c.fillStyle = '#f97316';
            c.fillRect(20, 40, 16, 4);
            c.fillStyle = '#0f172a';
            const cMeta = MODEL_META.find(m => m.id === state.compareModel) || { name: state.compareModel };
            c.fillText(`${cMeta.name} (So sánh đối chứng)`, 44, 44);
        }
    }

    // 2. Neural Signal Flow & Computation Architecture Graph
    function drawNetworkDiagram(c, w, h) {
        c.clearRect(0, 0, w, h);
        const activeRes = state.allPredictions[state.activeModel];
        if (!activeRes) return;

        const leftX = 100;
        const midX1 = w * 0.35;
        const midX2 = w * 0.52;
        const midX3 = w * 0.68;
        const rightX = w - 140;

        // 1. Draw Inputs on the left
        const inY = [h * 0.36, h * 0.64];
        const inLabels = ['x₁ (Cột 7)', 'x₂ (Cột 8)'];
        const inVals = [`${state.x1.toFixed(2)} dB`, `${state.x2.toFixed(2)} dB`];

        // 2. Draw Outputs on the right
        const outY = [];
        const outH = h - 60;
        for (let i = 0; i < 6; i++) {
            outY.push(35 + (i * outH) / 5);
        }

        // Synapses & Connections
        c.lineWidth = 1;
        state.animTick = (state.animTick + 0.04) % 1;

        // Draw connections from input to mid
        inY.forEach((iy, inIdx) => {
            outY.forEach((oy, outIdx) => {
                c.strokeStyle = 'rgba(226, 232, 240, 0.7)';
                c.beginPath();
                c.moveTo(leftX + 40, iy);
                c.bezierCurveTo(midX1, iy, midX3, oy, rightX - 40, oy);
                c.stroke();

                // Animated signal pulse particle
                const t = (state.animTick + (inIdx * 0.3) + (outIdx * 0.15)) % 1;
                const px = Math.pow(1 - t, 3) * (leftX + 40) +
                           3 * Math.pow(1 - t, 2) * t * midX1 +
                           3 * (1 - t) * Math.pow(t, 2) * midX3 +
                           Math.pow(t, 3) * (rightX - 40);
                const py = Math.pow(1 - t, 3) * iy +
                           3 * Math.pow(1 - t, 2) * t * iy +
                           3 * (1 - t) * Math.pow(t, 2) * oy +
                           Math.pow(t, 3) * oy;

                c.fillStyle = inIdx === 0 ? 'rgba(37, 99, 235, 0.75)' : 'rgba(8, 145, 178, 0.75)';
                c.beginPath();
                c.arc(px, py, 2.5, 0, Math.PI * 2);
                c.fill();
            });
        });

        // Middle Processing Architecture Block
        const midBlockW = 180;
        const midBlockH = 260;
        const midBlockX = w * 0.44 - midBlockW / 2;
        const midBlockY = h / 2 - midBlockH / 2;

        c.fillStyle = '#f8fafc';
        c.strokeStyle = '#cbd5e1';
        c.lineWidth = 1.5;
        c.beginPath();
        c.roundRect ? c.roundRect(midBlockX, midBlockY, midBlockW, midBlockH, 12) : c.rect(midBlockX, midBlockY, midBlockW, midBlockH);
        c.fill();
        c.stroke();

        // Model internal representation inside box
        c.fillStyle = '#0f172a';
        c.font = 'bold 12px Inter, sans-serif';
        c.textAlign = 'center';
        const meta = MODEL_META.find(m => m.id === state.activeModel) || MODEL_META[0];
        c.fillText(meta.name, midBlockX + midBlockW / 2, midBlockY + 28);

        c.font = '10px JetBrains Mono, monospace';
        c.fillStyle = '#64748b';
        if (state.activeModel === 'NeuralNetwork') {
            c.fillText('Tầng 1: 64 ReLU', midBlockX + midBlockW / 2, midBlockY + 68);
            c.fillText('Tầng 2: 64 ReLU', midBlockX + midBlockW / 2, midBlockY + 98);
            c.fillText('Tầng 3: 32 ReLU', midBlockX + midBlockW / 2, midBlockY + 128);
            c.fillText('Tầng ra: 6 Linear', midBlockX + midBlockW / 2, midBlockY + 158);
            c.fillText('Mất mát MSE Adam', midBlockX + midBlockW / 2, midBlockY + 195);
        } else if (state.activeModel === 'SVM') {
            c.fillText('Siêu phẳng W·x + b', midBlockX + midBlockW / 2, midBlockY + 70);
            c.fillText('Hàm mất mát ε-L2', midBlockX + midBlockW / 2, midBlockY + 105);
            c.fillText('Khắc phục quá khớp', midBlockX + midBlockW / 2, midBlockY + 140);
            c.fillText('6 vector lề cực đại', midBlockX + midBlockW / 2, midBlockY + 175);
            c.fillText('Xuất số thực liên tục', midBlockX + midBlockW / 2, midBlockY + 210);
        } else if (state.activeModel === 'Ensemble') {
            c.fillText('Nhánh 1: 40% MLP', midBlockX + midBlockW / 2, midBlockY + 70);
            c.fillText('Nhánh 2: 35% SVM', midBlockX + midBlockW / 2, midBlockY + 105);
            c.fillText('Nhánh 3: 25% Poly', midBlockX + midBlockW / 2, midBlockY + 140);
            c.fillText('Bộ cộng trọng số Σ', midBlockX + midBlockW / 2, midBlockY + 180);
            c.fillText('Tối ưu hóa đa mô hình', midBlockX + midBlockW / 2, midBlockY + 215);
        } else {
            c.fillText('Ánh xạ affine ma trận', midBlockX + midBlockW / 2, midBlockY + 80);
            c.fillText('y = W · x + b', midBlockX + midBlockW / 2, midBlockY + 120);
            c.fillText('6 Chiều độc lập', midBlockX + midBlockW / 2, midBlockY + 160);
            c.fillText('Không gian số thực', midBlockX + midBlockW / 2, midBlockY + 200);
        }

        // Draw Input nodes
        inY.forEach((iy, i) => {
            c.fillStyle = '#ffffff';
            c.strokeStyle = i === 0 ? '#2563eb' : '#0891b2';
            c.lineWidth = 2.5;
            c.beginPath();
            c.arc(leftX, iy, 22, 0, Math.PI * 2);
            c.fill();
            c.stroke();

            c.fillStyle = i === 0 ? '#2563eb' : '#0891b2';
            c.font = 'bold 12px Inter, sans-serif';
            c.textAlign = 'center';
            c.textBaseline = 'middle';
            c.fillText(`x${i+1}`, leftX, iy - 6);

            c.font = '10px JetBrains Mono, monospace';
            c.fillStyle = '#475569';
            c.fillText(inVals[i], leftX, iy + 9);

            c.font = '11px Inter, sans-serif';
            c.fillStyle = '#0f172a';
            c.textAlign = 'right';
            c.fillText(inLabels[i], leftX - 30, iy);
        });

        // Draw Output nodes
        outY.forEach((oy, i) => {
            const val = activeRes.y[i].toFixed(2);
            c.fillStyle = '#ffffff';
            c.strokeStyle = OUTPUT_COLORS[i];
            c.lineWidth = 2.5;
            c.beginPath();
            c.arc(rightX, oy, 16, 0, Math.PI * 2);
            c.fill();
            c.stroke();

            c.fillStyle = OUTPUT_COLORS[i];
            c.font = 'bold 10px Inter, sans-serif';
            c.textAlign = 'center';
            c.textBaseline = 'middle';
            c.fillText(`y${i+1}`, rightX, oy);

            // Output Tag with value
            c.textAlign = 'left';
            c.font = 'bold 12px JetBrains Mono, monospace';
            c.fillStyle = '#0f172a';
            c.fillText(val, rightX + 26, oy);

            c.font = '10px Inter, sans-serif';
            c.fillStyle = '#64748b';
            c.fillText(OUTPUT_NAMES[i], rightX + 80, oy);
        });
    }

    // 3. Impact Breakdown Chart
    function drawBreakdownChart(c, w, h) {
        c.clearRect(0, 0, w, h);
        const activeRes = state.allPredictions[state.activeModel];
        if (!activeRes) return;

        const startY = 40;
        const rowH = (h - 60) / 6;
        const leftMargin = 120;
        const barMaxW = w - leftMargin - 140;

        // Top Legend
        c.font = 'bold 11px Inter, sans-serif';
        c.textAlign = 'left';

        c.fillStyle = '#2563eb';
        c.fillRect(leftMargin, 15, 12, 12);
        c.fillStyle = '#334155';
        c.fillText('Đóng góp x₁', leftMargin + 18, 25);

        c.fillStyle = '#0891b2';
        c.fillRect(leftMargin + 120, 15, 12, 12);
        c.fillStyle = '#334155';
        c.fillText('Đóng góp x₂', leftMargin + 138, 25);

        c.fillStyle = '#8b5cf6';
        c.fillRect(leftMargin + 240, 15, 12, 12);
        c.fillStyle = '#334155';
        c.fillText('Hệ số gốc (Bias)', leftMargin + 258, 25);

        c.fillStyle = '#059669';
        c.fillRect(leftMargin + 370, 15, 12, 12);
        c.fillStyle = '#334155';
        c.fillText('Giá trị dự đoán cuối', leftMargin + 388, 25);

        const x1 = state.x1;
        const x2 = state.x2;

        for (let i = 0; i < 6; i++) {
            const yPos = startY + i * rowH + 16;
            const finalVal = activeRes.y[i];

            // Row Label
            c.fillStyle = OUTPUT_COLORS[i];
            c.font = 'bold 12px Inter, sans-serif';
            c.textAlign = 'right';
            c.fillText(OUTPUT_LABELS[i], leftMargin - 15, yPos + 12);

            // Background bar track
            c.fillStyle = '#f1f5f9';
            c.beginPath();
            c.roundRect ? c.roundRect(leftMargin, yPos, barMaxW, 16, 4) : c.rect(leftMargin, yPos, barMaxW, 16);
            c.fill();

            // Compute components
            let w1 = 1, w2 = 1, b = 60;
            if (models.SVM && models.SVM.coef) {
                w1 = models.SVM.coef[i][0];
                w2 = models.SVM.coef[i][1];
                b = models.SVM.intercept[i];
            }
            const c1 = Math.abs(w1 * x1);
            const c2 = Math.abs(w2 * x2);
            const cb = Math.abs(b);
            const sumC = c1 + c2 + cb || 1;

            const finalW = (finalVal / 140) * barMaxW;

            // Fill Bar with final value
            c.fillStyle = OUTPUT_COLORS[i];
            c.beginPath();
            c.roundRect ? c.roundRect(leftMargin, yPos, Math.max(8, finalW), 16, 4) : c.rect(leftMargin, yPos, Math.max(8, finalW), 16);
            c.fill();

            // Final value text
            c.font = 'bold 12px JetBrains Mono, monospace';
            c.fillStyle = '#0f172a';
            c.textAlign = 'left';
            c.fillText(finalVal.toFixed(2), leftMargin + barMaxW + 15, yPos + 13);
        }
    }

    // Switch Visualizer Tab
    function switchVizTab(tabName) {
        state.vizTab = tabName;
        [tabRadar, tabNetwork, tabBreakdown].forEach(btn => {
            if (btn) btn.classList.remove('active');
        });

        if (tabName === 'radar') {
            if (tabRadar) tabRadar.classList.add('active');
            if (vizDynamicTitle) vizDynamicTitle.textContent = 'Biểu Đồ Mạng Nhện Không Gian 6 Chiều (Radar View)';
            if (vizFooterHint) vizFooterHint.textContent = '💡 Kéo thanh trượt x₁, x₂ ở Bước 1 để quan sát đa giác 6 chiều và luồng tín hiệu biến đổi mượt mà theo thời gian thực';
            if (document.getElementById('vizCompareWrap')) document.getElementById('vizCompareWrap').style.display = 'inline-flex';
        } else if (tabName === 'network') {
            if (tabNetwork) tabNetwork.classList.add('active');
            if (vizDynamicTitle) vizDynamicTitle.textContent = 'Sơ Đồ Luồng Tín Hiệu & Mạng Nơ-ron (Signal Propagation)';
            if (vizFooterHint) vizFooterHint.textContent = '💡 Hạt xung tín hiệu chuyển động mô phỏng quá trình lan truyền tiến (Forward Pass) từ 2 đầu vào qua tầng ẩn sang 6 đầu ra';
            if (document.getElementById('vizCompareWrap')) document.getElementById('vizCompareWrap').style.display = 'none';
        } else if (tabName === 'breakdown') {
            if (tabBreakdown) tabBreakdown.classList.add('active');
            if (vizDynamicTitle) vizDynamicTitle.textContent = 'Phân Tích Thành Phần Đóng Góp & Trọng Số (Feature Impact)';
            if (vizFooterHint) vizFooterHint.textContent = '💡 Thể hiện tỷ trọng tác động của x₁ (f1) và x₂ (f2) cùng hệ số tự do lên từng biến đầu ra';
            if (document.getElementById('vizCompareWrap')) document.getElementById('vizCompareWrap').style.display = 'none';
        }

        renderVisualizer();
    }

    // Full UI Update
    function updateUI() {
        x1Badge.textContent = state.x1.toFixed(2);
        x2Badge.textContent = state.x2.toFixed(2);

        state.allPredictions = computeAllModels(state.x1, state.x2);
        renderComparisonTable(state.allPredictions);
        updateActiveModelDetails(state.allPredictions);
        updatePipelineAndDecomp();
        renderVisualizer();
    }

    // Event listeners
    sliderX1.addEventListener('input', (e) => {
        state.x1 = parseFloat(e.target.value);
        inputX1.value = state.x1.toFixed(1);
        updateUI();
    });

    inputX1.addEventListener('change', (e) => {
        let v = parseFloat(e.target.value);
        if (isNaN(v)) v = -15.0;
        v = Math.max(-52.07, Math.min(0.0, v));
        state.x1 = v;
        sliderX1.value = v;
        e.target.value = v.toFixed(1);
        updateUI();
    });

    sliderX2.addEventListener('input', (e) => {
        state.x2 = parseFloat(e.target.value);
        inputX2.value = state.x2.toFixed(1);
        updateUI();
    });

    inputX2.addEventListener('change', (e) => {
        let v = parseFloat(e.target.value);
        if (isNaN(v)) v = -5.0;
        v = Math.max(-42.08, Math.min(0.0, v));
        state.x2 = v;
        sliderX2.value = v;
        e.target.value = v.toFixed(1);
        updateUI();
    });

    resetBtn.addEventListener('click', () => {
        state.x1 = -15.0;
        state.x2 = -5.0;
        sliderX1.value = -15.0;
        inputX1.value = (-15.0).toFixed(1);
        sliderX2.value = -5.0;
        inputX2.value = (-5.0).toFixed(1);
        state.activeModel = 'Ensemble';
        updateUI();
    });

    copyActiveBtn.addEventListener('click', () => {
        const res = state.allPredictions[state.activeModel];
        if (!res) return;
        const text = `t3 (y1) = ${res.y[0].toFixed(2)}\nt2 (y2) = ${res.y[1].toFixed(2)}\nt1 (y3) = ${res.y[2].toFixed(2)}\nz3 (y4) = ${res.y[3].toFixed(2)}\nz2 (y5) = ${res.y[4].toFixed(2)}\nz1 (y6) = ${res.y[5].toFixed(2)}\n(Model: ${state.activeModel}, Input: x1=${state.x1.toFixed(2)}, x2=${state.x2.toFixed(2)})`;
        navigator.clipboard.writeText(text).then(() => {
            const orig = copyActiveBtn.innerHTML;
            copyActiveBtn.innerHTML = `✓ Đã sao chép 6 số thập phân!`;
            setTimeout(() => { copyActiveBtn.innerHTML = orig; }, 1600);
        });
    });

    copyFormulaBtn.addEventListener('click', () => {
        const text = formulaCode.textContent;
        navigator.clipboard.writeText(text).then(() => {
            const orig = copyFormulaBtn.innerHTML;
            copyFormulaBtn.innerHTML = `✓ Đã chép công thức!`;
            setTimeout(() => { copyFormulaBtn.innerHTML = orig; }, 1600);
        });
    });

    exportCsvBtn.addEventListener('click', () => {
        const rows = [
            ['Model_ID', 'Model_Name', 'Input_x1', 'Input_x2', 't3_y1', 't2_y2', 't1_y3', 'z3_y4', 'z2_y5', 'z1_y6', 'Simulated_x1', 'Simulated_x2', 'Mean_Delta_x', 'Rating']
        ];

        MODEL_META.forEach(meta => {
            const r = state.allPredictions[meta.id];
            rows.push([
                meta.id,
                `"${meta.name}"`,
                state.x1.toFixed(2),
                state.x2.toFixed(2),
                r.y[0].toFixed(2),
                r.y[1].toFixed(2),
                r.y[2].toFixed(2),
                r.y[3].toFixed(2),
                r.y[4].toFixed(2),
                r.y[5].toFixed(2),
                r.sim[0].toFixed(3),
                r.sim[1].toFixed(3),
                r.meanErr.toFixed(3),
                `"${meta.rank}"`
            ]);
        });

        const csv = 'data:text/csv;charset=utf-8,' + rows.map(r => r.join(',')).join('\n');
        const link = document.createElement('a');
        link.setAttribute('href', encodeURI(csv));
        link.setAttribute('download', `continuous_ml_predictions_6outputs_x1_${state.x1}_x2_${state.x2}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // Visualizer Tab Listeners
    if (tabRadar) tabRadar.addEventListener('click', () => switchVizTab('radar'));
    if (tabNetwork) tabNetwork.addEventListener('click', () => switchVizTab('network'));
    if (tabBreakdown) tabBreakdown.addEventListener('click', () => switchVizTab('breakdown'));
    if (vizCompareSelect) {
        vizCompareSelect.addEventListener('change', (e) => {
            state.compareModel = e.target.value;
            renderVisualizer();
        });
    }

    // Animation Loop for live signal particle propagation
    function animLoop() {
        if (state.vizTab === 'network') {
            renderVisualizer();
        }
        requestAnimationFrame(animLoop);
    }
    requestAnimationFrame(animLoop);

    // Resize handling for Canvas
    window.addEventListener('resize', () => {
        renderVisualizer();
    });

    // Initial render
    updateUI();
});
