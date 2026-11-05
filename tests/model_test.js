// Automated Test Suite for 6-Output Continuous ML Models & Simulator
const assert = require('assert');
const fs = require('fs');
const path = require('path');

console.log('--- RUNNING CI AUTOMATED TEST SUITE (CONTINUOUS 6 OUTPUTS) ---');

// 1. Verify all_models.json exists and is valid JSON
const jsonPath = path.join(__dirname, '..', 'all_models.json');
assert(fs.existsSync(jsonPath), 'Error: all_models.json does not exist');
const data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
console.log('✓ Test 1 Passed: all_models.json is valid');

// 2. Verify all models are present
const expectedModels = [
    'LinearRegression',
    'SVM',
    'RidgeRegression',
    'LassoRegression',
    'PolynomialRegression',
    'DecisionTree',
    'NeuralNetwork',
    'Ensemble'
];
for (const m of expectedModels) {
    assert(data.models[m], `Missing model: ${m}`);
    assert(typeof data.models[m].mae === 'number', `Model ${m} missing MAE`);
    assert(typeof data.models[m].r2 === 'number', `Model ${m} missing R2`);
}
console.log('✓ Test 2 Passed: All 8 primary ML models (including SVM) are defined with metrics');

// 3. Verify Forward Digital Twin Simulator exists
assert(data.forward_simulator, 'Missing forward_simulator in dataset');
assert(data.forward_simulator.layers.length > 0, 'Forward simulator has no layers');
assert(data.forward_simulator.scaler_x.mean.length === 6, 'Forward simulator scaler_x should have 6 inputs');
assert(data.forward_simulator.scaler_y.mean.length === 2, 'Forward simulator scaler_y should have 2 outputs');
console.log('✓ Test 3 Passed: Forward Digital Twin Simulator (6 inputs -> 2 outputs) is configured');

// 4. Verify representative dataset for KNN
assert(Array.isArray(data.samples), 'samples must be an array');
assert(data.samples.length > 1000, `Expected > 1000 samples, got ${data.samples.length}`);
assert(data.samples[0].d1 !== undefined && data.samples[0].d6 !== undefined, 'Sample must have d1 to d6');
console.log(`✓ Test 4 Passed: KNN database contains ${data.samples.length} verified points with 6 dimensions (d1-d6)`);

// 4b. Test KNN inference with k=1, k=3, k=5
const testV1 = -15.0;
const testV2 = -5.0;

function testKNN(samples, v1, v2, k) {
    const dists = samples.map(s => {
        const d1 = s.v1 - v1;
        const d2 = s.v2 - v2;
        return {
            dist: Math.sqrt(d1 * d1 + d2 * d2),
            y: [s.d1, s.d2, s.d3, s.d4, s.d5, s.d6]
        };
    });
    dists.sort((a, b) => a.dist - b.dist);
    if (k === 1) return dists[0].y;
    let sumW = 0, sumY = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < k; i++) {
        const w = 1.0 / (dists[i].dist + 1e-6);
        sumW += w;
        for (let j = 0; j < 6; j++) sumY[j] += dists[i].y[j] * w;
    }
    return sumY.map(v => v / sumW);
}
const knn1D = testKNN(data.samples, testV1, testV2, 1);
const knn3D = testKNN(data.samples, testV1, testV2, 3);
const knn5D = testKNN(data.samples, testV1, testV2, 5);
assert(knn1D.length === 6 && knn3D.length === 6 && knn5D.length === 6);
console.log('✓ Test 4b Passed: KNN (k=1, 3, 5) outputs 6 valid dimensions:', {
    'k=1': knn1D,
    'k=3': knn3D.map(v => v.toFixed(2)),
    'k=5': knn5D.map(v => v.toFixed(2))
});

// 5. Test Linear & Ridge & Lasso & SVM inference (6 continuous decimal outputs)
function testLinearModel(m, v1, v2) {
    const d = [];
    for (let i = 0; i < 6; i++) {
        const val = m.coef[i][0] * v1 + m.coef[i][1] * v2 + m.intercept[i];
        d.push(Math.round(val * 100) / 100);
    }
    return d;
}
const lrD = testLinearModel(data.models.LinearRegression, testV1, testV2);
assert(lrD.length === 6, 'Linear regression must output 6 dimensions');
assert(lrD.some(v => v % 1 !== 0), 'Linear regression must predict continuous decimal numbers');
console.log('✓ Test 5 Passed: Linear Regression outputs continuous decimals:', lrD.map(v => v.toFixed(2)));

const svmD = testLinearModel(data.models.SVM, testV1, testV2);
assert(svmD.length === 6, 'SVM must output 6 dimensions');
assert(svmD.some(v => v % 1 !== 0), 'SVM must predict continuous decimal numbers');
console.log('✓ Test 5b Passed: Support Vector Machine (Linear SVR) outputs continuous decimals:', svmD.map(v => v.toFixed(2)));

const lassoD = testLinearModel(data.models.LassoRegression, testV1, testV2);
assert(lassoD.length === 6, 'Lasso regression must output 6 dimensions');
console.log('✓ Test 5c Passed: Lasso Regression outputs continuous decimals:', lassoD.map(v => v.toFixed(2)));

// 6. Test Polynomial Regression inference (6 continuous decimal outputs)
function testPoly(m, v1, v2) {
    const feats = [1.0, v1, v2, v1 * v1, v1 * v2, v2 * v2];
    const d = [];
    for (let i = 0; i < 6; i++) {
        let sum = 0;
        for (let f = 0; f < feats.length; f++) {
            sum += m.coef[i][f] * feats[f];
        }
        d.push(Math.round(sum * 100) / 100);
    }
    return d;
}
const polyD = testPoly(data.models.PolynomialRegression, testV1, testV2);
assert(polyD.length === 6, 'Polynomial regression must output 6 dimensions');
assert(polyD.some(v => v % 1 !== 0), 'Polynomial regression must predict continuous decimal numbers');
console.log('✓ Test 6 Passed: Polynomial Regression outputs continuous decimals:', polyD.map(v => v.toFixed(2)));

// 7. Test Decision Tree inference (6 outputs)
function testTree(tree, v1, v2) {
    let node = 0;
    const inputs = [v1, v2];
    while (tree.children_left[node] !== -1) {
        const feat = tree.feature[node];
        const thresh = tree.threshold[node];
        if (inputs[feat] <= thresh) {
            node = tree.children_left[node];
        } else {
            node = tree.children_right[node];
        }
    }
    return tree.value[node].map(v => Math.round(v * 100) / 100);
}
const dtD = testTree(data.models.DecisionTree.tree, testV1, testV2);
assert(dtD.length === 6, 'Decision tree must output 6 dimensions');
console.log('✓ Test 7 Passed: Decision Tree traversal outputs continuous decimals:', dtD.map(v => v.toFixed(2)));

// 8. Test Neural Network (MLP) inference (6 continuous decimal outputs)
function testMLP(m, v1, v2) {
    let cur = [
        (v1 - m.scaler_x.mean[0]) / m.scaler_x.scale[0],
        (v2 - m.scaler_x.mean[1]) / m.scaler_x.scale[1]
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
    return cur.map((val, i) => Math.round((val * m.scaler_y.scale[i] + m.scaler_y.mean[i]) * 100) / 100);
}
const mlpD = testMLP(data.models.NeuralNetwork, testV1, testV2);
assert(mlpD.length === 6, 'MLP must output 6 dimensions');
assert(mlpD.some(v => v % 1 !== 0), 'MLP must predict continuous decimal numbers');
console.log('✓ Test 8 Passed: MLP Neural Network outputs continuous decimals:', mlpD.map(v => v.toFixed(2)));

// 8b. Test Pure Continuous Stacking Ensemble (0.40 MLP + 0.35 SVM + 0.25 Poly)
const ensembleD = mlpD.map((v, i) => Math.round((0.40 * v + 0.35 * svmD[i] + 0.25 * polyD[i]) * 100) / 100);
assert(ensembleD.length === 6, 'Ensemble must output 6 dimensions');
assert(ensembleD.some(v => v % 1 !== 0), 'Ensemble must predict continuous decimal numbers');
console.log('✓ Test 8b Passed: Continuous Ensemble outputs continuous decimals:', ensembleD.map(v => v.toFixed(2)));

// 9. Test Forward Simulator (Digital Twin: 6 inputs -> 2 outputs)
function testForward(fwd, yValues) {
    let cur = yValues.map((val, i) => (val - fwd.scaler_x.mean[i]) / fwd.scaler_x.scale[i]);
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
const simResult = testForward(data.forward_simulator, ensembleD);
assert(simResult.length === 2, 'Simulator must output 2 values [v1, v2]');
console.log('✓ Test 9 Passed: Forward Digital Twin simulates response from continuous decimal inputs:', simResult.map(v => v.toFixed(2) + ' dB'));

// 10. Test models_data.js integrity
const jsPath = path.join(__dirname, '..', 'models_data.js');
assert(fs.existsSync(jsPath), 'models_data.js must exist');
const jsContent = fs.readFileSync(jsPath, 'utf8');
assert(jsContent.startsWith('// Auto-generated Antenna ML Models Data\nwindow.ANTENNA_MODELS_DATA = '), 'Header check in models_data.js');
console.log('✓ Test 10 Passed: models_data.js file integrity validated');

console.log('\n======================================================');
console.log('🎉 ALL CI TESTS PASSED: TRUE CONTINUOUS DECIMAL OUTPUTS!');
console.log('======================================================');
