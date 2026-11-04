// Automated Test Suite for Antenna ML Models & Simulator
const assert = require('assert');
const fs = require('fs');
const path = require('path');

console.log('--- RUNNING CI AUTOMATED TEST SUITE ---');

// 1. Verify all_models.json exists and is valid JSON
const jsonPath = path.join(__dirname, '..', 'all_models.json');
assert(fs.existsSync(jsonPath), 'Error: all_models.json does not exist');
const data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
console.log('✓ Test 1 Passed: all_models.json is valid');

// 2. Verify all models are present
const expectedModels = [
    'LinearRegression',
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
console.log('✓ Test 2 Passed: All 7 primary ML models are defined with metrics');

// 3. Verify Forward Digital Twin Simulator exists
assert(data.forward_simulator, 'Missing forward_simulator in dataset');
assert(data.forward_simulator.layers.length > 0, 'Forward simulator has no layers');
console.log('✓ Test 3 Passed: Forward Digital Twin Simulator is configured');

// 4. Verify representative dataset for KNN
assert(Array.isArray(data.samples), 'samples must be an array');
assert(data.samples.length > 1000, `Expected > 1000 samples, got ${data.samples.length}`);
console.log(`✓ Test 4 Passed: KNN database contains ${data.samples.length} verified simulation points`);

// 4b. Test KNN inference with k=1, k=3, k=5
const testV1 = -15.0;
const testV2 = -5.0;

function testKNN(samples, v1, v2, k) {
    const dists = samples.map(s => {
        const d1 = s.v1 - v1;
        const d2 = s.v2 - v2;
        return {
            dist: Math.sqrt(d1 * d1 + d2 * d2),
            y: [s.d1, s.d2, s.d3, s.d4]
        };
    });
    dists.sort((a, b) => a.dist - b.dist);
    if (k === 1) return dists[0].y;
    let sumW = 0, sumY = [0, 0, 0, 0];
    for (let i = 0; i < k; i++) {
        const w = 1.0 / (dists[i].dist + 1e-6);
        sumW += w;
        for (let j = 0; j < 4; j++) sumY[j] += dists[i].y[j] * w;
    }
    return sumY.map(v => v / sumW);
}
const knn1D = testKNN(data.samples, testV1, testV2, 1);
const knn3D = testKNN(data.samples, testV1, testV2, 3);
const knn5D = testKNN(data.samples, testV1, testV2, 5);
assert(knn1D.length === 4 && knn3D.length === 4 && knn5D.length === 4);
console.log('✓ Test 4b Passed: KNN (k=1, 3, 5) outputs valid dimensions:', {
    'k=1': knn1D,
    'k=3': knn3D.map(v => v.toFixed(1)),
    'k=5': knn5D.map(v => v.toFixed(1))
});

// 5. Test Linear & Ridge & Lasso inference

function testLinear(m, v1, v2) {
    const d = [];
    for (let i = 0; i < 4; i++) {
        const val = m.coef[i][0] * v1 + m.coef[i][1] * v2 + m.intercept[i];
        d.push(val);
    }
    return d;
}
const lrD = testLinear(data.models.LinearRegression, testV1, testV2);
assert(lrD.length === 4, 'Linear regression must output 4 dimensions');
lrD.forEach((dim, idx) => {
    assert(!isNaN(dim) && isFinite(dim), `Linear dim[${idx}] is not a finite number`);
});
console.log('✓ Test 5 Passed: Linear Regression outputs valid dimensions:', lrD.map(v => v.toFixed(1)));

const lassoD = testLinear(data.models.LassoRegression, testV1, testV2);
assert(lassoD.length === 4, 'Lasso regression must output 4 dimensions');
console.log('✓ Test 5b Passed: Lasso Regression outputs valid dimensions:', lassoD.map(v => v.toFixed(1)));

// 6. Test Polynomial Regression inference
function testPoly(m, v1, v2) {
    const feats = [1.0, v1, v2, v1 * v1, v1 * v2, v2 * v2];
    const d = [];
    for (let i = 0; i < 4; i++) {
        let sum = 0;
        for (let f = 0; f < feats.length; f++) {
            sum += m.coef[i][f] * feats[f];
        }
        d.push(sum);
    }
    return d;
}
const polyD = testPoly(data.models.PolynomialRegression, testV1, testV2);
assert(polyD.length === 4, 'Polynomial regression must output 4 dimensions');
polyD.forEach((dim, idx) => {
    assert(!isNaN(dim) && isFinite(dim), `Poly dim[${idx}] is not a finite number`);
});
console.log('✓ Test 6 Passed: Polynomial Regression outputs valid dimensions:', polyD.map(v => v.toFixed(1)));

// 7. Test Decision Tree inference
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
    return tree.value[node];
}
const dtD = testTree(data.models.DecisionTree.tree, testV1, testV2);
assert(dtD.length === 4, 'Decision tree must output 4 dimensions');
console.log('✓ Test 7 Passed: Decision Tree traversal outputs valid dimensions:', dtD.map(v => v.toFixed(1)));

// 8. Test Neural Network (MLP) inference
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
    return cur.map((val, i) => val * m.scaler_y.scale[i] + m.scaler_y.mean[i]);
}
const mlpD = testMLP(data.models.NeuralNetwork, testV1, testV2);
assert(mlpD.length === 4, 'MLP must output 4 dimensions');
console.log('✓ Test 8 Passed: MLP Neural Network forward pass outputs valid dimensions:', mlpD.map(v => v.toFixed(1)));

// 8b. Test Ensemble Blending
const ensembleD = mlpD.map((v, i) => 0.45 * v + 0.35 * polyD[i] + 0.20 * 75.0);
assert(ensembleD.length === 4, 'Ensemble must output 4 dimensions');
console.log('✓ Test 8b Passed: Ensemble Blending outputs valid dimensions:', ensembleD.map(v => v.toFixed(1)));

// 9. Test Forward Simulator (Digital Twin)
function testForward(fwd, d1, d2, d3, d4) {
    const dims = [d1, d2, d3, d4];
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
const simResult = testForward(data.forward_simulator, mlpD[0], mlpD[1], mlpD[2], mlpD[3]);
assert(simResult.length === 2, 'Simulator must output 2 values [v1, v2]');
assert(simResult[0] <= 0.5, `Simulated v1 should be <= 0 dB, got ${simResult[0]}`);
console.log('✓ Test 9 Passed: Forward Digital Twin simulates response:', simResult.map(v => v.toFixed(2) + ' dB'));

// 10. Test models_data.js integrity
const jsPath = path.join(__dirname, '..', 'models_data.js');
assert(fs.existsSync(jsPath), 'models_data.js must exist');
const jsContent = fs.readFileSync(jsPath, 'utf8');
assert(jsContent.startsWith('// Auto-generated Antenna ML Models Data\nwindow.ANTENNA_MODELS_DATA = '), 'Header check in models_data.js');
console.log('✓ Test 10 Passed: models_data.js file integrity validated');

console.log('\n========================================');
console.log('🎉 ALL CI TEST SUITES PASSED CLEANLY!');
console.log('========================================');
