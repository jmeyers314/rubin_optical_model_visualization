const {DeckGL, OrthographicView, ScatterplotLayer, COORDINATE_SYSTEM} = window.deck;

let N;
const WORLD_RADIUS = 2;
const VIEW_PADDING = 0.9;
const VALUE_DECIMALS = 3;
const TARGET_VMODE_RMS_WORLD_UNITS = 0.12;
const MODE_RANGE_MIN = 0.01;
const MODE_RANGE_MAX = 100.0;
const MODE_RANGE_SAMPLE_COUNT = 6000;
const VMODE_DEBUG = new URLSearchParams(window.location.search).has('vmodeDebug');
const CONTROL_NAMES = [
    'M2 dz [µm]', 'M2 dx [µm]', 'M2 dy [µm]', 'M2 rx [arcsec]', 'M2 ry [arcsec]',
    'Cam dz [µm]', 'Cam dx [µm]', 'Cam dy [µm]', 'Cam rx [arcsec]', 'Cam ry [arcsec]',
    'M1M3 B1 [µm]', 'M1M3 B2 [µm]', 'M1M3 B3 [µm]', 'M1M3 B4 [µm]', 'M1M3 B5 [µm]',
    'M1M3 B6 [µm]', 'M1M3 B7 [µm]', 'M1M3 B8 [µm]', 'M1M3 B9 [µm]', 'M1M3 B10 [µm]',
    'M1M3 B11 [µm]', 'M1M3 B12 [µm]', 'M1M3 B13 [µm]', 'M1M3 B14 [µm]', 'M1M3 B15 [µm]',
    'M1M3 B16 [µm]', 'M1M3 B17 [µm]', 'M1M3 B18 [µm]', 'M1M3 B19 [µm]', 'M1M3 B20 [µm]',
    'M2 B1 [µm]', 'M2 B2 [µm]', 'M2 B3 [µm]', 'M2 B4 [µm]', 'M2 B5 [µm]',
    'M2 B6 [µm]', 'M2 B7 [µm]', 'M2 B8 [µm]', 'M2 B9 [µm]', 'M2 B10 [µm]',
    'M2 B11 [µm]', 'M2 B12 [µm]', 'M2 B13 [µm]', 'M2 B14 [µm]', 'M2 B15 [µm]',
    'M2 B16 [µm]', 'M2 B17 [µm]', 'M2 B18 [µm]', 'M2 B19 [µm]', 'M2 B20 [µm]'
];
const CONTROL_RANGES = [
  200.0, 2500.0, 2500.0, 120.0, 120.0,
  200.0, 6000.0, 6000.0, 120.0, 120.0,
  2.5, 2.5, 2.5, 2.0, 2.0,
  2.0, 2.0, 1.25, 1.25, 1.25,
  1.25, 1.0, 1.0, 1.0, 1.0,
  1.0, 0.5, 0.5, 0.5, 0.5,

  2.5, 2.5, 2.5, 2.5, 2.0,
  2.0, 2.0, 2.0, 2.0, 1.25,
  1.0, 1.0, 1.0, 1.0, 1.0,
  0.5, 0.5, 0.5, 0.5, 0.5,
];

let K = CONTROL_NAMES.length;
const DEFAULT_COMBO_RANGE = 1.0;
const DEFAULT_COMBO_STEP = 0.01;

let p = new Float32Array(K);
let pBase = new Float32Array(K);
let comboCount = 0;
let combo = new Float32Array(0);
let comboMatrix = new Float32Array(0);
let comboLabels = [];
let comboRanges = [];
let comboSteps = [];
let comboModeMatrix = [];
let comboSingularValues = [];
let comboSelectedDofs = [];
let activeDofs = new Set();
let dofRanges = [];
let dofPower = [];
let defaultUseDof = [];
let defaultNkeep = null;
let x0, y0, Sx, Sy;
let modelRadius = WORLD_RADIUS;
let positions = new Float32Array(0);
let data = [];

function estimateRadius(x, y) {
  let maxR2 = 0;
  for (let i = 0; i < x.length; i++) {
    const r2 = x[i] * x[i] + y[i] * y[i];
    if (r2 > maxR2) maxR2 = r2;
  }
  return Math.max(1e-6, Math.sqrt(maxR2));
}

function getControlSpec(index) {
  const label = CONTROL_NAMES[index] ?? `p${index}`;
  const range = CONTROL_RANGES[index] ?? 1;
  const step = Math.max(range / 200, 1e-4);
  return {label, range, step};
}

function getComboSpec(index) {
  const range = comboRanges[index] ?? DEFAULT_COMBO_RANGE;
  return {
    label: comboLabels[index] ?? `Vmode ${index + 1}`,
    range,
    step: comboSteps[index] ?? Math.max(range / 200, DEFAULT_COMBO_STEP)
  };
}

function clampControlValue(index, value) {
  const range = CONTROL_RANGES[index] ?? Infinity;
  return Math.max(-range, Math.min(range, value));
}

function parseUseDofString(text, controlCount) {
  const result = [];
  const seen = new Set();
  const cleaned = (text ?? '').trim();
  if (!cleaned) {
    throw new Error('use_dof cannot be empty');
  }
  for (const rawPart of cleaned.split(',')) {
    const part = rawPart.trim();
    if (!part) continue;
    if (part.includes('-')) {
      const pieces = part.split('-').map((v) => v.trim());
      if (pieces.length !== 2) {
        throw new Error(`Invalid use_dof range '${part}'`);
      }
      const start = Number(pieces[0]);
      const end = Number(pieces[1]);
      if (!Number.isInteger(start) || !Number.isInteger(end)) {
        throw new Error(`Invalid use_dof range '${part}'`);
      }
      const lo = Math.min(start, end);
      const hi = Math.max(start, end);
      for (let i = lo; i <= hi; i++) {
        if (i < 0 || i >= controlCount) {
          throw new Error(`use_dof index ${i} out of bounds [0, ${controlCount - 1}]`);
        }
        if (!seen.has(i)) {
          seen.add(i);
          result.push(i);
        }
      }
    } else {
      const value = Number(part);
      if (!Number.isInteger(value)) {
        throw new Error(`Invalid use_dof index '${part}'`);
      }
      if (value < 0 || value >= controlCount) {
        throw new Error(`use_dof index ${value} out of bounds [0, ${controlCount - 1}]`);
      }
      if (!seen.has(value)) {
        seen.add(value);
        result.push(value);
      }
    }
  }
  if (result.length === 0) {
    throw new Error('use_dof must include at least one index');
  }
  result.sort((a, b) => a - b);
  return result;
}

function formatUseDofString(indices) {
  if (!Array.isArray(indices) || indices.length === 0) return '';
  const sorted = Array.from(new Set(indices.map((v) => Number(v)).filter(Number.isInteger))).sort((a, b) => a - b);
  if (sorted.length === 0) return '';

  const chunks = [];
  let start = sorted[0];
  let prev = sorted[0];

  for (let i = 1; i < sorted.length; i++) {
    const value = sorted[i];
    if (value === prev + 1) {
      prev = value;
      continue;
    }
    chunks.push(start === prev ? `${start}` : `${start}-${prev}`);
    start = value;
    prev = value;
  }
  chunks.push(start === prev ? `${start}` : `${start}-${prev}`);
  return chunks.join(',');
}

function summarizeArray(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return {min: NaN, max: NaN, mean: NaN};
  }
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
    sum += value;
  }
  return {
    min,
    max,
    mean: sum / values.length
  };
}

function computeComboConfigFromSelection(selectedDofs, requestedNkeep) {
  if (!Sx || !Sy || !Number.isInteger(N) || N <= 0) {
    throw new Error('Sensitivity arrays are not loaded');
  }
  if (!window.numeric || typeof window.numeric.svd !== 'function') {
    throw new Error('numeric.js SVD is unavailable');
  }
  const localDofCount = selectedDofs.length;
  const nkeep = Math.max(1, Math.min(requestedNkeep, localDofCount));

  const weights = new Float64Array(localDofCount);
  for (let localIndex = 0; localIndex < localDofCount; localIndex++) {
    const dof = selectedDofs[localIndex];
    weights[localIndex] = (Number(dofRanges[dof]) || 1) * (Number(dofPower[dof]) || 1);
  }

  const gram = Array.from({length: localDofCount}, () => new Array(localDofCount).fill(0));
  for (let a = 0; a < localDofCount; a++) {
    const dofA = selectedDofs[a];
    const offA = dofA * N;
    const wa = weights[a];
    for (let b = a; b < localDofCount; b++) {
      const dofB = selectedDofs[b];
      const offB = dofB * N;
      const wb = weights[b];
      let dot = 0;
      for (let i = 0; i < N; i++) {
        const sxA = Sx[offA + i];
        const sxB = Sx[offB + i];
        if (Number.isFinite(sxA) && Number.isFinite(sxB)) {
          dot += sxA * sxB;
        }
        const syA = Sy[offA + i];
        const syB = Sy[offB + i];
        if (Number.isFinite(syA) && Number.isFinite(syB)) {
          dot += syA * syB;
        }
      }
      const value = dot * wa * wb;
      gram[a][b] = value;
      gram[b][a] = value;
    }
  }

  const svd = window.numeric.svd(gram);
  const v = svd?.V;
  const s = svd?.S;
  if (!Array.isArray(v) || v.length !== localDofCount) {
    throw new Error('numeric.svd returned an unexpected V matrix');
  }
  if (!Array.isArray(s) || s.length === 0) {
    throw new Error('numeric.svd returned unexpected singular values');
  }

  const controlMajor = new Float32Array(K * nkeep);
  const localModeVectors = Array.from({length: nkeep}, () => new Float64Array(localDofCount));
  for (let modeIndex = 0; modeIndex < nkeep; modeIndex++) {
    for (let localIndex = 0; localIndex < localDofCount; localIndex++) {
      const dof = selectedDofs[localIndex];
      const coeff = Number(v[localIndex][modeIndex]);
      localModeVectors[modeIndex][localIndex] = coeff;
      controlMajor[dof * nkeep + modeIndex] = coeff;
    }
  }

  const sampleStride = Math.max(1, Math.floor(N / MODE_RANGE_SAMPLE_COUNT));
  const ranges = new Array(nkeep);
  const steps = new Array(nkeep);

  for (let modeIndex = 0; modeIndex < nkeep; modeIndex++) {
    const modeVec = localModeVectors[modeIndex];
    let sumR2 = 0;
    let sampleCount = 0;

    for (let i = 0; i < N; i += sampleStride) {
      let dx = 0;
      let dy = 0;

      for (let localIndex = 0; localIndex < localDofCount; localIndex++) {
        const coeff = modeVec[localIndex];
        if (coeff === 0) continue;
        const dof = selectedDofs[localIndex];
        const off = dof * N + i;

        const sx = Sx[off];
        if (Number.isFinite(sx)) dx += sx * coeff;

        const sy = Sy[off];
        if (Number.isFinite(sy)) dy += sy * coeff;
      }

      sumR2 += dx * dx + dy * dy;
      sampleCount++;
    }

    const rms = Math.sqrt(sumR2 / Math.max(1, sampleCount));
    const rawRange = TARGET_VMODE_RMS_WORLD_UNITS / Math.max(rms, 1e-12);
    const range = Math.max(MODE_RANGE_MIN, Math.min(MODE_RANGE_MAX, rawRange));
    ranges[modeIndex] = Number(range.toFixed(4));
    steps[modeIndex] = Number(Math.max(range / 200, 0.001).toFixed(5));
  }

  if (VMODE_DEBUG) {
    const singularValues = Array.isArray(svd?.S) ? svd.S.map((value) => Number(value)) : [];
    const gramDiag = new Array(localDofCount);
    for (let i = 0; i < localDofCount; i++) {
      gramDiag[i] = Number(gram[i][i]);
    }
    const gramSummary = summarizeArray(gramDiag);
    const singularSummary = summarizeArray(singularValues.slice(0, nkeep));
    const rangeSummary = summarizeArray(ranges);

    console.groupCollapsed('[vmode debug] combo build');
    console.log('selected_dofs', selectedDofs);
    console.log('requested_nkeep', requestedNkeep, 'effective_nkeep', nkeep);
    console.log('gram_diag_summary', gramSummary);
    console.log('sv_top', singularValues.slice(0, Math.min(10, singularValues.length)));
    console.log('sv_summary_kept', singularSummary);
    console.log('range_summary', rangeSummary);
    console.log('ranges', ranges);
    console.log('steps', steps);
    console.groupEnd();
  }

  return {
    comboCount: nkeep,
    controlMajor,
    labels: Array.from({length: nkeep}, (_, i) => `Vmode ${i + 1}`),
    ranges,
    steps,
    modeMatrix: v,
    singularValues: s.map((value) => Number(value))
  };
}

function setDofActivity(selectedDofs) {
  activeDofs = new Set(selectedDofs);
  for (let k = 0; k < K; k++) {
    if (!activeDofs.has(k)) {
      pBase[k] = 0;
      p[k] = 0;
    }
  }
}

function applyModeSelection(selectedDofs, requestedNkeep) {
  const comboConfig = computeComboConfigFromSelection(selectedDofs, requestedNkeep);
  setDofActivity(selectedDofs);
  comboSelectedDofs = selectedDofs.slice();
  comboCount = comboConfig.comboCount;
  comboLabels = comboConfig.labels ?? [];
  comboRanges = comboConfig.ranges ?? [];
  comboSteps = comboConfig.steps ?? [];
  comboModeMatrix = comboConfig.modeMatrix ?? [];
  comboSingularValues = comboConfig.singularValues ?? [];
  combo = new Float32Array(comboCount);
  comboMatrix = comboConfig.controlMajor;
  buildSliders();
  renderVmodeDiagnostics();
  resetControls();
}

function applyModeSelectionFromInputs() {
  if (!Sx || !Sy) return;
  const useDofText = useDofInput?.value ?? '';
  const selectedDofs = parseUseDofString(useDofText, K);
  if (useDofInput) {
    useDofInput.value = formatUseDofString(selectedDofs);
  }
  const requestedNkeep = Number(nkeepInput?.value ?? 1);
  if (!Number.isInteger(requestedNkeep) || requestedNkeep <= 0) {
    throw new Error('nkeep must be a positive integer');
  }
  applyModeSelection(selectedDofs, requestedNkeep);
}

function formatControlValue(value) {
  return value.toFixed(VALUE_DECIMALS);
}

async function loadPackedModel(metaUrl, modelUrl) {
  const metaResponse = await fetch(metaUrl);
  if (!metaResponse.ok) {
    throw new Error(`Failed to load ${metaUrl}: ${metaResponse.status} ${metaResponse.statusText}`);
  }
  const meta = await metaResponse.json();

  const modelBuffer = await fetchArrayBufferWithProgress(modelUrl, updateLoadingProgress);
  const packed = new Float32Array(modelBuffer);

  const n = Number(meta.N);
  const k = Number(meta.K);
  const metaDofRanges = meta.dof_ranges;
  const metaDofPower = meta.dof_power;
  const metaDefaultUseDof = meta.default_use_dof;
  const metaDefaultNkeep = Number(meta.default_nkeep);
  if (!Number.isInteger(n) || n <= 0 || !Number.isInteger(k) || k <= 0) {
    throw new Error('Metadata must contain positive integer N and K');
  }
  if (!Array.isArray(metaDofRanges) || metaDofRanges.length !== k) {
    throw new Error('Metadata must contain dof_ranges with length K');
  }
  if (!Array.isArray(metaDofPower) || metaDofPower.length !== k) {
    throw new Error('Metadata must contain dof_power with length K');
  }

  const initSpec = meta.layout?.init_xy;
  const sxSpec = meta.layout?.Sx;
  const sySpec = meta.layout?.Sy;
  if (!initSpec || !sxSpec || !sySpec) {
    throw new Error('Metadata layout must include init_xy, Sx, and Sy blocks');
  }

  const initOffset = Number(initSpec.offset_f32);
  const initLength = Number(initSpec.length_f32);
  const sxOffset = Number(sxSpec.offset_f32);
  const sxLength = Number(sxSpec.length_f32);
  const syOffset = Number(sySpec.offset_f32);
  const syLength = Number(sySpec.length_f32);

  if (
    ![initOffset, initLength, sxOffset, sxLength, syOffset, syLength].every(Number.isInteger)
  ) {
    throw new Error('All metadata offsets/lengths must be integers');
  }

  if (initLength !== 2 * n || sxLength !== k * n || syLength !== k * n) {
    throw new Error('Metadata block lengths do not match N/K');
  }

  const totalExpected = Math.max(initOffset + initLength, sxOffset + sxLength, syOffset + syLength);
  if (packed.length < totalExpected) {
    throw new Error(`Packed model too short: expected at least ${totalExpected} float32 values, got ${packed.length}`);
  }

  const initFlat = packed.subarray(initOffset, initOffset + initLength);
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = initFlat[2 * i];
    y[i] = initFlat[2 * i + 1];
  }

  const loadedSx = new Float32Array(packed.subarray(sxOffset, sxOffset + sxLength));
  const loadedSy = new Float32Array(packed.subarray(syOffset, syOffset + syLength));

  return {
    x,
    y,
    sx: loadedSx,
    sy: loadedSy,
    k,
    dofRanges: metaDofRanges.map((v) => Number(v)),
    dofPower: metaDofPower.map((v) => Number(v)),
    defaultUseDof: Array.isArray(metaDefaultUseDof) ? metaDefaultUseDof.map((v) => Number(v)) : [],
    defaultNkeep: Number.isInteger(metaDefaultNkeep) && metaDefaultNkeep > 0 ? metaDefaultNkeep : null
  };
}

async function fetchArrayBufferWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status} ${response.statusText}`);
  }

  const totalHeader = response.headers.get('content-length');
  const totalBytes = totalHeader ? Number(totalHeader) : NaN;
  const reader = response.body?.getReader?.();

  if (!reader) {
    const fallback = await response.arrayBuffer();
    onProgress?.(fallback.byteLength, fallback.byteLength, true);
    return fallback;
  }

  let receivedBytes = 0;
  const chunks = [];
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      receivedBytes += value.byteLength;
      onProgress?.(receivedBytes, totalBytes, false);
    }
  }

  const merged = new Uint8Array(receivedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }

  onProgress?.(receivedBytes, totalBytes, true);
  return merged.buffer;
}

function initModel(initialX, initialY, initialSx, initialSy, loadedK) {
  N = initialX.length;
  if (Number.isInteger(loadedK) && loadedK > 0) {
    K = loadedK;
  }
  p = new Float32Array(K);
  pBase = new Float32Array(K);
  comboCount = 0;
  combo = new Float32Array(0);
  comboMatrix = new Float32Array(0);
  comboLabels = [];
  comboRanges = [];
  comboSteps = [];
  activeDofs = new Set(Array.from({length: K}, (_, i) => i));
  x0 = initialX;
  y0 = initialY;
  positions = new Float32Array(N * 2);
  data = Array.from({length: N}, (_, i) => i);
  Sx = new Float32Array(K * N);
  Sy = new Float32Array(K * N);

  const R = estimateRadius(x0, y0);
  modelRadius = R;

  if (!initialSx || !initialSy) {
    throw new Error('Missing Sx/Sy sensitivity arrays in loaded model');
  }

  if (initialSx.length !== K * N || initialSy.length !== K * N) {
    throw new Error('Loaded sensitivity sizes do not match K*N');
  }
  Sx.set(initialSx);
  Sy.set(initialSy);
}

function updatePositions() {
  for (let i = 0; i < N; i++) {
    positions[2 * i] = x0[i];
    positions[2 * i + 1] = y0[i];
  }
  for (let k = 0; k < K; k++) {
    const pk = p[k];
    if (pk === 0) continue;
    const off = k * N;
    for (let i = 0; i < N; i++) {
      positions[2 * i] += Sx[off + i] * pk;
      positions[2 * i + 1] += Sy[off + i] * pk;
    }
  }
}

const vis = document.getElementById('vis');
const gridCanvas = document.getElementById('grid-canvas');
const canvas = document.getElementById('deck-canvas');
const resetControlsBtn = document.getElementById('reset-controls');
const useDofInput = document.getElementById('use-dof-input');
const nkeepInput = document.getElementById('nkeep-input');
const applyModesBtn = document.getElementById('apply-modes');
const comboSlidersRoot = document.getElementById('combo-sliders');
const vmodeTabPlotsBtn = document.getElementById('vmode-tab-plots');
const vmodeTabCoeffBtn = document.getElementById('vmode-tab-coeff');
const vmodePanelPlots = document.getElementById('vmode-panel-plots');
const vmodePanelCoeff = document.getElementById('vmode-panel-coeff');
const vmodeMatrixCanvas = document.getElementById('vmode-matrix-canvas');
const vmodeSpectrumCanvas = document.getElementById('vmode-spectrum-canvas');
const vmodeCoeffOutput = document.getElementById('vmode-coeff-output');
const mouseCoordsEl = document.getElementById('mouse-coords');
const scaleBarLineEl = document.getElementById('scale-bar-line');
const loadingOverlayEl = document.getElementById('loading-overlay');
const loadingFillEl = document.getElementById('loading-fill');
const loadingTextEl = document.getElementById('loading-text');
const SCALE_BAR_WORLD_UNITS = 0.24;
const GRID_PITCH_WORLD_UNITS = 0.048;
const GRID_MAJOR_EVERY = 5;
const GRID_MINOR_COLOR = 'rgba(122, 178, 255, 0.04)';
const GRID_MAJOR_COLOR = 'rgba(122, 178, 255, 0.05)';
let sliderInputs = [];
let sliderValues = [];
let comboSliderInputs = [];
let comboSliderValues = [];
let syncingBaseSliderUI = false;
let currentViewState = { target: [0, 0, 0], zoom: 0 };

function setVmodeDiagTab(tabName) {
  const showPlots = tabName !== 'coeff';
  if (vmodePanelPlots) vmodePanelPlots.classList.toggle('is-active', showPlots);
  if (vmodePanelCoeff) vmodePanelCoeff.classList.toggle('is-active', !showPlots);
  if (vmodeTabPlotsBtn) vmodeTabPlotsBtn.classList.toggle('is-active', showPlots);
  if (vmodeTabCoeffBtn) vmodeTabCoeffBtn.classList.toggle('is-active', !showPlots);
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function blueWhiteRed(value) {
  const t = clamp01((value + 1) * 0.5);
  let r;
  let g;
  let b;
  if (t <= 0.5) {
    const local = t / 0.5;
    r = Math.round(64 + local * (255 - 64));
    g = Math.round(128 + local * (255 - 128));
    b = Math.round(255);
  } else {
    const local = (t - 0.5) / 0.5;
    r = Math.round(255);
    g = Math.round(255 + local * (64 - 255));
    b = Math.round(255 + local * (64 - 255));
  }
  return `rgb(${r}, ${g}, ${b})`;
}

function setupHiDpiCanvas(canvasEl) {
  if (!canvasEl) return null;
  const rect = canvasEl.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  canvasEl.width = width;
  canvasEl.height = height;
  const context = canvasEl.getContext('2d');
  if (!context) return null;
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, width, height);
  return {context, width, height, dpr};
}

function renderVmodeMatrix() {
  const canvasState = setupHiDpiCanvas(vmodeMatrixCanvas);
  if (!canvasState) return;
  const {context, width, height} = canvasState;

  const matrix = comboModeMatrix;
  const nRows = Array.isArray(matrix) ? matrix.length : 0;
  const nCols = nRows > 0 && Array.isArray(matrix[0]) ? matrix[0].length : 0;
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, width, height);
  if (nRows === 0 || nCols === 0) return;

  const left = 28;
  const right = Math.max(left + 10, width - 8);
  const top = 8;
  const bottom = Math.max(top + 10, height - 20);
  const plotW = Math.max(1, right - left);
  const plotH = Math.max(1, bottom - top);

  let vmax = 0;
  for (let row = 0; row < nRows; row++) {
    for (let col = 0; col < nCols; col++) {
      const value = Math.abs(Number(matrix[row][col]));
      if (value > vmax) vmax = value;
    }
  }
  vmax = Math.max(vmax, 1e-12);

  const cellW = plotW / nCols;
  const cellH = plotH / nRows;
  for (let row = 0; row < nRows; row++) {
    for (let col = 0; col < nCols; col++) {
      const value = Number(matrix[row][col]) / vmax;
      context.fillStyle = blueWhiteRed(value);
      const x0 = Math.floor(left + col * cellW);
      const y0 = Math.floor(top + row * cellH);
      const x1 = Math.ceil(left + (col + 1) * cellW);
      const y1 = Math.ceil(top + (row + 1) * cellH);
      context.fillRect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
    }
  }

  context.strokeStyle = 'rgba(0, 0, 0, 0.12)';
  context.lineWidth = 1;
  for (let c = 0; c <= nCols; c++) {
    const x = Math.round(left + c * cellW) + 0.5;
    context.beginPath();
    context.moveTo(x, top);
    context.lineTo(x, bottom);
    context.stroke();
  }
  for (let r = 0; r <= nRows; r++) {
    const y = Math.round(top + r * cellH) + 0.5;
    context.beginPath();
    context.moveTo(left, y);
    context.lineTo(right, y);
    context.stroke();
  }

  const keepCols = Math.max(0, Math.min(comboCount, nCols));
  if (keepCols < nCols) {
    const fadeX = left + keepCols * cellW;
    context.fillStyle = 'rgba(80, 80, 80, 0.28)';
    context.fillRect(fadeX, top, right - fadeX, bottom - top);

    context.strokeStyle = 'rgba(40, 40, 40, 0.65)';
    context.lineWidth = 1.5;
    context.beginPath();
    context.moveTo(Math.round(fadeX) + 0.5, top);
    context.lineTo(Math.round(fadeX) + 0.5, bottom);
    context.stroke();
  }

  const dofGroupIndex = (dof) => {
    if (dof <= 4) return 0;
    if (dof <= 9) return 1;
    if (dof <= 29) return 2;
    return 3;
  };

  if (comboSelectedDofs.length === nRows) {
    context.strokeStyle = 'rgba(0, 0, 0, 0.55)';
    context.lineWidth = 2;
    for (let row = 1; row < nRows; row++) {
      const prevGroup = dofGroupIndex(comboSelectedDofs[row - 1]);
      const thisGroup = dofGroupIndex(comboSelectedDofs[row]);
      if (prevGroup === thisGroup) continue;
      const y = Math.round(top + row * cellH) + 0.5;
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(right, y);
      context.stroke();
    }
  }

  context.lineWidth = 1;

  context.fillStyle = '#444';
  context.font = '11px system-ui, sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'alphabetic';
  context.fillText('mode index', (left + right) / 2, height - 4);

  context.save();
  context.translate(10, (top + bottom) / 2);
  context.rotate(-Math.PI / 2);
  context.textAlign = 'center';
  context.textBaseline = 'alphabetic';
  context.fillText('selected dof index', 0, 0);
  context.restore();
}

function renderVmodeSpectrum() {
  const canvasState = setupHiDpiCanvas(vmodeSpectrumCanvas);
  if (!canvasState) return;
  const {context, width, height} = canvasState;

  const values = comboSingularValues;
  const n = Array.isArray(values) ? values.length : 0;
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, width, height);
  if (n === 0) return;

  const left = 36;
  const right = Math.max(left + 10, width - 8);
  const top = 8;
  const bottom = Math.max(top + 10, height - 18);
  const plotW = Math.max(1, right - left);
  const plotH = Math.max(1, bottom - top);

  const logs = values.map((value) => Math.log10(Math.max(Number(value), 1e-20)));
  const maxLog = Math.max(...logs);
  const minLog = Math.min(...logs);
  const denom = Math.max(1e-12, maxLog - minLog);

  context.strokeStyle = 'rgba(0, 0, 0, 0.18)';
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(left + 0.5, top + 0.5);
  context.lineTo(left + 0.5, bottom + 0.5);
  context.lineTo(right + 0.5, bottom + 0.5);
  context.stroke();

  context.strokeStyle = '#1f4e9a';
  context.lineWidth = 1.5;
  context.beginPath();
  for (let i = 0; i < n; i++) {
    const x = left + (n === 1 ? 0 : (i / (n - 1)) * plotW);
    const y = top + ((maxLog - logs[i]) / denom) * plotH;
    if (i === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  }
  context.stroke();

  const keepIndex = Math.max(1, Math.min(comboCount, n)) - 1;
  const keepX = left + (n === 1 ? 0 : (keepIndex / (n - 1)) * plotW);
  context.strokeStyle = 'rgba(180, 30, 30, 0.7)';
  context.setLineDash([4, 3]);
  context.beginPath();
  context.moveTo(keepX + 0.5, top);
  context.lineTo(keepX + 0.5, bottom);
  context.stroke();
  context.setLineDash([]);

  context.fillStyle = '#444';
  context.font = '11px system-ui, sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'alphabetic';
  context.fillText('mode index', (left + right) / 2, height - 4);

  context.save();
  context.translate(11, (top + bottom) / 2);
  context.rotate(-Math.PI / 2);
  context.textAlign = 'center';
  context.textBaseline = 'alphabetic';
  context.fillText('log10(singular value)', 0, 0);
  context.restore();

  context.textAlign = 'right';
  context.textBaseline = 'middle';
  context.fillText(maxLog.toFixed(2), left - 4, top);
  context.fillText(minLog.toFixed(2), left - 4, bottom);

  context.textAlign = 'left';
  context.textBaseline = 'top';
  context.fillStyle = 'rgba(180, 30, 30, 0.8)';
  context.fillText(`nkeep=${comboCount}`, Math.min(keepX + 4, right - 52), top + 2);
}

function renderVmodeCoefficients() {
  if (!vmodeCoeffOutput) return;
  const matrix = comboModeMatrix;
  const nRows = Array.isArray(matrix) ? matrix.length : 0;
  const nCols = nRows > 0 && Array.isArray(matrix[0]) ? matrix[0].length : 0;
  if (nRows === 0 || nCols === 0 || comboSelectedDofs.length !== nRows) {
    vmodeCoeffOutput.value = 'No vmode coefficients available yet. Click Apply to compute modes.';
    return;
  }

  const header = ['dof_index', 'dof_name'];
  for (let col = 0; col < nCols; col++) {
    header.push(`mode_${col + 1}`);
  }

  const rows = [header.join(',')];
  for (let row = 0; row < nRows; row++) {
    const dof = comboSelectedDofs[row];
    const label = CONTROL_NAMES[dof] ?? `p${dof}`;
    const values = [String(dof), `"${label.replaceAll('"', '""')}"`];
    for (let col = 0; col < nCols; col++) {
      values.push(Number(matrix[row][col]).toExponential(8));
    }
    rows.push(values.join(','));
  }

  rows.push('');
  rows.push(`nkeep,${comboCount}`);
  if (Array.isArray(comboSingularValues) && comboSingularValues.length > 0) {
    rows.push(`singular_values,${comboSingularValues.map((v) => Number(v).toExponential(8)).join(',')}`);
  }
  vmodeCoeffOutput.value = rows.join('\n');
}

function renderVmodeDiagnostics() {
  renderVmodeMatrix();
  renderVmodeSpectrum();
  renderVmodeCoefficients();
}

function getComboContributionForControl(index) {
  if (comboCount === 0) return 0;
  let value = 0;
  const rowOffset = index * comboCount;
  for (let j = 0; j < comboCount; j++) {
    value += comboMatrix[rowOffset + j] * combo[j];
  }
  return value;
}

function syncBaseSlidersFromState() {
  syncingBaseSliderUI = true;
  for (let k = 0; k < K; k++) {
    if (sliderInputs[k]) sliderInputs[k].value = String(p[k]);
    if (sliderValues[k]) sliderValues[k].textContent = formatControlValue(p[k]);
  }
  syncingBaseSliderUI = false;
}

function recomputeControls(syncBaseUI) {
  for (let k = 0; k < K; k++) {
    const combined = pBase[k] + getComboContributionForControl(k);
    p[k] = clampControlValue(k, combined);
  }
  if (syncBaseUI) {
    syncBaseSlidersFromState();
  }
}

function setControlsReady(ready) {
  if (!resetControlsBtn) return;
  resetControlsBtn.disabled = !ready;
  resetControlsBtn.textContent = ready ? 'Reset (r)' : 'Loading…';
  if (applyModesBtn) applyModesBtn.disabled = !ready;
}

function formatMiB(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function updateLoadingProgress(loadedBytes, totalBytes, done) {
  if (!loadingOverlayEl) return;
  if (!Number.isFinite(loadedBytes) || loadedBytes < 0) loadedBytes = 0;

  const hasTotal = Number.isFinite(totalBytes) && totalBytes > 0;
  const ratio = hasTotal ? Math.min(1, loadedBytes / totalBytes) : 0;

  if (loadingFillEl) {
    loadingFillEl.style.width = `${(ratio * 100).toFixed(1)}%`;
  }

  if (loadingTextEl) {
    if (hasTotal) {
      const percent = Math.min(100, ratio * 100).toFixed(1);
      loadingTextEl.textContent = `${percent}% (${formatMiB(loadedBytes)} / ${formatMiB(totalBytes)})`;
    } else {
      loadingTextEl.textContent = `Downloaded ${formatMiB(loadedBytes)}…`;
    }
  }

  if (done) {
    if (loadingFillEl) loadingFillEl.style.width = '100%';
    loadingOverlayEl.classList.add('is-hidden');
  }
}

function showLoadingError(errorMessage) {
  if (!loadingOverlayEl || !loadingTextEl) return;
  if (loadingFillEl) loadingFillEl.style.width = '0%';
  loadingTextEl.textContent = errorMessage;
}

const deckgl = new DeckGL({
  canvas,
  views: [new OrthographicView({id: 'ortho'})],
  initialViewState: { target: [0, 0, 0], zoom: 0 },
  controller: false,
  layers: []
});

function getFitZoom(width, height, radius) {
  const diameter = Math.max(1e-6, 2 * radius);
  const usablePixels = Math.max(1, Math.min(width, height) * VIEW_PADDING);
  const scale = usablePixels / diameter;
  return Math.log2(scale);
}

function resizeDeck() {
  const r = vis.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (gridCanvas) {
    gridCanvas.width = Math.max(1, Math.floor(r.width * dpr));
    gridCanvas.height = Math.max(1, Math.floor(r.height * dpr));
  }
  canvas.width = Math.max(1, Math.floor(r.width * dpr));
  canvas.height = Math.max(1, Math.floor(r.height * dpr));
  const zoom = getFitZoom(r.width, r.height, modelRadius);
  currentViewState = { target: [0, 0, 0], zoom };
  drawGrid(r.width, r.height);
  updateScaleBar();
  deckgl.setProps({
    width: r.width,
    height: r.height,
    viewState: currentViewState
  });
}

function drawGrid(width, height) {
  if (!gridCanvas) return;
  const context = gridCanvas.getContext('2d');
  if (!context) return;

  const dpr = window.devicePixelRatio || 1;
  const canvasWidth = Math.max(1, Math.floor(width * dpr));
  const canvasHeight = Math.max(1, Math.floor(height * dpr));

  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, canvasWidth, canvasHeight);

  const scale = Math.pow(2, currentViewState.zoom);
  const pitchDevicePx = Math.max(Math.round(2 * dpr), Math.round(GRID_PITCH_WORLD_UNITS * scale * dpr));
  const majorPitchDevicePx = pitchDevicePx * GRID_MAJOR_EVERY;

  const originScreenDeviceX = (width / 2 + (0 - currentViewState.target[0]) * scale) * dpr;
  const originScreenDeviceY = (height / 2 - (0 - currentViewState.target[1]) * scale) * dpr;

  const offsetDeviceX = ((originScreenDeviceX % pitchDevicePx) + pitchDevicePx) % pitchDevicePx;
  const offsetDeviceY = ((originScreenDeviceY % pitchDevicePx) + pitchDevicePx) % pitchDevicePx;
  const majorOffsetDeviceX = ((originScreenDeviceX % majorPitchDevicePx) + majorPitchDevicePx) % majorPitchDevicePx;
  const majorOffsetDeviceY = ((originScreenDeviceY % majorPitchDevicePx) + majorPitchDevicePx) % majorPitchDevicePx;

  context.lineWidth = 1;

  context.strokeStyle = GRID_MINOR_COLOR;
  context.beginPath();
  for (let ix = 0, x = offsetDeviceX; x <= canvasWidth; ix++, x += pitchDevicePx) {
    if (ix % GRID_MAJOR_EVERY === 0) continue;
    const xf = Math.round(x) + 0.5;
    context.moveTo(xf, 0);
    context.lineTo(xf, canvasHeight);
  }
  for (let iy = 0, y = offsetDeviceY; y <= canvasHeight; iy++, y += pitchDevicePx) {
    if (iy % GRID_MAJOR_EVERY === 0) continue;
    const yf = Math.round(y) + 0.5;
    context.moveTo(0, yf);
    context.lineTo(canvasWidth, yf);
  }
  context.stroke();

  context.strokeStyle = GRID_MAJOR_COLOR;
  context.beginPath();
  for (let x = majorOffsetDeviceX; x <= canvasWidth; x += majorPitchDevicePx) {
    const xf = Math.round(x) + 0.5;
    context.moveTo(xf, 0);
    context.lineTo(xf, canvasHeight);
  }
  for (let y = majorOffsetDeviceY; y <= canvasHeight; y += majorPitchDevicePx) {
    const yf = Math.round(y) + 0.5;
    context.moveTo(0, yf);
    context.lineTo(canvasWidth, yf);
  }
  context.stroke();
}

function updateScaleBar() {
  if (!scaleBarLineEl) return;
  const pxPerWorldUnit = Math.pow(2, currentViewState.zoom);
  const widthPx = Math.max(1, SCALE_BAR_WORLD_UNITS * pxPerWorldUnit);
  scaleBarLineEl.style.width = `${widthPx.toFixed(1)}px`;
}

function updateMouseCoords(event) {
  if (!mouseCoordsEl) return;
  const rect = vis.getBoundingClientRect();
  const px = event.clientX - rect.left;
  const py = event.clientY - rect.top;
  const scale = Math.pow(2, currentViewState.zoom);
  const wx = (px - rect.width / 2) / scale + currentViewState.target[0];
  const wy = (rect.height / 2 - py) / scale + currentViewState.target[1];
  mouseCoordsEl.textContent = `x: ${wx.toFixed(3)}, y: ${wy.toFixed(3)}`;
}

function clearMouseCoords() {
  if (!mouseCoordsEl) return;
  mouseCoordsEl.textContent = 'x: ---, y: ---';
}

let posVersion = 0;

function render() {
  const layer = new ScatterplotLayer({
    id: 'pts',
    data,
    coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
    getPosition: (i) => [positions[2 * i], positions[2 * i + 1], 0],
    updateTriggers: {
      getPosition: posVersion
    },
    getRadius: 1.0,
    radiusUnits: 'pixels',
    stroked: false,
    getFillColor: [120, 180, 255, 60]
  });

  deckgl.setProps({layers: [layer]});
}

let pending = false;
function requestUpdate() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    updatePositions();
    posVersion++;
    render();
  });
}

function buildSliders() {
  const root = document.getElementById('sliders');
  root.innerHTML = '';
  if (comboSlidersRoot) comboSlidersRoot.innerHTML = '';
  sliderInputs = [];
  sliderValues = [];
  comboSliderInputs = [];
  comboSliderValues = [];

  for (let k = 0; k < K; k++) {
    const spec = getControlSpec(k);
    const row = document.createElement('div');
    row.className = 'row';
    const isActive = activeDofs.has(k);
    if (!isActive) {
      row.classList.add('is-inactive');
    }

    const label = document.createElement('label');
    label.textContent = spec.label;

    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(-spec.range);
    input.max = String(spec.range);
    input.step = String(spec.step);
    input.value = '0';
    input.disabled = !isActive;

    const val = document.createElement('span');
    val.textContent = formatControlValue(0);

    input.addEventListener('input', () => {
      if (syncingBaseSliderUI) return;
      const v = Number(input.value);
      const comboContribution = getComboContributionForControl(k);
      pBase[k] = clampControlValue(k, v - comboContribution);
      recomputeControls(true);
      requestUpdate();
    });

    row.appendChild(label);
    row.appendChild(input);
    row.appendChild(val);
    root.appendChild(row);
    sliderInputs.push(input);
    sliderValues.push(val);
  }

  if (!comboSlidersRoot) return;

  for (let j = 0; j < comboCount; j++) {
    const spec = getComboSpec(j);
    const row = document.createElement('div');
    row.className = 'row';

    const label = document.createElement('label');
    label.textContent = spec.label;

    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(-spec.range);
    input.max = String(spec.range);
    input.step = String(spec.step);
    input.value = '0';

    const val = document.createElement('span');
    val.textContent = formatControlValue(0);

    input.addEventListener('input', () => {
      const v = Number(input.value);
      combo[j] = v;
      val.textContent = formatControlValue(v);
      recomputeControls(true);
      requestUpdate();
    });

    row.appendChild(label);
    row.appendChild(input);
    row.appendChild(val);
    comboSlidersRoot.appendChild(row);
    comboSliderInputs.push(input);
    comboSliderValues.push(val);
  }
}

function resetControls() {
  for (let k = 0; k < K; k++) {
    pBase[k] = 0;
    p[k] = 0;
    if (sliderInputs[k]) sliderInputs[k].value = '0';
    if (sliderValues[k]) sliderValues[k].textContent = formatControlValue(0);
  }
  for (let j = 0; j < comboCount; j++) {
    combo[j] = 0;
    if (comboSliderInputs[j]) comboSliderInputs[j].value = '0';
    if (comboSliderValues[j]) comboSliderValues[j].textContent = formatControlValue(0);
  }
  recomputeControls(true);
  requestUpdate();
}

async function start() {
  setControlsReady(false);
  updateLoadingProgress(0, NaN, false);
  const model = await loadPackedModel('./model_meta.json', './model.f32');
  dofRanges = model.dofRanges;
  dofPower = model.dofPower;
  defaultUseDof = model.defaultUseDof;
  defaultNkeep = model.defaultNkeep;

  initModel(model.x, model.y, model.sx, model.sy, model.k);
  if (useDofInput) {
    const defaults = defaultUseDof.length
      ? formatUseDofString(defaultUseDof)
      : '0-9,10-16,30-34';
    useDofInput.value = defaults;
  }
  if (nkeepInput) {
    const initialNkeep = Number.isInteger(defaultNkeep)
      ? defaultNkeep
      : Math.min(12, K);
    nkeepInput.value = String(initialNkeep);
  }

  buildSliders();
  try {
    applyModeSelectionFromInputs();
  } catch (error) {
    showLoadingError(`Initial mode setup failed: ${error.message}`);
  }
  resizeDeck();
  updatePositions();
  posVersion++;
  render();
  setControlsReady(true);
}

start().catch((error) => {
  console.error(error);
  setControlsReady(false);
  showLoadingError(`Load failed: ${error.message}`);
});

window.addEventListener('resize', () => {
  renderVmodeDiagnostics();
  resizeDeck();
  render();
});

if (resetControlsBtn) {
  resetControlsBtn.addEventListener('click', resetControls);
}

if (applyModesBtn) {
  applyModesBtn.addEventListener('click', () => {
    try {
      applyModeSelectionFromInputs();
      requestUpdate();
    } catch (error) {
      showLoadingError(`Mode update failed: ${error.message}`);
    }
  });
}

function handleModeInputEnter(event) {
  if (event.key !== 'Enter') return;
  event.preventDefault();
  try {
    applyModeSelectionFromInputs();
    requestUpdate();
  } catch (error) {
    showLoadingError(`Mode update failed: ${error.message}`);
  }
}

if (useDofInput) {
  useDofInput.addEventListener('keydown', handleModeInputEnter);
}

if (nkeepInput) {
  nkeepInput.addEventListener('keydown', handleModeInputEnter);
}

if (vmodeTabPlotsBtn) {
  vmodeTabPlotsBtn.addEventListener('click', () => setVmodeDiagTab('plots'));
}

if (vmodeTabCoeffBtn) {
  vmodeTabCoeffBtn.addEventListener('click', () => setVmodeDiagTab('coeff'));
}

setVmodeDiagTab('plots');

vis.addEventListener('mousemove', updateMouseCoords);
vis.addEventListener('mouseleave', clearMouseCoords);

window.addEventListener('keydown', (event) => {
  if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
  const tag = event.target && event.target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  if (event.key === 'r' || event.key === 'R') {
    event.preventDefault();
    resetControls();
  }
});
