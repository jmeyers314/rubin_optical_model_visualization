let DeckGL;
let OrthographicView;
let ScatterplotLayer;
let COORDINATE_SYSTEM;

let N;
const WORLD_RADIUS = 2;
const VIEW_PADDING = 0.9;
const VALUE_DECIMALS = 3;
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
const DONUT_CORNERS = 4;
const DONUT_ZERNIKE_TERMS = 29;
const WHEEL_ZOOM_SENSITIVITY = 0.0008;
const WHEEL_ZOOM_MAX_STEP = 0.12;
const SPOT_SCALE = 3e-3;
const DONUT_SPOT_SCALE = 5e-4;
const ARCSEC_PER_DEGREE = 3600;
const ANGULAR_DOF_INDICES = new Set([3, 4, 8, 9]);
const DEFAULT_RANGE_NORM_EXP = 1.0;
const DEFAULT_FWHM_NORM_EXP = 1.0;
const DEFAULT_USE_DOF = '0-16,30-34';
const DEFAULT_NKEEP = 12;

let p = new Float32Array(K);
let scienceN = 0;
let donutN = 0;
let scienceX0, scienceY0, scienceSx, scienceSy;
let scienceFieldX, scienceFieldY;
let donutX0, donutY0, donutSx, donutSy;
let donutFieldX, donutFieldY;
let donutCenterX, donutCenterY;
let spotScale = SPOT_SCALE;
let donutSpotScale = DONUT_SPOT_SCALE;
let donutClockAngleDeg = 10;
let donutClockRadiusScale = 1.4;
let zk0 = new Float32Array(0);
let dZk = new Float32Array(0);
let modelRadius = WORLD_RADIUS;
let positions = new Float32Array(0);
let data = [];

// Vmode state
let norm = null;          // Float64Array(K) — normalization weights
let rangeWeights = null;  // Float64Array(K)
let fwhmWeights = null;   // Float64Array(K)
let rangeNormExp = DEFAULT_RANGE_NORM_EXP;
let fwhmNormExp = DEFAULT_FWHM_NORM_EXP;
let wfSens = null;        // Float32Array — wavefront sensitivity matrix (nObs × K)
let wfSensRows = 0;       // number of observation rows in wfSens
let currentUseDof = [];   // Int array of active DOF indices
let currentNkeep = DEFAULT_NKEEP;
let Vh = null;            // Float64Array — (nkeep × nActive) mixing matrix
let fullMixMatrix = null; // Float64Array — (nActive × nActive) full eigenvector matrix
let nActive = 0;          // len(currentUseDof)
let vValues = [];         // current vmode slider values
let vmodeSliderInputs = [];
let vmodeParamInputs = [];
const loadTelemetry = {
  startupStartIso: null,
  startupEndIso: null,
  startupDurationMs: null,
  metaFetchDurationMs: null,
  metaBytes: null,
  metaHttpStatus: null,
  modelFetchDurationMs: null,
  modelBytesExpected: null,
  modelBytesReceived: null,
  modelHttpStatus: null,
  modelUsedStreamReader: null,
  modelDownloadCompleted: false,
};

function nowMs() {
  if (typeof performance !== 'undefined' && performance && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function formatDuration(value) {
  if (!Number.isFinite(value) || value < 0) return 'unknown';
  return value.toFixed(1);
}

function estimateRadius(x, y) {
  let maxR2 = 0;
  for (let i = 0; i < x.length; i++) {
    const r2 = x[i] * x[i] + y[i] * y[i];
    if (r2 > maxR2) maxR2 = r2;
  }
  return Math.max(1e-6, Math.sqrt(maxR2));
}

function getControlSpec(index) {
  const label = CONTROL_NAMES[index] != null ? CONTROL_NAMES[index] : `p${index}`;
  const range = CONTROL_RANGES[index] != null ? CONTROL_RANGES[index] : 1;
  const step = Math.max(range / 200, 1e-4);
  return {label, range, step};
}

function clampControlValue(index, value) {
  const range = CONTROL_RANGES[index] != null ? CONTROL_RANGES[index] : Infinity;
  return Math.max(-range, Math.min(range, value));
}

function formatControlValue(value) {
  return value.toFixed(VALUE_DECIMALS);
}

function dofValueToModelUnits(dofIndex, value) {
  if (ANGULAR_DOF_INDICES.has(dofIndex)) {
    return value / ARCSEC_PER_DEGREE;
  }
  return value;
}

function computeClockedCenter(x, y, isIntra) {
  const theta = (isIntra ? donutClockAngleDeg : -donutClockAngleDeg) * Math.PI / 180;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return {
    x: donutClockRadiusScale * (c * x - s * y),
    y: donutClockRadiusScale * (s * x + c * y),
  };
}

// --- use_dof string parser (mirrors Python StateFactory) ---
function parseUseDof(s) {
  if (typeof s !== 'string') return [];
  s = s.replace(/\s/g, '').trim();
  if (!s) return [];
  const result = [];
  for (const part of s.split(',')) {
    if (part.includes('-')) {
      const [a, b] = part.split('-').map(Number);
      if (!Number.isInteger(a) || !Number.isInteger(b)) continue;
      for (let i = a; i <= b; i++) result.push(i);
    } else {
      const v = Number(part);
      if (Number.isInteger(v)) result.push(v);
    }
  }
  return [...new Set(result)].sort((a, b) => a - b);
}

// --- Format sorted int array back to compact range string ---
function formatUseDof(arr) {
  if (!arr || arr.length === 0) return '';
  const parts = [];
  let start = arr[0], end = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] === end + 1) {
      end = arr[i];
    } else {
      parts.push(start === end ? String(start) : `${start}-${end}`);
      start = arr[i];
      end = arr[i];
    }
  }
  parts.push(start === end ? String(start) : `${start}-${end}`);
  return parts.join(',');
}

// --- Jacobi eigendecomposition for real symmetric matrix ---
function jacobiEigen(G, maxIter) {
  const n = Math.round(Math.sqrt(G.length));
  if (n * n !== G.length) throw new Error('jacobiEigen: not a square matrix');
  if (typeof maxIter !== 'number') maxIter = 100 * n * n;

  // Copy G into A (we'll destroy it)
  const A = new Float64Array(G);
  // V starts as identity
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;

  for (let iter = 0; iter < maxIter; iter++) {
    // Find largest off-diagonal element
    let maxVal = 0, ip = 0, iq = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const v = Math.abs(A[i * n + j]);
        if (v > maxVal) { maxVal = v; ip = i; iq = j; }
      }
    }
    if (maxVal < 1e-15) break;

    const aij = A[ip * n + iq];
    const aii = A[ip * n + ip];
    const ajj = A[iq * n + iq];
    const theta = 0.5 * Math.atan2(2 * aij, aii - ajj);
    const c = Math.cos(theta);
    const s = Math.sin(theta);

    // Rotate rows/cols ip, iq of A
    for (let k = 0; k < n; k++) {
      const aik = A[ip * n + k];
      const ajk = A[iq * n + k];
      A[ip * n + k] = c * aik + s * ajk;
      A[iq * n + k] = -s * aik + c * ajk;
    }
    for (let k = 0; k < n; k++) {
      const aki = A[k * n + ip];
      const akj = A[k * n + iq];
      A[k * n + ip] = c * aki + s * akj;
      A[k * n + iq] = -s * aki + c * akj;
    }

    // Accumulate V
    for (let k = 0; k < n; k++) {
      const vki = V[k * n + ip];
      const vkj = V[k * n + iq];
      V[k * n + ip] = c * vki + s * vkj;
      V[k * n + iq] = -s * vki + c * vkj;
    }
  }

  // Extract eigenvalues from diagonal of A
  const eigenvalues = new Float64Array(n);
  for (let i = 0; i < n; i++) eigenvalues[i] = A[i * n + i];

  // Sort by decreasing eigenvalue, return eigenvectors as columns of V
  const order = Array.from({length: n}, (_, i) => i);
  order.sort((a, b) => eigenvalues[b] - eigenvalues[a]);

  const sortedVals = new Float64Array(n);
  const sortedVecs = new Float64Array(n * n); // each column is an eigenvector
  for (let rank = 0; rank < n; rank++) {
    const orig = order[rank];
    sortedVals[rank] = eigenvalues[orig];
    for (let row = 0; row < n; row++) {
      sortedVecs[row * n + rank] = V[row * n + orig];
    }
  }

  return {values: sortedVals, vectors: sortedVecs};
}

// --- Compute Vh from wavefront sensitivity, norm, useDof, nkeep ---
// Mirrors StateFactory.__init__() from state.py
function computeVh(sens, sensRows, sensK, normArr, useDof, nkeep) {
  const nAct = useDof.length;
  if (nAct === 0 || nkeep <= 0) return null;

  // A_norm = sens * diag(norm)  (column scaling)
  // A_sliced = A_norm[:, useDof]
  // G = A_sliced^T @ A_sliced   (nAct x nAct)
  const G = new Float64Array(nAct * nAct);
  for (let i = 0; i < nAct; i++) {
    for (let j = i; j < nAct; j++) {
      let dot = 0;
      const di = useDof[i];
      const dj = useDof[j];
      const ni = normArr[di];
      const nj = normArr[dj];
      for (let r = 0; r < sensRows; r++) {
        dot += (sens[r * sensK + di] * ni) * (sens[r * sensK + dj] * nj);
      }
      G[i * nAct + j] = dot;
      G[j * nAct + i] = dot;
    }
  }

  const {vectors} = jacobiEigen(G);

  // Vh = top nkeep eigenvectors as rows, denormalized
  // vectors is column-major: vectors[row * nAct + col] where col = eigenvector index
  const nk = Math.min(nkeep, nAct);
  const vh = new Float64Array(nk * nAct);
  const fullMatrix = new Float64Array(nAct * nAct);
  for (let mode = 0; mode < nAct; mode++) {
    for (let j = 0; j < nAct; j++) {
      const val = vectors[j * nAct + mode] * normArr[useDof[j]];
      fullMatrix[mode * nAct + j] = val;
      if (mode < nk) vh[mode * nAct + j] = val;
    }
  }
  return {vh, fullMatrix};
}

async function loadPackedModel(metaUrl, modelUrl) {
  const metaStart = nowMs();
  const metaResponse = await fetch(metaUrl);
  loadTelemetry.metaHttpStatus = metaResponse.status;
  const metaEnd = nowMs();
  loadTelemetry.metaFetchDurationMs = Math.max(0, metaEnd - metaStart);
  const metaLengthHeader = metaResponse.headers.get('content-length');
  const metaBytes = metaLengthHeader ? Number(metaLengthHeader) : NaN;
  loadTelemetry.metaBytes = Number.isFinite(metaBytes) && metaBytes >= 0 ? metaBytes : null;
  if (!metaResponse.ok) {
    throw new Error(`Failed to load ${metaUrl}: ${metaResponse.status} ${metaResponse.statusText}`);
  }
  const meta = await metaResponse.json();

  const modelBuffer = await fetchArrayBufferWithProgress(modelUrl, updateLoadingProgress);
  const packed = new Float32Array(modelBuffer);

  const n = Number(meta.N);
  const k = Number(meta.K);
  if (!Number.isInteger(n) || n <= 0 || !Number.isInteger(k) || k <= 0) {
    throw new Error('Metadata must contain positive integer N and K');
  }

  const layout = meta && meta.layout ? meta.layout : null;
  const spotSpec = layout ? layout.spot_xy : undefined;
  const sxSpec = layout ? layout.Sx : undefined;
  const sySpec = layout ? layout.Sy : undefined;
  const fieldSpec = layout ? layout.field_xy : undefined;
  const donutSpotSpec = layout ? layout.donut_spot_xy : undefined;
  const donutSxSpec = layout ? layout.donut_Sx : undefined;
  const donutSySpec = layout ? layout.donut_Sy : undefined;
  const donutFieldSpec = layout ? layout.donut_field_xy : undefined;
  const rangeWeightsSpec = layout ? layout.range_weights : undefined;
  const fwhmWeightsSpec = layout ? layout.fwhm_weights : undefined;
  const wfSensSpec = layout ? layout.wf_sens : undefined;
  if (!spotSpec || !sxSpec || !sySpec || !fieldSpec) {
    throw new Error('Metadata layout must include spot_xy, Sx, Sy, and field_xy blocks');
  }

  const spotOffset = Number(spotSpec.offset_f32);
  const spotLength = Number(spotSpec.length_f32);
  const sxOffset = Number(sxSpec.offset_f32);
  const sxLength = Number(sxSpec.length_f32);
  const syOffset = Number(sySpec.offset_f32);
  const syLength = Number(sySpec.length_f32);
  const fieldOffset = Number(fieldSpec.offset_f32);
  const fieldLength = Number(fieldSpec.length_f32);

  const donutCount = Number(meta.donut_N || 0);
  const hasDonuts = donutCount > 0 && donutSpotSpec && donutSxSpec && donutSySpec && donutFieldSpec;

  let donutSpotOffset = 0;
  let donutSpotLength = 0;
  let donutSxOffset = 0;
  let donutSxLength = 0;
  let donutSyOffset = 0;
  let donutSyLength = 0;
  let donutFieldOffset = 0;
  let donutFieldLength = 0;
  if (hasDonuts) {
    donutSpotOffset = Number(donutSpotSpec.offset_f32);
    donutSpotLength = Number(donutSpotSpec.length_f32);
    donutSxOffset = Number(donutSxSpec.offset_f32);
    donutSxLength = Number(donutSxSpec.length_f32);
    donutSyOffset = Number(donutSySpec.offset_f32);
    donutSyLength = Number(donutSySpec.length_f32);
    donutFieldOffset = Number(donutFieldSpec.offset_f32);
    donutFieldLength = Number(donutFieldSpec.length_f32);
  }

  const hasVmodeData = !!(rangeWeightsSpec && fwhmWeightsSpec && wfSensSpec);
  let rangeWeightsOffset = 0;
  let rangeWeightsLength = 0;
  let fwhmWeightsOffset = 0;
  let fwhmWeightsLength = 0;
  let wfSensOffset = 0;
  let wfSensLength = 0;
  let wfRows = 0;
  let wfCols = 0;
  if (hasVmodeData) {
    rangeWeightsOffset = Number(rangeWeightsSpec.offset_f32);
    rangeWeightsLength = Number(rangeWeightsSpec.length_f32);
    fwhmWeightsOffset = Number(fwhmWeightsSpec.offset_f32);
    fwhmWeightsLength = Number(fwhmWeightsSpec.length_f32);
    wfSensOffset = Number(wfSensSpec.offset_f32);
    wfSensLength = Number(wfSensSpec.length_f32);
    wfRows = wfSensSpec.shape ? Number(wfSensSpec.shape[0]) : 0;
    wfCols = wfSensSpec.shape ? Number(wfSensSpec.shape[1]) : 0;
  }

  if (
    ![spotOffset, spotLength, sxOffset, sxLength, syOffset, syLength, fieldOffset, fieldLength].every(Number.isInteger)
  ) {
    throw new Error('All metadata offsets/lengths must be integers');
  }
  if (hasDonuts && ![
    donutSpotOffset,
    donutSpotLength,
    donutSxOffset,
    donutSxLength,
    donutSyOffset,
    donutSyLength,
    donutFieldOffset,
    donutFieldLength,
  ].every(Number.isInteger)) {
    throw new Error('All donut metadata offsets/lengths must be integers');
  }
  if (hasVmodeData && ![
    rangeWeightsOffset,
    rangeWeightsLength,
    fwhmWeightsOffset,
    fwhmWeightsLength,
    wfSensOffset,
    wfSensLength,
  ].every(Number.isInteger)) {
    throw new Error('All vmode metadata offsets/lengths must be integers');
  }

  if (spotLength !== 2 * n || sxLength !== k * n || syLength !== k * n || fieldLength !== 2 * n) {
    throw new Error('Metadata block lengths do not match N/K');
  }
  if (hasDonuts) {
    if (donutSpotLength !== 2 * donutCount || donutSxLength !== k * donutCount || donutSyLength !== k * donutCount || donutFieldLength !== 2 * donutCount) {
      throw new Error('Donut metadata block lengths do not match donut_N/K');
    }
  }
  if (hasVmodeData) {
    if (rangeWeightsLength !== k || fwhmWeightsLength !== k) {
      throw new Error('Vmode weight lengths do not match K');
    }
    if (!Number.isInteger(wfRows) || !Number.isInteger(wfCols) || wfRows <= 0 || wfCols !== k) {
      throw new Error('wf_sens shape is invalid or does not match K');
    }
    if (wfSensLength !== wfRows * wfCols) {
      throw new Error('wf_sens metadata length does not match its shape');
    }
  }

  const zk0Spec = layout ? layout.zk0 : undefined;
  const dzkSpec = layout ? layout.dzk : undefined;
  const hasZernikes = !!(zk0Spec && dzkSpec);
  let zk0Offset = 0, zk0Length = 0, dzkOffset = 0, dzkLength = 0;
  let zkCorners = 0, zkTerms = 0;
  if (hasZernikes) {
    zk0Offset = Number(zk0Spec.offset_f32);
    zk0Length = Number(zk0Spec.length_f32);
    dzkOffset = Number(dzkSpec.offset_f32);
    dzkLength = Number(dzkSpec.length_f32);
    zkCorners = zk0Spec.shape ? Number(zk0Spec.shape[0]) : DONUT_CORNERS;
    zkTerms = zk0Spec.shape ? Number(zk0Spec.shape[1]) : DONUT_ZERNIKE_TERMS;
  }

  const totalExpected = Math.max(
    spotOffset + spotLength,
    sxOffset + sxLength,
    syOffset + syLength,
    fieldOffset + fieldLength,
    hasDonuts ? donutSpotOffset + donutSpotLength : 0,
    hasDonuts ? donutSxOffset + donutSxLength : 0,
    hasDonuts ? donutSyOffset + donutSyLength : 0,
    hasDonuts ? donutFieldOffset + donutFieldLength : 0,
    hasVmodeData ? rangeWeightsOffset + rangeWeightsLength : 0,
    hasVmodeData ? fwhmWeightsOffset + fwhmWeightsLength : 0,
    hasVmodeData ? wfSensOffset + wfSensLength : 0,
    hasZernikes ? zk0Offset + zk0Length : 0,
    hasZernikes ? dzkOffset + dzkLength : 0
  );
  if (packed.length < totalExpected) {
    throw new Error(`Packed model too short: expected at least ${totalExpected} float32 values, got ${packed.length}`);
  }

  const spotFlat = packed.subarray(spotOffset, spotOffset + spotLength);
  const fieldFlat = packed.subarray(fieldOffset, fieldOffset + fieldLength);
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  const loadedFieldX = new Float32Array(n);
  const loadedFieldY = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = spotFlat[2 * i];
    y[i] = spotFlat[2 * i + 1];
    loadedFieldX[i] = fieldFlat[2 * i];
    loadedFieldY[i] = fieldFlat[2 * i + 1];
  }

  const loadedSx = new Float32Array(packed.subarray(sxOffset, sxOffset + sxLength));
  const loadedSy = new Float32Array(packed.subarray(syOffset, syOffset + syLength));

  let loadedDonutX = new Float32Array(0);
  let loadedDonutY = new Float32Array(0);
  let loadedDonutFieldX = new Float32Array(0);
  let loadedDonutFieldY = new Float32Array(0);
  let loadedDonutSx = new Float32Array(0);
  let loadedDonutSy = new Float32Array(0);
  if (hasDonuts) {
    const donutSpotFlat = packed.subarray(donutSpotOffset, donutSpotOffset + donutSpotLength);
    const donutFieldFlat = packed.subarray(donutFieldOffset, donutFieldOffset + donutFieldLength);
    loadedDonutX = new Float32Array(donutCount);
    loadedDonutY = new Float32Array(donutCount);
    loadedDonutFieldX = new Float32Array(donutCount);
    loadedDonutFieldY = new Float32Array(donutCount);
    for (let i = 0; i < donutCount; i++) {
      loadedDonutX[i] = donutSpotFlat[2 * i];
      loadedDonutY[i] = donutSpotFlat[2 * i + 1];
      loadedDonutFieldX[i] = donutFieldFlat[2 * i];
      loadedDonutFieldY[i] = donutFieldFlat[2 * i + 1];
    }
    loadedDonutSx = new Float32Array(packed.subarray(donutSxOffset, donutSxOffset + donutSxLength));
    loadedDonutSy = new Float32Array(packed.subarray(donutSyOffset, donutSyOffset + donutSyLength));
  }

  let loadedZk0 = new Float32Array(0);
  let loadedDzk = new Float32Array(0);
  if (hasZernikes) {
    loadedZk0 = new Float32Array(packed.subarray(zk0Offset, zk0Offset + zk0Length));
    loadedDzk = new Float32Array(packed.subarray(dzkOffset, dzkOffset + dzkLength));
  }

  let loadedRangeWeights = new Float32Array(0);
  let loadedFwhmWeights = new Float32Array(0);
  let loadedWfSens = new Float32Array(0);
  if (hasVmodeData) {
    loadedRangeWeights = new Float32Array(packed.subarray(rangeWeightsOffset, rangeWeightsOffset + rangeWeightsLength));
    loadedFwhmWeights = new Float32Array(packed.subarray(fwhmWeightsOffset, fwhmWeightsOffset + fwhmWeightsLength));
    loadedWfSens = new Float32Array(packed.subarray(wfSensOffset, wfSensOffset + wfSensLength));
  }

  return {
    x,
    y,
    fieldX: loadedFieldX,
    fieldY: loadedFieldY,
    sx: loadedSx,
    sy: loadedSy,
    donutX: loadedDonutX,
    donutY: loadedDonutY,
    donutFieldX: loadedDonutFieldX,
    donutFieldY: loadedDonutFieldY,
    donutSx: loadedDonutSx,
    donutSy: loadedDonutSy,
    donutN: donutCount,
    donutNfieldIntra: Number(meta.donut_nfield_intra || 0),
    donutNfieldExtra: Number(meta.donut_nfield_extra || 0),
    donutNray: Number(meta.donut_nray || 0),
    donutClockAngleDeg: Number(meta.donut_clock_angle_deg || 10),
    donutClockRadiusScale: Number(meta.donut_clock_radius_scale || 1.4),
    rangeWeights: loadedRangeWeights,
    fwhmWeights: loadedFwhmWeights,
    wfSens: loadedWfSens,
    wfSensRows: wfRows,
    zk0: loadedZk0,
    dzk: loadedDzk,
    zkCorners,
    zkTerms,
    k,
  };
}

async function fetchArrayBufferWithProgress(url, onProgress) {
  const fetchStart = nowMs();
  const response = await fetch(url);
  loadTelemetry.modelHttpStatus = response.status;
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status} ${response.statusText}`);
  }

  const totalHeader = response.headers.get('content-length');
  const totalBytes = totalHeader ? Number(totalHeader) : NaN;
  loadTelemetry.modelBytesExpected = Number.isFinite(totalBytes) && totalBytes >= 0 ? totalBytes : null;
  const responseBody = response.body;
  const reader = responseBody && typeof responseBody.getReader === 'function'
    ? responseBody.getReader()
    : null;
  loadTelemetry.modelUsedStreamReader = !!reader;

  if (!reader) {
    const fallback = await response.arrayBuffer();
    loadTelemetry.modelBytesReceived = fallback.byteLength;
    loadTelemetry.modelDownloadCompleted = true;
    loadTelemetry.modelFetchDurationMs = Math.max(0, nowMs() - fetchStart);
    if (typeof onProgress === 'function') {
      onProgress(fallback.byteLength, fallback.byteLength, true);
    }
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
      loadTelemetry.modelBytesReceived = receivedBytes;
      if (typeof onProgress === 'function') {
        onProgress(receivedBytes, totalBytes, false);
      }
    }
  }

  const merged = new Uint8Array(receivedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }

  if (typeof onProgress === 'function') {
    onProgress(receivedBytes, totalBytes, true);
  }
  loadTelemetry.modelBytesReceived = receivedBytes;
  loadTelemetry.modelDownloadCompleted = true;
  loadTelemetry.modelFetchDurationMs = Math.max(0, nowMs() - fetchStart);
  return merged.buffer;
}

function initModel(model) {
  const initialX = model.x;
  const initialY = model.y;
  const initialFieldX = model.fieldX;
  const initialFieldY = model.fieldY;
  const initialSx = model.sx;
  const initialSy = model.sy;
  const loadedK = model.k;

  scienceN = initialX.length;
  donutN = model.donutN || 0;
  N = scienceN + donutN;
  if (Number.isInteger(loadedK) && loadedK > 0) {
    K = loadedK;
  }
  p = new Float32Array(K);
  scienceX0 = initialX;
  scienceY0 = initialY;
  scienceFieldX = initialFieldX;
  scienceFieldY = initialFieldY;
  scienceSx = new Float32Array(K * scienceN);
  scienceSy = new Float32Array(K * scienceN);

  donutX0 = model.donutX || new Float32Array(0);
  donutY0 = model.donutY || new Float32Array(0);
  donutFieldX = model.donutFieldX || new Float32Array(0);
  donutFieldY = model.donutFieldY || new Float32Array(0);
  donutSx = new Float32Array(K * donutN);
  donutSy = new Float32Array(K * donutN);
  donutCenterX = new Float32Array(donutN);
  donutCenterY = new Float32Array(donutN);

  spotScale = SPOT_SCALE;
  donutSpotScale = DONUT_SPOT_SCALE;
  donutClockAngleDeg = model.donutClockAngleDeg || 10;
  donutClockRadiusScale = model.donutClockRadiusScale || 1.4;

  positions = new Float32Array(N * 2);
  data = Array.from({length: N}, (_, i) => i);

  if (!scienceFieldX || !scienceFieldY || scienceFieldX.length !== scienceN || scienceFieldY.length !== scienceN) {
    throw new Error('Missing or invalid field_xy arrays in loaded model');
  }

  if (!initialSx || !initialSy) {
    throw new Error('Missing Sx/Sy arrays in loaded model');
  }

  if (initialSx.length !== K * scienceN || initialSy.length !== K * scienceN) {
    throw new Error('Loaded sensitivity sizes do not match K*N');
  }

  if (donutN > 0) {
    if (donutX0.length !== donutN || donutY0.length !== donutN || donutFieldX.length !== donutN || donutFieldY.length !== donutN) {
      throw new Error('Loaded donut arrays do not match donut_N');
    }
    if (!model.donutSx || !model.donutSy || model.donutSx.length !== K * donutN || model.donutSy.length !== K * donutN) {
      throw new Error('Loaded donut sensitivities do not match K*donut_N');
    }
  }

  scienceSx.set(initialSx);
  scienceSy.set(initialSy);
  if (donutN > 0) {
    donutSx.set(model.donutSx);
    donutSy.set(model.donutSy);
  }

  if (model.zk0 && model.zk0.length > 0) {
    zk0 = model.zk0;
    dZk = model.dzk;
  }

  rangeWeights = null;
  fwhmWeights = null;
  wfSens = null;
  wfSensRows = 0;
  if (model.rangeWeights && model.fwhmWeights && model.wfSens) {
    if (model.rangeWeights.length === K && model.fwhmWeights.length === K && model.wfSens.length > 0) {
      rangeWeights = new Float64Array(model.rangeWeights);
      fwhmWeights = new Float64Array(model.fwhmWeights);
      wfSens = model.wfSens;
      wfSensRows = Number(model.wfSensRows) || 0;
    }
  }

  if (donutN > 0) {
    const nfieldIntra = Number(model.donutNfieldIntra || 0);
    const nray = Number(model.donutNray || 0);
    const intraPointCount = nfieldIntra * nray;
    for (let i = 0; i < donutN; i++) {
      const isIntra = i < intraPointCount;
      const center = computeClockedCenter(donutFieldX[i], donutFieldY[i], isIntra);
      donutCenterX[i] = center.x;
      donutCenterY[i] = center.y;
    }
  }

  buildZernikeOutputs();

  let maxR2 = 0;
  for (let i = 0; i < scienceN; i++) {
    const xx = scienceFieldX[i] + scienceX0[i] * spotScale;
    const yy = scienceFieldY[i] + scienceY0[i] * spotScale;
    const r2 = xx * xx + yy * yy;
    if (r2 > maxR2) maxR2 = r2;
  }
  for (let i = 0; i < donutN; i++) {
    const xx = donutCenterX[i] + donutX0[i] * donutSpotScale;
    const yy = donutCenterY[i] + donutY0[i] * donutSpotScale;
    const r2 = xx * xx + yy * yy;
    if (r2 > maxR2) maxR2 = r2;
  }
  modelRadius = Math.max(1e-6, Math.sqrt(maxR2));
}

function updatePositions() {
  for (let i = 0; i < scienceN; i++) {
    positions[2*i] = scienceFieldX[i] + scienceX0[i] * spotScale;
    positions[2*i+1] = scienceFieldY[i] + scienceY0[i] * spotScale;
  }
  for (let i = 0; i < donutN; i++) {
    const idx = scienceN + i;
    positions[2*idx] = donutCenterX[i] + donutX0[i] * donutSpotScale;
    positions[2*idx+1] = donutCenterY[i] + donutY0[i] * donutSpotScale;
  }

  for (let k = 0; k < K; k++) {
    const pk = dofValueToModelUnits(k, p[k]);
    if (pk === 0) continue;
    const scienceOff = k * scienceN;
    for (let i = 0; i < scienceN; i++) {
      positions[2*i] += scienceSx[scienceOff + i] * pk * spotScale;
      positions[2*i+1] += scienceSy[scienceOff + i] * pk * spotScale;
    }

    const donutOff = k * donutN;
    for (let i = 0; i < donutN; i++) {
      const idx = scienceN + i;
      positions[2*idx] += donutSx[donutOff + i] * pk * donutSpotScale;
      positions[2*idx+1] += donutSy[donutOff + i] * pk * donutSpotScale;
    }
  }
}

const vis = document.getElementById('vis');
const gridCanvas = document.getElementById('grid-canvas');
const canvas = document.getElementById('deck-canvas');
const resetControlsBtn = document.getElementById('reset-controls');
const includeIntrinsicsCheckbox = document.getElementById('include-intrinsics');
const mouseCoordsEl = document.getElementById('mouse-coords');
const scaleBarLineEl = document.getElementById('scale-bar-line');
const loadingOverlayEl = document.getElementById('loading-overlay');
const loadingFillEl = document.getElementById('loading-fill');
const loadingTextEl = document.getElementById('loading-text');
const copyDiagnosticsBtn = document.getElementById('copy-diagnostics');
const copyDiagnosticsStatusEl = document.getElementById('copy-diagnostics-status');
const zoomInBtn = document.getElementById('zoom-in');
const zoomOutBtn = document.getElementById('zoom-out');
const zoomResetBtn = document.getElementById('zoom-reset');
const SCALE_BAR_WORLD_UNITS = 0.1;
const GRID_PITCH_WORLD_UNITS = 0.048;
const GRID_MAJOR_EVERY = 5;
const GRID_MINOR_COLOR = 'rgba(122, 178, 255, 0.04)';
const GRID_MAJOR_COLOR = 'rgba(122, 178, 255, 0.05)';
const CORNER_OVERLAY_POSITIONS = [
  {left: '4%', top: '80%'},
  {left: '4%', top: '8%'},
  {right: '4%', top: '80%'},
  {right: '4%', top: '8%'},
];
const ZERNIKE_DISPLAY_ROWS = [
  {label: 'Z4:', terms: [4]},
  {label: 'Z5,6:', terms: [5, 6]},
  {label: 'Z7,8:', terms: [7, 8]},
  {label: 'Z9,10:', terms: [9, 10]},
  {label: 'Z11:', terms: [11]},
  {label: 'Z12,13:', terms: [12, 13]},
  {label: 'Z14,15:', terms: [14, 15]},
  {label: 'Z16,17:', terms: [16, 17]},
  {label: 'Z18,19:', terms: [18, 19]},
  {label: 'Z20,21:', terms: [20, 21]},
  {label: 'Z22:', terms: [22]},
  {label: 'Z23,24:', terms: [23, 24]},
  {label: 'Z25,26:', terms: [25, 26]},
  {label: 'Z27,28:', terms: [27, 28]},
];
let sliderInputs = [];
let parameterInputs = [];
let currentViewState = { target: [0, 0, 0], zoom: Number.NaN };
let deckgl = null;
let lastStartupError = '';
let zernikeCornerEls = [];
let includeIntrinsics = true;
let isDraggingView = false;
let dragLastClientX = 0;
let dragLastClientY = 0;

function setControlsReady(ready) {
  if (!resetControlsBtn) return;
  resetControlsBtn.disabled = !ready;
  resetControlsBtn.textContent = ready ? 'Reset (r)' : 'Loading…';
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

function setCopyDiagnosticsStatus(message) {
  if (!copyDiagnosticsStatusEl) return;
  copyDiagnosticsStatusEl.textContent = message;
}

function zernikeIndex(cornerIndex, termIndex, dofIndex) {
  return (cornerIndex * DONUT_ZERNIKE_TERMS + termIndex) * K + dofIndex;
}

function evaluateCornerZernike(cornerIndex, termIndex) {
  const base = includeIntrinsics ? zk0[cornerIndex * DONUT_ZERNIKE_TERMS + termIndex] : 0;
  let delta = 0;
  for (let k = 0; k < K; k++) {
    delta += dZk[zernikeIndex(cornerIndex, termIndex, k)] * dofValueToModelUnits(k, p[k]);
  }
  return base + delta;
}

function formatZernikeFixed5(value) {
  if (!Number.isFinite(value)) return '-----';
  const rounded = Math.round(value);
  if (rounded < -9999 || rounded > 99999) return '#####';
  return `${rounded}`.padStart(5, ' ');
}

function formatZernikeRowText(rowLabel, rowTerms, rowValuesByTerm) {
  const label = rowLabel.padEnd(8, ' ');
  const blank = ' '.repeat(5);

  if (rowTerms.length === 1) {
    const center = formatZernikeFixed5(rowValuesByTerm[rowTerms[0]]);
    return `${label} ${blank} ${center} ${blank}`;
  }

  const left = formatZernikeFixed5(rowValuesByTerm[rowTerms[0]]);
  const right = formatZernikeFixed5(rowValuesByTerm[rowTerms[1]]);
  return `${label} ${left} ${blank} ${right}`;
}

function buildZernikeOutputs() {
  const root = document.getElementById('zernike-overlay');
  if (!root) return;

  root.innerHTML = '';
  zernikeCornerEls = [];

  for (let cornerIndex = 0; cornerIndex < DONUT_CORNERS; cornerIndex++) {
    const card = document.createElement('div');
    card.className = 'zernike-corner';
    const anchor = CORNER_OVERLAY_POSITIONS[cornerIndex] || {left: '50%', top: '50%'};
    if (anchor.left != null) card.style.left = anchor.left;
    if (anchor.right != null) card.style.right = anchor.right;
    card.style.top = anchor.top;

    const rowsEl = document.createElement('div');
    rowsEl.className = 'zernike-rows';

    const cornerRowEls = [];
    for (const rowSpec of ZERNIKE_DISPLAY_ROWS) {
      const rowEl = document.createElement('div');
      rowEl.className = 'zernike-row';

      const valueEl = document.createElement('span');
      valueEl.className = 'zernike-value';
      valueEl.textContent = formatZernikeRowText(rowSpec.label, rowSpec.terms, {});
      rowEl.appendChild(valueEl);

      rowsEl.appendChild(rowEl);
      cornerRowEls.push({label: rowSpec.label, terms: rowSpec.terms, valueEl});
    }

    card.appendChild(rowsEl);
    root.appendChild(card);
    zernikeCornerEls.push(cornerRowEls);
  }
}

function updateZernikeOutputs() {
  if (!zernikeCornerEls.length) return;

  for (let cornerIndex = 0; cornerIndex < DONUT_CORNERS; cornerIndex++) {
    const cornerRows = zernikeCornerEls[cornerIndex];
    if (!cornerRows) continue;

    for (const row of cornerRows) {
      const rowValuesByTerm = {};
      for (const term of row.terms) {
        rowValuesByTerm[term] = evaluateCornerZernike(cornerIndex, term);
      }
      row.valueEl.textContent = formatZernikeRowText(row.label, row.terms, rowValuesByTerm);
    }
  }
}

function initializeDeck() {
  if (!window.deck) {
    throw new Error('Visualization library failed to load (deck.gl script unavailable).');
  }

  DeckGL = window.deck.DeckGL;
  OrthographicView = window.deck.OrthographicView;
  ScatterplotLayer = window.deck.ScatterplotLayer;
  COORDINATE_SYSTEM = window.deck.COORDINATE_SYSTEM;

  if (!DeckGL || !OrthographicView || !ScatterplotLayer || !COORDINATE_SYSTEM) {
    throw new Error('Visualization library loaded incompletely.');
  }

  deckgl = new DeckGL({
    canvas,
    views: [new OrthographicView({id: 'ortho'})],
    initialViewState: { target: [0, 0, 0], zoom: 0 },
    controller: false,
    layers: []
  });
}

function getCompatibilityIssues() {
  const issues = [];

  if (typeof window.fetch !== 'function') issues.push('missing Fetch API');
  if (typeof window.Promise !== 'function') issues.push('missing Promise support');
  if (typeof window.Float32Array !== 'function') issues.push('missing typed array support');
  if (typeof window.requestAnimationFrame !== 'function') issues.push('missing animation frame support');
  if (!canvas || typeof canvas.getContext !== 'function') {
    issues.push('missing canvas support');
    return issues;
  }

  const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
  if (!gl) issues.push('WebGL unavailable');

  return issues;
}

function buildDiagnosticsText() {
  const issues = getCompatibilityIssues();
  const navigatorInfo = typeof navigator !== 'undefined' ? navigator : null;
  const locationInfo = typeof window !== 'undefined' ? window.location : null;
  const webgl2Supported = canvas && canvas.getContext ? !!canvas.getContext('webgl2') : false;
  const webglSupported = canvas && canvas.getContext
    ? (!!canvas.getContext('webgl') || !!canvas.getContext('experimental-webgl'))
    : false;

  const rows = [
    `timestamp_utc: ${new Date().toISOString()}`,
    `user_agent: ${navigatorInfo && navigatorInfo.userAgent ? navigatorInfo.userAgent : 'unknown'}`,
    `language: ${navigatorInfo && navigatorInfo.language ? navigatorInfo.language : 'unknown'}`,
    `platform: ${navigatorInfo && navigatorInfo.platform ? navigatorInfo.platform : 'unknown'}`,
    `url: ${locationInfo && locationInfo.href ? locationInfo.href : 'unknown'}`,
    `protocol: ${locationInfo && locationInfo.protocol ? locationInfo.protocol : 'unknown'}`,
    `deck_global_present: ${!!window.deck}`,
    `fetch_supported: ${typeof window.fetch === 'function'}`,
    `promise_supported: ${typeof window.Promise === 'function'}`,
    `typedarray_supported: ${typeof window.Float32Array === 'function'}`,
    `raf_supported: ${typeof window.requestAnimationFrame === 'function'}`,
    `webgl2_supported: ${webgl2Supported}`,
    `webgl_supported: ${webglSupported}`,
    `compatibility_issues: ${issues.length ? issues.join('; ') : 'none'}`,
    `startup_started_utc: ${loadTelemetry.startupStartIso || 'unknown'}`,
    `startup_finished_utc: ${loadTelemetry.startupEndIso || 'unknown'}`,
    `startup_duration_ms: ${formatDuration(loadTelemetry.startupDurationMs)}`,
    `meta_http_status: ${loadTelemetry.metaHttpStatus != null ? loadTelemetry.metaHttpStatus : 'unknown'}`,
    `meta_fetch_duration_ms: ${formatDuration(loadTelemetry.metaFetchDurationMs)}`,
    `meta_content_length_bytes: ${loadTelemetry.metaBytes != null ? loadTelemetry.metaBytes : 'unknown'}`,
    `model_http_status: ${loadTelemetry.modelHttpStatus != null ? loadTelemetry.modelHttpStatus : 'unknown'}`,
    `model_fetch_duration_ms: ${formatDuration(loadTelemetry.modelFetchDurationMs)}`,
    `model_content_length_bytes: ${loadTelemetry.modelBytesExpected != null ? loadTelemetry.modelBytesExpected : 'unknown'}`,
    `model_bytes_received: ${loadTelemetry.modelBytesReceived != null ? loadTelemetry.modelBytesReceived : 'unknown'}`,
    `model_streaming_reader_used: ${loadTelemetry.modelUsedStreamReader != null ? loadTelemetry.modelUsedStreamReader : 'unknown'}`,
    `model_download_completed: ${loadTelemetry.modelDownloadCompleted}`,
    `last_startup_error: ${lastStartupError || 'none'}`,
  ];

  return rows.join('\n');
}

async function copyDiagnosticsToClipboard() {
  const text = buildDiagnosticsText();

  if (navigator && navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
    await navigator.clipboard.writeText(text);
    return true;
  }

  return false;
}

function buildLoadErrorMessage(error) {
  const raw = error && error.message ? error.message : String(error);

  if (/missing|unavailable|unsupported|WebGL/i.test(raw)) {
    return `Browser compatibility error: ${raw}. Please update your browser and ensure hardware acceleration is enabled.`;
  }

  if (/deck\.gl|Visualization library/i.test(raw)) {
    return `${raw} This can happen if the CDN script is blocked by a network policy, ad blocker, or privacy extension.`;
  }

  if (/Failed to fetch|NetworkError|Load failed/i.test(raw)) {
    return `Network error while downloading model files: ${raw}. Check connection quality, proxy/firewall settings, and that the page is served over HTTP(S).`;
  }

  if (/Metadata|Packed model|N\/K/i.test(raw)) {
    return `Model data error: ${raw}. The model files may be incomplete or from mismatched versions.`;
  }

  return `Load failed: ${raw}`;
}

function getFitZoom(width, height, radius) {
  const diameter = Math.max(1e-6, 2 * radius);
  const usablePixels = Math.max(1, Math.min(width, height) * VIEW_PADDING);
  const scale = usablePixels / diameter;
  return Math.log2(scale);
}

function getFitViewState() {
  const r = vis.getBoundingClientRect();
  const zoom = getFitZoom(r.width, r.height, modelRadius);
  return {target: [0, 0, 0], zoom};
}

function applyViewState() {
  const r = vis.getBoundingClientRect();
  drawGrid(r.width, r.height);
  updateScaleBar();
  if (deckgl) {
    deckgl.setProps({ viewState: currentViewState });
  }
}

function setZoom(nextZoom) {
  const clampedZoom = Math.max(-4, Math.min(18, nextZoom));
  currentViewState = {
    target: currentViewState.target,
    zoom: clampedZoom
  };
  applyViewState();
}

function zoomBy(delta) {
  setZoom(currentViewState.zoom + delta);
}

function resetView() {
  currentViewState = getFitViewState();
  applyViewState();
}

function panByPixels(deltaX, deltaY) {
  const scale = Math.pow(2, currentViewState.zoom);
  const targetX = currentViewState.target[0] - deltaX / scale;
  const targetY = currentViewState.target[1] - deltaY / scale;
  currentViewState = {
    target: [targetX, targetY, 0],
    zoom: currentViewState.zoom
  };
  applyViewState();
}

function resizeDeck() {
  const r = vis.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (gridCanvas) {
    gridCanvas.width = Math.max(1, Math.floor(r.width * dpr));
    gridCanvas.height = Math.max(1, Math.floor(r.height * dpr));
  }
  canvas.width  = Math.max(1, Math.floor(r.width  * dpr));
  canvas.height = Math.max(1, Math.floor(r.height * dpr));
  if (!Number.isFinite(currentViewState.zoom)) {
    currentViewState = getFitViewState();
  }
  drawGrid(r.width, r.height);
  updateScaleBar();
  if (deckgl) {
    deckgl.setProps({
      width: r.width,
      height: r.height,
      viewState: currentViewState
    });
  }
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
  if (!deckgl || !ScatterplotLayer || !COORDINATE_SYSTEM) return;

  const layer = new ScatterplotLayer({
    id: 'pts',
    data,
    coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
    getPosition: (i) => [positions[2*i], positions[2*i + 1], 0],
    updateTriggers: {
      getPosition: posVersion
    },
    getRadius: 1.0,
    radiusUnits: 'pixels',
    stroked: false,
    getFillColor: [120, 180, 255, 60]
  });

  deckgl.setProps({ layers: [layer] });
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
    updateZernikeOutputs();
  });
}

function buildSliders() {
  const root = document.getElementById('sliders');
  root.innerHTML = '';
  sliderInputs = [];
  parameterInputs = [];

  function applyControlValue(controlIndex, rawValue) {
    const parsed = Number(rawValue);
    if (!Number.isFinite(parsed)) return;
    const clamped = clampControlValue(controlIndex, parsed);
    p[controlIndex] = clamped;

    if (sliderInputs[controlIndex]) {
      sliderInputs[controlIndex].value = String(clamped);
    }
    if (parameterInputs[controlIndex]) {
      parameterInputs[controlIndex].value = formatControlValue(clamped);
    }

    requestUpdate();
  }

  function commitNumericInput(controlIndex) {
    const input = parameterInputs[controlIndex];
    if (!input) return;

    const raw = input.value.trim();
    if (raw === '') {
      input.value = formatControlValue(p[controlIndex]);
      return;
    }

    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) {
      input.value = formatControlValue(p[controlIndex]);
      return;
    }

    applyControlValue(controlIndex, parsed);
  }

  for (let k = 0; k < K; k++) {
    const spec = getControlSpec(k);
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

    const numericInput = document.createElement('input');
    numericInput.type = 'number';
    numericInput.min = String(-spec.range);
    numericInput.max = String(spec.range);
    numericInput.step = String(spec.step);
    numericInput.value = formatControlValue(0);

    input.addEventListener('input', () => {
      applyControlValue(k, input.value);
    });

    numericInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        commitNumericInput(k);
        return;
      }

      if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
        requestAnimationFrame(() => {
          commitNumericInput(k);
        });
      }
    });

    numericInput.addEventListener('change', () => {
      commitNumericInput(k);
    });

    numericInput.addEventListener('blur', () => {
      commitNumericInput(k);
    });

    row.appendChild(label);
    row.appendChild(input);
    row.appendChild(numericInput);
    root.appendChild(row);
    sliderInputs.push(input);
    parameterInputs.push(numericInput);
  }
}

function resetControls() {
  const rangeExpInput = document.getElementById('range-exp-input');
  const fwhmExpInput = document.getElementById('fwhm-exp-input');
  const useDofInput = document.getElementById('use-dof-input');
  const nkeepInput = document.getElementById('nkeep-input');

  for (let k = 0; k < K; k++) {
    p[k] = 0;
    if (sliderInputs[k]) sliderInputs[k].value = '0';
    if (parameterInputs[k]) parameterInputs[k].value = formatControlValue(0);
  }
  for (let m = 0; m < vValues.length; m++) {
    vValues[m] = 0;
    if (vmodeSliderInputs[m]) vmodeSliderInputs[m].value = '0';
    if (vmodeParamInputs[m]) vmodeParamInputs[m].value = formatControlValue(0);
  }

  rangeNormExp = DEFAULT_RANGE_NORM_EXP;
  fwhmNormExp = DEFAULT_FWHM_NORM_EXP;
  currentNkeep = DEFAULT_NKEEP;
  if (rangeExpInput) rangeExpInput.value = String(DEFAULT_RANGE_NORM_EXP);
  if (fwhmExpInput) fwhmExpInput.value = String(DEFAULT_FWHM_NORM_EXP);
  if (useDofInput) useDofInput.value = DEFAULT_USE_DOF;
  if (nkeepInput) nkeepInput.value = String(DEFAULT_NKEEP);

  recomputeVmodes();
}

function buildNormVector() {
  if (!rangeWeights || !fwhmWeights || rangeWeights.length !== K || fwhmWeights.length !== K) {
    return null;
  }
  const out = new Float64Array(K);
  for (let i = 0; i < K; i++) {
    out[i] = Math.pow(rangeWeights[i], rangeNormExp) * Math.pow(fwhmWeights[i], fwhmNormExp);
  }
  return out;
}

function recomputeVmodes() {
  const rangeExpInput = document.getElementById('range-exp-input');
  const fwhmExpInput = document.getElementById('fwhm-exp-input');
  const useDofInput = document.getElementById('use-dof-input');
  const nkeepInput = document.getElementById('nkeep-input');
  if (!rangeExpInput || !fwhmExpInput || !useDofInput || !nkeepInput) return;

  const parsedRangeExp = Number(rangeExpInput.value);
  const parsedFwhmExp = Number(fwhmExpInput.value);
  rangeNormExp = Number.isFinite(parsedRangeExp) ? parsedRangeExp : DEFAULT_RANGE_NORM_EXP;
  fwhmNormExp = Number.isFinite(parsedFwhmExp) ? parsedFwhmExp : DEFAULT_FWHM_NORM_EXP;
  rangeExpInput.value = String(rangeNormExp);
  fwhmExpInput.value = String(fwhmNormExp);

  currentUseDof = parseUseDof(useDofInput.value).filter((dof) => dof >= 0 && dof < K);
  useDofInput.value = formatUseDof(currentUseDof);
  nActive = currentUseDof.length;
  if (nActive === 0) {
    currentNkeep = 0;
    nkeepInput.value = '0';
  } else {
    currentNkeep = Math.max(1, Math.min(nActive, Number(nkeepInput.value) || DEFAULT_NKEEP));
    nkeepInput.value = String(currentNkeep);
  }

  norm = buildNormVector();

  if (norm && wfSens && wfSensRows > 0 && nActive > 0) {
    const result = computeVh(wfSens, wfSensRows, K, norm, currentUseDof, currentNkeep);
    Vh = result.vh;
    fullMixMatrix = result.fullMatrix;
  } else {
    Vh = null;
    fullMixMatrix = null;
  }

  // Reset vmode values and rebuild sliders
  vValues = new Array(currentNkeep).fill(0);
  buildVmodeSliders();
  drawMixingMatrix();
}

function buildVmodeSliders() {
  const root = document.getElementById('vmode-sliders');
  if (!root) return;
  root.innerHTML = '';
  vmodeSliderInputs = [];
  vmodeParamInputs = [];

  if (!Vh || currentNkeep <= 0) return;

  // Compute feasible per-mode delta bounds so applying this mode alone
  // cannot drive any active DOF outside its configured range.
  const vmodeMins = new Float64Array(currentNkeep);
  const vmodeMaxs = new Float64Array(currentNkeep);
  for (let m = 0; m < currentNkeep; m++) {
    const rowOff = m * nActive;
    let lower = -Infinity;
    let upper = Infinity;
    for (let j = 0; j < nActive; j++) {
      const dofIdx = currentUseDof[j];
      const coeff = Vh[rowOff + j];
      const dofRange = CONTROL_RANGES[dofIdx] != null ? CONTROL_RANGES[dofIdx] : Infinity;
      const p0 = p[dofIdx];
      if (Math.abs(coeff) <= 1e-15 || !isFinite(dofRange)) {
        continue;
      }

      // p0 + delta*coeff must stay in [-dofRange, +dofRange]
      const a = (-dofRange - p0) / coeff;
      const b = (dofRange - p0) / coeff;
      const thisLow = Math.min(a, b);
      const thisHigh = Math.max(a, b);
      if (thisLow > lower) lower = thisLow;
      if (thisHigh < upper) upper = thisHigh;
    }

    if (!isFinite(lower) || !isFinite(upper) || upper <= lower) {
      lower = -1;
      upper = 1;
    }
    vmodeMins[m] = lower;
    vmodeMaxs[m] = upper;
  }

  for (let m = 0; m < currentNkeep; m++) {
    const row = document.createElement('div');
    row.className = 'row';

    const label = document.createElement('label');
    label.textContent = `V${m + 1}`;

    const modeMin = vmodeMins[m];
    const modeMax = vmodeMaxs[m];
    const VMODE_STEP = Math.max((modeMax - modeMin) / 400, 1e-6);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(modeMin);
    slider.max = String(modeMax);
    slider.step = String(VMODE_STEP);
    slider.value = '0';

    const numInput = document.createElement('input');
    numInput.type = 'number';
    numInput.min = String(modeMin);
    numInput.max = String(modeMax);
    numInput.step = String(VMODE_STEP);
    numInput.value = formatControlValue(0);

    const modeIndex = m; // capture for closure

    function applyVmodeDelta(newVal) {
      const clamped = Math.max(modeMin, Math.min(modeMax, newVal));
      const delta = clamped - vValues[modeIndex];
      vValues[modeIndex] = clamped;

      // Propagate delta into DOF sliders: p[useDof[j]] += delta * Vh[mode][j]
      if (Vh && Math.abs(delta) > 1e-15) {
        const rowOff = modeIndex * nActive;
        for (let j = 0; j < nActive; j++) {
          const dofIdx = currentUseDof[j];
          p[dofIdx] += delta * Vh[rowOff + j];
          // Update DOF slider display
          const clamped2 = clampControlValue(dofIdx, p[dofIdx]);
          p[dofIdx] = clamped2;
          if (sliderInputs[dofIdx]) sliderInputs[dofIdx].value = String(clamped2);
          if (parameterInputs[dofIdx]) parameterInputs[dofIdx].value = formatControlValue(clamped2);
        }
      }

      // Sync vmode slider/input displays
      if (vmodeSliderInputs[modeIndex]) vmodeSliderInputs[modeIndex].value = String(clamped);
      if (vmodeParamInputs[modeIndex]) vmodeParamInputs[modeIndex].value = formatControlValue(clamped);

      requestUpdate();
    }

    slider.addEventListener('input', () => {
      applyVmodeDelta(Number(slider.value));
    });

    function commitVmodeInput() {
      const raw = numInput.value.trim();
      if (raw === '') {
        numInput.value = formatControlValue(vValues[modeIndex]);
        return;
      }
      const parsed = Number(raw);
      if (!Number.isFinite(parsed)) {
        numInput.value = formatControlValue(vValues[modeIndex]);
        return;
      }
      applyVmodeDelta(parsed);
    }

    numInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') { event.preventDefault(); commitVmodeInput(); return; }
      if (event.key === 'Escape') { event.preventDefault(); numInput.blur(); return; }
      if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
        requestAnimationFrame(() => commitVmodeInput());
      }
    });
    numInput.addEventListener('change', commitVmodeInput);
    numInput.addEventListener('blur', commitVmodeInput);

    row.appendChild(label);
    row.appendChild(slider);
    row.appendChild(numInput);
    root.appendChild(row);
    vmodeSliderInputs.push(slider);
    vmodeParamInputs.push(numInput);
  }
}

function bwrColor(t) {
  t = Math.max(-1, Math.min(1, t));
  let r, g, b;
  if (t < 0) {
    const s = 1 + t;
    r = Math.round(255 * s);
    g = Math.round(255 * s);
    b = 255;
  } else {
    const s = 1 - t;
    r = 255;
    g = Math.round(255 * s);
    b = Math.round(255 * s);
  }
  return `rgb(${r},${g},${b})`;
}

function computeGroupBoundaries(useDof) {
  const m2Hex = [], camHex = [], m1m3Bend = [], m2Bend = [];
  for (let i = 0; i < useDof.length; i++) {
    const d = useDof[i];
    if (d < 5) m2Hex.push(i);
    else if (d < 10) camHex.push(i);
    else if (d < 30) m1m3Bend.push(i);
    else m2Bend.push(i);
  }
  const boundaries = [];
  if (m2Hex.length > 0 && (camHex.length + m1m3Bend.length + m2Bend.length) > 0)
    boundaries.push(m2Hex[m2Hex.length - 1]);
  if (camHex.length > 0 && (m1m3Bend.length + m2Bend.length) > 0)
    boundaries.push(camHex[camHex.length - 1]);
  if (m1m3Bend.length > 0 && m2Bend.length > 0)
    boundaries.push(m1m3Bend[m1m3Bend.length - 1]);
  return boundaries;
}

function drawMixingMatrix() {
  const canvas = document.getElementById('mixing-matrix-canvas');
  if (!canvas) return;

  if (!fullMixMatrix || nActive <= 0) {
    canvas.width = 0;
    canvas.height = 0;
    canvas.style.width = '0';
    canvas.style.height = '0';
    return;
  }

  const CELL = 14;
  const LABEL_W = 75;
  const BOTTOM_H = 19;
  const TOP_PAD = 2;
  const nModes = nActive;
  const nDofs = nActive;

  const canvasW = LABEL_W + nModes * CELL;
  const canvasH = TOP_PAD + nDofs * CELL + BOTTOM_H;

  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.ceil(canvasW * dpr);
  canvas.height = Math.ceil(canvasH * dpr);
  canvas.style.width = canvasW + 'px';
  canvas.style.height = canvasH + 'px';

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvasW, canvasH);

  // Normalize: mix @ diag(1/norm[use_dof])
  const normMix = new Float64Array(fullMixMatrix.length);
  for (let mode = 0; mode < nModes; mode++) {
    for (let j = 0; j < nDofs; j++) {
      const dofIdx = currentUseDof[j];
      const n = (norm && norm[dofIdx]) ? norm[dofIdx] : 1;
      normMix[mode * nActive + j] = fullMixMatrix[mode * nActive + j] / n;
    }
  }

  let vmax = 0;
  for (let i = 0; i < normMix.length; i++) {
    vmax = Math.max(vmax, Math.abs(normMix[i]));
  }
  if (vmax < 1e-15) vmax = 1;

  // Draw cells (origin="lower": DOF index 0 at bottom)
  for (let mode = 0; mode < nModes; mode++) {
    for (let j = 0; j < nDofs; j++) {
      const val = normMix[mode * nActive + j];
      ctx.fillStyle = bwrColor(val / vmax);
      ctx.fillRect(LABEL_W + mode * CELL, TOP_PAD + (nDofs - 1 - j) * CELL, CELL, CELL);
    }
  }

  // Red overlay for modes beyond nkeep
  if (currentNkeep < nModes) {
    ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
    ctx.fillRect(
      LABEL_W + currentNkeep * CELL, TOP_PAD,
      (nModes - currentNkeep) * CELL, nDofs * CELL
    );
  }

  // Group separator lines
  const bounds = computeGroupBoundaries(currentUseDof);
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
  ctx.lineWidth = 1;
  for (const bIdx of bounds) {
    const y = TOP_PAD + (nDofs - 1 - bIdx) * CELL;
    ctx.beginPath();
    ctx.moveTo(LABEL_W, y);
    ctx.lineTo(LABEL_W + nModes * CELL, y);
    ctx.stroke();
  }

  // Y-axis labels (DOF names)
  ctx.fillStyle = '#333';
  ctx.font = '8px ui-monospace, Menlo, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let j = 0; j < nDofs; j++) {
    const dofIdx = currentUseDof[j];
    const full = CONTROL_NAMES[dofIdx] || ('DOF ' + dofIdx);
    const name = full.replace(/\s*\[.*\]$/, '');
    ctx.fillText(name, LABEL_W - 3, TOP_PAD + (nDofs - 1 - j) * CELL + CELL / 2);
  }

  // X-axis labels (mode numbers)
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const xStep = nModes > 30 ? 5 : (nModes > 15 ? 2 : 1);
  for (let m = 0; m < nModes; m++) {
    if (m % xStep !== 0 && m !== nModes - 1) continue;
    ctx.fillText(String(m + 1), LABEL_W + m * CELL + CELL / 2, TOP_PAD + nDofs * CELL + 2);
  }
}

async function start() {
  const startupStart = nowMs();
  loadTelemetry.startupStartIso = new Date().toISOString();
  loadTelemetry.startupEndIso = null;
  loadTelemetry.startupDurationMs = null;
  loadTelemetry.modelDownloadCompleted = false;
  setControlsReady(false);
  updateLoadingProgress(0, NaN, false);

  const compatibilityIssues = getCompatibilityIssues();
  if (compatibilityIssues.length) {
    throw new Error(`unsupported browser features: ${compatibilityIssues.join(', ')}`);
  }

  initializeDeck();
  const model = await loadPackedModel('./model_meta.json', './model.f32');

  initModel(model);

  buildSliders();
  recomputeVmodes();
  resizeDeck();
  updatePositions();
  posVersion++;
  render();
  updateZernikeOutputs();
  setControlsReady(true);
  loadTelemetry.startupEndIso = new Date().toISOString();
  loadTelemetry.startupDurationMs = Math.max(0, nowMs() - startupStart);
}

start().catch((error) => {
  loadTelemetry.startupEndIso = new Date().toISOString();
  if (loadTelemetry.startupStartIso && !Number.isFinite(loadTelemetry.startupDurationMs)) {
    const fallbackStart = Date.parse(loadTelemetry.startupStartIso);
    if (Number.isFinite(fallbackStart)) {
      loadTelemetry.startupDurationMs = Math.max(0, Date.now() - fallbackStart);
    }
  }
  lastStartupError = error && error.message ? error.message : String(error);
  console.error(error);
  setControlsReady(false);
  showLoadingError(buildLoadErrorMessage(error));
});

if (copyDiagnosticsBtn) {
  copyDiagnosticsBtn.addEventListener('click', async () => {
    setCopyDiagnosticsStatus('');
    copyDiagnosticsBtn.disabled = true;

    try {
      const copied = await copyDiagnosticsToClipboard();
      if (copied) {
        setCopyDiagnosticsStatus('Copied');
      } else {
        const text = buildDiagnosticsText();
        window.prompt('Copy diagnostics:', text);
        setCopyDiagnosticsStatus('Opened copy dialog');
      }
    } catch (error) {
      const text = buildDiagnosticsText();
      window.prompt('Copy diagnostics:', text);
      setCopyDiagnosticsStatus('Clipboard blocked; opened dialog');
    } finally {
      copyDiagnosticsBtn.disabled = false;
    }
  });
}

window.addEventListener('resize', () => { resizeDeck(); render(); });

if (resetControlsBtn) {
  resetControlsBtn.addEventListener('click', resetControls);
}

if (includeIntrinsicsCheckbox) {
  includeIntrinsicsCheckbox.checked = includeIntrinsics;
  includeIntrinsicsCheckbox.addEventListener('change', () => {
    includeIntrinsics = includeIntrinsicsCheckbox.checked;
    updateZernikeOutputs();
  });
}

// Apply vmodes button
const applyVmodesBtn = document.getElementById('apply-vmodes');
if (applyVmodesBtn) {
  applyVmodesBtn.addEventListener('click', recomputeVmodes);
}
const rangeExpInputEl = document.getElementById('range-exp-input');
if (rangeExpInputEl) {
  rangeExpInputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); recomputeVmodes(); }
    if (e.key === 'Escape') { e.preventDefault(); rangeExpInputEl.blur(); }
  });
  rangeExpInputEl.addEventListener('change', recomputeVmodes);
}
const fwhmExpInputEl = document.getElementById('fwhm-exp-input');
if (fwhmExpInputEl) {
  fwhmExpInputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); recomputeVmodes(); }
    if (e.key === 'Escape') { e.preventDefault(); fwhmExpInputEl.blur(); }
  });
  fwhmExpInputEl.addEventListener('change', recomputeVmodes);
}
// Also allow Enter in the use_dof and nkeep inputs
const useDofInputEl = document.getElementById('use-dof-input');
const nkeepInputEl = document.getElementById('nkeep-input');
if (useDofInputEl) {
  useDofInputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); recomputeVmodes(); }
    if (e.key === 'Escape') { e.preventDefault(); useDofInputEl.blur(); }
  });
}
if (nkeepInputEl) {
  nkeepInputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); recomputeVmodes(); }
    if (e.key === 'Escape') { e.preventDefault(); nkeepInputEl.blur(); }
  });
}

vis.addEventListener('mousemove', updateMouseCoords);
vis.addEventListener('mouseleave', clearMouseCoords);
vis.addEventListener('wheel', (event) => {
  event.preventDefault();
  const rawStep = -event.deltaY * WHEEL_ZOOM_SENSITIVITY;
  const zoomStep = Math.max(-WHEEL_ZOOM_MAX_STEP, Math.min(WHEEL_ZOOM_MAX_STEP, rawStep));
  zoomBy(zoomStep);
}, {passive: false});

vis.addEventListener('mousedown', (event) => {
  if (event.button !== 0) return;
  isDraggingView = true;
  dragLastClientX = event.clientX;
  dragLastClientY = event.clientY;
  vis.style.cursor = 'grabbing';
});

vis.addEventListener('mousemove', (event) => {
  if (!isDraggingView) return;
  const dx = event.clientX - dragLastClientX;
  const dy = event.clientY - dragLastClientY;
  dragLastClientX = event.clientX;
  dragLastClientY = event.clientY;
  panByPixels(dx, dy);
});

window.addEventListener('mouseup', () => {
  if (!isDraggingView) return;
  isDraggingView = false;
  vis.style.cursor = '';
});

vis.addEventListener('mouseleave', () => {
  if (!isDraggingView) return;
  isDraggingView = false;
  vis.style.cursor = '';
});

if (zoomInBtn) {
  zoomInBtn.addEventListener('click', () => zoomBy(0.3));
}

if (zoomOutBtn) {
  zoomOutBtn.addEventListener('click', () => zoomBy(-0.3));
}

if (zoomResetBtn) {
  zoomResetBtn.addEventListener('click', resetView);
}

window.addEventListener('keydown', (event) => {
  if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
  const tag = event.target && event.target.tagName;
  if (tag === 'TEXTAREA' || tag === 'SELECT' || tag === 'INPUT') return;
  if (event.key === 'r' || event.key === 'R') {
    event.preventDefault();
    resetControls();
    return;
  }
  if (event.key === '+' || event.key === '=') {
    event.preventDefault();
    zoomBy(0.3);
    return;
  }
  if (event.key === '-') {
    event.preventDefault();
    zoomBy(-0.3);
    return;
  }
  if (event.key === '0') {
    event.preventDefault();
    resetView();
  }
});
