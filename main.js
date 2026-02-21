const {DeckGL, OrthographicView, ScatterplotLayer, COORDINATE_SYSTEM} = window.deck;

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
  100.0, 1500.0, 1500.0, 60.0, 60.0,
  100.0, 3000.0, 3000.0, 60.0, 60.0,
  1.5, 1.5, 1.5, 1.0, 1.0,
  1.0, 1.0, 1.0, 0.75, 0.75,
  0.5, 0.5, 0.5, 0.5, 0.5,
  0.25, 0.25, 0.25, 0.25, 0.25,

  1.5, 1.5, 1.5, 1.0, 1.0,
  1.0, 1.0, 1.0, 0.75, 0.75,
  0.5, 0.5, 0.5, 0.5, 0.5,
  0.25, 0.25, 0.25, 0.25, 0.25,
];

let K = CONTROL_NAMES.length;

let p = new Float32Array(K);
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
  if (!Number.isInteger(n) || n <= 0 || !Number.isInteger(k) || k <= 0) {
    throw new Error('Metadata must contain positive integer N and K');
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

  return {x, y, sx: loadedSx, sy: loadedSy, k};
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
    positions[2*i]   = x0[i];
    positions[2*i+1] = y0[i];
  }
  for (let k = 0; k < K; k++) {
    const pk = p[k];
    if (pk === 0) continue;
    const off = k * N;
    for (let i = 0; i < N; i++) {
      positions[2*i]   += Sx[off + i] * pk;
      positions[2*i+1] += Sy[off + i] * pk;
    }
  }
}

const vis = document.getElementById('vis');
const canvas = document.getElementById('deck-canvas');
const resetControlsBtn = document.getElementById('reset-controls');
const mouseCoordsEl = document.getElementById('mouse-coords');
const scaleBarLineEl = document.getElementById('scale-bar-line');
const loadingOverlayEl = document.getElementById('loading-overlay');
const loadingFillEl = document.getElementById('loading-fill');
const loadingTextEl = document.getElementById('loading-text');
const SCALE_BAR_WORLD_UNITS = 0.24;
const GRID_PITCH_WORLD_UNITS = 0.048;
const GRID_MAJOR_EVERY = 5;
let sliderInputs = [];
let sliderValues = [];
let currentViewState = { target: [0, 0, 0], zoom: 0 };

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
  canvas.width  = Math.max(1, Math.floor(r.width  * dpr));
  canvas.height = Math.max(1, Math.floor(r.height * dpr));
  const zoom = getFitZoom(r.width, r.height, modelRadius);
  currentViewState = { target: [0, 0, 0], zoom };
  updateGridBackground(r.width, r.height);
  updateScaleBar();
  deckgl.setProps({
    width: r.width,
    height: r.height,
    viewState: currentViewState
  });
}

function updateGridBackground(width, height) {
  if (!vis) return;
  const scale = Math.pow(2, currentViewState.zoom);
  const pitchPx = Math.max(2, GRID_PITCH_WORLD_UNITS * scale);
  const majorPitchPx = Math.max(2, pitchPx * GRID_MAJOR_EVERY);
  const originScreenX = width / 2 + (0 - currentViewState.target[0]) * scale;
  const originScreenY = height / 2 - (0 - currentViewState.target[1]) * scale;
  const offsetX = ((originScreenX % pitchPx) + pitchPx) % pitchPx;
  const offsetY = ((originScreenY % pitchPx) + pitchPx) % pitchPx;
  const majorOffsetX = ((originScreenX % majorPitchPx) + majorPitchPx) % majorPitchPx;
  const majorOffsetY = ((originScreenY % majorPitchPx) + majorPitchPx) % majorPitchPx;

  vis.style.setProperty('--grid-pitch-px', `${pitchPx.toFixed(2)}px`);
  vis.style.setProperty('--grid-major-pitch-px', `${majorPitchPx.toFixed(2)}px`);
  vis.style.setProperty('--grid-offset-x', `${offsetX.toFixed(2)}px`);
  vis.style.setProperty('--grid-offset-y', `${offsetY.toFixed(2)}px`);
  vis.style.setProperty('--grid-major-offset-x', `${majorOffsetX.toFixed(2)}px`);
  vis.style.setProperty('--grid-major-offset-y', `${majorOffsetY.toFixed(2)}px`);
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
  });
}

function buildSliders() {
  const root = document.getElementById('sliders');
  root.innerHTML = '';
  sliderInputs = [];
  sliderValues = [];
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

    const val = document.createElement('span');
    val.textContent = formatControlValue(0);

    input.addEventListener('input', () => {
      const v = Number(input.value);
      p[k] = v;
      val.textContent = formatControlValue(v);
      requestUpdate();
    });

    row.appendChild(label);
    row.appendChild(input);
    row.appendChild(val);
    root.appendChild(row);
    sliderInputs.push(input);
    sliderValues.push(val);
  }
}

function resetControls() {
  for (let k = 0; k < K; k++) {
    p[k] = 0;
    if (sliderInputs[k]) sliderInputs[k].value = '0';
    if (sliderValues[k]) sliderValues[k].textContent = formatControlValue(0);
  }
  requestUpdate();
}

async function start() {
  setControlsReady(false);
  updateLoadingProgress(0, NaN, false);
  const model = await loadPackedModel('./model_meta.json', './model.f32');

  initModel(model.x, model.y, model.sx, model.sy, model.k);
  buildSliders();
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

window.addEventListener('resize', () => { resizeDeck(); render(); });

if (resetControlsBtn) {
  resetControlsBtn.addEventListener('click', resetControls);
}

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
