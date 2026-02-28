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

let p = new Float32Array(K);
let x0, y0, Sx, Sy;
let modelRadius = WORLD_RADIUS;
let positions = new Float32Array(0);
let data = [];
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
  const initSpec = layout ? layout.init_xy : undefined;
  const sxSpec = layout ? layout.Sx : undefined;
  const sySpec = layout ? layout.Sy : undefined;
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
    k
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
const gridCanvas = document.getElementById('grid-canvas');
const canvas = document.getElementById('deck-canvas');
const resetControlsBtn = document.getElementById('reset-controls');
const mouseCoordsEl = document.getElementById('mouse-coords');
const scaleBarLineEl = document.getElementById('scale-bar-line');
const loadingOverlayEl = document.getElementById('loading-overlay');
const loadingFillEl = document.getElementById('loading-fill');
const loadingTextEl = document.getElementById('loading-text');
const copyDiagnosticsBtn = document.getElementById('copy-diagnostics');
const copyDiagnosticsStatusEl = document.getElementById('copy-diagnostics-status');
const SCALE_BAR_WORLD_UNITS = 0.24;
const GRID_PITCH_WORLD_UNITS = 0.048;
const GRID_MAJOR_EVERY = 5;
const GRID_MINOR_COLOR = 'rgba(122, 178, 255, 0.04)';
const GRID_MAJOR_COLOR = 'rgba(122, 178, 255, 0.05)';
let sliderInputs = [];
let sliderValues = [];
let currentViewState = { target: [0, 0, 0], zoom: 0 };
let deckgl = null;
let lastStartupError = '';

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

function resizeDeck() {
  const r = vis.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (gridCanvas) {
    gridCanvas.width = Math.max(1, Math.floor(r.width * dpr));
    gridCanvas.height = Math.max(1, Math.floor(r.height * dpr));
  }
  canvas.width  = Math.max(1, Math.floor(r.width  * dpr));
  canvas.height = Math.max(1, Math.floor(r.height * dpr));
  const zoom = getFitZoom(r.width, r.height, modelRadius);
  currentViewState = { target: [0, 0, 0], zoom };
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
      p[k] = clampControlValue(k, v);
      val.textContent = formatControlValue(p[k]);
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

  initModel(model.x, model.y, model.sx, model.sy, model.k);
  buildSliders();
  resizeDeck();
  updatePositions();
  posVersion++;
  render();
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
