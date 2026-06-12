/* =====================================================================
   MI300X Dashboard — Lightweight SVG chart library (no dependencies)
   All renderers take a container element and write an inline SVG so the
   dashboard works fully offline from file://.
   ===================================================================== */

const NS = 'http://www.w3.org/2000/svg';
const C = {
  red: '#ED1C24', redDim: '#7a1418',
  teal: '#00C2B2', amber: '#FFB100', green: '#39D353',
  blue: '#3FA7FF', purple: '#B07CFF',
  grid: '#222a38', axis: '#5b6675', text: '#9aa6b8', textHi: '#e8edf5',
  track: '#1b2230',
};

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function svgRoot(container, w, h) {
  container.innerHTML = '';
  const svg = el('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%', height: '100%',
    preserveAspectRatio: 'xMidYMid meet' }, container);
  return svg;
}
function fmtNum(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  if (n >= 100) return Math.round(n).toString();
  if (n >= 1) return n.toFixed(1);
  return n.toFixed(3);
}

/* ---- Radial gauge ---------------------------------------------------- */
function gauge(container, value, max, opts = {}) {
  const w = 180, h = 130, cx = w / 2, cy = 105, r = 72;
  const svg = svgRoot(container, w, h);
  const frac = clamp(value / max, 0, 1);
  const a0 = Math.PI, a1 = 0;                 // 180° arc, left→right
  const arc = (from, to, color, width) => {
    const x0 = cx + r * Math.cos(from), y0 = cy + r * Math.sin(from);
    const x1 = cx + r * Math.cos(to), y1 = cy + r * Math.sin(to);
    const large = Math.abs(to - from) > Math.PI ? 1 : 0;
    const sweep = to > from ? 1 : 0;
    el('path', { d: `M ${x0} ${y0} A ${r} ${r} 0 ${large} ${sweep} ${x1} ${y1}`,
      fill: 'none', stroke: color, 'stroke-width': width, 'stroke-linecap': 'round' }, svg);
  };
  arc(a0, a1, C.track, 12);
  const col = opts.color || (frac > 0.85 ? C.red : frac > 0.6 ? C.amber : C.teal);
  arc(a0, a0 + (a1 - a0) * frac, col, 12);
  const t1 = el('text', { x: cx, y: cy - 8, fill: C.textHi, 'font-size': 26,
    'font-weight': 700, 'text-anchor': 'middle' }, svg);
  t1.textContent = opts.label != null ? opts.label : fmtNum(value);
  const t2 = el('text', { x: cx, y: cy + 14, fill: C.text, 'font-size': 11,
    'text-anchor': 'middle' }, svg);
  t2.textContent = opts.unit || '';
}

/* ---- Horizontal bar meter ------------------------------------------- */
function meter(container, value, max, opts = {}) {
  const w = 260, h = 26;
  const svg = svgRoot(container, w, h);
  const frac = clamp(value / max, 0, 1);
  el('rect', { x: 0, y: 6, width: w, height: 12, rx: 6, fill: C.track }, svg);
  const col = opts.color || (frac > 0.85 ? C.red : frac > 0.6 ? C.amber : C.teal);
  el('rect', { x: 0, y: 6, width: Math.max(4, w * frac), height: 12, rx: 6, fill: col }, svg);
}

/* ---- Line / area chart ---------------------------------------------- */
function lineChart(container, series, opts = {}) {
  const w = opts.w || 520, h = opts.h || 200, pad = { l: 46, r: 14, t: 16, b: 26 };
  const svg = svgRoot(container, w, h);
  const all = series.flatMap(s => s.data);
  const maxY = opts.maxY || Math.max(...all) * 1.15 || 1;
  const minY = 0;
  const n = series[0].data.length;
  const X = i => pad.l + (w - pad.l - pad.r) * (i / Math.max(n - 1, 1));
  const Y = v => h - pad.b - (h - pad.t - pad.b) * ((v - minY) / (maxY - minY));

  // grid + y labels
  for (let g = 0; g <= 4; g++) {
    const yv = minY + (maxY - minY) * g / 4;
    const y = Y(yv);
    el('line', { x1: pad.l, y1: y, x2: w - pad.r, y2: y, stroke: C.grid, 'stroke-width': 1 }, svg);
    const t = el('text', { x: pad.l - 8, y: y + 4, fill: C.text, 'font-size': 10,
      'text-anchor': 'end' }, svg);
    t.textContent = fmtNum(yv);
  }
  series.forEach(s => {
    const pts = s.data.map((v, i) => `${X(i)},${Y(v)}`).join(' ');
    if (opts.area) {
      el('polygon', { points: `${X(0)},${Y(0)} ${pts} ${X(n - 1)},${Y(0)}`,
        fill: s.color, opacity: 0.12 }, svg);
    }
    el('polyline', { points: pts, fill: 'none', stroke: s.color,
      'stroke-width': 2.4, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
    const last = s.data[s.data.length - 1];
    el('circle', { cx: X(n - 1), cy: Y(last), r: 3.5, fill: s.color }, svg);
  });
  if (opts.xlabel) {
    const t = el('text', { x: (w + pad.l) / 2, y: h - 4, fill: C.text, 'font-size': 10,
      'text-anchor': 'middle' }, svg);
    t.textContent = opts.xlabel;
  }
}

/* ---- Bar chart ------------------------------------------------------- */
function barChart(container, bars, opts = {}) {
  const w = opts.w || 520, h = opts.h || 220, pad = { l: 48, r: 14, t: 18, b: 46 };
  const svg = svgRoot(container, w, h);
  const maxY = opts.maxY || Math.max(...bars.map(b => b.value)) * 1.15 || 1;
  const Y = v => h - pad.b - (h - pad.t - pad.b) * (v / maxY);
  const bw = (w - pad.l - pad.r) / bars.length;
  for (let g = 0; g <= 4; g++) {
    const yv = maxY * g / 4, y = Y(yv);
    el('line', { x1: pad.l, y1: y, x2: w - pad.r, y2: y, stroke: C.grid }, svg);
    const t = el('text', { x: pad.l - 8, y: y + 4, fill: C.text, 'font-size': 10,
      'text-anchor': 'end' }, svg);
    t.textContent = fmtNum(yv);
  }
  bars.forEach((b, i) => {
    const x = pad.l + i * bw + bw * 0.18, bwidth = bw * 0.64;
    el('rect', { x, y: Y(b.value), width: bwidth, height: h - pad.b - Y(b.value),
      rx: 4, fill: b.color || C.teal }, svg);
    const lab = el('text', { x: x + bwidth / 2, y: h - pad.b + 16, fill: C.text,
      'font-size': 10, 'text-anchor': 'middle' }, svg);
    lab.textContent = b.label;
    if (b.sub) {
      const s = el('text', { x: x + bwidth / 2, y: h - pad.b + 30, fill: C.axis,
        'font-size': 9, 'text-anchor': 'middle' }, svg);
      s.textContent = b.sub;
    }
    const val = el('text', { x: x + bwidth / 2, y: Y(b.value) - 6, fill: C.textHi,
      'font-size': 11, 'font-weight': 600, 'text-anchor': 'middle' }, svg);
    val.textContent = b.valLabel || fmtNum(b.value);
  });
}

/* ---- Donut ----------------------------------------------------------- */
function donut(container, segments, opts = {}) {
  const w = 200, h = 200, cx = 100, cy = 100, r = 72, rin = 46;
  const svg = svgRoot(container, w, h);
  const total = segments.reduce((s, x) => s + x.value, 0) || 1;
  let ang = -Math.PI / 2;
  segments.forEach(seg => {
    const a0 = ang, a1 = ang + (seg.value / total) * Math.PI * 2;
    ang = a1;
    const large = a1 - a0 > Math.PI ? 1 : 0;
    const p = (rad, a) => `${cx + rad * Math.cos(a)} ${cy + rad * Math.sin(a)}`;
    el('path', {
      d: `M ${p(r, a0)} A ${r} ${r} 0 ${large} 1 ${p(r, a1)} L ${p(rin, a1)} A ${rin} ${rin} 0 ${large} 0 ${p(rin, a0)} Z`,
      fill: seg.color }, svg);
  });
  if (opts.center) {
    const t = el('text', { x: cx, y: cy - 2, fill: C.textHi, 'font-size': 22,
      'font-weight': 700, 'text-anchor': 'middle' }, svg);
    t.textContent = opts.center;
    if (opts.centerSub) {
      const s = el('text', { x: cx, y: cy + 18, fill: C.text, 'font-size': 11,
        'text-anchor': 'middle' }, svg);
      s.textContent = opts.centerSub;
    }
  }
}

/* ---- Roofline -------------------------------------------------------- */
function roofline(container, point, opts = {}) {
  const w = opts.w || 520, h = opts.h || 240, pad = { l: 54, r: 16, t: 18, b: 40 };
  const svg = svgRoot(container, w, h);
  const peakT = opts.peakTflops, bwTBs = opts.peakBwTBs;
  // Roofline: y_TFLOPS = min(peakT, bandwidth[TB/s] * intensity[FLOP/byte]).
  // (TB/s = 1e12 byte/s, so bw * intensity is already in TFLOP/s.)
  const ridge = peakT / bwTBs;                 // FLOP/byte at the ridge point
  const xMin = 0.01, xMax = 10000, yMin = 0.1, yMax = peakT * 1.2;
  const lx = v => pad.l + (w - pad.l - pad.r) * (Math.log10(v) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin));
  const ly = v => h - pad.b - (h - pad.t - pad.b) * (Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin));
  // grid
  [0.01, 0.1, 1, 10, 100, 1000, 10000].forEach(gx => {
    el('line', { x1: lx(gx), y1: pad.t, x2: lx(gx), y2: h - pad.b, stroke: C.grid }, svg);
    const t = el('text', { x: lx(gx), y: h - pad.b + 14, fill: C.text, 'font-size': 9,
      'text-anchor': 'middle' }, svg); t.textContent = gx;
  });
  [0.1, 1, 10, 100, 1000].forEach(gy => {
    if (gy > yMax) return;
    el('line', { x1: pad.l, y1: ly(gy), x2: w - pad.r, y2: ly(gy), stroke: C.grid }, svg);
    const t = el('text', { x: pad.l - 6, y: ly(gy) + 3, fill: C.text, 'font-size': 9,
      'text-anchor': 'end' }, svg); t.textContent = gy;
  });
  // roofline: memory-bound slope (bw * intensity) then flat compute ceiling
  const rl = [];
  for (let lxi = Math.log10(xMin); lxi <= Math.log10(xMax); lxi += 0.1) {
    const xi = Math.pow(10, lxi);
    rl.push(`${lx(xi)},${ly(Math.max(yMin, Math.min(peakT, bwTBs * xi)))}`);
  }
  el('polyline', { points: rl.join(' '), fill: 'none', stroke: C.amber, 'stroke-width': 2 }, svg);
  // ridge marker
  el('line', { x1: lx(ridge), y1: pad.t, x2: lx(ridge), y2: h - pad.b,
    stroke: C.axis, 'stroke-width': 1, 'stroke-dasharray': '3 3' }, svg);
  // achieved point
  const px = lx(clamp(point.x, xMin, xMax)), py = ly(clamp(point.y, yMin, yMax));
  el('circle', { cx: px, cy: py, r: 6, fill: C.red, stroke: '#fff', 'stroke-width': 1.5 }, svg);
  const lab = el('text', { x: px + 10, y: py - 8, fill: C.textHi, 'font-size': 11,
    'font-weight': 600 }, svg); lab.textContent = `${fmtNum(point.y)} TFLOPS`;
  // axis labels
  const ax = el('text', { x: (w + pad.l) / 2, y: h - 2, fill: C.text, 'font-size': 10,
    'text-anchor': 'middle' }, svg); ax.textContent = 'Arithmetic intensity (FLOP/byte)';
}

/* ---- Histogram ------------------------------------------------------- */
function histogram(container, center, opts = {}) {
  const w = opts.w || 520, h = opts.h || 180, pad = { l: 40, r: 14, t: 14, b: 28 };
  const svg = svgRoot(container, w, h);
  const bins = 22;
  const data = [];
  for (let i = 0; i < bins; i++) {
    const dx = (i - bins * 0.42) / (bins * 0.18);
    data.push(Math.exp(-0.5 * dx * dx) + 0.04 * Math.max(0, dx)); // right-skewed bell
  }
  const maxV = Math.max(...data);
  const bw = (w - pad.l - pad.r) / bins;
  data.forEach((v, i) => {
    const bh = (h - pad.t - pad.b) * (v / maxV);
    el('rect', { x: pad.l + i * bw + 1, y: h - pad.b - bh, width: bw - 2, height: bh,
      rx: 2, fill: i > bins * 0.72 ? C.amber : C.teal, opacity: 0.85 }, svg);
  });
  const t = el('text', { x: (w + pad.l) / 2, y: h - 4, fill: C.text, 'font-size': 10,
    'text-anchor': 'middle' }, svg);
  t.textContent = opts.xlabel || `latency (p50 ≈ ${fmtNum(center)} ms)`;
}

/* ---- Kernel timeline ------------------------------------------------- */
function timeline(container, kernels, opts = {}) {
  const w = opts.w || 520, h = opts.h || 120, pad = { l: 10, r: 10, t: 10, b: 24 };
  const svg = svgRoot(container, w, h);
  const total = kernels.reduce((s, k) => s + k.dur, 0) || 1;
  let x = pad.l; const lane = 34;
  const palette = [C.teal, C.blue, C.purple, C.amber, C.green, C.red];
  kernels.forEach((k, i) => {
    const bw = (w - pad.l - pad.r) * (k.dur / total);
    el('rect', { x, y: pad.t, width: Math.max(2, bw - 2), height: lane, rx: 3,
      fill: palette[i % palette.length], opacity: 0.85 }, svg);
    if (bw > 46) {
      const t = el('text', { x: x + 5, y: pad.t + 21, fill: '#06121a', 'font-size': 10,
        'font-weight': 600 }, svg); t.textContent = k.name;
    }
    x += bw;
  });
  const t = el('text', { x: pad.l, y: h - 6, fill: C.text, 'font-size': 10 }, svg);
  t.textContent = opts.xlabel || 'one iteration — kernel dispatch timeline (relative)';
}

/* ---- Parity plot: predicted/measured ratio vs ±20% band -------------- */
function parityPlot(container, pairs, opts = {}) {
  const w = opts.w || 520, h = opts.h || 220, pad = { l: 44, r: 14, t: 16, b: 40 };
  const svg = svgRoot(container, w, h);
  const yMin = 0.6, yMax = 1.4;
  const band = (opts.target || 20) / 100;          // ±band fraction
  const Y = r => h - pad.b - (h - pad.t - pad.b) * ((clamp(r, yMin, yMax) - yMin) / (yMax - yMin));
  const n = pairs.length;
  const X = i => pad.l + (w - pad.l - pad.r) * ((i + 0.5) / n);
  // ±band shading
  el('rect', { x: pad.l, y: Y(1 + band), width: w - pad.l - pad.r, height: Y(1 - band) - Y(1 + band),
    fill: C.green, opacity: 0.10 }, svg);
  el('line', { x1: pad.l, y1: Y(1 + band), x2: w - pad.r, y2: Y(1 + band), stroke: C.green,
    'stroke-width': 1, 'stroke-dasharray': '4 3', opacity: 0.6 }, svg);
  el('line', { x1: pad.l, y1: Y(1 - band), x2: w - pad.r, y2: Y(1 - band), stroke: C.green,
    'stroke-width': 1, 'stroke-dasharray': '4 3', opacity: 0.6 }, svg);
  // perfect-prediction line at ratio = 1
  el('line', { x1: pad.l, y1: Y(1), x2: w - pad.r, y2: Y(1), stroke: C.axis, 'stroke-width': 1.5 }, svg);
  // y labels
  [0.6, 0.8, 1.0, 1.2, 1.4].forEach(r => {
    const t = el('text', { x: pad.l - 6, y: Y(r) + 3, fill: C.text, 'font-size': 9, 'text-anchor': 'end' }, svg);
    t.textContent = (r * 100 - 100 > 0 ? '+' : '') + Math.round(r * 100 - 100) + '%';
  });
  pairs.forEach((p, i) => {
    const col = p.within ? C.green : C.red;
    el('line', { x1: X(i), y1: Y(1), x2: X(i), y2: Y(p.ratio), stroke: col, 'stroke-width': 2, opacity: 0.5 }, svg);
    el('circle', { cx: X(i), cy: Y(p.ratio), r: 5.5, fill: col, stroke: '#0a0d14', 'stroke-width': 1.5 }, svg);
    const lab = el('text', { x: X(i), y: h - pad.b + 14, fill: C.text, 'font-size': 8.5, 'text-anchor': 'middle' }, svg);
    lab.textContent = (p.k.length > 12 ? p.k.slice(0, 11) + '…' : p.k);
    const e = el('text', { x: X(i), y: h - pad.b + 26, fill: col, 'font-size': 9, 'font-weight': 600, 'text-anchor': 'middle' }, svg);
    e.textContent = p.errPct.toFixed(1) + '%';
  });
  const t = el('text', { x: pad.l, y: 12, fill: C.text, 'font-size': 9 }, svg);
  t.textContent = 'predicted vs measured (0% = perfect; green band = ±' + (opts.target || 20) + '%)';
}

/* ---- Budget bar: used vs deadline budget ----------------------------- */
function budgetBar(container, used, budget, opts = {}) {
  const w = opts.w || 520, h = opts.h || 64, pad = { l: 10, r: 10, t: 14, b: 22 };
  const svg = svgRoot(container, w, h);
  const scaleMax = Math.max(budget * 1.25, used * 1.1);
  const X = v => pad.l + (w - pad.l - pad.r) * (v / scaleMax);
  el('rect', { x: pad.l, y: pad.t, width: w - pad.l - pad.r, height: 18, rx: 5, fill: C.track }, svg);
  const over = used > budget;
  el('rect', { x: pad.l, y: pad.t, width: Math.max(3, X(Math.min(used, scaleMax)) - pad.l), height: 18, rx: 5,
    fill: over ? C.red : used > budget * 0.85 ? C.amber : C.teal }, svg);
  // budget marker
  el('line', { x1: X(budget), y1: pad.t - 4, x2: X(budget), y2: pad.t + 24, stroke: C.amber, 'stroke-width': 2 }, svg);
  const bl = el('text', { x: X(budget), y: pad.t - 6, fill: C.amber, 'font-size': 9, 'text-anchor': 'middle' }, svg);
  bl.textContent = 'budget ' + budget.toFixed(1) + 'ms';
  if (opts.p99 != null) {
    el('line', { x1: X(Math.min(opts.p99, scaleMax)), y1: pad.t, x2: X(Math.min(opts.p99, scaleMax)), y2: pad.t + 18,
      stroke: '#fff', 'stroke-width': 1.5, 'stroke-dasharray': '2 2' }, svg);
  }
  const ul = el('text', { x: pad.l, y: h - 6, fill: over ? C.red : C.text, 'font-size': 10 }, svg);
  ul.textContent = 'used ' + used.toFixed(2) + ' ms' + (opts.p99 != null ? '  (p99 ' + opts.p99.toFixed(2) + ' ms)' : '') + (over ? '  — DEADLINE MISS' : '  — within budget');
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
