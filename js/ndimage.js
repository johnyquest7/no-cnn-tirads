// JavaScript equivalents of the scipy.ndimage and numpy routines that segment.py, features.py
// and pipeline.py call. Each function names the routine it reproduces. Images are flat typed
// arrays in row-major order with explicit height h and width w; masks are Uint8Array of 0/1.
//
// Arithmetic follows the originals: scipy filters accumulate in double precision in the same
// order as scipy's C code and are then stored at the precision of the output array, and
// float32 results are rounded exactly where numpy rounds them (Math.fround / Float32Array).
(function () {
'use strict';
const IWSP = window.IWSP = window.IWSP || {};
const EPS = 2.220446049250313e-16;

// scipy mode='reflect':  d c b a | a b c d | d c b a
function reflect(i, n) {
  if (i >= 0 && i < n) return i;
  if (n === 1) return 0;
  const p = 2 * n;
  i %= p; if (i < 0) i += p;
  return i < n ? i : p - 1 - i;
}

// Copies one line of `src` (along `axis`) into buf, extended by s1 samples before and s2 after.
function loadLine(src, base, stride, n, s1, s2, buf) {
  for (let i = 0; i < n; i++) buf[i + s1] = src[base + i * stride];
  for (let i = 0; i < s1; i++) buf[i] = src[base + reflect(i - s1, n) * stride];
  for (let i = 0; i < s2; i++) buf[n + s1 + i] = src[base + reflect(n + i, n) * stride];
}

// scipy.ndimage.correlate1d (NI_Correlate1D), mode='reflect', origin 0.
function correlate1d(src, h, w, axis, wts, Out) {
  const n = axis === 0 ? h : w, lines = axis === 0 ? w : h, stride = axis === 0 ? w : 1;
  const K = wts.length, s1 = K >> 1, s2 = K - s1 - 1;
  let sym = 0;
  if (K & 1) {
    sym = 1;
    for (let ii = 1; ii <= s1; ii++) if (Math.abs(wts[ii + s1] - wts[s1 - ii]) > EPS) { sym = 0; break; }
    if (sym === 0) {
      sym = -1;
      for (let ii = 1; ii <= s1; ii++) if (Math.abs(wts[s1 + ii] + wts[s1 - ii]) > EPS) { sym = 0; break; }
    }
  }
  const out = new Out(h * w), buf = new Float64Array(n + s1 + s2);
  for (let L = 0; L < lines; L++) {
    const base = axis === 0 ? L : L * w;
    loadLine(src, base, stride, n, s1, s2, buf);
    for (let l = 0; l < n; l++) {
      const c = l + s1;
      let o;
      if (sym > 0) {
        o = buf[c] * wts[s1];
        for (let jj = -s1; jj < 0; jj++) o += (buf[c + jj] + buf[c - jj]) * wts[s1 + jj];
      } else if (sym < 0) {
        o = buf[c] * wts[s1];
        for (let jj = -s1; jj < 0; jj++) o += (buf[c + jj] - buf[c - jj]) * wts[s1 + jj];
      } else {
        o = buf[c + s2] * wts[s1 + s2];
        for (let jj = -s1; jj < s2; jj++) o += buf[c + jj] * wts[s1 + jj];
      }
      out[base + l * stride] = o;
    }
  }
  return out;
}

function kernel(sigma, order) {
  const k = IWSP.MODEL.gauss_kernels[Number.isInteger(sigma) ? sigma.toFixed(1) : String(sigma)];
  if (!k) throw new Error('No exported Gaussian kernel for sigma ' + sigma);
  return k[order];
}

// scipy.ndimage.gaussian_filter(src, sigma): axis 0 then axis 1, the intermediate held at output precision.
function gaussianFilter(src, h, w, sigma, Out) {
  const k0 = kernel(sigma, 0);
  return correlate1d(correlate1d(src, h, w, 0, k0, Out), h, w, 1, k0, Out);
}

// scipy.ndimage.gaussian_laplace(src float32, sigma) -> float32:
// (second derivative along rows, smoothing along columns) + (smoothing, second derivative).
function gaussianLaplace(src, h, w, sigma) {
  const k0 = kernel(sigma, 0), k2 = kernel(sigma, 2), F = Float32Array;
  const a = correlate1d(correlate1d(src, h, w, 0, k2, F), h, w, 1, k0, F);
  const b = correlate1d(correlate1d(src, h, w, 0, k0, F), h, w, 1, k2, F);
  for (let i = 0; i < a.length; i++) a[i] = a[i] + b[i];
  return a;
}

// scipy.ndimage.uniform_filter1d (NI_UniformFilter1D): a running sum, mode='reflect', origin 0.
function uniformFilter1d(src, h, w, axis, size, Out) {
  const n = axis === 0 ? h : w, lines = axis === 0 ? w : h, stride = axis === 0 ? w : 1;
  const s1 = size >> 1, s2 = size - s1 - 1;
  const out = new Out(h * w), buf = new Float64Array(n + s1 + s2);
  for (let L = 0; L < lines; L++) {
    const base = axis === 0 ? L : L * w;
    loadLine(src, base, stride, n, s1, s2, buf);
    let tmp = 0;
    for (let l = 0; l < size; l++) tmp += buf[l];
    out[base] = tmp / size;
    for (let l = 1; l < n; l++) {
      tmp += buf[l + size - 1] - buf[l - 1];
      out[base + l * stride] = tmp / size;
    }
  }
  return out;
}
function uniformFilter(src, h, w, size, Out) {
  return uniformFilter1d(uniformFilter1d(src, h, w, 0, size, Out), h, w, 1, size, Out);
}

// scipy.ndimage.maximum_filter(src, size) with a square window (odd size), mode='reflect'.
function maximumFilter(src, h, w, size) {
  const r = size >> 1, Out = src.constructor;
  let cur = src;
  for (let axis = 0; axis < 2; axis++) {
    const n = axis === 0 ? h : w, lines = axis === 0 ? w : h, stride = axis === 0 ? w : 1;
    const out = new Out(h * w), buf = new Float64Array(n + 2 * r);
    for (let L = 0; L < lines; L++) {
      const base = axis === 0 ? L : L * w;
      loadLine(cur, base, stride, n, r, r, buf);
      for (let l = 0; l < n; l++) {
        let m = buf[l];
        for (let k = 1; k <= 2 * r; k++) if (buf[l + k] > m) m = buf[l + k];
        out[base + l * stride] = m;
      }
    }
    cur = out;
  }
  return cur;
}

// scipy.ndimage.maximum_filter1d(x, size, mode='wrap') for a 1-D array.
function maximumFilter1dWrap(x, size) {
  const n = x.length, r = size >> 1, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let m = -Infinity;
    for (let k = -r; k <= r; k++) { const v = x[((i + k) % n + n) % n]; if (v > m) m = v; }
    out[i] = m;
  }
  return out;
}

// scipy.ndimage.median_filter(src float32, size=5), mode='reflect'.
function medianFilter5(src, h, w) {
  const out = new Float32Array(h * w), win = new Float32Array(25);
  const rows = new Int32Array(h * 5), cols = new Int32Array(w * 5);
  for (let y = 0; y < h; y++) for (let k = 0; k < 5; k++) rows[y * 5 + k] = reflect(y + k - 2, h) * w;
  for (let x = 0; x < w; x++) for (let k = 0; k < 5; k++) cols[x * 5 + k] = reflect(x + k - 2, w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let n = 0;
      for (let a = 0; a < 5; a++) {
        const ro = rows[y * 5 + a];
        for (let b = 0; b < 5; b++) {                      // insertion sort as the window is filled
          const v = src[ro + cols[x * 5 + b]];
          let j = n++;
          while (j > 0 && win[j - 1] > v) { win[j] = win[j - 1]; j--; }
          win[j] = v;
        }
      }
      out[y * w + x] = win[12];
    }
  }
  return out;
}

// ---------------------------------------------------------------- binary morphology
// scipy's default structuring element is the 4-neighbour cross, and `iterations=k` of it is a
// city-block ball of radius k. Both are computed here from an exact city-block distance map.
const BIG = 1 << 29;

function cityblockToSet(mask, h, w) {                       // distance to the nearest 1-pixel
  const d = new Int32Array(h * w);
  for (let i = 0; i < d.length; i++) d[i] = mask[i] ? 0 : BIG;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x; let v = d[i];
    if (y > 0 && d[i - w] + 1 < v) v = d[i - w] + 1;
    if (x > 0 && d[i - 1] + 1 < v) v = d[i - 1] + 1;
    d[i] = v;
  }
  for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
    const i = y * w + x; let v = d[i];
    if (y < h - 1 && d[i + w] + 1 < v) v = d[i + w] + 1;
    if (x < w - 1 && d[i + 1] + 1 < v) v = d[i + 1] + 1;
    d[i] = v;
  }
  return d;
}
function cityblockToBackground(mask, h, w) {                // nearest 0-pixel; outside the image counts as 0
  const d = new Int32Array(h * w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!mask[i]) { d[i] = 0; continue; }
    const up = y > 0 ? d[i - w] : 0, left = x > 0 ? d[i - 1] : 0;
    d[i] = (up < left ? up : left) + 1;
  }
  for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
    const i = y * w + x;
    if (!mask[i]) continue;
    const dn = y < h - 1 ? d[i + w] : 0, right = x < w - 1 ? d[i + 1] : 0;
    const v = (dn < right ? dn : right) + 1;
    if (v < d[i]) d[i] = v;
  }
  return d;
}
// scipy.ndimage.binary_dilation(mask, iterations=k)
function dilate(mask, h, w, k) {
  const d = cityblockToSet(mask, h, w), out = new Uint8Array(h * w);
  for (let i = 0; i < out.length; i++) out[i] = d[i] <= k ? 1 : 0;
  return out;
}
// scipy.ndimage.binary_erosion(mask, iterations=k)   (border_value=0)
function erode(mask, h, w, k) {
  const d = cityblockToBackground(mask, h, w), out = new Uint8Array(h * w);
  for (let i = 0; i < out.length; i++) out[i] = d[i] > k ? 1 : 0;
  return out;
}
// scipy.ndimage.binary_opening(mask, iterations=1)
function opening(mask, h, w) { return dilate(erode(mask, h, w, 1), h, w, 1); }

// scipy.ndimage.binary_fill_holes(mask): everything not 4-connected to the outside through background.
function fillHoles(mask, h, w) {
  const seen = new Uint8Array(h * w), stack = new Int32Array(h * w);
  let sp = 0;
  const push = i => { if (!mask[i] && !seen[i]) { seen[i] = 1; stack[sp++] = i; } };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (sp) {
    const i = stack[--sp], x = i % w;
    if (i >= w) push(i - w);
    if (i < (h - 1) * w) push(i + w);
    if (x > 0) push(i - 1);
    if (x < w - 1) push(i + 1);
  }
  const out = new Uint8Array(h * w);
  for (let i = 0; i < out.length; i++) out[i] = seen[i] ? 0 : 1;
  return out;
}

// scipy.ndimage.label(mask): 4-connected components, numbered in raster order of their first pixel.
function label(mask, h, w) {
  const lab = new Int32Array(h * w), stack = new Int32Array(h * w), sizes = [0];
  let n = 0;
  for (let s = 0; s < lab.length; s++) {
    if (!mask[s] || lab[s]) continue;
    n++;
    let sp = 0, count = 0;
    lab[s] = n; stack[sp++] = s;
    while (sp) {
      const i = stack[--sp], x = i % w;
      count++;
      if (i >= w && mask[i - w] && !lab[i - w]) { lab[i - w] = n; stack[sp++] = i - w; }
      if (i < (h - 1) * w && mask[i + w] && !lab[i + w]) { lab[i + w] = n; stack[sp++] = i + w; }
      if (x > 0 && mask[i - 1] && !lab[i - 1]) { lab[i - 1] = n; stack[sp++] = i - 1; }
      if (x < w - 1 && mask[i + 1] && !lab[i + 1]) { lab[i + 1] = n; stack[sp++] = i + 1; }
    }
    sizes.push(count);
  }
  return {lab, n, sizes};                                   // sizes[k] = pixels with label k (np.bincount)
}

// scipy.ndimage.distance_transform_edt(mask): exact Euclidean distance to the nearest 0-pixel
// (Meijster's two-pass algorithm in integer arithmetic).
function edt(mask, h, w) {
  const INF = h + w, g = new Int32Array(h * w);
  for (let x = 0; x < w; x++) {
    g[x] = mask[x] ? INF : 0;
    for (let y = 1; y < h; y++) g[y * w + x] = mask[y * w + x] ? Math.min(INF, g[(y - 1) * w + x] + 1) : 0;
    for (let y = h - 2; y >= 0; y--) if (g[(y + 1) * w + x] + 1 < g[y * w + x]) g[y * w + x] = g[(y + 1) * w + x] + 1;
  }
  const out = new Float64Array(h * w), s = new Int32Array(w), t = new Int32Array(w);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    const fn = (x, i) => (x - i) * (x - i) + g[o + i] * g[o + i];
    const sep = (i, u) => Math.floor((u * u - i * i + g[o + u] * g[o + u] - g[o + i] * g[o + i]) / (2 * (u - i)));
    let q = 0; s[0] = 0; t[0] = 0;
    for (let u = 1; u < w; u++) {
      while (q >= 0 && fn(t[q], s[q]) > fn(t[q], u)) q--;
      if (q < 0) { q = 0; s[0] = u; }
      else { const ww = 1 + sep(s[q], u); if (ww < w) { q++; s[q] = u; t[q] = ww; } }
    }
    for (let u = w - 1; u >= 0; u--) {
      out[o + u] = Math.sqrt(fn(u, s[q]));
      if (u === t[q]) q--;
    }
  }
  return out;
}

// np.gradient(a) for a 2-D array: central differences inside, one-sided at the edges.
function gradient(src, h, w, Out) {
  const gy = new Out(h * w), gx = new Out(h * w), f32 = Out === Float32Array, fr = Math.fround;
  const diff = f32 ? (a, b) => fr(a - b) : (a, b) => a - b;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    gy[i] = h < 2 ? 0 : y === 0 ? diff(src[i + w], src[i]) : y === h - 1 ? diff(src[i], src[i - w]) : diff(src[i + w], src[i - w]) / 2;
    gx[i] = w < 2 ? 0 : x === 0 ? diff(src[i + 1], src[i]) : x === w - 1 ? diff(src[i], src[i - 1]) : diff(src[i + 1], src[i - 1]) / 2;
  }
  return {gy, gx};
}

// ---------------------------------------------------------------- numpy helpers
// np.add.reduce on a contiguous float array: numpy's pairwise summation (blocks of up to 128
// values with eight running sums, split recursively above that). `rnd` is Math.fround for
// float32 input and the identity for float64, so partial sums round exactly as numpy's do.
const ident = v => v;
function pairwise(a, lo, n, rnd) {
  if (n < 8) { let r = 0; for (let i = 0; i < n; i++) r = rnd(r + a[lo + i]); return r; }
  if (n <= 128) {
    let r0 = a[lo], r1 = a[lo + 1], r2 = a[lo + 2], r3 = a[lo + 3], r4 = a[lo + 4], r5 = a[lo + 5], r6 = a[lo + 6], r7 = a[lo + 7];
    let i = 8;
    for (; i < n - (n % 8); i += 8) {
      r0 = rnd(r0 + a[lo + i]); r1 = rnd(r1 + a[lo + i + 1]); r2 = rnd(r2 + a[lo + i + 2]); r3 = rnd(r3 + a[lo + i + 3]);
      r4 = rnd(r4 + a[lo + i + 4]); r5 = rnd(r5 + a[lo + i + 5]); r6 = rnd(r6 + a[lo + i + 6]); r7 = rnd(r7 + a[lo + i + 7]);
    }
    let res = rnd(rnd(rnd(r0 + r1) + rnd(r2 + r3)) + rnd(rnd(r4 + r5) + rnd(r6 + r7)));
    for (; i < n; i++) res = rnd(res + a[lo + i]);
    return res;
  }
  let n2 = n >> 1; n2 -= n2 % 8;
  return rnd(pairwise(a, lo, n2, rnd) + pairwise(a, lo + n2, n - n2, rnd));
}
const sum32 = v => pairwise(v, 0, v.length, Math.fround);          // float32 array .sum()
const sum64 = v => pairwise(v, 0, v.length, ident);                // float64 array .sum()
const mean32 = v => Math.fround(sum32(v) / v.length);              // float32 array .mean()
const mean64 = v => sum64(v) / v.length;
// float32 array .std(): population standard deviation, every step in float32
function std32(v) {
  const m = mean32(v), x = new Float32Array(v.length);
  for (let i = 0; i < x.length; i++) { const d = Math.fround(v[i] - m); x[i] = d * d; }
  return Math.fround(Math.sqrt(Math.fround(sum32(x) / x.length)));
}
// np.median of float32 values, returned as the float32 numpy returns (mean of the two middle
// values is taken in float32). NaN for an empty input.
function median32(vals) {
  const n = vals.length;
  if (!n) return NaN;
  const s = Float32Array.from(vals).sort();
  return n & 1 ? s[n >> 1] : Math.fround(s[n / 2 - 1] + s[n / 2]) / 2;
}
function median64(vals) {
  const n = vals.length;
  if (!n) return NaN;
  const s = Float64Array.from(vals).sort();
  return n & 1 ? s[n >> 1] : (s[n / 2 - 1] + s[n / 2]) / 2;
}
// Values of `src` where `mask` is set, in raster order (numpy boolean indexing).
function pick(src, mask) {
  let n = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) n++;
  const out = new src.constructor(n);
  for (let i = 0, k = 0; i < mask.length; i++) if (mask[i]) out[k++] = src[i];
  return out;
}
function count(mask) { let n = 0; for (let i = 0; i < mask.length; i++) if (mask[i]) n++; return n; }
// Python's round(): half to even.
function rint(x) {
  const r = Math.round(x);
  return (r - x === 0.5 && r % 2 !== 0) ? r - 1 : r;
}
// np.remainder for a positive divisor.
function pymod(a, b) { let m = a % b; if (m !== 0 && m < 0) m += b; return m; }

// np.interp(x, xp, fp, period=period)
function interpPeriodic(x, xp, fp, period) {
  const n = xp.length, idx = new Array(n), xm = new Float64Array(n);
  for (let i = 0; i < n; i++) { xm[i] = pymod(xp[i], period); idx[i] = i; }
  idx.sort((a, b) => xm[a] - xm[b] || a - b);
  const X = new Float64Array(n + 2), F = new Float64Array(n + 2);
  for (let i = 0; i < n; i++) { X[i + 1] = xm[idx[i]]; F[i + 1] = fp[idx[i]]; }
  X[0] = X[n] - period; F[0] = F[n];
  X[n + 1] = X[1] + period; F[n + 1] = F[1];
  const m = n + 2, out = new Float64Array(x.length);
  for (let q = 0; q < x.length; q++) {
    const xv = pymod(x[q], period);
    if (xv > X[m - 1]) { out[q] = F[m - 1]; continue; }
    if (xv < X[0]) { out[q] = F[0]; continue; }
    let lo = 0, hi = m;                                    // largest j with X[j] <= xv
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (X[mid] <= xv) lo = mid; else hi = mid; }
    const j = lo;
    if (j === m - 1) { out[q] = F[j]; continue; }
    if (X[j] === xv) { out[q] = F[j]; continue; }
    const slope = (F[j + 1] - F[j]) / (X[j + 1] - X[j]);
    let r = slope * (xv - X[j]) + F[j];
    if (Number.isNaN(r)) {
      r = slope * (xv - X[j + 1]) + F[j + 1];
      if (Number.isNaN(r) && F[j] === F[j + 1]) r = F[j];
    }
    out[q] = r;
  }
  return out;
}

// |np.fft.rfft(x)|**2
function rfftPower(x) {
  const n = x.length, out = new Float64Array((n >> 1) + 1);
  for (let k = 0; k < out.length; k++) {
    let re = 0, im = 0;
    for (let t = 0; t < n; t++) {
      const a = -2 * Math.PI * ((k * t) % n) / n;
      re += x[t] * Math.cos(a); im += x[t] * Math.sin(a);
    }
    out[k] = re * re + im * im;
  }
  return out;
}

IWSP.nd = {reflect, correlate1d, gaussianFilter, gaussianLaplace, uniformFilter1d, uniformFilter, maximumFilter,
           maximumFilter1dWrap, medianFilter5, dilate, erode, opening, fillHoles, label, edt, gradient,
           median32, median64, pick, count, rint, pymod, interpPeriodic, rfftPower,
           sum32, sum64, mean32, mean64, std32};
})();
