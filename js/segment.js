// JavaScript port of segment.py (the frozen stage-1 segmenter, md5 4734c58d...).
//
//   segment(img, h, w, box) -> Uint8Array mask
//     img : Float32Array in [0, 1], row-major       box : [x1, y1, x2, y2] in pixels
//
// The method is unchanged: the image is resampled into polar coordinates around the centre of the
// loose box, the radial brightness step is measured at every angle, and dynamic programming picks
// one radius per angle. Every line below corresponds to a line of segment.py, with numpy's
// float32 rounding reproduced explicitly (fr = Math.fround).
(function () {
'use strict';
const IWSP = window.IWSP = window.IWSP || {};
const fr = Math.fround;

function ellipseMask(h, w, cx, cy, ax, ay) {
  const m = new Uint8Array(h * w), rint = IWSP.nd.rint;
  IWSP.cv.fillEllipse(m, h, w, rint(cx), rint(cy), Math.max(2, Math.trunc(ax)), Math.max(2, Math.trunc(ay)));
  return m;
}

// np.percentile(a, q) for a float32 array, including numpy's float32 interpolation arithmetic.
function percentile32(values, q) {
  const s = Float32Array.from(values).sort(), n = s.length;
  const vi = (n - 1) * (q / 100);                              // the index is float64, the weight float32
  const lo = Math.floor(vi), hi = Math.min(lo + 1, n - 1), g = fr(vi - lo);
  const a = s[lo], b = s[hi], d = fr(b - a);
  return g >= 0.5 ? fr(b - fr(d * fr(1 - g))) : fr(a + fr(d * g));
}

// _dp(cost): cheapest radius per angle with a limited radial step; two passes to close the curve.
function dp(cost, nr, na, maxStep) {
  const offs = [];
  for (let o = -maxStep; o <= maxStep; o++) offs.push(o);
  const run = (first, single) => {
    const rnd = single ? fr : (v => v);
    let acc = Float64Array.from(first);
    const back = new Int16Array(nr * na);
    for (let j = 1; j < na; j++) {
      const next = new Float64Array(nr);
      for (let i = 0; i < nr; i++) {
        let best = Infinity, bo = 0;
        for (const o of offs) {                               // np.argmin keeps the first minimum
          const src = i + o, c = (src < 0 || src >= nr) ? 1e9 : acc[src];
          if (c < best) { best = c; bo = o; }
        }
        next[i] = rnd(best + cost[i * na + j]);
        back[i * na + j] = i + bo;
      }
      acc = next;
    }
    const path = new Int32Array(na);
    let arg = 0;
    for (let i = 1; i < nr; i++) if (acc[i] < acc[arg]) arg = i;
    path[na - 1] = arg;
    for (let j = na - 1; j > 0; j--) path[j - 1] = back[path[j] * na + j];
    return path;
  };
  const first = new Float64Array(nr);
  for (let i = 0; i < nr; i++) first[i] = cost[i * na];
  const start = run(first, true)[0];                          // float32 arithmetic, as in numpy
  for (let i = 0; i < nr; i++) first[i] = cost[i * na] + 1e6 * (i !== start ? 1 : 0);
  return run(first, false);                                   // this pass is float64 in numpy
}

function segment(img, h, w, box, debug) {
  const P = IWSP.MODEL.seg, cv = IWSP.cv, NR = P.N_RAD, NA = P.N_ANGLE;
  const [x1, y1, x2, y2] = box;
  const cx = (x1 + x2) / 2.0, cy = (y1 + y2) / 2.0;
  const ax = (x2 - x1) * P.WIDTH_FRAC / 2.0, ay = (y2 - y1) * P.HEIGHT_FRAC / 2.0;
  if (ax < 5 || ay < 5) return ellipseMask(h, w, cx, cy, ax, ay);

  // im = cv2.medianBlur((np.clip(img, 0, 1) * 255).astype(np.uint8), 5).astype(np.float32) / 255.0
  const u8 = new Uint8Array(h * w);
  for (let i = 0; i < u8.length; i++) u8[i] = Math.trunc(fr(Math.min(1, Math.max(0, img[i])) * 255));
  const med = cv.medianBlur5(u8, h, w), im = new Float32Array(h * w);
  for (let i = 0; i < im.length; i++) im[i] = med[i] / 255;

  // _polar: sample on a polar grid, radius 1 = the prior ellipse
  const xs = new Float32Array(NR * NA), ys = new Float32Array(NR * NA);
  const ax32 = fr(ax), ay32 = fr(ay), cx32 = fr(cx), cy32 = fr(cy);
  for (let i = 0; i < NR; i++) {
    const rx = fr(P.rad32[i] * ax32), ry = fr(P.rad32[i] * ay32);
    for (let j = 0; j < NA; j++) {
      xs[i * NA + j] = cx32 + fr(rx * P.cos32[j]);
      ys[i * NA + j] = cy32 + fr(ry * P.sin32[j]);
    }
  }
  const pol = cv.gaussianBlur5(cv.remapLinear(im, h, w, xs, ys), NR, NA);

  // grad = np.gradient(pol, axis=0): positive = brighter going outward
  const grad = new Float32Array(NR * NA);
  for (let i = 0; i < NR; i++) for (let j = 0; j < NA; j++) {
    const k = i * NA + j;
    grad[k] = i === 0 ? pol[k + NA] - pol[k] : i === NR - 1 ? pol[k] - pol[k - NA] : fr(pol[k + NA] - pol[k - NA]) / 2;
  }
  // pol[:N_RAD // 3] and pol[-N_RAD // 3:]  -- Python floors -64 / 3 to -22, so the outer band has one more row
  const nIn = Math.floor(NR / 3), nOut = -Math.floor(-NR / 3);
  const inside = IWSP.nd.mean32(pol.subarray(0, nIn * NA)), outside = IWSP.nd.mean32(pol.subarray((NR - nOut) * NA));
  if (inside > outside) for (let k = 0; k < grad.length; k++) grad[k] = -grad[k];   // hyperechoic nodule

  const absg = new Float32Array(grad.length);
  for (let k = 0; k < grad.length; k++) absg[k] = Math.abs(grad[k]);
  const scale = fr(percentile32(absg, P.EDGE_PCTILE) + fr(1e-6));
  const cost = new Float32Array(NR * NA), clip = P.EDGE_CLIP;
  for (let i = 0; i < NR; i++) {
    const d = fr(P.rad32[i] - 1), prior = fr(fr(P.PRIOR_WEIGHT) * fr(d * d));
    for (let j = 0; j < NA; j++) {
      const e = Math.min(clip, Math.max(-clip, fr(grad[i * NA + j] / scale)));
      cost[i * NA + j] = -e + prior;
    }
  }

  const path = dp(cost, NR, NA, P.MAX_STEP);
  const pts = [];
  for (let j = 0; j < NA; j++) {
    const r = P.rad32[path[j]];
    pts.push([cv.cvRound(cx + fr(r * ax32) * P.cos64[j]), cv.cvRound(cy + fr(r * ay32) * P.sin64[j])]);
  }
  const m = new Uint8Array(h * w);
  cv.fillPoly(m, h, w, pts);
  if (debug) Object.assign(debug, {med, pol, grad, scale, cost, path, pts});
  if (IWSP.nd.count(m) < 0.05 * Math.PI * ax * ay) return ellipseMask(h, w, cx, cy, ax, ay);
  return m;
}

IWSP.segment = segment;
IWSP.segmentMethod = () => IWSP.MODEL.method;
})();
