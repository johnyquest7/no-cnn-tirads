// JavaScript equivalents of the scikit-image routines features.py calls: regionprops,
// convex_hull_image, perimeter, find_contours and graycomatrix. Ported from scikit-image 0.26.
(function () {
'use strict';
const IWSP = window.IWSP = window.IWSP || {};

// skimage.measure.regionprops(mask)[0]: the properties features.py reads.
function regionprops(mask, h, w) {
  let minr = h, minc = w, maxr = -1, maxc = -1, n = 0, sr = 0, sc = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) {
    n++; sr += y; sc += x;
    if (y < minr) minr = y; if (y > maxr) maxr = y;
    if (x < minc) minc = x; if (x > maxc) maxc = x;
  }
  if (!n) return null;
  const cy = sr / n, cx = sc / n;
  // central second moments about the centroid, in bounding-box coordinates as skimage does
  const ly = (sr - minr * n) / n, lx = (sc - minc * n) / n;
  let mu20 = 0, mu02 = 0, mu11 = 0;
  for (let y = minr; y <= maxr; y++) for (let x = minc; x <= maxc; x++) if (mask[y * w + x]) {
    const dy = (y - minr) - ly, dx = (x - minc) - lx;
    mu20 += dy * dy; mu02 += dx * dx; mu11 += dy * dx;
  }
  const a = mu02 / n, b = -mu11 / n, c = mu20 / n;
  const orientation = (a - c === 0) ? (b < 0 ? Math.PI / 4 : -Math.PI / 4) : 0.5 * Math.atan2(-2 * b, c - a);
  return {bbox: [minr, minc, maxr + 1, maxc + 1], area: n, centroid: [cy, cx], orientation,
          equivalentDiameter: Math.sqrt(4 * n / Math.PI)};
}

// skimage.measure.grid_points_in_poly's point test (_shared/geometry.pyx): 0 outside, else inside/edge/vertex.
function pointInPolygon(xp, yp, x, y) {
  const n = xp.length, eps = 1e-12;
  let x1 = xp[n - 1] - x, y1 = yp[n - 1] - y, l = 0, r = 0;
  for (let i = 0; i < n; i++) {
    const x0 = xp[i] - x, y0 = yp[i] - y;
    if (-eps < x0 && x0 < eps && -eps < y0 && y0 < eps) return 2;
    if ((y0 > 0) !== (y1 > 0)) { if ((x0 * y1 - x1 * y0) / (y1 - y0) > 0) r++; }
    if ((y0 < 0) !== (y1 < 0)) { if ((x0 * y1 - x1 * y0) / (y1 - y0) < 0) l++; }
    x1 = x0; y1 = y0;
  }
  if ((r & 1) !== (l & 1)) return 3;
  return r & 1;
}

// skimage.morphology.convex_hull_image(mask): the convex hull of the pixel "diamonds"
// (each pixel centre offset by half a pixel along each axis), borders included.
function convexHullImage(mask, h, w) {
  const pts = [];
  for (let y = 0; y < h; y++) {
    let first = -1, last = -1;
    for (let x = 0; x < w; x++) if (mask[y * w + x]) { if (first < 0) first = x; last = x; }
    if (first < 0) continue;
    for (const x of (first === last ? [first] : [first, last]))
      pts.push([y - 0.5, x], [y + 0.5, x], [y, x - 0.5], [y, x + 0.5]);
  }
  const out = new Uint8Array(h * w);
  if (!pts.length) return out;
  pts.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const p of pts) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop(); lower.push(p); }
  for (let i = pts.length - 1; i >= 0; i--) { const p = pts[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop(); upper.push(p); }
  const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
  const xp = Float64Array.from(hull, p => p[0]), yp = Float64Array.from(hull, p => p[1]);
  let r0 = Infinity, r1 = -Infinity, c0 = Infinity, c1 = -Infinity;
  for (const p of hull) { r0 = Math.min(r0, p[0]); r1 = Math.max(r1, p[0]); c0 = Math.min(c0, p[1]); c1 = Math.max(c1, p[1]); }
  r0 = Math.max(0, Math.floor(r0)); r1 = Math.min(h - 1, Math.ceil(r1));
  c0 = Math.max(0, Math.floor(c0)); c1 = Math.min(w - 1, Math.ceil(c1));
  for (let y = r0; y <= r1; y++) {
    // the hull is convex, so each row is one run: find its ends, then fill
    let a = c0, b = c1;
    while (a <= b && !pointInPolygon(xp, yp, y, a)) a++;
    while (b >= a && !pointInPolygon(xp, yp, y, b)) b--;
    for (let x = a; x <= b; x++) out[y * w + x] = 1;
  }
  return out;
}

// skimage.measure.perimeter(mask, neighborhood=4)
const SQRT2 = Math.sqrt(2);
const PERIM_W = (() => {
  const t = new Float64Array(50);
  for (const i of [5, 7, 15, 17, 25, 27]) t[i] = 1;
  for (const i of [21, 33]) t[i] = SQRT2;
  for (const i of [13, 23]) t[i] = (1 + SQRT2) / 2;
  return t;
})();
function perimeter(mask, h, w) {
  const border = new Uint8Array(h * w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!mask[i]) continue;
    const inner = y > 0 && y < h - 1 && x > 0 && x < w - 1 && mask[i - w] && mask[i + w] && mask[i - 1] && mask[i + 1];
    border[i] = inner ? 0 : 1;
  }
  const hist = new Float64Array(50);
  const at = (y, x) => (y >= 0 && y < h && x >= 0 && x < w) ? border[y * w + x] : 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = at(y, x) + 2 * (at(y - 1, x) + at(y + 1, x) + at(y, x - 1) + at(y, x + 1)) +
              10 * (at(y - 1, x - 1) + at(y - 1, x + 1) + at(y + 1, x - 1) + at(y + 1, x + 1));
    hist[v]++;
  }
  let total = 0;
  for (let i = 0; i < 50; i++) total += hist[i] * PERIM_W[i];
  return total;
}

// skimage.measure.find_contours(mask.astype(float), 0.5): marching squares with low-valued
// pixels vertex-connected, segments joined exactly as skimage's _assemble_contours does.
// Returns contours as arrays of [row, col]; coordinates are multiples of 0.5.
function findContours(mask, h, w, box) {
  const [ra, ca, rb, cb] = box || [0, 0, h, w];              // squares outside the mask's box are empty
  const r0s = Math.max(0, ra - 1), r0e = Math.min(h - 1, rb + 1), c0s = Math.max(0, ca - 1), c0e = Math.min(w - 1, cb + 1);
  const W2 = 2 * w + 2, key = p => p[0] * 2 * W2 + p[1] * 2;
  const contours = new Map(), starts = new Map(), ends = new Map();
  let current = 0;
  const add = (from, to) => {
    const kf = key(from), kt = key(to);
    if (kf === kt) return;
    const tail = starts.get(kt), head = ends.get(kf);
    if (tail) starts.delete(kt);
    if (head) ends.delete(kf);
    if (tail && head) {
      if (tail.c === head.c) { head.c.push(to); }
      else if (tail.num > head.num) {
        for (const p of tail.c) head.c.push(p);
        contours.delete(tail.num);
        starts.set(key(head.c[0]), head); ends.set(key(head.c[head.c.length - 1]), head);
      } else {
        const merged = head.c.concat(tail.c);
        starts.delete(key(head.c[0]));
        tail.c.length = 0; for (const p of merged) tail.c.push(p);
        contours.delete(head.num);
        starts.set(key(tail.c[0]), tail); ends.set(key(tail.c[tail.c.length - 1]), tail);
      }
    } else if (!tail && !head) {
      const rec = {c: [from, to], num: current};
      contours.set(current, rec); starts.set(kf, rec); ends.set(kt, rec); current++;
    } else if (!head) { tail.c.unshift(from); starts.set(kf, tail); }
    else { head.c.push(to); ends.set(kt, head); }
  };
  for (let r0 = r0s; r0 < r0e; r0++) for (let c0 = c0s; c0 < c0e; c0++) {
    const r1 = r0 + 1, c1 = c0 + 1;
    const ul = mask[r0 * w + c0], ur = mask[r0 * w + c1], ll = mask[r1 * w + c0], lr = mask[r1 * w + c1];
    const sq = (ul ? 1 : 0) + (ur ? 2 : 0) + (ll ? 4 : 0) + (lr ? 8 : 0);
    if (sq === 0 || sq === 15) continue;
    const top = [r0, c0 + (ul === ur ? 0 : 0.5)], bottom = [r1, c0 + (ll === lr ? 0 : 0.5)];
    const left = [r0 + (ul === ll ? 0 : 0.5), c0], right = [r0 + (ur === lr ? 0 : 0.5), c1];
    switch (sq) {
      case 1: add(top, left); break;
      case 2: add(right, top); break;
      case 3: add(right, left); break;
      case 4: add(left, bottom); break;
      case 5: add(top, bottom); break;
      case 6: add(right, top); add(left, bottom); break;
      case 7: add(right, bottom); break;
      case 8: add(bottom, right); break;
      case 9: add(top, left); add(bottom, right); break;
      case 10: add(bottom, top); break;
      case 11: add(bottom, left); break;
      case 12: add(left, right); break;
      case 13: add(top, right); break;
      case 14: add(left, top); break;
    }
  }
  return [...contours.keys()].sort((a, b) => a - b).map(k => contours.get(k).c);
}

// skimage.feature.graycomatrix(q, distances, angles=[0, pi/2], levels, symmetric=True), counts only.
// Returns P[d][a] as Float64Array(levels*levels).
function graycomatrix(q, h, w, distances, levels) {
  return distances.map(d => [[0, d], [d, 0]].map(([dr, dc]) => {
    const P = new Float64Array(levels * levels);
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      const rr = r + dr, cc = c + dc;
      if (rr >= h || cc >= w) continue;
      const i = q[r * w + c], j = q[rr * w + cc];
      P[i * levels + j]++; P[j * levels + i]++;
    }
    return P;
  }));
}

IWSP.sk = {regionprops, convexHullImage, perimeter, findContours, graycomatrix};
})();
