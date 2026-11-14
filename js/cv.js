// JavaScript equivalents of the OpenCV routines that segment.py, features.py and pipeline.py call,
// ported from OpenCV's own source (modules/imgproc/src/drawing.cpp, resize.cpp, color conversion)
// so that integer results are identical and float32 results are rounded the same way.
(function () {
'use strict';
const IWSP = window.IWSP = window.IWSP || {};
const fr = Math.fround, XY_SHIFT = 16, XY_ONE = 65536, HALF = 32768;
const shr16 = v => Math.floor(v / XY_ONE);                   // arithmetic >> 16 on values beyond 32 bits

// cvRound / np.round: round half to even.
function cvRound(x) {
  const r = Math.round(x);
  return (r - x === 0.5 && r % 2 !== 0) ? r - 1 : r;
}

// cv2.cvtColor(BGR2GRAY) on 8-bit input: 15-bit fixed point, exactly as OpenCV.
function grayFromRGBA(rgba, n) {
  const out = new Uint8Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4)
    out[i] = (rgba[p] * 9798 + rgba[p + 1] * 19235 + rgba[p + 2] * 3735 + 16384) >> 15;
  return out;
}

// cv2.resize(src uint8, (dw, dh), interpolation=cv2.INTER_AREA) for shrinking.
function resizeArea(src, sh, sw, dh, dw) {
  const scaleX = sw / dw, scaleY = sh / dh;
  const ix = Math.round(scaleX), iy = Math.round(scaleY);
  const dst = new Uint8Array(dh * dw);
  const sat = v => { const r = cvRound(v); return r < 0 ? 0 : r > 255 ? 255 : r; };
  if (Math.abs(scaleX - ix) < 2.220446049250313e-16 && Math.abs(scaleY - iy) < 2.220446049250313e-16) {
    // integer factors: plain block average (ResizeAreaFast)
    const area = ix * iy, scale = fr(1 / area);
    for (let y = 0; y < dh; y++) for (let x = 0; x < dw; x++) {
      let sum = 0;
      for (let a = 0; a < iy; a++) for (let b = 0; b < ix; b++) sum += src[(y * iy + a) * sw + x * ix + b];
      dst[y * dw + x] = (ix === 2 && iy === 2) ? (sum + 2) >> 2 : sat(fr(sum * scale));
    }
    return dst;
  }
  // general case (computeResizeAreaTab + ResizeArea_Invoker), float32 accumulation
  const tab = (ssize, dsize, scale) => {
    const t = [];
    for (let d = 0; d < dsize; d++) {
      const fsx1 = d * scale, fsx2 = fsx1 + scale, cell = Math.min(scale, ssize - fsx1);
      let sx1 = Math.ceil(fsx1), sx2 = Math.floor(fsx2);
      sx2 = Math.min(sx2, ssize - 1); sx1 = Math.min(sx1, sx2);
      if (sx1 - fsx1 > 1e-3) t.push({di: d, si: sx1 - 1, alpha: fr((sx1 - fsx1) / cell)});
      for (let sx = sx1; sx < sx2; sx++) t.push({di: d, si: sx, alpha: fr(1.0 / cell)});
      if (fsx2 - sx2 > 1e-3) t.push({di: d, si: sx2, alpha: fr(Math.min(Math.min(fsx2 - sx2, 1.0), cell) / cell)});
    }
    return t;
  };
  const xt = tab(sw, dw, scaleX), yt = tab(sh, dh, scaleY);
  const buf = new Float32Array(dw), sum = new Float32Array(dw);
  let prev = yt.length ? yt[0].di : 0;
  for (let j = 0; j < yt.length; j++) {
    const beta = yt[j].alpha, dy = yt[j].di, so = yt[j].si * sw;
    buf.fill(0);
    for (let k = 0; k < xt.length; k++) buf[xt[k].di] = buf[xt[k].di] + fr(src[so + xt[k].si] * xt[k].alpha);
    if (dy !== prev) {
      for (let x = 0; x < dw; x++) { dst[prev * dw + x] = sat(sum[x]); sum[x] = fr(beta * buf[x]); }
      prev = dy;
    } else {
      for (let x = 0; x < dw; x++) sum[x] = sum[x] + fr(beta * buf[x]);
    }
  }
  for (let x = 0; x < dw; x++) dst[prev * dw + x] = sat(sum[x]);
  return dst;
}

// cv2.medianBlur(src uint8, 5): 5x5 median, replicated border.
function medianBlur5(src, h, w) {
  const out = new Uint8Array(h * w), win = new Uint8Array(25);
  const cl = (v, n) => v < 0 ? 0 : v >= n ? n - 1 : v;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let n = 0;
    for (let a = -2; a <= 2; a++) {
      const ro = cl(y + a, h) * w;
      for (let b = -2; b <= 2; b++) {
        const v = src[ro + cl(x + b, w)];
        let j = n++;
        while (j > 0 && win[j - 1] > v) { win[j] = win[j - 1]; j--; }
        win[j] = v;
      }
    }
    out[y * w + x] = win[12];
  }
  return out;
}

// cv2.remap(src float32, xs, ys, INTER_LINEAR, borderMode=BORDER_REPLICATE) as OpenCV 5 computes it:
// exact bilinear interpolation, horizontal then vertical, each step a fused multiply-add
// (a + t*(b - a) with one rounding). Verified bit for bit against OpenCV 5.0.
function remapLinear(src, h, w, xs, ys) {
  const out = new Float32Array(xs.length);
  const cl = (v, n) => v < 0 ? 0 : v >= n ? n - 1 : v;
  for (let i = 0; i < xs.length; i++) {
    const x0 = Math.floor(xs[i]), y0 = Math.floor(ys[i]);
    const fx = fr(xs[i] - x0), fy = fr(ys[i] - y0);
    const xa = cl(x0, w), xb = cl(x0 + 1, w), ya = cl(y0, h) * w, yb = cl(y0 + 1, h) * w;
    const top = fr(src[ya + xa] + fx * fr(src[ya + xb] - src[ya + xa]));
    const bot = fr(src[yb + xa] + fx * fr(src[yb + xb] - src[yb + xa]));
    out[i] = top + fy * fr(bot - top);
  }
  return out;
}

// cv2.GaussianBlur(src float32, (5, 5), 0): the fixed kernel [1 4 6 4 1]/16, BORDER_REFLECT_101,
// rows then columns. The rounding pattern (one fused step in the row pass) matches OpenCV 5.0.
function gaussianBlur5(src, h, w) {
  const K0 = 0.0625, K1 = 0.25, K2 = 0.375;
  const r101 = (i, n) => { if (n === 1) return 0; while (i < 0 || i >= n) i = i < 0 ? -i : 2 * (n - 1) - i; return i; };
  const tmp = new Float32Array(h * w), out = new Float32Array(h * w);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) {
      const c = src[o + x];
      const p1 = fr(src[o + r101(x - 1, w)] + src[o + r101(x + 1, w)]);
      const p0 = fr(src[o + r101(x - 2, w)] + src[o + r101(x + 2, w)]);
      tmp[o + x] = fr(c * K2 + p1 * K1) + p0 * K0;
    }
  }
  for (let y = 0; y < h; y++) {
    const a = r101(y - 2, h) * w, b = r101(y - 1, h) * w, c = y * w, d = r101(y + 1, h) * w, e = r101(y + 2, h) * w;
    for (let x = 0; x < w; x++) {
      const p1 = fr(tmp[b + x] + tmp[d + x]), p0 = fr(tmp[a + x] + tmp[e + x]);
      out[c + x] = fr(fr(tmp[c + x] * K2) + p1 * K1) + p0 * K0;
    }
  }
  return out;
}

// ---------------------------------------------------------------- drawing (drawing.cpp)
// clipLine(Size2l, Point2l&, Point2l&); p1, p2 are {x, y} and are modified in place.
function clipLine(width, height, p1, p2) {
  const right = width - 1, bottom = height - 1;
  if (width <= 0 || height <= 0) return false;
  let c1 = (p1.x < 0 ? 1 : 0) + (p1.x > right ? 2 : 0) + (p1.y < 0 ? 4 : 0) + (p1.y > bottom ? 8 : 0);
  let c2 = (p2.x < 0 ? 1 : 0) + (p2.x > right ? 2 : 0) + (p2.y < 0 ? 4 : 0) + (p2.y > bottom ? 8 : 0);
  if ((c1 & c2) === 0 && (c1 | c2) !== 0) {
    let a;
    if (c1 & 12) {
      a = c1 < 8 ? 0 : bottom;
      p1.x += Math.trunc((a - p1.y) * (p2.x - p1.x) / (p2.y - p1.y));
      p1.y = a;
      c1 = (p1.x < 0 ? 1 : 0) + (p1.x > right ? 2 : 0);
    }
    if (c2 & 12) {
      a = c2 < 8 ? 0 : bottom;
      p2.x += Math.trunc((a - p2.y) * (p2.x - p1.x) / (p2.y - p1.y));
      p2.y = a;
      c2 = (p2.x < 0 ? 1 : 0) + (p2.x > right ? 2 : 0);
    }
    if ((c1 & c2) === 0 && (c1 | c2) !== 0) {
      if (c1) {
        a = c1 === 1 ? 0 : right;
        p1.y += Math.trunc((a - p1.x) * (p2.y - p1.y) / (p2.x - p1.x));
        p1.x = a; c1 = 0;
      }
      if (c2) {
        a = c2 === 1 ? 0 : right;
        p2.y += Math.trunc((a - p2.x) * (p2.y - p1.y) / (p2.x - p1.x));
        p2.x = a; c2 = 0;
      }
    }
  }
  return (c1 | c2) === 0;
}

// Line(img, pt1, pt2, color, 8): LineIterator with 8-connectivity, drawn left to right.
function line(img, h, w, a, b) {
  let p1 = {x: a.x, y: a.y}, p2 = {x: b.x, y: b.y};
  if (p1.x < 0 || p1.x >= w || p2.x < 0 || p2.x >= w || p1.y < 0 || p1.y >= h || p2.y < 0 || p2.y >= h)
    if (!clipLine(w, h, p1, p2)) return;
  let dx = p2.x - p1.x, dy = p2.y - p1.y, deltaX = 1, deltaY = 1;
  if (dx < 0) { dx = -dx; dy = -dy; p1 = p2; }
  if (dy < 0) { dy = -dy; deltaY = -1; }
  const vert = dy > dx;
  if (vert) { let t = dx; dx = dy; dy = t; t = deltaX; deltaX = deltaY; deltaY = t; }
  let err = dx - (dy + dy);
  const plusDelta = dx + dx, minusDelta = -(dy + dy);
  let minusShift = deltaX, plusShift = 0, minusStep = 0, plusStep = deltaY;
  const cnt = dx + 1;
  if (vert) { let t = plusStep; plusStep = plusShift; plusShift = t; t = minusStep; minusStep = minusShift; minusShift = t; }
  let x = p1.x, y = p1.y;
  for (let i = 0; i < cnt; i++) {
    img[y * w + x] = 1;
    const m = err < 0;
    err += minusDelta + (m ? plusDelta : 0);
    x += minusShift + (m ? plusShift : 0);
    y += minusStep + (m ? plusStep : 0);
  }
}

// Line2(img, pt1, pt2, color): fixed-point (16.16) line used by FillConvexPoly.
function line2(img, h, w, a, b) {
  const pt1 = {x: a.x, y: a.y}, pt2 = {x: b.x, y: b.y};
  if (!clipLine(w * XY_ONE, h * XY_ONE, pt1, pt2)) return;
  let dx = pt2.x - pt1.x, dy = pt2.y - pt1.y;
  const ax = Math.abs(dx), ay = Math.abs(dy);
  const swap = () => { let t = pt1.x; pt1.x = pt2.x; pt2.x = t; t = pt1.y; pt1.y = pt2.y; pt2.y = t; };
  const or1 = v => (v % 2 === 0 ? v + 1 : v);
  let xStep, yStep, ecount;
  if (ax > ay) {
    if (dx < 0) { dy = -dy; swap(); }
    xStep = XY_ONE; yStep = Math.trunc(dy * XY_ONE / or1(ax)); ecount = shr16(pt2.x - pt1.x);
  } else {
    if (dy < 0) { dx = -dx; swap(); }
    xStep = Math.trunc(dx * XY_ONE / or1(ay)); yStep = XY_ONE; ecount = shr16(pt2.y - pt1.y);
  }
  pt1.x += HALF; pt1.y += HALF;
  const put = (x, y) => { if (x >= 0 && x < w && y >= 0 && y < h) img[y * w + x] = 1; };
  put(shr16(pt2.x + HALF), shr16(pt2.y + HALF));
  if (ax > ay) {
    pt1.x = shr16(pt1.x);
    while (ecount >= 0) { put(pt1.x, shr16(pt1.y)); pt1.x++; pt1.y += yStep; ecount--; }
  } else {
    pt1.y = shr16(pt1.y);
    while (ecount >= 0) { put(shr16(pt1.x), pt1.y); pt1.x += xStep; pt1.y++; ecount--; }
  }
}

function hline(img, w, y, x1, x2) { for (let x = x1; x <= x2; x++) img[y * w + x] = 1; }

// cv2.fillPoly(img, [pts], 1) for one polygon with integer vertices (CollectPolyEdges + FillEdgeCollection).
function fillPoly(img, h, w, pts) {
  const edges = [], n = pts.length;
  let pt0 = {x: pts[n - 1][0] * XY_ONE, y: pts[n - 1][1]};
  for (let i = 0; i < n; i++) {
    const pt1 = {x: pts[i][0] * XY_ONE, y: pts[i][1]};
    const pt0c = {x: pt0.x, y: pt0.y}, pt1c = {x: pt1.x, y: pt1.y};
    const t0 = {x: shr16(pt0.x + HALF), y: pt0.y}, t1 = {x: shr16(pt1.x + HALF), y: pt1.y};
    line(img, h, w, t0, t1);
    if (t0.x < 0 || t0.x >= w || t1.x < 0 || t1.x >= w || t0.y < 0 || t0.y >= h || t1.y < 0 || t1.y >= h) {
      clipLine(w, h, t0, t1);
      if (t0.y !== t1.y) { pt0c.y = t0.y; pt1c.y = t1.y; }
    }
    pt0c.x = t0.x * XY_ONE; pt1c.x = t1.x * XY_ONE;
    if (pt0.y !== pt1.y) {
      const dx = Math.trunc((pt1c.x - pt0c.x) / (pt1c.y - pt0c.y));
      if (pt0.y < pt1.y) edges.push({y0: pt0.y, y1: pt1.y, x: pt0c.x + (pt0.y - pt0c.y) * dx, dx, next: null});
      else edges.push({y0: pt1.y, y1: pt0.y, x: pt1c.x + (pt1.y - pt1c.y) * dx, dx, next: null});
    }
    pt0 = pt1;
  }
  fillEdgeCollection(img, h, w, edges);
}

function fillEdgeCollection(img, h, w, edges) {
  const total = edges.length, delta = XY_ONE - 1;
  if (total < 2) return;
  let yMax = -Infinity, yMin = Infinity, xMax = -1, xMin = Infinity;
  for (const e1 of edges) {
    const x1 = e1.x + (e1.y1 - e1.y0) * e1.dx;
    yMin = Math.min(yMin, e1.y0); yMax = Math.max(yMax, e1.y1);
    xMin = Math.min(xMin, e1.x, x1); xMax = Math.max(xMax, e1.x, x1);
  }
  if (yMax < 0 || yMin >= h || xMax < 0 || xMin >= w * XY_ONE) return;
  edges.sort((a, b) => (a.y0 - b.y0) || (a.x - b.x) || (a.dx - b.dx));
  const tmp = {y0: Infinity, next: null, x: 0, dx: 0, y1: 0};
  edges.push({y0: Infinity, next: null, x: 0, dx: 0, y1: 0});
  let i = 0, e = edges[0];
  yMax = Math.min(yMax, h);
  for (let y = e.y0; y < yMax; y++) {
    let draw = 0, prelast = tmp, last = tmp.next, keep;
    const clipline = y < 0;
    while (last || e.y0 === y) {
      if (last && last.y1 === y) { prelast.next = last.next; last = last.next; continue; }
      keep = prelast;
      if (last && (e.y0 > y || last.x < e.x)) { prelast = last; last = last.next; }
      else if (i < total) { prelast.next = e; e.next = last; prelast = e; e = edges[++i]; }
      else break;
      if (draw) {
        if (!clipline) {
          let x1, x2;
          if (keep.x > prelast.x) { x1 = shr16(prelast.x + delta); x2 = shr16(keep.x); }
          else { x1 = shr16(keep.x + delta); x2 = shr16(prelast.x); }
          if (x1 < w && x2 >= 0) { if (x1 < 0) x1 = 0; if (x2 >= w) x2 = w - 1; hline(img, w, y, x1, x2); }
        }
        keep.x += keep.dx; prelast.x += prelast.dx;
      }
      draw ^= 1;
    }
    // keep the active list sorted by x (bubble sort, as in OpenCV)
    keep = null;
    do {
      prelast = tmp; last = tmp.next;
      let lastExchange = null;
      while (last !== keep && last && last.next) {
        const te = last.next;
        if (last.x > te.x) { prelast.next = te; last.next = te.next; te.next = last; prelast = te; lastExchange = prelast; }
        else { prelast = last; last = te; }
      }
      if (lastExchange === null) break;
      keep = lastExchange;
    } while (keep !== tmp.next && keep !== tmp);
  }
}

const SIN_TABLE = Float32Array.from([0.0000000,0.0174524,0.0348995,0.0523360,0.0697565,0.0871557,0.1045285,0.1218693,0.1391731,0.1564345,0.1736482,0.1908090,0.2079117,0.2249511,0.2419219,0.2588190,0.2756374,0.2923717,0.3090170,0.3255682,0.3420201,0.3583679,0.3746066,0.3907311,0.4067366,0.4226183,0.4383711,0.4539905,0.4694716,0.4848096,0.5000000,0.5150381,0.5299193,0.5446390,0.5591929,0.5735764,0.5877853,0.6018150,0.6156615,0.6293204,0.6427876,0.6560590,0.6691306,0.6819984,0.6946584,0.7071068,0.7193398,0.7313537,0.7431448,0.7547096,0.7660444,0.7771460,0.7880108,0.7986355,0.8090170,0.8191520,0.8290376,0.8386706,0.8480481,0.8571673,0.8660254,0.8746197,0.8829476,0.8910065,0.8987940,0.9063078,0.9135455,0.9205049,0.9271839,0.9335804,0.9396926,0.9455186,0.9510565,0.9563048,0.9612617,0.9659258,0.9702957,0.9743701,0.9781476,0.9816272,0.9848078,0.9876883,0.9902681,0.9925462,0.9945219,0.9961947,0.9975641,0.9986295,0.9993908,0.9998477,1.0000000,0.9998477,0.9993908,0.9986295,0.9975641,0.9961947,0.9945219,0.9925462,0.9902681,0.9876883,0.9848078,0.9816272,0.9781476,0.9743701,0.9702957,0.9659258,0.9612617,0.9563048,0.9510565,0.9455186,0.9396926,0.9335804,0.9271839,0.9205049,0.9135455,0.9063078,0.8987940,0.8910065,0.8829476,0.8746197,0.8660254,0.8571673,0.8480481,0.8386706,0.8290376,0.8191520,0.8090170,0.7986355,0.7880108,0.7771460,0.7660444,0.7547096,0.7431448,0.7313537,0.7193398,0.7071068,0.6946584,0.6819984,0.6691306,0.6560590,0.6427876,0.6293204,0.6156615,0.6018150,0.5877853,0.5735764,0.5591929,0.5446390,0.5299193,0.5150381,0.5000000,0.4848096,0.4694716,0.4539905,0.4383711,0.4226183,0.4067366,0.3907311,0.3746066,0.3583679,0.3420201,0.3255682,0.3090170,0.2923717,0.2756374,0.2588190,0.2419219,0.2249511,0.2079117,0.1908090,0.1736482,0.1564345,0.1391731,0.1218693,0.1045285,0.0871557,0.0697565,0.0523360,0.0348995,0.0174524,0.0000000,-0.0174524,-0.0348995,-0.0523360,-0.0697565,-0.0871557,-0.1045285,-0.1218693,-0.1391731,-0.1564345,-0.1736482,-0.1908090,-0.2079117,-0.2249511,-0.2419219,-0.2588190,-0.2756374,-0.2923717,-0.3090170,-0.3255682,-0.3420201,-0.3583679,-0.3746066,-0.3907311,-0.4067366,-0.4226183,-0.4383711,-0.4539905,-0.4694716,-0.4848096,-0.5000000,-0.5150381,-0.5299193,-0.5446390,-0.5591929,-0.5735764,-0.5877853,-0.6018150,-0.6156615,-0.6293204,-0.6427876,-0.6560590,-0.6691306,-0.6819984,-0.6946584,-0.7071068,-0.7193398,-0.7313537,-0.7431448,-0.7547096,-0.7660444,-0.7771460,-0.7880108,-0.7986355,-0.8090170,-0.8191520,-0.8290376,-0.8386706,-0.8480481,-0.8571673,-0.8660254,-0.8746197,-0.8829476,-0.8910065,-0.8987940,-0.9063078,-0.9135455,-0.9205049,-0.9271839,-0.9335804,-0.9396926,-0.9455186,-0.9510565,-0.9563048,-0.9612617,-0.9659258,-0.9702957,-0.9743701,-0.9781476,-0.9816272,-0.9848078,-0.9876883,-0.9902681,-0.9925462,-0.9945219,-0.9961947,-0.9975641,-0.9986295,-0.9993908,-0.9998477,-1.0000000,-0.9998477,-0.9993908,-0.9986295,-0.9975641,-0.9961947,-0.9945219,-0.9925462,-0.9902681,-0.9876883,-0.9848078,-0.9816272,-0.9781476,-0.9743701,-0.9702957,-0.9659258,-0.9612617,-0.9563048,-0.9510565,-0.9455186,-0.9396926,-0.9335804,-0.9271839,-0.9205049,-0.9135455,-0.9063078,-0.8987940,-0.8910065,-0.8829476,-0.8746197,-0.8660254,-0.8571673,-0.8480481,-0.8386706,-0.8290376,-0.8191520,-0.8090170,-0.7986355,-0.7880108,-0.7771460,-0.7660444,-0.7547096,-0.7431448,-0.7313537,-0.7193398,-0.7071068,-0.6946584,-0.6819984,-0.6691306,-0.6560590,-0.6427876,-0.6293204,-0.6156615,-0.6018150,-0.5877853,-0.5735764,-0.5591929,-0.5446390,-0.5299193,-0.5150381,-0.5000000,-0.4848096,-0.4694716,-0.4539905,-0.4383711,-0.4226183,-0.4067366,-0.3907311,-0.3746066,-0.3583679,-0.3420201,-0.3255682,-0.3090170,-0.2923717,-0.2756374,-0.2588190,-0.2419219,-0.2249511,-0.2079117,-0.1908090,-0.1736482,-0.1564345,-0.1391731,-0.1218693,-0.1045285,-0.0871557,-0.0697565,-0.0523360,-0.0348995,-0.0174524,-0.0000000,0.0174524,0.0348995,0.0523360,0.0697565,0.0871557,0.1045285,0.1218693,0.1391731,0.1564345,0.1736482,0.1908090,0.2079117,0.2249511,0.2419219,0.2588190,0.2756374,0.2923717,0.3090170,0.3255682,0.3420201,0.3583679,0.3746066,0.3907311,0.4067366,0.4226183,0.4383711,0.4539905,0.4694716,0.4848096,0.5000000,0.5150381,0.5299193,0.5446390,0.5591929,0.5735764,0.5877853,0.6018150,0.6156615,0.6293204,0.6427876,0.6560590,0.6691306,0.6819984,0.6946584,0.7071068,0.7193398,0.7313537,0.7431448,0.7547096,0.7660444,0.7771460,0.7880108,0.7986355,0.8090170,0.8191520,0.8290376,0.8386706,0.8480481,0.8571673,0.8660254,0.8746197,0.8829476,0.8910065,0.8987940,0.9063078,0.9135455,0.9205049,0.9271839,0.9335804,0.9396926,0.9455186,0.9510565,0.9563048,0.9612617,0.9659258,0.9702957,0.9743701,0.9781476,0.9816272,0.9848078,0.9876883,0.9902681,0.9925462,0.9945219,0.9961947,0.9975641,0.9986295,0.9993908,0.9998477,1.0000000]);

// FillConvexPoly(img, v, npts, color, 8, XY_SHIFT); v are {x, y} in 16.16 fixed point.
function fillConvexPoly(img, h, w, v) {
  const npts = v.length, delta = HALF, edge = [{}, {}];
  let imin = 0, edges = npts, xmin = v[0].x, xmax = v[0].x, ymin = v[0].y, ymax = v[0].y;
  let p0 = v[npts - 1];
  for (let i = 0; i < npts; i++) {
    const p = v[i];
    if (p.y < ymin) { ymin = p.y; imin = i; }
    ymax = Math.max(ymax, p.y); xmax = Math.max(xmax, p.x); xmin = Math.min(xmin, p.x);
    line2(img, h, w, p0, p);
    p0 = p;
  }
  xmin = shr16(xmin + delta); xmax = shr16(xmax + delta); ymin = shr16(ymin + delta); ymax = shr16(ymax + delta);
  if (npts < 3 || xmax < 0 || ymax < 0 || xmin >= w || ymin >= h) return;
  ymax = Math.min(ymax, h - 1);
  let y = ymin;
  edge[0] = {idx: imin, ye: y, di: 1, x: -XY_ONE, dx: 0};
  edge[1] = {idx: imin, ye: y, di: npts - 1, x: -XY_ONE, dx: 0};
  do {
    for (let i = 0; i < 2; i++) {
      if (y >= edge[i].ye) {
        let idx0 = edge[i].idx, idx = idx0 + edge[i].di;
        const di = edge[i].di;
        if (idx >= npts) idx -= npts;
        for (; edges-- > 0;) {
          const ty = shr16(v[idx].y + delta);
          if (ty > y) {
            const xs = v[idx0].x, xe = v[idx].x;
            edge[i].ye = ty;
            edge[i].dx = Math.trunc(((xe - xs) * 2 + (ty - y)) / (2 * (ty - y)));
            edge[i].x = xs; edge[i].idx = idx;
            break;
          }
          idx0 = idx; idx += di;
          if (idx >= npts) idx -= npts;
        }
      }
    }
    if (edges < 0) break;
    if (y >= 0) {
      let left = 0, right = 1;
      if (edge[0].x > edge[1].x) { left = 1; right = 0; }
      let xx1 = shr16(edge[left].x + HALF), xx2 = shr16(edge[right].x + HALF);
      if (xx2 >= 0 && xx1 < w) { if (xx1 < 0) xx1 = 0; if (xx2 >= w) xx2 = w - 1; hline(img, w, y, xx1, xx2); }
    }
    edge[0].x += edge[0].dx; edge[1].x += edge[1].dx;
  } while (++y <= ymax);
}

// cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, 1, -1): filled, axis-aligned ellipse with integer
// centre and half-axes (EllipseEx + ellipse2Poly + FillConvexPoly).
function fillEllipse(img, h, w, cx, cy, ax, ay) {
  const C = {x: cx * XY_ONE, y: cy * XY_ONE}, A = {w: Math.abs(ax) * XY_ONE, h: Math.abs(ay) * XY_ONE};
  let delta = shr16(Math.max(A.w, A.h) + HALF);
  delta = delta < 3 ? 90 : delta < 10 ? 30 : delta < 15 ? 18 : 5;
  const alpha = SIN_TABLE[450], beta = SIN_TABLE[0], v = [];
  let prev = null;
  for (let i = 0; i < 360 + delta; i += delta) {
    const angle = Math.min(i, 360);
    const x = A.w * SIN_TABLE[450 - angle], y = A.h * SIN_TABLE[angle];
    const px = C.x + x * alpha - y * beta, py = C.y + x * beta + y * alpha;
    const pt = {x: cvRound(px / XY_ONE) * XY_ONE, y: cvRound(py / XY_ONE) * XY_ONE};
    pt.x += cvRound(px - pt.x); pt.y += cvRound(py - pt.y);
    if (!prev || pt.x !== prev.x || pt.y !== prev.y) { v.push(pt); prev = pt; }
  }
  if (v.length <= 1) { v.length = 0; v.push(C, C); }
  fillConvexPoly(img, h, w, v);
}

// cv2.morphologyEx(src float32, MORPH_OPEN, disk(r)) evaluated on the rectangle [y0,y1) x [x0,x1) only.
// Pixels outside the image are ignored, as with OpenCV's default border. Returns a full-size
// array that is valid inside the rectangle. A disk is a stack of horizontal runs, so each
// distinct run length is handled with one sliding-window pass per row.
function morphOpenDisk(src, h, w, r, y0, y1, x0, x1) {
  const byHalf = new Map();                                 // skimage.morphology.disk: x^2 + y^2 <= r^2
  for (let dy = -r; dy <= r; dy++) {
    const hw = Math.floor(Math.sqrt(r * r - dy * dy));
    if (!byHalf.has(hw)) byHalf.set(hw, []);
    byHalf.get(hw).push(dy);
  }
  // extreme of `inp` over the disk, for the pixels of [ya,yb) x [xa,xb)
  const extreme = (inp, isMin, ya, yb, xa, xb) => {
    const out = new Float32Array(h * w), W = xb - xa;
    for (let y = ya; y < yb; y++) out.fill(isMin ? Infinity : -Infinity, y * w + xa, y * w + xb);
    const ra = Math.max(0, ya - r), rb = Math.min(h, yb + r);
    const tmp = new Float32Array((rb - ra) * W), dq = new Int32Array(w);
    for (const [hw, dys] of byHalf) {
      for (let row = ra; row < rb; row++) {                 // sliding window [x-hw, x+hw] clipped to the image
        const o = row * w, lo = Math.max(0, xa - hw), hi = Math.min(w - 1, xb - 1 + hw);
        let head = 0, tail = 0, next = lo;
        for (let x = xa; x < xb; x++) {
          const right = Math.min(hi, x + hw), left = x - hw;
          for (; next <= right; next++) {
            const v = inp[o + next];
            while (tail > head && (isMin ? inp[o + dq[tail - 1]] >= v : inp[o + dq[tail - 1]] <= v)) tail--;
            dq[tail++] = next;
          }
          while (dq[head] < left) head++;
          tmp[(row - ra) * W + x - xa] = inp[o + dq[head]];
        }
      }
      for (const dy of dys) for (let y = ya; y < yb; y++) {
        const row = y + dy;
        if (row < 0 || row >= h) continue;
        const t = (row - ra) * W, o = y * w + xa;
        if (isMin) { for (let k = 0; k < W; k++) if (tmp[t + k] < out[o + k]) out[o + k] = tmp[t + k]; }
        else { for (let k = 0; k < W; k++) if (tmp[t + k] > out[o + k]) out[o + k] = tmp[t + k]; }
      }
    }
    return out;
  };
  // the dilation at a pixel needs the erosion within r of it
  const ey0 = Math.max(0, y0 - r), ey1 = Math.min(h, y1 + r), ex0 = Math.max(0, x0 - r), ex1 = Math.min(w, x1 + r);
  const er = extreme(src, true, ey0, ey1, ex0, ex1);
  // the dilation only ever reads the eroded rectangle: mark everything else as "outside"
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (y < ey0 || y >= ey1 || x < ex0 || x >= ex1) er[y * w + x] = -Infinity;
  return extreme(er, false, y0, y1, x0, x1);
}

IWSP.cv = {cvRound, grayFromRGBA, resizeArea, medianBlur5, remapLinear, gaussianBlur5, clipLine, line, fillPoly,
           fillEllipse, morphOpenDisk};
})();
