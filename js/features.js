// JavaScript port of features.py: the 56 named TI-RADS measurements.
//
//   extract(img, mask, h, w, keep) -> { feature_name: number }
//     img  : Float32Array in [0, 1], B-mode grayscale, depth increases downward
//     mask : Uint8Array, 1 inside the nodule
//
// The code follows features.py section by section and keeps its variable names, so the two files
// can be read side by side. Where numpy works in float32 the rounding is reproduced with
// fr = Math.fround; thresholds compared against float32 images are rounded to float32 first,
// because that is what numpy does with a Python float. The ten measurements features.py computes
// and then discards (_UNUSED) are not computed here.
(function () {
'use strict';
const IWSP = window.IWSP = window.IWSP || {};
const fr = Math.fround, F32 = Float32Array, F64 = Float64Array;
const int = Math.trunc;

function and(a, b) { const o = new Uint8Array(a.length); for (let i = 0; i < o.length; i++) o[i] = a[i] & b[i]; return o; }
function andNot(a, b) { const o = new Uint8Array(a.length); for (let i = 0; i < o.length; i++) o[i] = a[i] && !b[i] ? 1 : 0; return o; }
// numpy-exact sums, means and standard deviation (see ndimage.js)
const mean64 = v => IWSP.nd.mean64(F64.from(v));
const mean32 = v => IWSP.nd.mean32(v instanceof F32 ? v : F32.from(v));
const std32 = v => IWSP.nd.std32(v instanceof F32 ? v : F32.from(v));

// robust z-score of a float32 response map against the values in `sel`:
// med = median, s = 1.4826 * MAD + 1e-6, z = (map - med) / s   (all float32, as in numpy)
function robustZ(map, sel) {
  const nd = IWSP.nd, vals = nd.pick(map, sel);
  const med = nd.median32(vals), dev = new F32(vals.length);
  for (let i = 0; i < vals.length; i++) dev[i] = Math.abs(fr(vals[i] - med));
  const s = fr(nd.median32(dev) * 1.4826 + 1e-6), z = new F32(map.length);
  for (let i = 0; i < map.length; i++) z[i] = fr(map[i] - med) / s;
  return z;
}
// -gaussian_laplace(img, sg) * sg * sg  (or +, with sign = 1), float32 throughout
function scaledLoG(img, h, w, sg, sign, scalePower) {
  const l = IWSP.nd.gaussianLaplace(img, h, w, sg), s = fr(sg);
  for (let i = 0; i < l.length; i++) {
    let v = sign < 0 ? -l[i] : l[i];
    for (let k = 0; k < scalePower; k++) v = fr(v * s);
    l[i] = v;
  }
  return l;
}

// _strip_ratio: mean of img[r0:r1, c0:c1] divided by the mean of equal strips to its left and right
function stripRatio(img, h, w, r0, r1, c0, c1, gap) {
  r0 = Math.max(0, r0); r1 = Math.min(h, r1);
  if (r1 - r0 < 2 || c1 - c0 < 1) return NaN;
  const wd = c1 - c0;
  const block = (a, b) => { const v = []; for (let y = r0; y < r1; y++) for (let x = a; x < b; x++) v.push(img[y * w + x]); return v; };
  const centre = block(Math.max(0, c0), Math.min(w, c1));
  const sides = block(Math.max(0, c0 - gap - wd), Math.max(0, c0 - gap)).concat(block(Math.min(w, c1 + gap), Math.min(w, c1 + gap + wd)));
  if (centre.length === 0 || sides.length === 0) return NaN;
  return fr(mean32(centre) / fr(mean32(sides) + fr(1e-6)));
}

function extract(img, mask, h, w, keep) {
  const nd = IWSP.nd, sk = IWSP.sk, cv = IWSP.cv, f = {}, N = h * w;
  const sm = nd.medianFilter5(img, h, w);                                  // speckle suppression
  const props = sk.regionprops(mask, h, w);
  const [minr, minc, maxr, maxc] = props.bbox;
  const nh = maxr - minr, nw = maxc - minc;
  const diam = Math.max(props.equivalentDiameter, 4.0);
  const area = props.area;
  const hull = sk.convexHullImage(mask, h, w);

  // ------------------------------------------------ shape
  f.shape_ap_over_trans = nh / Math.max(nw, 1);
  f.shape_int_ap_robust = f.shape_ap_over_trans;
  f.shape_orientation_deg = 90.0 - Math.abs(props.orientation * (180 / Math.PI));   // 0 = horizontal

  // ------------------------------------------------ reference tissue
  const ringW = Math.max(3, int(0.2 * diam));
  const ring = andNot(nd.dilate(mask, h, w, ringW), mask);
  let refMed = nd.median32(nd.pick(sm, ring));
  if (Number.isNaN(refMed)) { const notMask = new Uint8Array(N); for (let i = 0; i < N; i++) notMask[i] = mask[i] ? 0 : 1; refMed = nd.median32(nd.pick(sm, notMask)); }
  if (Number.isNaN(refMed)) refMed = 0.5;
  const ref = refMed + 1e-6;

  // ------------------------------------------------ composition
  const inside = nd.pick(sm, mask);
  const fluidThr = fr(Math.max(0.06, 0.35 * ref));
  const fluid = new Uint8Array(N);
  let fluidN = 0;
  for (let i = 0; i < N; i++) if (mask[i] && sm[i] < fluidThr) { fluid[i] = 1; fluidN++; }
  f.comp_fluid_fraction = fluidN / area;
  const fl = nd.label(fluid, h, w);
  let smallN = 0, smallSum = 0;
  for (let k = 1; k <= fl.n; k++) if (fl.sizes[k] >= 3 && fl.sizes[k] < 0.02 * area) { smallN++; smallSum += fl.sizes[k]; }
  f.comp_microcyst_density = fl.n ? smallN / (area / 100.0) : 0.0;
  f.comp_microcyst_area_frac = fl.n ? smallSum / area : 0.0;

  const fracBelow = thr => { let c = 0; for (let i = 0; i < inside.length; i++) if (inside[i] < thr) c++; return c / inside.length; };
  f.comp_fluid_frac_strict = fracBelow(fr(Math.max(0.04, 0.20 * ref)));
  f.comp_fluid_frac_loose = fracBelow(fr(Math.max(0.08, 0.50 * ref)));
  // mirror of the bright-dot test: cystic spaces are dark blobs that beat the speckle downward
  let dcount = 0, dscale = 0, dspread = 0, bestResp = 0;
  const posY = [], posX = [];
  if (inside.length >= 20) {
    for (const sg of [1.5, 3.0, 6.0]) {
      const dl = scaledLoG(sm, h, w, sg, 1, 2);                             // positive on dark blobs
      const dz = robustZ(dl, mask), mx = nd.maximumFilter(dl, h, w, 5);
      let zmax = -Infinity, any = false;
      for (let i = 0; i < N; i++) if (mask[i] && dl[i] === mx[i] && dz[i] > 6.0) {
        dcount++; any = true; if (dz[i] > zmax) zmax = dz[i];
        posY.push(int(i / w)); posX.push(i % w);
      }
      if (any && zmax > bestResp) { bestResp = zmax; dscale = sg; }
    }
  }
  if (posY.length >= 3) {
    const variance = v => { const m = mean64(v); return mean64(v.map(x => (x - m) * (x - m))); };
    dspread = Math.sqrt(variance(posY) + variance(posX)) / diam;
  }
  f.comp_z_dark_count = Math.log1p(dcount);
  f.comp_z_dark_scale = dscale;
  f.comp_z_dark_spread = Math.min(2.0, Math.max(0.0, dspread));
  f.comp_dark_vs_own_median = fracBelow(fr(fr(0.4) * fr(nd.median32(inside) + fr(1e-3))));

  // ------------------------------------------------ echogenicity
  const solid = andNot(mask, fluid);
  const solidVals = nd.count(solid) > 20 ? nd.pick(sm, solid) : inside;
  const sMed = nd.median32(solidVals);
  f.echo_solid_ratio_ring = sMed / ref;
  const top = Math.max(0, minr - int(0.6 * nh));
  const antVals = [];
  for (let y = top; y < Math.min(h, Math.max(top + 1, minr - 2)); y++) for (let x = minc; x < maxc; x++) antVals.push(sm[y * w + x]);
  const antMed = (antVals.length ? nd.median32(antVals) : ref) + 1e-6;
  f.echo_solid_ratio_anterior = sMed / antMed;
  f.echo_solid_cv = fr(std32(solidVals) / fr(mean32(solidVals) + fr(1e-6)));
  f.echo_solid_abs = sMed;
  f.echo_ref_abs = ref;
  f.echo_solid_minus_ref = sMed - ref;
  const lw = Math.max(6, int(0.4 * nw));
  const band0 = minr + Math.floor(nh / 4), band1 = Math.min(h, Math.max(minr + Math.floor(nh / 4) + 1, maxr - Math.floor(nh / 4)));
  const dil3 = nd.dilate(mask, h, w, 3), thr003 = fr(0.03);
  const side = (xa, xb) => { const v = []; for (let y = band0; y < band1; y++) for (let x = xa; x < xb; x++) if (!dil3[y * w + x]) v.push(sm[y * w + x]); return v; };
  const lft = side(Math.max(0, minc - 3 - lw), Math.max(0, minc - 3)), rgt = side(Math.min(w, maxc + 3), Math.min(w, maxc + 3 + lw));
  const lftB = lft.filter(v => v > thr003), rgtB = rgt.filter(v => v > thr003), lat = lftB.concat(rgtB);
  f.echo_solid_ratio_lateral = sMed / ((lat.length ? nd.median32(lat) : ref) + 1e-6);
  const ml = nd.median32(lftB), mr = nd.median32(rgtB);
  f.echo_lateral_asymmetry = (Number.isFinite(ml) && Number.isFinite(mr)) ? Math.abs(Math.log((ml + 1e-3) / (mr + 1e-3))) : 0.0;

  // ------------------------------------------------ texture (co-occurrence) inside nodule
  if (area > 50) {
    const q = new Uint8Array(nh * nw);
    for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) {
      const i = (y + minr) * w + x + minc;
      q[y * nw + x] = mask[i] ? int(Math.min(15, Math.max(0, img[i] * 16))) + 1 : 0;      // 0 = outside nodule
    }
    const P = sk.graycomatrix(q, nh, nw, [2, 6], 17);
    const avg = d => {                                                    // drop level 0, normalise, mean over angles
      const g = new F64(256);
      for (const Pa of P[d]) {
        let s = 0;
        for (let i = 1; i < 17; i++) for (let j = 1; j < 17; j++) s += Pa[i * 17 + j];
        s = Math.max(s, 1);
        for (let i = 1; i < 17; i++) for (let j = 1; j < 17; j++) g[(i - 1) * 16 + j - 1] += Pa[i * 17 + j] / s;
      }
      for (let i = 0; i < 256; i++) g[i] /= 2;
      return g;
    };
    const g2 = avg(0), g6 = avg(1);
    const total = fn => { const t = new F64(256); for (let i = 0; i < 16; i++) for (let j = 0; j < 16; j++) t[i * 16 + j] = fn(i, j, i * 16 + j); return nd.sum64(t); };
    f.tex_contrast = total((i, j, k) => g2[k] * ((i - j) * (i - j)));
    f.tex_homogeneity = total((i, j, k) => g2[k] / (1 + (i - j) * (i - j)));
    const mi = total((i, j, k) => g6[k] * i), mj = total((i, j, k) => g6[k] * j);
    const si = Math.sqrt(total((i, j, k) => g6[k] * ((i - mi) * (i - mi)))), sj = Math.sqrt(total((i, j, k) => g6[k] * ((j - mj) * (j - mj))));
    f.tex_correlation_far = total((i, j, k) => g6[k] * (i - mi) * (j - mj)) / (si * sj + 1e-9);
  } else { f.tex_contrast = 0.0; f.tex_homogeneity = 1.0; f.tex_correlation_far = 0.0; }

  // taller-than-wide read off the brightness outline at several thresholds, not off the mask
  const ratios = [];
  const zoneS = nd.dilate(mask, h, w, Math.max(2, int(0.1 * diam)));
  const cyR = nd.rint(props.centroid[0]), cxR = nd.rint(props.centroid[1]);
  for (const frac of [0.2, 0.35, 0.5, 0.65, 0.8]) {
    const t = fr(sMed + frac * (ref - sMed)), bin = new Uint8Array(N), darker = ref > sMed;
    for (let i = 0; i < N; i++) bin[i] = zoneS[i] && (darker ? sm[i] < t : sm[i] > t) ? 1 : 0;
    const lb = nd.label(nd.opening(bin, h, w), h, w);
    if (lb.n === 0) continue;
    let k = lb.lab[cyR * w + cxR];
    if (k === 0) { k = 1; for (let c = 2; c <= lb.n; c++) if (lb.sizes[c] > lb.sizes[k]) k = c; }
    if (lb.sizes[k] < 20) continue;
    let y0 = h, y1 = -1, x0 = w, x1 = -1;
    for (let i = 0; i < N; i++) if (lb.lab[i] === k) { const y = int(i / w), x = i % w; if (y < y0) y0 = y; if (y > y1) y1 = y; if (x < x0) x0 = x; if (x > x1) x1 = x; }
    ratios.push((y1 - y0 + 1) / Math.max(1.0, x1 - x0 + 1.0));
  }
  if (ratios.length) f.shape_int_ap_robust = Math.min(5.0, Math.max(0.0, nd.median64(ratios)));

  // ------------------------------------------------ margin: outline geometry
  f.margin_solidity = area / nd.count(hull);
  f.margin_perimeter_ratio = sk.perimeter(mask, h, w) / (sk.perimeter(hull, h, w) + 1e-6);
  const contours = sk.findContours(mask, h, w, props.bbox);
  let cnt = [];
  for (const c of contours) if (c.length > cnt.length) cnt = c;            // max(..., key=len): first longest
  const [cy, cx] = props.centroid;
  if (cnt.length >= 16) {
    const ang = new F64(cnt.length), rad = new F64(cnt.length);
    for (let i = 0; i < cnt.length; i++) { ang[i] = Math.atan2(cnt[i][0] - cy, cnt[i][1] - cx); rad[i] = Math.hypot(cnt[i][0] - cy, cnt[i][1] - cx); }
    const grid = new F64(128);
    for (let i = 0; i < 128; i++) grid[i] = i * (2 * Math.PI / 128) + (-Math.PI);
    const r = nd.interpPeriodic(grid, ang, rad, 2 * Math.PI), rm = mean64(r), centred = new F64(128);
    for (let i = 0; i < 128; i++) centred[i] = r[i] - rm;
    const spec = nd.rfftPower(centred);
    f.margin_radial_hf_energy = nd.sum64(spec.subarray(7)) / (nd.sum64(spec.subarray(1)) + 1e-9);
    // indentations: dips of the radius below its smooth envelope
    const env = nd.maximumFilter1dWrap(r, 15);
    let ndips = 0, prev = false;
    for (let i = 0; i < 128; i++) { const dip = (env[i] - r[i]) > 0.04 * diam; if (dip && !prev) ndips++; prev = dip; }
    f.margin_lobulation_count = ndips;
  } else { f.margin_radial_hf_energy = 0.0; f.margin_lobulation_count = 0.0; }

  // ------------------------------------------------ margin: edge quality
  const g15 = nd.gradient(nd.gaussianFilter(img, h, w, 1.5, F32), h, w, F32);
  const boundary = andNot(mask, nd.erode(mask, h, w, 2));
  const bvals = [];
  for (let i = 0; i < N; i++) if (boundary[i]) bvals.push(fr(Math.sqrt(g15.gx[i] * g15.gx[i] + g15.gy[i] * g15.gy[i])));
  f.margin_edge_sharpness_cv = bvals.length ? fr(std32(bvals) / fr(mean32(bvals) + fr(1e-6))) : 0.0;
  const band = Math.max(2, int(0.08 * diam));
  const inner = andNot(mask, nd.erode(mask, h, w, band));
  const outer = andNot(nd.dilate(mask, h, w, band), mask);
  const pad = 3 * band + 6;
  const R0 = Math.max(0, minr - pad), R1 = Math.min(h, maxr + pad), C0 = Math.max(0, minc - pad), C1 = Math.min(w, maxc + pad);
  // theta is only defined on the window [R0:R1, C0:C1]; sector16/sector36 are -1 elsewhere
  const sector16 = new Int8Array(N).fill(-1), sector36 = new Int8Array(N).fill(-1);
  for (let y = R0; y < R1; y++) for (let x = C0; x < C1; x++) {
    const u = (Math.atan2(y - cy, x - cx) + Math.PI) / (2 * Math.PI);
    sector16[y * w + x] = int(u * 16) % 16; sector36[y * w + x] = int(u * 36) % 36;
  }
  const contrasts = [], ref32 = fr(ref);
  for (let s = 0; s < 16; s++) {
    const a = [], b = [];
    for (let i = 0; i < N; i++) if (sector16[i] === s) { if (inner[i]) a.push(sm[i]); if (outer[i]) b.push(sm[i]); }
    if (a.length > 5 && b.length > 5) contrasts.push(fr(Math.abs(fr(nd.median32(b) - nd.median32(a))) / ref32));
  }
  f.margin_min_sector_contrast = contrasts.length ? Math.min(...contrasts) : 0.0;
  f.margin_lowcontrast_frac = contrasts.length ? contrasts.filter(v => v < fr(0.08)).length / contrasts.length : 1.0;

  // ------------------------------------------------ margin: gradient-normal alignment
  const notMask = new Uint8Array(N); for (let i = 0; i < N; i++) notMask[i] = mask[i] ? 0 : 1;
  const dIn = nd.edt(mask, h, w), dOut = nd.edt(notMask, h, w), sdist = new F64(N);
  for (let i = 0; i < N; i++) sdist[i] = mask[i] ? -dIn[i] : dOut[i];
  const nrm = nd.gradient(nd.gaussianFilter(sdist, h, w, 1.0, F64), h, w, F64);
  const sm2 = nd.gaussianFilter(img, h, w, 2.0, F32);
  const g2 = nd.gradient(sm2, h, w, F32);
  const nbW = Math.max(2, int(0.04 * diam)) + 2;
  const allW = [], allCW = [], secW = [], secCW = [];
  for (let q = 0; q < 16; q++) { secW.push([]); secCW.push([]); }
  for (let i = 0; i < N; i++) {
    if (!(Math.abs(sdist[i]) <= nbW)) continue;
    const gm = fr(Math.sqrt(g2.gx[i] * g2.gx[i] + g2.gy[i] * g2.gy[i]));
    const cosv = Math.abs(g2.gx[i] * nrm.gx[i] + g2.gy[i] * nrm.gy[i]) / (gm * Math.hypot(nrm.gx[i], nrm.gy[i]) + 1e-9);
    allW.push(gm); allCW.push(cosv * gm);
    const q = sector16[i];
    if (q >= 0) { secW[q].push(gm); secCW[q].push(cosv * gm); }
  }
  const wsum = nd.sum32(F32.from(allW));
  f.margin_normal_coherence = wsum > 0 ? nd.sum64(F64.from(allCW)) / wsum : 0.5;
  const coh = [];
  for (let q = 0; q < 16; q++) {
    const den = nd.sum32(F32.from(secW[q]));
    if (secW[q].length > 5 && den > 0) coh.push(nd.sum64(F64.from(secCW[q])) / den);
  }
  f.margin_coherent_sector_frac = coh.length ? coh.filter(v => v > 0.75).length / coh.length : 0.0;

  // ------------------------------------------------ margin: intensity-defined outline
  const wband = Math.max(3, int(0.15 * diam));
  const zone = nd.dilate(mask, h, w, wband), core = nd.erode(mask, h, w, wband);
  const mid = fr(0.5 * (sMed + ref)), darkerNodule = sMed < ref;
  let seg = new Uint8Array(N);
  for (let i = 0; i < N; i++) seg[i] = core[i] || (zone[i] && (darkerNodule ? sm2[i] < mid : sm2[i] > mid)) ? 1 : 0;
  seg = nd.opening(seg, h, w);
  const sl = nd.label(seg, h, w);
  if (sl.n) {
    // largest component by overlap with the mask (or by size if none overlaps)
    const inMask = new Int32Array(sl.n + 1);
    let anyIn = false, maxIn = 0;
    for (let i = 0; i < N; i++) if (mask[i] && sl.lab[i]) { inMask[sl.lab[i]]++; anyIn = true; if (sl.lab[i] > maxIn) maxIn = sl.lab[i]; }
    let k = 1;
    if (anyIn) { for (let c = 2; c <= maxIn; c++) if (inMask[c] > inMask[k]) k = c; }
    else { for (let c = 2; c <= sl.n; c++) if (sl.sizes[c] > sl.sizes[k]) k = c; }
    const only = new Uint8Array(N);
    for (let i = 0; i < N; i++) only[i] = sl.lab[i] === k ? 1 : 0;
    seg = nd.fillHoles(only, h, w);
  }
  const segN = nd.count(seg);
  if (segN > 30) {
    const sh = sk.convexHullImage(seg, h, w);
    f.margin_int_solidity = segN / nd.count(sh);
    f.margin_int_perimeter_ratio = sk.perimeter(seg, h, w) / (sk.perimeter(sh, h, w) + 1e-6);
  } else { f.margin_int_solidity = 1.0; f.margin_int_perimeter_ratio = 1.0; }
  let bzN = 0, bzDiff = 0;
  for (let i = 0; i < N; i++) if (zone[i] && !core[i]) { bzN++; if (seg[i] !== mask[i]) bzDiff++; }
  f.margin_int_mismatch = bzN ? bzDiff / bzN : 0.0;

  // ------------------------------------------------ echogenic foci
  const g = nd.gaussianFilter(img, h, w, 1.0, F32);
  const radTh = Math.max(2, int(0.06 * diam));
  const search = nd.dilate(mask, h, w, 2);
  const sy0 = Math.max(0, minr - 2), sy1 = Math.min(h, maxr + 2), sx0 = Math.max(0, minc - 2), sx1 = Math.min(w, maxc + 2);
  const opened = cv.morphOpenDisk(g, h, w, radTh, sy0, sy1, sx0, sx1);     // only needed where `search` is set
  const thr010 = fr(0.10), thrBright = fr(1.3 * ref);
  let bright = new Uint8Array(N);
  for (let y = sy0; y < sy1; y++) for (let x = sx0; x < sx1; x++) {
    const i = y * w + x;
    if (search[i] && fr(g[i] - opened[i]) > thr010 && g[i] > thrBright) bright[i] = 1;
  }
  let bl = nd.label(bright, h, w);
  if (bl.n) {                                                             // drop specks under 3 px
    for (let i = 0; i < N; i++) if (bright[i] && bl.sizes[bl.lab[i]] < 3) bright[i] = 0;
    bl = nd.label(bright, h, w);
  }
  const edgeBand = andNot(search, nd.erode(mask, h, w, Math.max(2, int(0.1 * diam))));
  let smallFoci = 0, largeFoci = 0;
  const shadow = [], comet = [];
  if (bl.n) {
    const bb = [];
    for (let k = 0; k <= bl.n; k++) bb.push([h, -1, w, -1]);
    for (let i = 0; i < N; i++) if (bl.lab[i]) { const b = bb[bl.lab[i]], y = int(i / w), x = i % w; if (y < b[0]) b[0] = y; if (y > b[1]) b[1] = y; if (x < b[2]) b[2] = x; if (x > b[3]) b[3] = x; }
    for (let k = 1; k <= bl.n; k++) {
      const d = 2 * Math.sqrt(bl.sizes[k] / Math.PI);
      if (d < 0.08 * diam) smallFoci++; else largeFoci++;
      const r1 = bb[k][1] + 1, c0 = bb[k][2], c1 = bb[k][3] + 1, ch = bb[k][1] + 1 - bb[k][0];
      comet.push(stripRatio(g, h, w, r1, r1 + Math.max(3, 2 * ch), c0, c1, 2));
      shadow.push(stripRatio(g, h, w, r1 + 2, r1 + 2 + Math.max(6, 4 * ch), c0, c1, 3));
    }
  }
  const brightN = nd.count(bright);
  f.foci_count_small = smallFoci;
  f.foci_count_large = largeFoci;
  f.foci_area_frac = brightN / area;
  const rimHits = new Set();
  let periN = 0;
  for (let i = 0; i < N; i++) if (bright[i] && edgeBand[i]) { periN++; if (sector36[i] >= 0) rimHits.add(sector36[i]); }
  f.foci_rim_coverage = rimHits.size / 36.0;
  f.foci_peripheral_share = brightN ? periN / brightN : 0.0;
  const finite = v => v.filter(Number.isFinite);
  f.foci_min_shadow_ratio = finite(shadow).length ? Math.min(...finite(shadow)) : 1.0;
  f.foci_max_comet_ratio = finite(comet).length ? Math.max(...finite(comet)) : 1.0;

  // ------------------------------------------------ continuous punctate-focus strength
  let interior = nd.erode(mask, h, w, Math.max(2, int(0.06 * diam)));
  if (nd.count(interior) < 20) interior = mask;
  const uni = nd.uniformFilter(g, h, w, 9, F32), gmax = nd.maximumFilter(g, h, w, 5), thr012 = fr(0.12);
  let peaks = 0;
  for (let i = 0; i < N; i++) if (interior[i] && g[i] === gmax[i] && fr(g[i] - uni[i]) > thr012) peaks++;
  f.foci_peak_density = peaks / (nd.count(interior) / 1000.0);

  let solidInt = andNot(interior, fluid);
  if (nd.count(solidInt) < 20) solidInt = interior;
  const solidIntN = nd.count(solidInt);
  const lf = scaledLoG(img, h, w, 1.0, -1, 0), lfmax = nd.maximumFilter(lf, h, w, 5), thrPk = fr(0.15 * ref);
  let pk = 0;
  for (let i = 0; i < N; i++) if (solidInt[i] && lf[i] === lfmax[i] && lf[i] > thrPk) pk++;
  f.foci_solid_peak_count = Math.log1p(pk);

  // a punctate focus is a dot that beats the local speckle, so judge it in units of speckle
  const z = robustZ(lf, solidInt);
  let zpk = 0, zmaxS = -Infinity;
  for (let i = 0; i < N; i++) if (solidInt[i]) { if (z[i] > zmaxS) zmaxS = z[i]; if (lf[i] === lfmax[i] && z[i] > 6.0) zpk++; }
  f.foci_punctate_z_count = Math.log1p(zpk);
  f.foci_punctate_z_density = zpk / (solidIntN / 1000.0);
  f.foci_punctate_z_max = Math.min(40.0, Math.max(0.0, zmaxS));

  // the same speckle-relative test at coarser scales separates macrocalcifications from dots
  const zCounts = [];
  let zBest = 0.0, zBestSigma = 0.0;
  for (const sg of [2.0, 3.5]) {
    const l = scaledLoG(img, h, w, sg, -1, 2), zz = robustZ(l, solidInt), lmx = nd.maximumFilter(l, h, w, 7);
    let c = 0, pkv = -Infinity;
    for (let i = 0; i < N; i++) if (solidInt[i]) { if (zz[i] > pkv) pkv = zz[i]; if (l[i] === lmx[i] && zz[i] > 6.0) c++; }
    zCounts.push(c);
    if (pkv > zBest) { zBest = pkv; zBestSigma = sg; }
  }
  f.foci_z_count_mid = zCounts[0]; f.foci_z_count_large = zCounts[1];
  f.foci_z_max_scale = zBest > 6.0 ? zBestSigma : 0.0;

  // ------------------------------------------------ scale-space bright blobs
  const pad2 = 14;
  const q0 = Math.max(0, minr - pad2), q1 = Math.min(h, maxr + pad2), p0 = Math.max(0, minc - pad2), p1 = Math.min(w, maxc + pad2);
  const bh = q1 - q0, bw = p1 - p0, sub = new F32(bh * bw);
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) sub[y * bw + x] = img[(y + q0) * w + x + p0];
  const sigmas = [1.0, 2.0, 3.5, 6.0], best = new F32(bh * bw).fill(-Infinity), arg = new Uint8Array(bh * bw);
  sigmas.forEach((sg, k) => {
    const l = scaledLoG(sub, bh, bw, sg, -1, 2);
    for (let i = 0; i < l.length; i++) if (l[i] > best[i]) { best[i] = l[i]; arg[i] = k; }
  });
  const bmax = nd.maximumFilter(best, bh, bw, 7), thrBlob = fr(0.25 * ref);
  let tiny = 0, medium = 0, large = 0, strongest = -1, strongestVal = -Infinity;
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
    const i = y * bw + x;
    if (!(best[i] === bmax[i] && best[i] > thrBlob && search[(y + q0) * w + x + p0])) continue;
    if (arg[i] === 0) tiny++; else if (arg[i] === 1) medium++; else large++;
    if (best[i] > strongestVal) { strongestVal = best[i]; strongest = i; }
  }
  f.foci_blobs_tiny = tiny; f.foci_blobs_medium = medium; f.foci_blobs_large = large;
  f.foci_strongest_blob_scale = strongest >= 0 ? sigmas[arg[strongest]] : 0.0;

  // ------------------------------------------------ focal posterior shadow
  const pr0 = maxr + 2, pr1 = Math.min(h, maxr + 2 + Math.max(8, Math.floor(nh / 2)));
  if (pr1 - pr0 >= 3 && nw >= 10) {
    const colm = new F32(nw);                                             // column means, accumulated in float32
    for (let x = 0; x < nw; x++) { let s = 0; for (let y = pr0; y < pr1; y++) s = fr(s + sm[y * w + x + minc]); colm[x] = s / (pr1 - pr0); }
    const kw = Math.max(2, int(0.1 * nw));
    const filt = nd.uniformFilter1d(colm, 1, nw, 1, kw, F32);
    const strips = filt.subarray(kw >> 1, nw - (kw >> 1));
    if (strips.length) {
      let mn = Infinity; for (const v of strips) if (v < mn) mn = v;
      f.post_local_shadow = fr(mn / fr(nd.median32(strips) + fr(1e-4)));
    } else f.post_local_shadow = 1.0;
    if (strongest >= 0) {
      const xb = (strongest % bw) + p0 - minc;
      const cLo = Math.max(0, xb - kw), cHi = Math.min(nw, xb + kw + 1);
      if (cHi > cLo && nw - (cHi - cLo) > 0) {
        const under = colm.slice(cLo, cHi), rest = new F32(nw - (cHi - cLo));
        rest.set(colm.subarray(0, cLo)); rest.set(colm.subarray(cHi), cLo);
        f.post_shadow_under_brightest = fr(mean32(under) / fr(mean32(rest) + fr(1e-4)));
      } else f.post_shadow_under_brightest = 1.0;
    } else f.post_shadow_under_brightest = 1.0;
  } else { f.post_local_shadow = 1.0; f.post_shadow_under_brightest = 1.0; }

  if (keep) Object.assign(keep, {sm, props, diam, ring, ringW, ref, fluid, solidInt, lf, lfmax, z});
  return f;
}

// features.py returns its measurements in this order (the order the page lists them in).
const ORDER = ['shape_ap_over_trans', 'shape_int_ap_robust', 'shape_orientation_deg', 'comp_fluid_fraction',
  'comp_microcyst_density', 'comp_microcyst_area_frac', 'comp_fluid_frac_strict', 'comp_fluid_frac_loose',
  'comp_z_dark_count', 'comp_z_dark_scale', 'comp_z_dark_spread', 'comp_dark_vs_own_median', 'echo_solid_ratio_ring',
  'echo_solid_ratio_anterior', 'echo_solid_cv', 'echo_solid_abs', 'echo_ref_abs', 'echo_solid_minus_ref',
  'echo_solid_ratio_lateral', 'echo_lateral_asymmetry', 'tex_contrast', 'tex_homogeneity', 'tex_correlation_far',
  'margin_solidity', 'margin_perimeter_ratio', 'margin_radial_hf_energy', 'margin_lobulation_count',
  'margin_edge_sharpness_cv', 'margin_min_sector_contrast', 'margin_lowcontrast_frac', 'margin_normal_coherence',
  'margin_coherent_sector_frac', 'margin_int_solidity', 'margin_int_perimeter_ratio', 'margin_int_mismatch',
  'foci_count_small', 'foci_count_large', 'foci_area_frac', 'foci_rim_coverage', 'foci_peripheral_share',
  'foci_min_shadow_ratio', 'foci_max_comet_ratio', 'foci_peak_density', 'foci_solid_peak_count',
  'foci_punctate_z_count', 'foci_punctate_z_density', 'foci_punctate_z_max', 'foci_z_count_mid', 'foci_z_count_large',
  'foci_z_max_scale', 'foci_blobs_tiny', 'foci_blobs_medium', 'foci_blobs_large', 'foci_strongest_blob_scale',
  'post_local_shadow', 'post_shadow_under_brightest'];

IWSP.extractFeatures = function (img, mask, h, w, keep) {
  const f = extract(img, mask, h, w, keep), out = {};
  for (const k of ORDER) out[k] = f[k];
  return out;
};
IWSP.featureHelpers = {robustZ, scaledLoG};
})();
