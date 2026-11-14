// JavaScript port of web/pipeline.py: image preparation, the logistic readout, the ACR TI-RADS
// rules and the overlay geometry. Produces the same result object the Python server returned.
(function () {
'use strict';
const IWSP = window.IWSP = window.IWSP || {};

// pipeline.preprocess, part 1: decode a file to 8-bit grayscale exactly as stored.
// EXIF rotation, colour management and alpha premultiplication are switched off, because the
// Python pipeline (cv2.imdecode) applies none of them.
async function decodeFile(file) {
  let bmp;
  try {
    bmp = await createImageBitmap(file, {imageOrientation: 'none', premultiplyAlpha: 'none', colorSpaceConversion: 'none'});
  } catch (_) {
    throw new Error('That file could not be read as an image. Browsers can decode PNG, JPEG, BMP, GIF and WebP; ' +
                    'convert TIFF or DICOM images to PNG first.');
  }
  const w = bmp.width, h = bmp.height;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', {willReadFrequently: true, colorSpace: 'srgb'});
  ctx.drawImage(bmp, 0, 0);
  if (bmp.close) bmp.close();
  return {gray: IWSP.cv.grayFromRGBA(ctx.getImageData(0, 0, w, h).data, w * h), h, w};
}

// pipeline.preprocess, part 2: longest side <= 640 px, area-averaged, as in training.
function limitSize(gray, h, w) {
  const MAX = IWSP.MODEL.max_side, scale = Math.min(1.0, MAX / Math.max(h, w));
  if (scale >= 1.0) return {gray, h, w};
  const dw = Math.max(1, IWSP.nd.rint(w * scale)), dh = Math.max(1, IWSP.nd.rint(h * scale));
  return {gray: IWSP.cv.resizeArea(gray, h, w, dh, dw), h: dh, w: dw};
}

// pipeline.score: fill, standardise, clip, logistic.
function score(feats) {
  const R = IWSP.MODEL.readout, names = R.features, n = names.length;
  const x = new Float64Array(n), z = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const v = feats[names[k]];
    x[k] = Number.isFinite(v) ? v : R.fill_median[k];
    z[k] = Math.min(5, Math.max(-5, (x[k] - R.mean[k]) / R.scale[k]));
  }
  const out = {};
  for (const [lab, d] of Object.entries(R.labels)) {
    let dot = 0;
    const contrib = new Float64Array(n);
    for (let k = 0; k < n; k++) { contrib[k] = d.coef[k] * z[k]; dot += contrib[k]; }
    const logit = dot + d.intercept;
    const order = [...contrib.keys()].sort((a, b) => Math.abs(contrib[b]) - Math.abs(contrib[a])).slice(0, 5);
    out[lab] = {
      p: 1.0 / (1.0 + Math.exp(-logit)), logit, intercept: d.intercept,
      auc: d.auc_visible, n_pos: d.n_pos_visible, title: IWSP.MODEL.label_title[lab] || lab,
      drivers: order.map(k => ({feature: names[k], value: x[k], z: z[k], contribution: contrib[k],
                                doc: IWSP.MODEL.feature_docs[names[k]] || ''})),
    };
  }
  return out;
}

// pipeline.acr: ACR TI-RADS points and level (mirrors tirads.py).
function acr(probs) {
  const R = IWSP.MODEL.readout, cats = {}, points = {};
  let total = 0;
  for (const [g, spec] of Object.entries(R.groups)) {
    const cols = Object.keys(spec.labels).filter(c => c in probs);
    if (!cols.length) continue;
    const ranked = cols.slice().sort((a, b) => probs[b].p - probs[a].p);
    const chosen = spec.type === 'exclusive' ? [ranked[0]] : cols.filter(c => probs[c].p >= 0.5);
    cats[g] = {title: IWSP.MODEL.category_title[g] || g, type: spec.type, chosen, ranked, points_table: spec.labels};
  }
  const spongi = !!(cats.composition && cats.composition.chosen.includes(R.spongiform_col));
  if (R.spongiform_override && spongi) {
    for (const g in cats) points[g] = 0;
    return {cats, points, total: 0, tr: R.tr_from_points['0'], spongi: true};
  }
  for (const [g, spec] of Object.entries(R.groups)) {
    if (!(g in cats)) continue;
    const on = cats[g].chosen;
    const p = spec.type === 'additive' ? on.reduce((s, c) => s + spec.labels[c], 0)
                                       : on.reduce((m, c) => Math.max(m, spec.labels[c]), 0);
    points[g] = p; total += p;
  }
  return {cats, points, total, tr: R.tr_from_points[String(Math.min(total, 15))], spongi: false};
}

// Outlines of a binary mask as canvas polygons ([x, y] at pixel centres) for the overlays.
function outlines(mask, h, w) {
  return IWSP.sk.findContours(mask, h, w).filter(c => c.length >= 3).map(c => c.map(p => [p[1] + 0.5, p[0] + 0.5]));
}

// pipeline.evidence: the geometry behind the numbers, taken from what extract() actually measured.
function evidence(mask, h, w, k) {
  const nd = IWSP.nd, N = h * w;
  // bright punctate foci: the speckle-relative test (fine scale, z > 6), strongest 40
  const foci = [];
  for (let i = 0; i < N; i++) if (k.solidInt[i] && k.lf[i] === k.lfmax[i] && k.z[i] > 6.0) foci.push({x: i % w, y: Math.trunc(i / w), z: k.z[i]});
  foci.sort((a, b) => b.z - a.z);
  // dark cystic spaces: the mirrored test (z > 4 at three scales)
  const dark = new Uint8Array(N);
  if (nd.count(mask) >= 20) {
    for (const sg of [1.5, 3.0, 6.0]) {
      const dz = IWSP.featureHelpers.robustZ(IWSP.featureHelpers.scaledLoG(k.sm, h, w, sg, 1, 2), mask);
      for (let i = 0; i < N; i++) if (mask[i] && dz[i] > 4.0) dark[i] = 1;
    }
  }
  const [minr, minc, maxr, maxc] = k.props.bbox;
  return {
    nodule: outlines(mask, h, w),
    ring_outer: outlines(nd.dilate(mask, h, w, k.ringW), h, w),
    dark: outlines(dark, h, w),
    foci: foci.slice(0, 40).map(f => ({x: f.x, y: f.y, z: Math.round(f.z * 10) / 10})),
    bbox: {x: minc, y: minr, w: maxc - minc, h: maxr - minr},
    diam: Math.round(k.diam * 10) / 10,
    ref_brightness: Math.round(k.ref * 1e4) / 1e4,
  };
}

// pipeline.analyse for an already prepared 8-bit grayscale image.
function analyseGray(gray, h, w, box) {
  if (!box) box = [0.2 * w, 0.2 * h, 0.8 * w, 0.8 * h];
  box = Array.from(box, Number);
  const img = new Float32Array(h * w);
  for (let i = 0; i < img.length; i++) img[i] = gray[i] / 255;             // astype(float32) / 255.0
  const mask = IWSP.segment(img, h, w, box);
  if (!IWSP.nd.count(mask)) throw new Error('The outline came out empty. Draw the box around the nodule and try again.');
  const keep = {};
  const features = IWSP.extractFeatures(img, mask, h, w, keep);
  const probs = score(features), a = acr(probs);
  return {
    width: w, height: h, box, mask,
    mask_area_frac: Math.round(IWSP.nd.count(mask) / (h * w) * 1e4) / 1e4,
    evidence: evidence(mask, h, w, keep),
    features, feature_docs: IWSP.MODEL.feature_docs,
    probs, categories: a.cats, points: a.points, total_points: a.total, tr: a.tr,
    spongiform_override: a.spongi, method: IWSP.MODEL.method,
  };
}

IWSP.decodeFile = decodeFile;
IWSP.limitSize = limitSize;
IWSP.score = score;
IWSP.acr = acr;
IWSP.analyseGray = analyseGray;
})();
