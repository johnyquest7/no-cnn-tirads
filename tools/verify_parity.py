"""Checks the JavaScript port against the desktop Python pipeline on real dataset images.

    pip install playwright && playwright install chromium-headless-shell     # once
    python Java_web/tools/verify_parity.py

Developer tool; the website does not need it. It loads Java_web/js/*.js in headless Chromium and,
for each image, compares with web/pipeline.py on this machine:

  1. algorithm  - the same prepared grayscale pixels go to both; outline, all 56 measurements,
                  every probability and the TI-RADS call are compared.
  2. decoding   - the original file goes to the browser's own decoder; the resulting grayscale
                  pixels are compared with cv2's, and the end-to-end TI-RADS call is compared.
"""
import argparse
import base64
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT))
JS = ["model-data.js", "ndimage.js", "cv.js", "skimage.js", "segment.js", "features.js", "pipeline.js"]

PAGE_FN = """
window.b64u8 = s => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; };
window.u8b64 = u => { let s = ''; for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode.apply(null, u.subarray(i, i + 32768)); return btoa(s); };
window.slim = r => ({features: r.features, tr: r.tr, total_points: r.total_points,
                     chosen: Object.fromEntries(Object.entries(r.categories).map(([k, c]) => [k, c.chosen])),
                     probs: Object.fromEntries(Object.entries(r.probs).map(([k, p]) => [k, p.p])), mask: u8b64(r.mask)});
window.runGray = (b64, h, w, box) => { try { const t = performance.now(); const r = slim(IWSP.analyseGray(b64u8(b64), h, w, box)); r.ms = performance.now() - t; return r; }
                                       catch (e) { return {error: String(e && e.stack || e)}; } };
window.runFile = async (b64, mime, box256) => {
  try {
    const d = await IWSP.decodeFile(new Blob([b64u8(b64)], {type: mime}));
    const p = IWSP.limitSize(d.gray, d.h, d.w);
    const box = [box256[0] * p.w / 256, box256[1] * p.h / 256, box256[2] * p.w / 256, box256[3] * p.h / 256];
    const r = slim(IWSP.analyseGray(p.gray, p.h, p.w, box));
    r.gray = u8b64(p.gray); r.h = p.h; r.w = p.w; return r;
  } catch (e) { return {error: String(e && e.stack || e)}; }
};
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of dataset images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="print every measurement that differs by more than 1e-5")
    a = ap.parse_args()

    import pipeline
    from playwright.sync_api import sync_playwright

    rows = list(csv.DictReader(open(ROOT / "annot_lat4.csv")))
    rng = np.random.default_rng(a.seed)
    rows = [rows[i] for i in rng.choice(len(rows), size=min(a.n, len(rows)), replace=False)]

    n = same_mask = same_call = 0
    rels, over, worst, worst_p, ms = [], {}, (0.0, ""), (0.0, ""), []
    n_dec = dec_same = dec_call = 0
    dec_maxdiff = 0

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("about:blank")
        page.on("pageerror", lambda e: print("PAGE ERROR:", e))
        for f in JS:
            page.add_script_tag(path=str(HERE / "js" / f))
        page.add_script_tag(content=PAGE_FN)

        for r in rows:
            f = ROOT / "images2" / r["file"]
            if not f.exists():
                continue
            raw = f.read_bytes()
            u8 = pipeline.preprocess(raw)
            h, w = u8.shape
            b256 = [float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
            box = [b256[0] * w / 256, b256[1] * h / 256, b256[2] * w / 256, b256[3] * h / 256]
            try:
                ref = pipeline.analyse(raw, box)
                imgf = u8.astype(np.float32) / 255.0
                ref_mask = np.asarray(pipeline.SEG.segment(imgf, tuple(box)), bool)
            except Exception as exc:                                    # noqa: BLE001
                print(f"{r['file']}: desktop pipeline failed ({exc}); skipped")
                continue

            # ---- 1. algorithm parity on identical pixels
            got = page.evaluate("([b, h, w, box]) => runGray(b, h, w, box)",
                                [base64.b64encode(u8.tobytes()).decode(), h, w, box])
            if "error" in got:
                print(f"{r['file']}: JAVASCRIPT FAILED: {got['error'][:400]}")
                n += 1
                continue
            n += 1
            ms.append(got["ms"])
            gm = np.frombuffer(base64.b64decode(got["mask"]), np.uint8).reshape(h, w).astype(bool)
            same_mask += bool((gm == ref_mask).all())
            for k, v in ref["features"].items():
                g = got["features"].get(k)
                if g is None or not np.isfinite(v):
                    worst = (float("inf"), f"{r['file']}:{k} desktop={v!r} js={g!r}")
                    continue
                rel = abs(g - v) / (1e-3 + abs(v))
                rels.append(rel)
                if rel > 1e-5:
                    over[k] = over.get(k, 0) + 1
                    if a.verbose:
                        print(f"   {r['file']}:{k} desktop={v!r} js={g!r}")
                if rel > worst[0]:
                    worst = (rel, f"{r['file']}:{k} desktop={v!r} js={g!r}")
            for k, v in ref["probs"].items():
                d = abs(got["probs"][k] - v["p"])
                if d > worst_p[0]:
                    worst_p = (d, f"{r['file']}:{k}")
            ok = (got["tr"] == ref["tr"] and got["total_points"] == ref["total_points"]
                  and all(got["chosen"][g] == c["chosen"] for g, c in ref["categories"].items()))
            same_call += ok
            if not ok:
                print(f"   {r['file']}: call differs: desktop TR{ref['tr']} ({ref['total_points']} pts), "
                      f"js TR{got['tr']} ({got['total_points']} pts)")

            # ---- 2. the browser's own decoder
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "bmp": "image/bmp"}.get(
                f.suffix.lower().lstrip("."), "application/octet-stream")
            dec = page.evaluate("([b, m, box]) => runFile(b, m, box)", [base64.b64encode(raw).decode(), mime, b256])
            if "error" in dec:
                print(f"{r['file']}: browser could not decode: {dec['error'][:200]}")
                continue
            n_dec += 1
            if (dec["h"], dec["w"]) == (h, w):
                gg = np.frombuffer(base64.b64decode(dec["gray"]), np.uint8).reshape(h, w)
                d = int(np.abs(gg.astype(int) - u8.astype(int)).max())
                dec_maxdiff = max(dec_maxdiff, d)
                dec_same += d == 0
            dec_call += dec["tr"] == ref["tr"] and dec["total_points"] == ref["total_points"]
            if n % 25 == 0:
                print(f"  {n} images ...", flush=True)
        browser.close()

    rels = np.array(rels)
    print(f"\n1. ALGORITHM (identical input pixels)        images: {n}")
    print(f"   identical nodule outline                  {same_mask}/{n}")
    print(f"   identical TR level, points, descriptors   {same_call}/{n}")
    print(f"   measurements compared                     {rels.size}")
    print(f"     bit-identical                           {(rels == 0).sum()} ({(rels == 0).mean():.1%})")
    print(f"     within 1e-6 (relative)                  {(rels <= 1e-6).sum()} ({(rels <= 1e-6).mean():.2%})")
    print(f"     differing by more than 1e-5             {(rels > 1e-5).sum()}")
    if over:
        print("       " + ", ".join(f"{k} x{v}" for k, v in sorted(over.items(), key=lambda t: -t[1])))
    print(f"   largest measurement difference            {worst[0]:.2e}   {worst[1]}")
    print(f"   largest probability difference            {worst_p[0]:.2e}   {worst_p[1]}")
    print(f"   JavaScript time per image                 median {np.median(ms):.0f} ms, max {np.max(ms):.0f} ms")
    print(f"\n2. DECODING (browser decoder vs cv2)         images: {n_dec}")
    print(f"   identical grayscale pixels                {dec_same}/{n_dec}   (largest pixel difference {dec_maxdiff})")
    print(f"   identical TR level and points end to end  {dec_call}/{n_dec}")
    ok = n > 0 and same_call == n and same_mask == n
    print("\nPARITY OK" if ok else "\nPARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
