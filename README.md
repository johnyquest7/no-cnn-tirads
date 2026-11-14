# iWSP TI-RADS — static website (HTML, CSS and JavaScript only)

Upload a thyroid ultrasound image, box the nodule, and get an ACR TI-RADS score with the
evidence behind every point. The whole site is static files: no server, no Python, no
WebAssembly, no build step. The image never leaves the browser.

It has every feature of the local app in `web/`: nodule outline and overlays, the points
equation, per-descriptor probabilities and drivers, reliability chips, the table of all 56
measurements, and a sample image.

## Run it

Open `index.html` in a browser (double-click works), or serve the folder:

```bash
cd "/mnt/d/iwsp tirads/Java_web" && python3 -m http.server 8000 --bind 127.0.0.1
```

An analysis took a median of 0.15 s on the dataset's images (headless Chromium on this
machine), and up to 1.7 s for the largest.

## Publish on GitHub Pages

1. Create a repository and copy this folder's contents to its root (or to `docs/`):
   `index.html`, `css/`, `js/`, `.nojekyll`. The `tools/` and `assets/` folders and this
   README are not needed by the site.
2. Repository **Settings → Pages → Deploy from a branch**; choose the branch and folder.
3. The site appears at `https://<user>.github.io/<repo>/`.

The only thing loaded from elsewhere is the IBM Plex web font from Google Fonts; without
it the page falls back to system fonts.

**The sample image is a synthetic phantom**, so the published site contains no patient
data. To use a real image instead, replace `assets/sample.png`, set its box in
`assets/sample.json`, and re-run the export below — only with an image you may publish.

## What is in the folder

| file | what it is | Python it replaces |
|---|---|---|
| `index.html`, `css/style.css` | the page | `web/index.html` |
| `js/app.js` | image loading, the box, overlays, the report | the page's script |
| `js/pipeline.js` | image preparation, logistic readout, ACR rules, overlay geometry | `web/pipeline.py` |
| `js/segment.js` | the nodule outline | `segment.py` |
| `js/features.js` | the 56 measurements | `features.py` |
| `js/ndimage.js` | filters, morphology, labelling, distance transform, numpy-exact sums | `scipy.ndimage`, `numpy` |
| `js/cv.js` | median blur, polar resampling, blur, polygon and ellipse fill, resize | `cv2` |
| `js/skimage.js` | region properties, convex hull, perimeter, contours, co-occurrence | `skimage` |
| `js/model-data.js` | **generated**: readout coefficients, ACR rules, texts, numeric tables, sample | `web/readout.json` |
| `tools/export_model.py` | developer tool that writes `js/model-data.js` | |
| `tools/verify_parity.py` | developer tool that checks the JavaScript against the Python | |

The two tools are Python because they read the Python project. The website does not use them.

## How faithful the port is

The library routines were ported from the source of OpenCV 5.0, SciPy 1.17 and
scikit-image 0.26, and the arithmetic follows the originals step by step, including where
numpy rounds to 32-bit floats and the order in which it adds numbers up.

`tools/verify_parity.py` runs the JavaScript in headless Chromium on dataset images with
their annotated boxes and compares with `web/pipeline.py`. Result on 2026-11-13:

| | |
|---|---|
| identical nodule outline | every image |
| identical TI-RADS level, points and chosen descriptors | every image |
| measurements bit-identical | 94.9% |
| largest difference in any measurement | 1 part in 10 million |
| largest difference in any descriptor probability | 0.00000001 |
| browser-decoded pixels identical to OpenCV's (PNG and JPEG) | every image |
| JavaScript time per image | median 0.15 s, slowest 1.7 s |

Routines the dataset rarely exercises were also tested directly against the real
libraries, with no mismatches: the ellipse fallback (400 random cases), polygon fill with
vertices outside the image and self-intersections (400), area resize for images over
640 px (9 sizes), morphology, labelling and the distance transform (60), and the disk
opening (60).

The page also has a **self-check**, run on demand from the "Run self-check" link above the
image (it is not run at load, to keep start-up instant): it analyses the bundled sample and
compares all 56 measurements and the TI-RADS call with the Python reference. A mismatch is
shown as a warning.

## Keeping it in step with the Python

This is a hand port, so it does not update itself.

- **`web/readout.json` changed** (coefficients only): run
  `python Java_web/tools/export_model.py`. No JavaScript changes.
- **`features.py` or `segment.py` changed**: make the same change in `js/features.js` or
  `js/segment.js`, re-run the export, then
  `python Java_web/tools/verify_parity.py`. Do not publish until it prints
  `PARITY OK`. `js/model-data.js` records which `segment.py` and `features.py` (by md5)
  it was exported from.

```bash
source ~/venvs/iwsp/bin/activate
pip install playwright && playwright install chromium-headless-shell     # once, for verify_parity.py
python Java_web/tools/export_model.py
python Java_web/tools/verify_parity.py
```

## Limits

- Research tool, not a medical device.
- **File formats:** the browser decodes the image, so PNG, JPEG, BMP, GIF and WebP work;
  TIFF and DICOM do not (the local Python app reads TIFF). 16-bit PNGs are reduced to
  8 bits by the browser, where the Python pipeline rescales them, so those can differ.
- JPEGs are analysed as stored: EXIF rotation is not applied, in the page or in Python.
- Parity was measured in Chromium. Other browsers run the same JavaScript and report
  their own self-check when you run it; their JPEG decoders may differ from OpenCV's by a gray level.
- The box matters, as in the local app: the model was validated on boxes about
  1.0–1.3× the nodule.
