# iWSP TI-RADS — static website (HTML, CSS and JavaScript only)

Upload a thyroid ultrasound image, box the nodule, and get an ACR TI-RADS score with the
evidence behind every point. The whole site is static files: no server, no Python, no
WebAssembly, no build step. The image never leaves the browser.

It has every feature of the local app in `web/`: nodule outline and overlays, the points
equation, per-descriptor probabilities and drivers, reliability chips, the table of all 56
measurements, and a sample image.


The page also has a **self-check**, run on demand from the "Run self-check" link above the
image (it is not run at load, to keep start-up instant): it analyses the bundled sample and
compares all 56 measurements and the TI-RADS call with the Python reference. A mismatch is
shown as a warning.


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
