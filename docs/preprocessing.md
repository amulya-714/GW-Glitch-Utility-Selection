# Preprocessing

## Approved initial baseline (shipped in `configs/classifier.yaml`)

| setting | value |
|---|---|
| crop | `[101, 61, 674, 540]` (left, top, right, bottom; right/bottom exclusive) on 800x600 originals -> 573x479 |
| reference size | `[800, 600]` (width, height); any other PNG size is rejected, never mis-cropped |
| channels | 3 (RGB) |
| output size | `[128, 153]` (height, width) |
| resize | bilinear (antialiased) |
| alpha background | white |
| normalisation | `unit` (x / 255) |
| views | all four: `url1`..`url4` = 0.5 s, 1 s, 2 s, 4 s |
| model input per event | `(4, 3, 128, 153)` |

**No stretching.** The crop has aspect ratio 573/479 = 1.1962. 128 x 153 has 1.1953, a residual stretch of 0.08%
that is just integer rounding (x-scale 0.26702, y-scale 0.26722). A square output would stretch the crop by 16.4%.
This is enforced, not assumed: `data.crop.max_aspect_distortion` (default 0.02) makes any configuration that
stretches the crop more than 2% fail at startup, before any download, and the error lists the aspect-preserving
alternatives. Other aspect-preserving sizes for this crop: 96 -> 115, 160 -> 191, 224 -> 268 (width). Set the limit to
`null` only if stretching is deliberate.

Print what a config will do, reading no images or metadata:

```bash
python -m src.data.crop_preview describe
```

Provenance: the box and the settings above come from your visual review of real H1 O2 images. The code in this
repository has only ever been exercised on synthetic images; real-image compatibility is confirmed by the debug run
on your machine, not by the tests.

Every run records the exact preprocessing: `preprocessing.json` in the run directory, `preprocessing` in
`summary.json` and in each checkpoint, and a one-line log message with a hash. The hash covers geometry, the ordered
view columns, view durations and normalisation, so a result can always be tied to its preprocessing.

## Pipeline

```
PNG (800x600 RGBA)
  -> [size check, only if cropping]   reject files that are not the size the box was defined for
  -> alpha composite                   data.alpha_background: white | black | drop
  -> [crop]                            data.crop.box = [left, top, right, bottom] in original pixels
  -> [grayscale]                       data.channels: 1 (luma) | 3 (RGB)
  -> resize                            data.image_size (int or [H, W]), data.resize_mode
  -> uint8 array cache                 one array per split; key includes every option above
  -> normalise on the fly              data.normalization.mode
```

Geometry (everything above the cache) and normalisation have different lifetimes. The array-cache key is a hash of
the ordered event IDs, the crop box and reference size, channels, size, resize filter, alpha background and the ordered
view columns, so changing any of them builds a new array and a stale one is never reused. Normalisation is applied on
the fly to the cached uint8 pixels, so changing it needs no re-decoding; it is covered by the run signature instead.

## Options and the reasoning behind them

**Crop.** Enabled in the shipped baseline. Coordinates are in pixels of the *original*
image, right/bottom exclusive (PIL convention). `crop.reference_size` records the size the box was drawn for;
any file with another size is rejected with a named `ImageGeometryError` rather than silently mis-cropped. Enabling
the crop with no box is an error that points at the diagnostic tool; `enabled: false` turns it off. The crop is applied at native resolution, before
resizing, so furniture never bleeds into the resized pixels.

**Colour.** `channels: 3` keeps RGB. `channels: 1` is PIL luma (ITU-R 601). Luma is a single number per pixel
computed from the colour map, not the underlying spectrogram energy, and whether it is monotonic in energy depends
on the colour map. That is unverified for these images, so compare both on the preview sheets. RGB is a superset of
the information, at 3x the input channels (the model's first conv is the only change, driven by `data.channels`).

**Alpha.** RGBA is composited over `alpha_background` (default white) instead of silently discarding alpha. If every
pixel is opaque this changes nothing; the `suggest` step reports the minimum alpha and the fraction of non-opaque
pixels so you can see whether it matters.

**Resolution.** `image_size` is an int (square) or `[H, W]`. With a crop, choose sizes that keep the crop's aspect
ratio (see above); the preview tool computes them for you from heights 96/128/160/224. Compute grows roughly with
H x W on every view, which matters on an M1 CPU/MPS; the benefit of higher resolution is untested.

**Resize filter.** `bilinear` (default), `area`, `bicubic`, `lanczos`. All of Pillow's filters scale their support when
downsampling, so they are antialiased.

**Normalisation** (`data.normalization.mode`), no longer assumed to be per-image standardisation:

| mode | what it does | keeps | discards |
|---|---|---|---|
| `unit` (default) | `x / 255` | absolute and cross-view intensity | nothing; scale handled downstream by BatchNorm |
| `per_view_standardize` | each view image to mean 0 / std 1 (previous behaviour) | per-view contrast pattern | absolute brightness, brightness ratio between time scales |
| `per_sample_standardize` | one mean/std per event over all four views | relative brightness between views | event-level offset and scale |
| `fixed` | `(x/255 - mean) / std`, constants in config | everything, centred | nothing |

`unit` is the baseline because it throws away nothing. Whether intensity is informative or a run-dependent nuisance
(noise floor, calibration) is an empirical question, and it matters for cross-run shift, so treat the choice as an
ablation to run on the source validation set, never on the target.

`fixed` constants must be computed on **source training** data only (`compute_channel_stats` in
`preprocessing.py` returns per-channel mean/std in [0, 1] units from a cached train array). Using validation or O3a
images would leak information into preprocessing. All other modes use no dataset statistics.

Configs and checkpoints written before these options existed have no `crop` or `normalization` block; they load
with no crop and `per_view_standardize`, i.e. their original behaviour.

## Originals are preserved

Raw PNGs live in `data/raw/images/` and are only read. Nothing rewrites them. A valid PNG of the wrong size is
rejected by the size check and left on disk (tested by hashing every file before and after). The only deletion in
the pipeline is a cached PNG that genuinely fails to decode, so that a rerun re-downloads it.

## Re-checking or choosing a crop (source runs only)

All commands read source-run data only. Target runs are refused with `TargetLeakageError` and there is no override
flag. With the defaults, events you already downloaded are reused from the cache.

```bash
# Verify the shipped baseline on real images: original+box, crop RGB, crop gray, and the CONFIGURED size first
python -m src.data.crop_preview preview --from-config --max-events 6

# Propose a box from per-pixel variance across real events (a suggestion, not a fact)
python -m src.data.crop_preview suggest --ifo H1 --run O2

# Try a different box, or the full image for comparison
python -m src.data.crop_preview preview --box LEFT TOP RIGHT BOTTOM --max-events 6
python -m src.data.crop_preview preview --no-crop --max-events 3 --out-dir docs/preprocessing_preview/H1_O2_nocrop
```

`--resolutions` are output HEIGHTS; widths are derived from the crop's aspect ratio, and each column's title shows
its size and stretch. `--square` switches to R x R outputs, which stretch a non-square crop, and exists to show what
that costs. Sheets are written to `docs/preprocessing_preview/<ifo>_<run>/`.

How `suggest` works and where it can fail: titles, tick labels, frames and the colour bar are nearly identical between
images, whereas the plotted data changes every time, so across-image per-pixel standard deviation is high inside the
plotting region and low in the furniture. The box is the longest dense block of high-variance rows and columns. It can
be wrong if the figure carries event-specific text, if a colour bar or margin varies, or if the spectrograms are
unusually smooth. It warns about per-view disagreement (> 5 px), a box covering < 20% or > 95% of the image, a size
mismatch with `reference_size`, and non-opaque pixels. Always confirm with `preview`.

Choices still worth settling with source-validation experiments rather than by eye: RGB vs gray, resolution,
normalisation mode. Do not use O3a for any of them.
