# GW-Glitch-Utility-Selection: stage 1, real-image classifier

Source-run (O2) training of a Gravity Spy glitch classifier with a strict unseen-target-run (O3a) protocol.
Synthetic generation and gradient-based selection are **not** in this stage.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate     # skip if your venv exists
pip install -r requirements.txt
```

## Verify the install (no network needed)

```bash
pytest -q                                               # unit + end-to-end tests on fake data
python -m src.data.fake_data --out /tmp/fake_gw --events 600
python -m src.models.train --debug --device cpu --name fake \
  --metadata-dir /tmp/fake_gw/metadata --image-dir /tmp/fake_gw/img --processed-dir /tmp/fake_gw/proc \
  --set split.gap_seconds=1800 --set split.min_val_per_class=1
```

## Small real-data test (downloads about 100-200 events x 4 images)

```bash
python -m src.models.train --config configs/classifier.yaml --ifo H1 --model multiview --debug
# check experiments/runs/H1_multiview_seed42_debug/{preprocessing.json,summary.json,image_failures.csv}
```

## Inspect the real images first (recommended before the full run)

```bash
python -m src.data.inspect_images --ifo H1 --run O2 --events-per-class 2 --max-events 24
```

Downloads a few events and writes `docs/image_inspection/H1_O2/contact_sheet_*.png` (four views side by side at
original resolution, labelled with ID, label and confidence), `image_info.csv` (real width/height/mode/bytes of every
PNG) and `inspection_summary.json`, which prints the native sizes next to the configured `data.image_size`/`data.channels`.
Target-run images are refused unless `--allow-target-inspection` is passed (recorded in `inspection_log.json`).

## Preprocessing (approved baseline)

The shipped config uses crop `[101, 61, 674, 540]` on 800x600 images, RGB, `[128, 153]` (HxW, aspect-preserving),
bilinear, alpha over white, `unit` normalisation, all four views. A configuration that stretches the crop by more than
2% is rejected at startup. Details: [`docs/preprocessing.md`](docs/preprocessing.md).

```bash
python -m src.data.crop_preview describe                         # effective settings, reads no data
python -m src.data.crop_preview preview --from-config --max-events 6   # visual check on real source-run events
```

## Full H1 O2 training

```bash
python -m src.models.train --config configs/classifier.yaml --ifo H1 --model multiview
# baselines and ablations
python -m src.models.train --ifo H1 --model baseline_merged
python -m src.models.train --ifo H1 --model baseline_single --set model.single_view_index=1
python -m src.models.train --ifo H1 --model multiview --set model.film.enabled=false --name H1_mv_nofilm
python -m src.models.train --ifo H1 --model multiview --set model.fusion.type=concat --name H1_mv_concat
```

Outputs go to `experiments/runs/<name>/`: `config.yaml`, `class_mapping.json`, `splits/*.csv` (manifests, class counts
before/after filtering, purged and excluded events), `history.csv`, `training_curves.png`, `checkpoints/{best,last}.pt`,
`eval_val_best/` (metrics.json, per_class.csv, confusion_matrix.csv/png), `image_failures.csv`, `summary.json`, `train.log`.

## Optional hierarchical scale interaction

The default remains the shared CNN + time-scale FiLM + attention baseline. An opt-in first experiment adds
duration-ordered adjacent-view residual messages after per-view pooling/projection:

```bash
python -m src.models.train --ifo H1 --model multiview --set model.scale_interaction.enabled=true \
  --set loss.scale_consistency.weight=0.05 --set loss.scale_consistency.confidence_threshold=0.7
```

The interaction gates start at zero, so they initially preserve the baseline representation. The optional
consistency penalty is a directed KL from detached fused probabilities to individual view predictions and is
applied only when both fused and per-view maximum probabilities exceed the configured confidence threshold.
This reduces pressure on views that are individually uncertain; it is a training regularizer on source data, not
a demand that every view agree. Keep its weight at zero for the baseline and ablations. Evaluate its value only
against source validation and the existing one-shot frozen-model target protocol.

## Final cross-run evaluation (target run O3a)

Run **once** per frozen model, after all choices are made using source validation only:

```bash
python -m src.models.evaluate --run-dir experiments/runs/H1_multiview_seed42 --split target --final
```

By default (`evaluate.max_per_class_target: null`) every eligible O3a event is scored; a per-class cap is rejected
unless the run was trained with `--debug`. Each confidence threshold (none, 0.7, 0.9) is a subset of the same pool and
gets its own `report.json` with separate blocks for the shared-class closed set, legacy/excluded labels
(`None_of_the_Above`), the new O3 classes (`Blip_Low_Frequency`, `Fast_Scattering`) and strict accuracy
(outside-label-space events counted as wrong). `threshold_summary.csv` tabulates the sample counts per threshold.

Without `--final` the command refuses. A second run is refused unless `--allow-rerun` is passed; either way
`final_eval_log.json` records the checkpoint hash and config hash.

## Protocol guard rails (where they live)

| Rule | Enforced in |
|---|---|
| Training code cannot load O3a metadata | `splits.load_run_metadata` raises `TargetLeakageError` |
| Temporal split with purge gap, never random | `splits.temporal_split` + `assert_split_disjoint` |
| Split boundary independent of confidence threshold | split first, filter after (`build_source_splits`) |
| No dataset statistics in preprocessing | per-sample, per-view standardisation (`preprocessing.normalize_views`) |
| No detector/run ID as model input | model forward takes only the (B,V,C,H,W) image tensor |
| Model selection on source val only | `train.run` (best epoch by val macro-F1) |
| One-shot target evaluation | `evaluate --final`, `final_eval_log.json` |
