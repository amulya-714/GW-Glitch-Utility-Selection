# Single-view baselines and the unified evaluation pipeline

This document covers the single-view CNN baselines (one model per Omega-scan duration) and the pipeline that
compares them with the frozen multi-view run `H1_O2_colab_baseline_v4` on the same O2 validation events.
It also covers the read-only visualisation of the preliminary sampled O3a diagnostic.

No results are reported here. Every number comes from running the commands below.

---

## 1. Experiment design

| | Multi-view (`H1_O2_colab_baseline_v4`) | Single-view baselines |
|---|---|---|
| `model.type` | `multiview` | `single_view` (new) |
| Input per event | 4 views, tensor `(4, 3, 128, 153)` | **1 view**, tensor `(1, 3, 128, 153)` |
| Backbone | shared FiLM-conditioned CNN + cross-view attention fusion | existing `BaselineCNN` conv backbone: same `ConvStage` design, width schedule, `embed_dim`, dropout and head |
| Views used | 0.5 s, 1 s, 2 s, 4 s | exactly one of 0.5 s / 1 s / 2 s / 4 s |

**What is identical by construction.** Pass `--reference-run experiments/runs/H1_O2_colab_baseline_v4` and
the single-view run uses that run's saved `config.yaml` as its base config. It therefore inherits all of these unchanged:

- the chronological O2 train/val split, `val_fraction` and the 6 h purge gap;
- confidence filtering (`confidence.train` / `confidence.val`) and the per-class caps, with their seeds;
- the class list, its order and the `None_of_the_Above` exclusion;
- preprocessing: crop, size, RGB, normalisation;
- class weighting (`effective_number`), loss, AdamW, the learning-rate schedule, epochs, early stopping on
  source-val macro-F1, and the seed.

Only these settings change: `model.type=single_view`, `model.single_view_index`, the run name/device/debug flag, and
`loss.scale_consistency.weight=0`. That last one is a regulariser that only exists for the multi-view model.
Every difference from the reference config is written to `<run>/protocol_check.json`. If a protocol-critical
setting differs (anything under `data`, `classes`, `confidence`, `split`, `sampling`, `protocol`, or the seed),
the script refuses to train unless you pass `--allow-protocol-change`.

**Same events.** The single-view dataset reads the same aligned four-view array cache as the multi-view run and
hands the model only the selected view. An event with *any* failed view is therefore dropped for every model,
so all five models see the same train and validation events. Training itself is the unchanged
`src.models.train.run` (same loop, checkpointing, artifacts). O3a is never read during training:
`splits.load_run_metadata` raises `TargetLeakageError`, and a test asserts that only `O2` metadata is loaded.

**How it differs from the old `baseline_single`.** The legacy `baseline_single` type still receives all four
views and slices one inside `forward`. The new `single_view` type is built for one input view, and the data
loader delivers only that view. With the same weights the two give identical outputs (this is tested). The
legacy type is unchanged, so its old checkpoints still load.

---

## 2. Training

All commands run from the repository root. `--device cuda` stops with a clear message if no GPU is
visible. Add `--allow-cpu-fallback` to train on CPU anyway.

### One view at a time

```bash
python -m src.models.train_single_view --ifo H1 --view 0.5 --device cuda --reference-run experiments/runs/H1_O2_colab_baseline_v4
python -m src.models.train_single_view --ifo H1 --view 1   --device cuda --reference-run experiments/runs/H1_O2_colab_baseline_v4
python -m src.models.train_single_view --ifo H1 --view 2   --device cuda --reference-run experiments/runs/H1_O2_colab_baseline_v4
python -m src.models.train_single_view --ifo H1 --view 4   --device cuda --reference-run experiments/runs/H1_O2_colab_baseline_v4
```

Run names default to `H1_O2_single_view_0.5s`, `H1_O2_single_view_1s`, `H1_O2_single_view_2s` and
`H1_O2_single_view_4s`. Use `--name` to choose your own (only with a single `--view`).

### All four sequentially

```bash
python -m src.models.train_single_view --ifo H1 --view all --device cuda \
    --reference-run experiments/runs/H1_O2_colab_baseline_v4 --skip-existing
```

`--skip-existing` skips any view whose run already has `checkpoints/best.pt` and `summary.json`. That makes the same
command safe to re-run after a Colab disconnect. A run that was interrupted part-way has no
`summary.json`; restart that view with `--view <v> --overwrite`. If one view fails, the others still run, and the
command exits non-zero at the end with a list of failures.

### Overrides

Command-line overrides take precedence over the inherited config. Changing `epochs` or `lr` is allowed, but it
is logged and recorded as a difference from the reference, and it means the comparison is no longer
like-for-like:

```bash
--epochs 30 --batch-size 64 --lr 0.001 --workers 2 --patience 8 --seed 42
--set train.weight_decay=0.01            # any config key, repeatable
--metadata-dir ... --image-dir ... --processed-dir ... --output-dir ...
```

Without `--reference-run`, the base is `--config configs/classifier.yaml` and a warning says that
consistency with the multi-view run was not verified. Use this only if the reference run directory is unavailable.

### Quick smoke test (tiny, CPU)

```bash
python -m src.models.train_single_view --view 0.5 --debug --device cpu \
    --reference-run experiments/runs/H1_O2_colab_baseline_v4
```

### Outputs (per run, `experiments/runs/<name>/`)

These are the same artifacts as `train.py` produces: `config.yaml` (with a `provenance` block naming the input
view), `checkpoints/{best,last}.pt`, `history.csv`, `training_curves.png`, `train.log`, `summary.json` (best
epoch, epochs run, training seconds, parameters, `input_shape_per_event` = `[1, 3, 128, 153]`),
`eval_val_best/`, `splits/` and `image_failures.csv`. The run also gets a new `protocol_check.json`.

---

## 3. Evaluation and comparison (one command)

```bash
python -m src.evaluation.compare_models \
    --multiview-run experiments/runs/H1_O2_colab_baseline_v4 \
    --runs-root experiments/runs \
    --out experiments/comparisons/H1_O2_single_vs_multiview \
    --o3a-diagnostic-run experiments/runs/H1_O2_colab_baseline_v4_sample_diagnostic \
    --device auto --workers 2
```

This command:

1. **Finds checkpoints.** It uses the multi-view run, plus every finished run under `--runs-root` with
   `model.type: single_view` for the same detector and source run. `*_debug` runs are skipped unless you pass
   `--include-debug`. You can name runs explicitly with `--single-view-runs DIR [DIR ...]` and add others with
   `--extra-run DIR`. Unfinished runs are skipped with a warning.
2. **Loads each model.** Each `best.pt` is loaded with its own saved config. The architecture is rebuilt and
   the weights loaded strictly, and a preprocessing-hash mismatch is logged as a warning. Each checkpoint's
   SHA-256 is checked before and after evaluation, and every pre-existing file in each run directory is
   verified unchanged.
3. **Scores the validation set.** Each model's O2 validation split is rebuilt from its own config and scored.
   The pipeline then compares the populations: whether the split is identical, and whether the scored events
   are identical.
   - If any event differs (e.g. an image failure in one run), the differences are listed in
     `population/excluded_samples.csv`, and the comparison is computed on the common events only.
   - Each model's own-population numbers stay in its run directory.
   - The `Val accuracy in summary.json` column is a reproduction check. It should equal `Accuracy` when the
     populations are identical.
4. **Writes all metrics and plots** (layout below). It refuses to overwrite existing outputs unless you pass
   `--overwrite`, which replaces only these generated directories.

Useful flags: `--batch-size` (inference), `--bootstrap 1000` (0 disables the paired bootstrap),
`--min-blocks 10`, `--checkpoint best|last`, and `--metadata-dir/--image-dir/--processed-dir` if the data has
moved since training.

### Output layout

```
experiments/comparisons/H1_O2_single_vs_multiview/
  README.md                         headline table, training facts, paired differences, population
  comparison_table.csv              Model | Input | N | Accuracy | Macro-F1 | Weighted-F1 | Balanced Acc | ECE | AURC |
                                    Parameters | Best epoch | Epochs run | Best val macro-F1 | Training time (h) | ...
  summary.json                      machine-readable: everything above + per-class, thresholds, checkpoint hashes, git commit
  all_metrics_long.csv              every metric, long format (model, group, class, threshold, metric, value)
  per_class_{precision,recall,f1}.csv   rows = classes in mapping order (incl. zero-support), columns = models
  thresholds.csv                    none / 0.5 / 0.7 / 0.9 for BOTH threshold types (see section 4)
  paired_differences_vs_multiview.csv
  population/  population_by_model.csv, excluded_samples.csv, class_distribution_common.csv, image_failures_*.csv
  plots/       overall_metric_comparison, per_class_f1_comparison, per_class_recall_comparison,
               reliability_diagrams, confidence_distributions, coverage_vs_threshold, accuracy_vs_threshold,
               selective_risk_vs_coverage, confusion_matrices/<run>_{counts,normalised}   (.png 300 dpi + .pdf)

experiments/runs/<each run>/eval_val_comparison/
  metrics.json, per_class.csv, confusion_matrix.csv, threshold_table.csv, reliability_bins.csv,
  predictions.csv, probs.npy, plots/ (confusion matrix ×2, reliability, confidence distribution,
  coverage/accuracy vs threshold, selective risk vs coverage)
```

Colours are fixed per model throughout and checked for colour-blind separation, and every line also has its own
marker. Classes always appear in the checkpoint's class-mapping order. Classes with no validation events are
kept: their rows are grey and labelled `n=0`, and undefined values are NaN, never silently 0.

---

## 4. Interpreting the metrics

| Metric | Meaning |
|---|---|
| Accuracy | Fraction correct. Dominated by frequent classes (1080Lines, Blip, ...). |
| Macro-F1 | Mean F1 over classes with at least one validation event. This is `macro_f1_present`, the quantity `train.py` logs and uses for model selection, so it is directly comparable to the multi-view run's recorded validation macro-F1. The pipeline reports whether it reproduces that value. A present class that is never predicted counts as 0. |
| Weighted-F1 | F1 weighted by support. Close to accuracy when large classes dominate. |
| Balanced accuracy | Mean per-class recall over classes present. |
| ECE | Expected calibration error, 15 bins. Same function as `train.py`, so it is directly comparable to the recorded multi-view validation ECE. |
| AURC | Area under the selective risk–coverage curve. Lower means confidence ranks correct predictions first. |
| Per-class precision | `NaN` (undefined) if the model never predicts that class. |
| Per-class recall / F1 | `NaN` if the class has no validation events. Rows are never dropped. |

**Two confidence axes are always reported separately** (column `threshold_type`):

- `model_confidence` is the classifier's max softmax. Thresholds 0.5/0.7/0.9 define **selective prediction**:
  - `coverage` is the fraction of events retained;
  - `*_retained` columns describe only the retained subset;
  - `*_all_samples` columns repeat the unfiltered numbers.

  The coverage, accuracy-vs-threshold and risk–coverage plots all use this axis.
- `label_confidence` is Gravity Spy's `ml_confidence` for the reference label. It changes *which events are
  evaluated*, not which predictions are trusted. On O2 validation it is largely redundant, because validation is
  already filtered at `confidence.val` (0.7 in the shipped config). The thresholds none/0.5/0.7 therefore coincide
  there. That is expected, not a bug.

**Paired differences.** For each single-view model, the pipeline reports the difference from the multi-view model on
the common events, with a 95% interval from a paired bootstrap that resamples whole days. Glitches cluster in time,
so resampling single events would make the intervals too narrow. With fewer than `--min-blocks` days, no interval
is reported ("not assessed").

The interval covers evaluation-sample uncertainty only. Each model is trained once with one seed, so training
variance is not included. Only describe a model as better when its interval excludes zero, and even then
mention the single-seed caveat. In particular, do not describe the multi-view architecture as superior unless
the table and intervals support it.

---

## 5. Preliminary sampled O3a diagnostic (read-only)

Rendered by `compare_models --o3a-diagnostic-run ...`, or on its own:

```bash
python -m src.evaluation.o3a_diagnostic --run-dir experiments/runs/H1_O2_colab_baseline_v4_sample_diagnostic
```

- **Inputs.** It reads only the existing `eval_target_*/predictions.csv`, plus `class_mapping.json` or
  `config.yaml` for the label space. It does **not** load O3a metadata (including
  `data/raw/metadata/H1_O3a_sample.csv`), download images, run a model, or call `evaluate.py`. The
  `--final` / `final_eval_log.json` guard is untouched. If no `predictions.csv` exists, it skips with a warning
  and recomputes nothing.
- **Outputs.** It writes only to `<diag run>/preliminary_o3a_visualisation/`. All pre-existing files are hashed
  before and after, and any change raises an error.
- **Labelling.** Everything is marked PRELIMINARY / sampled, and the folder's `README.txt` states that this is
  not a complete O3a evaluation.
- **Category separation.** Shared trained classes, legacy-excluded labels (`None_of_the_Above`) and new O3
  classes (`Blip_Low_Frequency`, `Fast_Scattering`) are kept apart. Events whose label the model was never trained
  on are never scored as correct. They appear only as counts, in "where the model sends them"
  (`outside_label_space_predicted_as.png`), and in a model-confidence-by-category plot, which shows whether the
  model is confidently wrong on classes it cannot know.
- **Threshold tables.** `label_confidence_thresholds.csv` uses the same axis as your preliminary table and as
  `evaluate.py` (`ml_confidence`). It shows, per threshold: total / shared / legacy / new counts, coverage, shared-class
  accuracy, macro-F1 and balanced accuracy, and strict accuracy. `shared_model_confidence_thresholds.csv` uses the
  classifier's own confidence instead.

Check the `none`, `0.7` and `0.9` rows of `label_confidence_thresholds.csv` against the preliminary table
(1891 / 1378 / 1027 events; accuracy 0.4765 / 0.5849 / 0.6709; macro-F1 0.3914 / 0.4527 / 0.4722).
Compare `n_total_events` *and* `n_shared_class` to see which population those counts referred to. If neither
matches, the diagnostic was produced with different settings, and this should be resolved before the review.

---

## 6. Google Colab

```python
# 1. Runtime > Change runtime type > GPU, then:
from google.colab import drive; drive.mount('/content/drive')
```

```bash
%cd /content/drive/MyDrive/GW-Glitch-Utility-Selection      # wherever your clone lives
!git pull
!pip install -q -r requirements.txt
!python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 2. train (resumable: re-run the same line after a disconnect)
!python -m src.models.train_single_view --ifo H1 --view all --device cuda --workers 2 \
    --reference-run experiments/runs/H1_O2_colab_baseline_v4 --skip-existing

# 3. evaluate + plots + comparison
!python -m src.evaluation.compare_models \
    --multiview-run experiments/runs/H1_O2_colab_baseline_v4 \
    --runs-root experiments/runs --out experiments/comparisons/H1_O2_single_vs_multiview \
    --o3a-diagnostic-run experiments/runs/H1_O2_colab_baseline_v4_sample_diagnostic \
    --device cuda --workers 2
```

Notes:

- **Paths.** The reference run's `config.yaml` supplies the data paths. If the multi-view run was trained on Colab
  with the same layout, the cached images and processed arrays are reused and nothing is re-downloaded. If your
  data lives elsewhere, add `--metadata-dir/--image-dir/--processed-dir` to both commands.
- **Seeds.** Seeds are fixed and cuDNN runs deterministically (`src/utils.set_seed`). Re-runs on the same GPU type
  should reproduce closely. Different GPUs or library versions can differ slightly.
- **Run one view per session** if a session cannot hold all four: `--view 0.5`, then `--view 1`, and so on.

---

## 7. Tests

```bash
pytest -q                                              # whole suite
pytest -q tests/test_single_view_and_comparison.py     # this feature
```

The new tests cover:

- single-view dataset loading and view selection;
- the single-view forward pass, and its equivalence with the legacy baseline;
- checkpoint save/load and validation;
- metrics against sklearn, zero-support and never-predicted classes, and confusion-matrix class order;
- confidence thresholds and coverage, risk–coverage and reliability;
- the paired bootstrap, including the too-few-blocks guard;
- population checks and protocol-difference detection;
- an end-to-end run on synthetic data: multi-view reference, four single-view runs, comparison. It checks that
  the populations are identical, that the summary metrics are reproduced, and that nothing is overwritten;
- the O3a renderer: read-only, no metadata or image access, and its numbers match `evaluate.py`'s
  `threshold_summary.csv`.
