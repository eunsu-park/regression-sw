# Retrain Sweep — August 2026 (seed-paired direct + recursive)

Two sweeps on the GPU server, both ap30-side only (no hp30), both with
`experiment.seed=250104` (already fixed in `configs/base.yaml` and applied by
`setup_seed` to python/numpy/torch/cuda; cudnn deterministic).

| Sweep | Profile | Experiments | Target head | Loss |
|---|---|---|---|---|
| A — direct, short-horizon | `server_ap` | in {6h,12h,18h,1d} × out {1h..6h} × 14 models = **336**, names `ap_{io}_{model}` | ap30 (1 ch) | solar_wind_weighted |
| B — recursive | `server_ap_recursive` | in {6h,12h,18h,1d} × out {1h..6h} × 14 models = **336**, names `ap_recursive_{io}_{model}` | all 22 input channels | mse |
| C — storm-only | `server_ap_storm` | same grid as A = **336**, names `ap_storm_{io}_{model}` | ap30 (1 ch) | solar_wind_weighted |
| D — quiet-only | `server_ap_quiet` | same grid as A = **336**, names `ap_quiet_{io}_{model}` | ap30 (1 ch) | solar_wind_weighted |

Sweep C (`configs/server_ap_storm.yaml`, added 2026-08-19) trains on only
the anchors whose target window peaks at ap30 ≥ 48 (Kp ≥ 5, G1) — 4.6 %
(out1h) to 11.7 % (out6h) of the training split; the filter is
window-aware per io config. Validation stays the full index and the
normalization stats are shared, so C is directly comparable with A.
Launch: `./train.sh --config-name server_ap_storm --max-jobs 4`.

Sweep D (`configs/server_ap_quiet.yaml`, added 2026-08-23) is the exact
complement of C: training keeps only anchors whose target window has NO
point of ap30 ≥ 48 (`train_filter.peak_max`), so C ∪ D = the full
training split (verified on the table: 3,549 + 73,036 = 76,585 for
out1h). Launch: `./train.sh --config-name server_ap_quiet --max-jobs 4`.

Mac fallback (GPU server down): `mac_ap`, `mac_ap_storm`, `mac_ap_quiet`,
`mac_ap_recursive` are the same sweeps on MPS with Mac roots and
`attention.create_plots: false`; use `--max-jobs 2 --skip-existing`.

Sweep A (revised 2026-08-12) is the short-horizon grid: 20 new io configs
`in{6h,12h,18h,1d}_out{1..5}h` plus the existing `*_out6h` four. `train.sh`
and `run_pending.sh` apply this grid automatically for `server_ap`; the
legacy long-horizon grid (out12h/18h/24h, in2d/3d) remains available via an
explicit `--filter`, and stays the scan matrix for `server_hp`.

Sweep B (`configs/server_ap_recursive.yaml`) predicts every input variable
over a 1–6 h chunk so the chunk can be fed back for an iterated rollout to
longer leads (E1 deep arm of the preregistration). `train.sh` and
`run_pending.sh` automatically restrict it to the same short-horizon grid
as sweep A and prefix its experiment names with `ap_recursive_`, so it
never collides with the direct `ap_*` results. (History: B first ran as
out6h-only — those 56 runs are a subset of this grid, skipped on relaunch
by `--skip-existing`; widened to all chunk lengths 2026-08-19.)

## 0. Before launching sweep A — archive the previous mainline results

With the short-horizon grid, sweep A collides with existing results only on
the four `*_out6h` io combos (`ap_in{6h,12h,18h,1d}_out6h_{model}`, 56 dirs)
under `/home/eunsupark/Projects/GeoIndex/results`. Archive those before
launching (CV fold dirs `*_fold[1-5]` stay; the legacy long-horizon results
that are not being retrained also stay):

```bash
cd /home/eunsupark/Projects/GeoIndex/results
mkdir -p _archive_pre-retrain-2026-08
for io in in6h_out6h in12h_out6h in18h_out6h in1d_out6h; do
    for d in ap_${io}_*; do
        [[ "$d" == *_fold* ]] && continue  # keep CV fold results
        [[ -d "$d" ]] && mv "$d" _archive_pre-retrain-2026-08/
    done
done
ls _archive_pre-retrain-2026-08 | wc -l    # expect 56
```

This is a rename inside the cloud-synced tree (cheap for the sync client)
and fully reversible.

## 1. Launch (GPU server, `conda activate geoindex`)

```bash
cd ~/GitHub/njit-geoindex/geoindex-model   # or the server checkout path

# Sweep A — 336 direct models
./train.sh --config-name server_ap --max-jobs 4

# Sweep B — 336 recursive models (--skip-existing skips the 56 out6h runs already done)
./train.sh --config-name server_ap_recursive --max-jobs 4 --skip-existing
```

`--max-jobs 4` matches `environment.num_workers: 4` in `server_ap.yaml`
(4 jobs × 4 loader workers on the single RTX 3090). Sanity-check the queue
first with `--dry-run` (A and B each print 336 configs).

Restarting an interrupted or re-scoped sweep: add `--skip-existing` — runs
whose `{save_root}/{exp}/log/training_history.json` exists (written only on
normal completion) are dropped from the queue; interrupted runs re-train
from scratch.

Scope history for sweep B: launched 2026-08-12 as 84 (six input lengths,
out6h only), narrowed to 56 the same day (in2d/in3d dropped), widened to
the full 336-combo short-horizon grid on 2026-08-19. Finished runs from
any earlier scope are skipped by `--skip-existing`; stray
`ap_recursive_in2d/in3d_*` results are extra data the current filter and
run_pending matrix no longer touch.

Logs: `~/tmp/train_logs/{experiment}.log`.

## 2. After training — validation

```bash
./run_pending.sh --config-name server_ap           --epoch best --max-jobs 4
./run_pending.sh --config-name server_ap_recursive --epoch best --max-jobs 4
```

## 3. Per-sample plots (on demand, after validation)

Sweeps run with `validation.save_plots: false`; regenerate plots from the
validation archives (no model/GPU needed — reads
`validation/<epoch>/npz.zip`).

Scale check before a full render: one full-period validation index is
~23,514 anchors per run at ~51 KB per PNG, so the whole 336-run direct
sweep is ~7.9M plots / **~385 GB**. Run it on the GPU server (the archives
are local there — on a cloud-synced replica every npz.zip must download
first) and point `--output-root` OUTSIDE the synced tree:

```bash
# whole sweep as one plots.zip per run (policy since 2026-08-22):
# renders via a LOCAL scratch dir, absorbs + deletes any loose plots/ dir,
# and skips runs whose plots.zip already exists
nohup python analysis/plot_validation_samples.py \
    --results-dir /home/eunsupark/Projects/GeoIndex/results \
    --filter '^ap_(storm_)?in(6h|12h|18h|1d)_out[1-6]h_' \
    --zip --workers 12 --min-free-gb 25 > ~/tmp/plot_sweep.log 2>&1 &

# single run / quick look (loose PNGs, no zip)
python analysis/plot_validation_samples.py \
    --results-dir /home/eunsupark/Projects/GeoIndex/results \
    --experiment ap_in12h_out6h_linear --output-root /tmp/plots --limit 20
```

`--filter` is a regex on run names (the pattern above also catches the CV
fold runs — intended). One loose PNG per anchor scales terribly through a
sync client (millions of small files stall upload/eviction); plots.zip is
one ~1.2 GB file per run. Recursive runs render the grouped envelope-band
layout automatically; direct runs keep the classic single-panel layout.

## Notes and caveats

- **Stats cache.** Both sweeps read `table_stats_ap.pkl` (existing, computed
  over the same 22 variables); nothing is recomputed or overwritten.
- **Loss for sweep B.** `solar_wind_weighted` denormalizes with
  `target_variables[0]`'s stats and applies ap-tier weighting to the whole
  target tensor — wrong for a 22-channel mixed-normalization target, hence
  plain MSE in normalized space (equal per-channel weighting).
- **Validation headline for sweep B.** The per-variable metrics cover all 22
  channels, but anything keyed to `target_variables[0]` (e.g. the MC-dropout
  calibration block) refers to `v_avg`, not ap30 — ap30 is the **last**
  channel (index 21). Rollout evaluation should read the ap30 channel
  explicitly.
- **Channel order.** Recursive targets are in exactly the input-variable
  order, so a rollout appends the predicted chunk to the input window with
  no reindexing.
- **Smoke test.** All 14 architectures build, forward, and backprop with the
  22-channel head at in6h/in2d (verified 2026-08-12 on CPU).
