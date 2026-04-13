# ds003374 temporary pipeline (no edits to my_PreP)

`main_PreP.py` is not modified. This folder contains a standalone script that reuses `main_PreP` rendering helpers and `ms_lfp_transform` pseudo-LFP logic.

## Script

- `build_ds003374_pseudo_lfp_plotly.py`

## What it does

1. Scans `ds003374` NIX files (`bidsignore/data_NIX/*.h5`).
2. Extracts iEEG trial data and concatenates trials into one session timeline.
3. Extracts sorted spike times per unit and maps units to `T{shank}_{unit}` format:
   - `uAL` -> shank 1
   - `uAR` -> shank 2
4. Keeps only channels/units whose anatomy label contains `amyg` by default.
5. Builds pseudo-LFP from spikes and compares against recorded amygdala LFP in Plotly.
6. Writes dataset-structure summaries (`subject/unit/source inventory`).
7. Applies preprocessing for fair comparison:
   - Macro LFP: line-noise notch using BIDS JSON `PowerLineFrequency` harmonics.
   - Macro + Pseudo traces: z-score normalization before plotting.
8. Layout order per macro pair: `Macro LFP N -> Pseudo N -> Spike rows (units on that shank)`.
9. Builds a separate PSD HTML per subject using full time range:
   - Macro PSD and pseudo-LFP PSD are plotted together for each macro pair.
10. Builds a separate Power Time HTML per subject:
   - Uses ThetaPower-like mechanism (sliding PSD -> theta-band power time series).
   - If `mne` is available, uses multitaper PSD windows.
   - If `mne` is unavailable, falls back to Welch-window PSD with the same window/step settings.
   - Layout matches signal plot order (Macro -> Pseudo -> Spike rows).

## Example run

```bash
python ds003374_tmp_pipeline/build_ds003374_pseudo_lfp_plotly.py \
  --ds-root /Users/takamanabe/Documents/Git/ds003374 \
  --out-dir /Users/takamanabe/Documents/Git/MS_mod_tSC/ds003374_tmp_pipeline/output_ds003374_all_v2 \
  --verbose
```

## Key outputs

- `index.html`: subject-level plot index
- `sub-XX_ses-01_signal_pseudo_lfp.html`: Plotly comparison pages
- `sub-XX_ses-01_signal_pseudo_lfp_psd.html`: Plotly PSD pages (full-range PSD)
- `sub-XX_ses-01_signal_pseudo_lfp_power_time.html`: Plotly Power Time pages
- `subject_inventory.tsv`: per-subject data/layout summary
- `unit_inventory.tsv`: per-unit/per-trial spike inventory with source/anatomy labels
- `source_inventory.tsv`: iEEG source (macro channel, anatomy, SOZ) mapping
- `run_summary.json`: run parameters and output paths

## Notes

- Subjects with no sorted units still get LFP-only pages when amygdala LFP exists.
- Subjects with no channels passing `--anatomy-substring` are skipped.
- Output HTML files are large when `--plotly-js inline` is used. Default is `cdn`.
- `--anatomy-substring ""` disables anatomy filtering.
- `--APPLY_MACRO_LINE_NOISE false` disables macro notch filtering.
- `--APPLY_ZSCORE false` disables z-score normalization.
- `index.html` has `Plot`, `PSD`, and `Power Time` link columns.
