import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import numpy as np
from scipy import signal
from scipy.io import loadmat
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt, welch
from scipy.stats import t as student_t
from scipy.stats import wilcoxon
from tqdm import tqdm

from my_FileLoad import matFileLoad
from my_INITIALIZATION import initialization
from ms_lfp_transform import parse_unit_name, synthesize_shank_lfp

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as exc:  # pragma: no cover
    go = None
    make_subplots = None
    PLOTLY_IMPORT_ERROR = exc
else:
    PLOTLY_IMPORT_ERROR = None


HC_FILE_RE = re.compile(r"^s(?P<session>.+?)_HC_(?P<label>.+)\.mat$", re.IGNORECASE)
MS_FILE_RE = re.compile(r"^s(?P<session>.+?)_MS_spike\.mat$", re.IGNORECASE)
HC_LABEL_CH_RE = re.compile(r"(?:^|_)eeg_ch(?P<ch>\d+)(?:$|_)", re.IGNORECASE)

SUPPORTED_MODES = (
    "psd",
    "theta_power",
    "theta_delta_ratio",
    "entropy",
    "entropy_td",
    "signal",
    "hilbert",
    "spectrogram",
    "phaselag",
    "syncfc",
    "coherence_band",
    "pearson",
    "coherence",
    "granger",
)
TIME_SERIES_MODES = {
    "theta_power",
    "theta_delta_ratio",
    "entropy",
    "entropy_td",
    "signal",
    "hilbert",
    "spectrogram",
    "phaselag",
    "syncfc",
    "coherence_band",
    "pearson",
    "coherence",
    "granger",
}
THETA_FAMILY_MODES = {
    "theta_power",
    "theta_delta_ratio",
    "entropy",
    "entropy_td",
}
MODE_TITLE_MAP = {
    "psd": "PSD",
    "theta_power": "Theta Power",
    "theta_delta_ratio": "Theta/Delta Ratio",
    "entropy": "Entropy",
    "entropy_td": "Entropy TD Balance",
    "signal": "Signal",
    "hilbert": "Hilbert",
    "spectrogram": "Spectrogram",
    "phaselag": "PhaseLag",
    "syncfc": "SyncFC",
    "coherence_band": "Coherence Band",
    "pearson": "Pearson Band",
    "coherence": "Coherence",
    "granger": "Granger gPDC",
}
MODE_SHORT_TITLE_MAP = {
    "psd": "PSD",
    "theta_power": "Power",
    "theta_delta_ratio": "T/D Ratio",
    "entropy": "Entropy",
    "entropy_td": "Entropy TD",
    "signal": "Signal",
    "hilbert": "Hilbert",
    "spectrogram": "Spectrogram",
    "phaselag": "PhaseLag",
    "syncfc": "SyncFC",
    "coherence_band": "CohBand",
    "pearson": "Pearson",
    "coherence": "Coherence",
    "granger": "Granger",
}
MODE_ALIASES = {
    "hilphase": "phaselag",
    "hilbertphase": "phaselag",
    "zshift": "phaselag",
    "zlag": "phaselag",
    "fc": "syncfc",
    "functionalconnectivity": "syncfc",
    "synchrony": "syncfc",
    "synchronyfc": "syncfc",
    "wavelet": "spectrogram",
}
DEFAULT_THETA_BAND_TOKENS = [
    "1-4", 
    "4-8",
    "4-12",
    "8-12",
    "8-30",
    "12-30"
    # "4-12",
    # "12-30",
    # "35-55",
    # "65-85",
    # "90-115",
    # "120-145",
    # "165-185",
]
DEFAULT_GRANGER_STATS_BAND_TOKENS = [
    "1-4", 
    "4-8",
    "4-12",
    "8-12",
    "8-30",
    "12-30"
    # "4-12",
    # "12-30",
    # "35-55",
    # "65-85",
    # "90-115",
    # "120-145",
    # "165-185",
]
DELTA_BAND = (1.0, 4.0)
PLOTLY_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

_PLOTLY_RESET_LISTENER = """
<script>
(function () {
  function isAxisKey(k) {
    return /^xaxis\\d*$/.test(k) || /^yaxis\\d*$/.test(k);
  }

  function axisNumberFromKey(axisKey) {
    var m = String(axisKey || "").match(/^[xy]axis(\\d*)$/);
    if (!m) return Number.MAX_SAFE_INTEGER;
    if (!m[1]) return 1;
    var n = parseInt(m[1], 10);
    return isFinite(n) ? n : Number.MAX_SAFE_INTEGER;
  }

  function axisIndexFromKey(axisKey) {
    var m = String(axisKey || "").match(/^[xy]axis(\\d*)$/);
    if (!m) return null;
    if (!m[1]) return "1";
    var n = parseInt(m[1], 10);
    return isFinite(n) ? String(n) : null;
  }

  function axisMatchIdFromKey(axisKey) {
    var m = String(axisKey || "").match(/^([xy])axis(\\d*)$/);
    if (!m) return null;
    if (!m[2]) return m[1];
    var n = parseInt(m[2], 10);
    return isFinite(n) ? m[1] + String(n) : null;
  }

  function chooseSyncAxisIndexSet(axisKeys) {
    var xKeys = axisKeys.filter(function (k) { return k.indexOf("xaxis") === 0; });
    var out = {};
    for (var i = 0; i < xKeys.length; i++) {
      var idx = axisIndexFromKey(xKeys[i]);
      if (idx) out[idx] = true;
    }
    return out;
  }

  function sortAxisKeys(keys) {
    return keys.slice().sort(function (a, b) {
      return axisNumberFromKey(a) - axisNumberFromKey(b);
    });
  }

  function linkAxisMatchesGroup(patch, axisKeys) {
    if (!axisKeys || !axisKeys.length) return;
    var sorted = sortAxisKeys(axisKeys);
    if (sorted.length <= 1) {
      patch[sorted[0] + ".matches"] = null;
      return;
    }
    var anchor = sorted[0];
    var anchorId = axisMatchIdFromKey(anchor);
    if (!anchorId) return;
    patch[anchor + ".matches"] = null;
    for (var i = 1; i < sorted.length; i++) {
      patch[sorted[i] + ".matches"] = anchorId;
    }
  }

  function applySyncMatches(enabled) {
    try {
      if (!window.Plotly) return;
      var plots = Array.prototype.slice.call(document.querySelectorAll(".js-plotly-plot"));
      for (var i = 0; i < plots.length; i++) {
        var gd = plots[i];
        var layout = gd && gd.layout ? gd.layout : {};
        var axisKeys = Object.keys(layout).filter(function (k) { return isAxisKey(k); });
        var patch = {};
        var xAxisKeys = axisKeys.filter(function (k) { return k.indexOf("xaxis") === 0; });
        for (var ai = 0; ai < xAxisKeys.length; ai++) {
          patch[xAxisKeys[ai] + ".matches"] = null;
        }
        if (enabled) {
          var syncIdxSet = chooseSyncAxisIndexSet(axisKeys);
          var xSync = [];
          for (var bi = 0; bi < axisKeys.length; bi++) {
            var axisKey = axisKeys[bi];
            if (axisKey.indexOf("xaxis") === 0) {
              var idx = axisIndexFromKey(axisKey);
              if (!idx || !syncIdxSet[idx]) continue;
              xSync.push(axisKey);
            }
          }
          linkAxisMatchesGroup(patch, xSync);
        }
        if (Object.keys(patch).length) {
          try { window.Plotly.relayout(gd, patch); } catch (e) {}
        }
      }
    } catch (e) {}
  }

  function resetAll() {
    try {
      if (!window.Plotly) return;
      var plots = Array.prototype.slice.call(document.querySelectorAll(".js-plotly-plot"));
      for (var i = 0; i < plots.length; i++) {
        var gd = plots[i];
        var layout = gd && gd.layout ? gd.layout : {};
        var patch = {};
        Object.keys(layout).forEach(function (k) {
          if (isAxisKey(k)) {
            patch[k + ".autorange"] = true;
          }
        });
        if (!Object.keys(patch).length) {
          patch["xaxis.autorange"] = true;
          patch["yaxis.autorange"] = true;
        }
        try { window.Plotly.relayout(gd, patch); } catch (e) {}
      }
    } catch (e) {}
  }

  function applyModebarAction(action) {
    try {
      var a = String(action || "").toLowerCase();
      if (!a) return;
      if (a === "reset") {
        resetAll();
        return;
      }
      if (a !== "zoom" && a !== "pan") return;
      if (!window.Plotly) return;
      var plots = Array.prototype.slice.call(document.querySelectorAll(".js-plotly-plot"));
      for (var i = 0; i < plots.length; i++) {
        try { window.Plotly.relayout(plots[i], {dragmode: a}); } catch (e) {}
      }
    } catch (e) {}
  }

  window.addEventListener("message", function (ev) {
    var d = ev && ev.data ? ev.data : null;
    if (!d || typeof d !== "object") return;
    if (d.type === "reset-plotly-view") resetAll();
    if (d.type === "plotly-sync-all-subplots") applySyncMatches(!!d.enabled);
    if (d.type === "plotly-modebar-action") applyModebarAction(d.action);
  });
})();
</script>
"""


@dataclass(frozen=True)
class EEGEntry:
    subject: str
    session: str
    hc_name: str
    eeg_z: np.ndarray
    ms_units: tuple[tuple[str, np.ndarray], ...]


@dataclass(frozen=True)
class DownsampleConfig:
    up: int
    down: int
    sampling_rate_out: float
    aa_cutoff_hz: float
    aa_sos: np.ndarray


@dataclass(frozen=True)
class TraceSpec:
    name: str
    x: np.ndarray
    y: np.ndarray
    color: str
    width: float = 1.2
    dash: str = "solid"


@dataclass(frozen=True)
class ModeCell:
    subject: str
    session: str
    traces: tuple[TraceSpec, ...]
    payload: Any = None


@dataclass(frozen=True)
class ModePage:
    subject: str
    mode: str
    mode_title: str
    href: str


@dataclass(frozen=True)
class StatsPage:
    key: str
    title: str
    href: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-mode analysis for repacked subject/session EEG. "
            "Generates per-subject/per-mode child Plotly HTMLs and one parent subject x mode index."
        )
    )
    parser.add_argument(
        "--folder",
        "--dataset",
        dest="folder",
        type=str,
        default=None,
        help=(
            "Dataset folder name under /Volumes/T7_Taka/Minnesota/MSHC/"
            "Data_VargaV/23798184_repacked."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan subject/session EEG files without loading MAT contents.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(SUPPORTED_MODES), #["granger"], # 
        help=(
            "Analysis modes to run. Supported: "
            "psd theta_power theta_delta_ratio entropy entropy_td signal hilbert "
            "spectrogram phaselag syncfc coherence_band pearson coherence granger. "
            "Aliases: HilPhase/ZShift -> phaselag. "
            "If omitted, all implemented modes are executed."
        ),
    )
    parser.add_argument(
        "--theta_bands",
        nargs="+",
        type=str,
        metavar="BAND",
        default=DEFAULT_THETA_BAND_TOKENS,
        help=(
            "Theta bands for signal/hilbert/theta_power/theta_delta_ratio/entropy/entropy_td/"
            "phaselag/syncfc/coherence_band/pearson. "
            "Formats: '4-8 4-10' or flat list '4 8 4 10'."
        ),
    )
    parser.add_argument(
        "--spectrogram_normalize",
        choices=("mean", "none"),
        default="mean",
        help=(
            "Wavelet spectrogram normalization. 'mean' shows dB relative to each "
            "signal's mean wavelet power at each frequency; 'none' shows raw dB power. "
            "Default: mean."
        ),
    )
    parser.add_argument(
        "--spectrogram_aperiodic_mode",
        choices=("residual", "none"),
        default="residual",
        help=(
            "Spectrogram aperiodic handling across frequency for each time window. "
            "'residual' removes a linear 1/f trend in log10(freq); 'none' keeps dB power. "
            "Default: residual."
        ),
    )
    parser.add_argument(
        "--syncfc_n_surrogates",
        type=int,
        default=50,
        help=(
            "Number of SyncFC null permutations/surrogates. "
            "Use 0 to disable null-normalized statistics. Default: 50."
        ),
    )
    parser.add_argument(
        "--syncfc_null",
        choices=("window_shuffle", "circular_shift", "none"),
        default="window_shuffle",
        help=(
            "Null model for SyncFC statistics. 'window_shuffle' pairs HC and MS "
            "windows from different times within the same session; 'circular_shift' "
            "uses the previous whole-trace shift surrogate; 'none' disables null "
            "statistics. Default: window_shuffle."
        ),
    )
    parser.add_argument(
        "--syncfc_min_shift_sec",
        type=float,
        default=2.0,
        help=(
            "Minimum circular shift in seconds for SyncFC surrogates. "
            "This avoids near-zero shifts that preserve local synchrony. Default: 2.0."
        ),
    )
    parser.add_argument(
        "--syncfc_seed",
        type=int,
        default=0,
        help="Random seed for SyncFC null permutations/surrogates. Default: 0.",
    )
    parser.add_argument(
        "--syncfc_stats_pdf",
        type=str,
        default="AUTO",
        help=(
            "Output PDF path for group-level SyncFC statistics. Use AUTO to save next "
            "to the parent HTML, or NONE to disable. Default: AUTO."
        ),
    )
    parser.add_argument(
        "--phaselag_min_z",
        "--hilphase_min_z",
        dest="phaselag_min_z",
        type=float,
        default=0.0,
        help=(
            "Minimum max Rayleigh Z required to include a PhaseLag window. "
            "Windows below this threshold are excluded. Default: 0."
        ),
    )
    parser.add_argument(
        "--phaselag_min_plv",
        "--hilphase_min_plv",
        dest="phaselag_min_plv",
        type=float,
        default=0.2,
        help=(
            "Minimum PLV at the selected PhaseLag lag required to include a PhaseLag window. "
            "Windows below this threshold are excluded. Default: 0.2."
        ),
    )
    parser.add_argument(
        "--phaselag_min_peak_delta_z",
        "--hilphase_min_peak_delta_z",
        dest="phaselag_min_peak_delta_z",
        type=float,
        default=0.0,
        help=(
            "Minimum Rayleigh-Z separation between the best and second-best "
            "PhaseLag lag. Windows with flatter peaks are excluded. Default: 0."
        ),
    )
    parser.add_argument(
        "--phaselag_min_peak_delta_frac",
        "--hilphase_min_peak_delta_frac",
        dest="phaselag_min_peak_delta_frac",
        type=float,
        default=0.0,
        help=(
            "Minimum fractional Rayleigh-Z separation between the best and "
            "second-best PhaseLag lag, computed as (best-second)/best. "
            "Windows with flatter peaks are excluded. Default: 0."
        ),
    )
    parser.add_argument(
        "--phaselag_min_valid_ratio",
        type=float,
        default=0.1,
        help=(
            "Minimum fraction of usable PhaseLag windows required to keep a band/session. "
            "Bands below this value are hidden. Default: 0.1."
        ),
    )
    parser.add_argument(
        "--phaselag_include_edge_peaks",
        action="store_true",
        help=(
            "Include PhaseLag windows whose best lag is at the edge of the searched lag range. "
            "Default behavior excludes edge-hit windows from best-lag probability."
        ),
    )
    parser.add_argument(
        "--phaselag_edge_guard_bins",
        type=int,
        default=0,
        help=(
            "Extra number of lag bins inside each search edge to exclude from PhaseLag "
            "best-lag probability. 0 excludes only the exact endpoints. Default: 0."
        ),
    )
    parser.add_argument(
        "--phaselag_stats_pdf",
        "--hilphase_stats_pdf",
        dest="phaselag_stats_pdf",
        type=str,
        default="AUTO",
        help=(
            "Output PDF path for group-level PhaseLag statistics. Use AUTO to save next "
            "to the parent HTML, or NONE to disable. Default: AUTO."
        ),
    )
    parser.add_argument(
        "--time_range",
        nargs=2,
        type=float,
        metavar=("T_MIN", "T_MAX"),
        default=[0, 1000],
        help=(
            "Time range [T_MIN, T_MAX] in seconds for time-series modes. "
            "This pipeline requires T_MIN=0 and T_MAX>0."
        ),
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=1000.0,
        help="Sampling rate [Hz]. Default: 1000.",
    )
    parser.add_argument(
        "--APPLY_LINE_NOISE_REMOVAL",
        type=str,
        default="true",
        help=(
            "Apply harmonic notch line-noise removal to HC LFP before downsampling/z-score. "
            "Default: true."
        ),
    )
    parser.add_argument(
        "--LINE_NOISE_HZ",
        type=float,
        default=50.0,
        help="Power-line base frequency [Hz] for harmonic notch filtering. Default: 50.",
    )
    parser.add_argument(
        "--LINE_NOISE_Q",
        type=float,
        default=30.0,
        help="Q-factor for line-noise harmonic notch filters. Default: 30.",
    )
    parser.add_argument(
        "--MS_LFP",
        type=str,
        default="true",
        help=(
            "Switch MS rendering mode in time-series plots. "
            "true: pseudo-LFP per shank, false: legacy per-unit spike raster. "
            "Default: false."
        ),
    )
    parser.add_argument(
        "--MS_LFP_OVERLAY",
        type=str,
        default="true",
        help=(
            "When --MS_LFP=true, overlay legacy spike raster markers on each shank row. "
            "Default: true."
        ),
    )
    parser.add_argument(
        "--MS_LFP_SIGMA",
        type=float,
        default=0.004,
        help=(
            "Sigma [sec] for the first-derivative Gaussian MS kernel. "
            "The positive peak is aligned to each spike time. Default: 0.004."
        ),
    )
    parser.add_argument(
        "--MS_LFP_A",
        type=float,
        default=0.2,
        help="Amplitude A for AP kernel V(t). Default: 0.2.",
    )
    parser.add_argument(
        "--MS_LFP_A0",
        type=float,
        default=1.0,
        help="Global gain A0 in pseudo-LFP weighting A0/d_ij. Default: 1.0.",
    )
    parser.add_argument(
        "--MS_LFP_D",
        type=str,
        default="{}",
        help=(
            "Shank-specific distance map JSON for d_ij. "
            "Schema: {'xx': {'yy': distance, ...}, ...}. "
            "Example: '{\"01\":{\"1\":1.0,\"2\":1.3},\"07\":{\"3\":0.9}}'. "
            "Missing/invalid d_ij entries are excluded with warning."
        ),
    )
    parser.add_argument(
        "--MS_LFP_D_DEFAULT",
        type=float,
        default=3.0,
        help=(
            "Default scalar distance used only when --MS_LFP_D is empty JSON. "
            "For each shank, this value is expanded to all detected units "
            "(d=[d0, d0, ...]). Default: 3.0."
        ),
    )
    parser.add_argument(
        "--MS_LFP_POST_SMOOTH_SEC",
        type=float,
        default=0,#0.012,
        help=(
            "Gaussian post-smoothing sigma [sec] applied to pseudo-LFP "
            "after spike-kernel summation. Use 0 to disable. Default: 0."
        ),
    )
    parser.add_argument(
        "--downsample-rate",
        type=float,
        default=500.0,
        help=(
            "Target sampling rate [Hz] for preprocessing downsampling. "
            "If >0, anti-alias lowpass is auto-set from target Nyquist and applied before downsampling. "
            "If 0, AA lowpass/downsampling is disabled."
        ),
    )
    parser.add_argument(
        "--psd-nperseg",
        type=int,
        default=4096,
        help="nperseg for scipy.signal.welch in PSD mode. Default: 4096.",
    )
    parser.add_argument(
        "--max-freq",
        type=float,
        default=500.0,
        help=(
            "Upper x-limit [Hz] for PSD plot. "
            "<=0 means auto (=Nyquist of effective sampling rate). "
            "Values above Nyquist are clamped to Nyquist. Default: 500."
        ),
    )
    parser.add_argument(
        "--psd_aperiodic_mode",
        choices=("residual", "none"),
        default="residual",
        help=(
            "PSD aperiodic handling. "
            "'residual' removes 1/f trend by linear fit in log10(freq) on PSD(dB). "
            "'none' keeps raw PSD(dB). Default: residual."
        ),
    )
    parser.add_argument(
        "--psd_1f_fit_range",
        nargs=2,
        type=float,
        metavar=("FMIN", "FMAX"),
        default=[1.0, 50.0],
        help=(
            "Fit range [Hz] for PSD 1/f removal when --psd_aperiodic_mode residual. "
            "Default: 1 50."
        ),
    )
    parser.add_argument(
        "--tf_win_sec",
        type=float,
        default=1.0,
        help="Sliding window length (sec) for spectrogram/theta/syncfc/coherence/pearson/granger family modes.",
    )
    parser.add_argument(
        "--tf_step_sec",
        type=float,
        default=1.0,
        help="Sliding window step (sec) for spectrogram/theta/syncfc/coherence/pearson/granger family modes.",
    )
    parser.add_argument(
        "--freq_plot",
        nargs=2,
        type=float,
        metavar=("FMIN", "FMAX"),
        default=[1, 30.0],
        help=(
            "Frequency range [Hz] displayed/aggregated in spectrogram/coherence modes "
            "(wavelet power/coherence/PLV). Default: 1.5 10."
        ),
    )
    parser.add_argument(
        "--freq_calc",
        nargs=2,
        type=float,
        metavar=("FMIN", "FMAX"),
        default=[1.0, 32.0],
        help=(
            "Wavelet frequency calculation range [Hz] for spectrogram/coherence modes. "
            "Default: 1 15."
        ),
    )
    parser.add_argument(
        "--granger_freq",
        nargs=2,
        type=float,
        metavar=("FMIN", "FMAX"),
        default=[1.0, 120.0],
        help=(
            "Frequency range [Hz] for Granger gPDC curve display. "
            "Default: 1 120."
        ),
    )
    parser.add_argument(
        "--granger_n_freqs",
        type=int,
        default=50,
        help="Number of frequency bins for Granger gPDC curves. Default: 50.",
    )
    parser.add_argument(
        "--granger_order_max",
        type=int,
        default=15,
        help=(
            "Maximum candidate MVAR order p for Granger model-order selection. "
            "Default: 15."
        ),
    )
    parser.add_argument(
        "--granger_order_criterion",
        choices=("bic", "aic"),
        default="bic",
        help="Information criterion for Granger model-order selection. Default: bic.",
    )
    parser.add_argument(
        "--granger_order_mode",
        choices=("median", "per_window"),
        default="median",
        help=(
            "Order strategy for Granger MVAR. "
            "'median': choose p as median across windows, then refit all windows with fixed p. "
            "'per_window': choose p independently for each window. Default: median."
        ),
    )
    parser.add_argument(
        "--granger_fixed_order",
        type=int,
        default=0,
        help=(
            "Fixed MVAR order p for Granger. If >0, skips per-window order selection "
            "and fits every window with this p. Default: 0 (use order selection)."
        ),
    )
    parser.add_argument(
        "--granger_stats_pdf",
        type=str,
        default="AUTO",
        help=(
            "Output PDF path for group-level Granger statistics. Use AUTO to save next "
            "to the parent HTML, or NONE to disable. Default: AUTO."
        ),
    )
    parser.add_argument(
        "--granger_stats_bands",
        nargs="+",
        type=str,
        metavar="BAND",
        default=DEFAULT_GRANGER_STATS_BAND_TOKENS,
        help=(
            "Frequency bands for Granger statistics PDF. "
            "Formats: '1-4 4-10' or flat list '1 4 4 10'. "
            "Default: 1-4 4-10 18-35 30-50 50-70 70-100 100-140 160-200."
        ),
    )
    parser.add_argument(
        "--granger_epoch_jobs",
        type=int,
        default=8,
        help=(
            "Parallel workers for Granger epoch/window processing inside each session entry. "
            "Default: 8."
        ),
    )
    parser.add_argument(
        "--granger_progress",
        choices=("none", "entry", "epoch"),
        default="entry",
        help=(
            "Progress detail level for Granger. "
            "'none': only Mode[granger] bar. "
            "'entry': mode bar + per-entry postfix/timing. "
            "'epoch': adds per-window sub-bars (order/fit); forces Granger entry n_jobs=1. "
            "Default: entry."
        ),
    )
    parser.add_argument(
        "--ts_smooth_win_sec",
        type=str,
        default="10",
        help="Causal smoothing window (sec) for time-series modes. Use NONE to disable.",
    )
    parser.add_argument(
        "--apply_db",
        action="store_true",
        help="Apply dB transform to hilbert/theta_power before plotting.",
    )
    parser.add_argument(
        "--db_eps",
        type=float,
        default=float(np.finfo(np.float32).eps),
        help="Small epsilon used for dB conversion. Default: float32 eps.",
    )
    parser.add_argument(
        "--plotly-js",
        choices=("cdn", "inline"),
        default="cdn",
        help="How to include plotly.js in HTML. Default: cdn.",
    )
    parser.add_argument(
        "--interactive_js",
        choices=("cdn", "inline"),
        default=None,
        help="Alias for plotly JS mode. If set, overrides --plotly-js.",
    )
    parser.add_argument(
        "--interactive_max_points",
        type=int,
        default=2000,
        help="Max points per trace for interactive time-series HTML downsampling.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help=(
            "Parallel workers for per-EEG computations. "
            "0=auto, -1=all cores, -2=all-1, positive integer=fixed workers."
        ),
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default="",
        help=(
            "Output parent index HTML path. Default: "
            "<dataset_dir>/interactive_analysis_index_<dataset>.html"
        ),
    )
    parser.add_argument(
        "--stats-html",
        "--stats_html",
        dest="stats_html",
        type=str,
        default="AUTO",
        help=(
            "Output tabbed Plotly HTML for group-level Granger/PhaseLag/SyncFC statistics. "
            "Use AUTO to save next to the parent HTML, or NONE to disable. Default: AUTO."
        ),
    )
    return parser


def natural_key(text: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(text))
    out: list[Any] = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p.lower())
    return tuple(out)


def sanitize_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-_.")
    return token or "unknown"


def _resolve_n_jobs(n_jobs: int | None) -> int:
    cpu = int(os.cpu_count() or 1)
    if n_jobs is None:
        return 1
    n = int(n_jobs)
    if n == 0:
        return max(1, min(cpu, 8))
    if n < 0:
        return max(1, cpu + 1 + n)
    return max(1, n)


def _build_downsample_config(
    sampling_rate_in: float,
    downsample_rate: float,
    aa_order: int = 6,
) -> DownsampleConfig | None:
    sr_in = float(sampling_rate_in)
    sr_req = float(downsample_rate)
    if sr_req <= 0:
        return None
    if sr_req >= sr_in:
        raise ValueError(
            f"--downsample-rate must be smaller than --sampling-rate for downsampling "
            f"(got downsample={sr_req}, sampling={sr_in})."
        )
    if aa_order < 2:
        raise ValueError("Internal AA lowpass order must be >=2.")

    ratio = Fraction(sr_req / sr_in).limit_denominator(10000)
    up = int(ratio.numerator)
    down = int(ratio.denominator)
    if up <= 0 or down <= 0:
        raise ValueError(
            f"Failed to resolve a valid rational resample ratio for --downsample-rate={sr_req}."
        )

    sr_out = float(sr_in * (up / down))
    nyq_in = 0.5 * sr_in
    nyq_out = 0.5 * sr_out
    cutoff_hz = min(nyq_out, nyq_in * 0.999999)
    wn = cutoff_hz / nyq_in
    wn = min(max(wn, 1e-6), 0.999999)
    aa_sos = butter(int(aa_order), wn, btype="low", output="sos")

    return DownsampleConfig(
        up=up,
        down=down,
        sampling_rate_out=sr_out,
        aa_cutoff_hz=float(cutoff_hz),
        aa_sos=np.asarray(aa_sos, dtype=float),
    )


def _apply_aa_downsample(sig: np.ndarray, cfg: DownsampleConfig | None) -> np.ndarray:
    x = np.asarray(sig, dtype=float).reshape(-1)
    if cfg is None:
        return x
    x_lp = sosfiltfilt(cfg.aa_sos, x)
    y = resample_poly(x_lp, up=int(cfg.up), down=int(cfg.down))
    return np.asarray(y, dtype=float).reshape(-1)


def _apply_harmonic_notch(sig: np.ndarray, fs: float, line_hz: float, q: float) -> np.ndarray:
    y = np.asarray(sig, dtype=float).reshape(-1)
    if y.size < 16:
        return y
    if (not np.isfinite(fs)) or float(fs) <= 0:
        return y
    if (not np.isfinite(line_hz)) or float(line_hz) <= 0:
        return y
    if (not np.isfinite(q)) or float(q) <= 0:
        return y

    nyq = 0.5 * float(fs)
    out = y.copy()
    f = float(line_hz)
    while f < (nyq - 1e-9):
        w0 = f / nyq
        if not (0.0 < w0 < 1.0):
            break
        try:
            b, a = signal.iirnotch(w0=float(w0), Q=float(q))
            out = signal.filtfilt(b, a, out)
        except Exception:
            break
        f += float(line_hz)
    return np.asarray(out, dtype=float).reshape(-1)


def _parallel_map(items: list[Any], worker: Any, n_jobs: int) -> Any:
    if n_jobs <= 1:
        for item in items:
            yield worker(item)
        return
    with ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
        for res in ex.map(worker, items):
            yield res


def list_subject_dirs(dataset_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(dataset_dir.iterdir(), key=lambda x: x.name.lower()):
        if p.is_dir() and not p.name.startswith(".") and not p.name.startswith("._"):
            out.append(p)
    return out


def collect_sessions(subject_dir: Path) -> dict[str, dict[str, Any]]:
    session_map: dict[str, dict[str, Any]] = {}
    for f in sorted(subject_dir.glob("*.mat"), key=lambda x: x.name.lower()):
        mh = HC_FILE_RE.match(f.name)
        if mh:
            sess = mh.group("session")
            rec = session_map.setdefault(sess, {"hc_files": [], "ms_file": None})
            rec["hc_files"].append(f)
            continue
        mm = MS_FILE_RE.match(f.name)
        if mm:
            sess = mm.group("session")
            rec = session_map.setdefault(sess, {"hc_files": [], "ms_file": None})
            rec["ms_file"] = f
    return session_map


def choose_hc_file(hc_files: list[Path]) -> Path | None:
    if not hc_files:
        return None
    scored: list[tuple[int, str, Path]] = []
    for p in hc_files:
        m = HC_FILE_RE.match(p.name)
        label = m.group("label") if m else p.stem
        mch = HC_LABEL_CH_RE.search(label)
        ch = int(mch.group("ch")) if mch else 10**9
        scored.append((ch, p.name.lower(), p))
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[0][2]


def build_work_items(dataset_dir: Path) -> list[tuple[str, str, Path, Path | None]]:
    items: list[tuple[str, str, Path, Path | None]] = []
    for subject_dir in list_subject_dirs(dataset_dir):
        session_map = collect_sessions(subject_dir)
        for session in sorted(session_map.keys(), key=natural_key):
            rec = session_map[session]
            hc_path = choose_hc_file(rec["hc_files"])
            if hc_path is None:
                continue
            items.append((subject_dir.name, session, hc_path, rec["ms_file"]))
    return items


def to_numeric_1d(vec: Any) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _normalize_spike_array(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([], dtype=np.int64)
    x = np.rint(x).astype(np.int64)
    x = x[x > 0]
    if x.size == 0:
        return np.array([], dtype=np.int64)
    if np.any(np.diff(x) < 0):
        x = np.sort(x)
    return x


def load_ms_spike_units(path: Path) -> list[tuple[str, np.ndarray]]:
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    units: list[tuple[str, np.ndarray]] = []

    ms = mat.get("MS_spike")
    fields = getattr(ms, "_fieldnames", None)
    if ms is not None and fields:
        for field in fields:
            spikes = _normalize_spike_array(np.asarray(getattr(ms, field)))
            if spikes.size > 0:
                units.append((str(field), spikes))
        return units

    unit_id = mat.get("unit_id")
    spike_times = mat.get("spike_times")
    if unit_id is None or spike_times is None:
        return units

    unit_arr = np.asarray(unit_id).reshape(-1)
    spike_arr = np.asarray(spike_times).reshape(-1)
    n = min(unit_arr.size, spike_arr.size)
    for i in range(n):
        uid = str(unit_arr[i]).strip()
        spikes = _normalize_spike_array(np.asarray(spike_arr[i]))
        if uid and spikes.size > 0:
            units.append((uid, spikes))
    return units


def zscore_1d(vec: np.ndarray) -> np.ndarray | None:
    if vec.size < 2:
        return None
    mean = float(np.mean(vec))
    std = float(np.std(vec))
    if not np.isfinite(std) or std <= 0:
        return None
    return (vec - mean) / std


def compute_psd_db(vec: np.ndarray, sampling_rate: float, nperseg: int) -> tuple[np.ndarray, np.ndarray] | None:
    if vec.size < 8:
        return None
    seg = int(max(8, min(int(nperseg), vec.size)))
    noverlap = seg // 2 if seg > 8 else 0
    freq, psd = welch(
        vec,
        fs=float(sampling_rate),
        window="hann",
        nperseg=seg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    if freq.size == 0 or psd.size == 0:
        return None
    psd_safe = np.maximum(psd, np.finfo(np.float64).tiny)
    return freq, 10.0 * np.log10(psd_safe)


def _psd_remove_1f_residual(
    freq_hz: np.ndarray,
    psd_db: np.ndarray,
    fit_range_hz: tuple[float, float],
) -> np.ndarray | None:
    f = np.asarray(freq_hz, dtype=float).reshape(-1)
    y = np.asarray(psd_db, dtype=float).reshape(-1)
    n = min(f.size, y.size)
    if n < 4:
        return None
    f = f[:n]
    y = y[:n]

    lo, hi = float(fit_range_hz[0]), float(fit_range_hz[1])
    fit_mask = (
        np.isfinite(f)
        & np.isfinite(y)
        & (f > 0.0)
        & (f >= lo)
        & (f <= hi)
    )
    if int(np.sum(fit_mask)) < 3:
        return None

    logf_fit = np.log10(f[fit_mask])
    X_fit = np.vstack([logf_fit, np.ones_like(logf_fit)]).T
    beta, _, _, _ = np.linalg.lstsq(X_fit, y[fit_mask], rcond=None)

    all_mask = np.isfinite(f) & np.isfinite(y) & (f > 0.0)
    if not np.any(all_mask):
        return None
    y_out = np.asarray(y, dtype=float).copy()
    X_all = np.vstack([np.log10(f[all_mask]), np.ones(int(np.sum(all_mask)), dtype=float)]).T
    trend = X_all @ beta
    y_out[all_mask] = y_out[all_mask] - trend
    return y_out


def _require_plotly() -> None:
    if go is None or make_subplots is None:
        raise RuntimeError(f"plotly is required. import error: {PLOTLY_IMPORT_ERROR}")


def _resolve_plotly_js_mode(args: argparse.Namespace) -> str:
    interactive_js = getattr(args, "interactive_js", None)
    if interactive_js is not None:
        return str(interactive_js).strip().lower()
    return str(args.plotly_js).strip().lower()


def _normalize_modes(modes_raw: list[str] | None) -> list[str]:
    if not modes_raw:
        return []
    req: list[str] = []
    for m in modes_raw:
        x = str(m).strip().lower()
        if not x:
            continue
        x = MODE_ALIASES.get(x, x)
        if x not in SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode: {x}. Supported: {', '.join(SUPPORTED_MODES)}. "
                "Aliases: hilphase, zshift, zlag -> phaselag; "
                "fc, synchrony -> syncfc; wavelet -> spectrogram."
            )
        if x not in req:
            req.append(x)

    out: list[str] = []
    seen: set[str] = set()
    for m in req:
        candidates = [m]
        if m == "theta_power":
            candidates.append("theta_delta_ratio")
        if m == "entropy":
            candidates.append("entropy_td")
        for c in candidates:
            if c not in seen:
                out.append(c)
                seen.add(c)
    return out


def _fmt_freq(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:g}"


def _parse_freq_bands(
    tokens: list[str] | tuple[str, ...] | None,
    arg_name: str,
) -> list[tuple[str, tuple[float, float]]]:
    if not tokens:
        raise ValueError(f"{arg_name} must not be empty.")

    bands_raw: list[tuple[float, float]] = []
    flat_vals: list[float] = []
    pair_re = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*$")

    for tok in tokens:
        s = str(tok).strip()
        if not s:
            continue
        m = pair_re.match(s)
        if m:
            bands_raw.append((float(m.group(1)), float(m.group(2))))
            continue
        s2 = s.replace(",", " ")
        for part in s2.split():
            flat_vals.append(float(part))

    if flat_vals:
        if len(flat_vals) % 2 != 0:
            raise ValueError(
                f"Invalid {arg_name} flat list. Provide pairs: e.g. '4 8 4 10 4 12'."
            )
        for i in range(0, len(flat_vals), 2):
            bands_raw.append((float(flat_vals[i]), float(flat_vals[i + 1])))

    if not bands_raw:
        raise ValueError(f"Invalid {arg_name}. Provide e.g. '4-8 4-10' or '4 8 4 10'.")

    out: list[tuple[str, tuple[float, float]]] = []
    seen: set[tuple[float, float]] = set()
    for lo, hi in bands_raw:
        lo_f = float(lo)
        hi_f = float(hi)
        if lo_f >= hi_f:
            raise ValueError(f"Invalid {arg_name} band: [{lo_f}, {hi_f}] (must satisfy low < high).")
        if lo_f < 0:
            raise ValueError(f"Invalid {arg_name} band: [{lo_f}, {hi_f}] (low must be >= 0).")
        key = (lo_f, hi_f)
        if key in seen:
            continue
        seen.add(key)
        out.append((f"{_fmt_freq(lo_f)}-{_fmt_freq(hi_f)}", (lo_f, hi_f)))
    return out


def _parse_theta_bands(tokens: list[str] | tuple[str, ...] | None) -> list[tuple[str, tuple[float, float]]]:
    return _parse_freq_bands(tokens, "--theta_bands")


def _parse_time_range_for_timeseries(
    modes: list[str],
    time_range: list[float] | tuple[float, float] | None,
) -> tuple[float, float] | None:
    if not any(m in TIME_SERIES_MODES for m in modes):
        return None
    if not isinstance(time_range, (list, tuple)) or len(time_range) != 2:
        raise ValueError("--time_range is required for time-series modes and must be [0 T_MAX].")
    t0 = float(time_range[0])
    t1 = float(time_range[1])
    if abs(t0) > 1e-9:
        raise ValueError("--time_range must start at 0 for this dataset (no onset).")
    if t1 <= 0:
        raise ValueError("--time_range T_MAX must be > 0.")
    return (0.0, t1)


def _parse_smooth_window_sec(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.lower() in {"none", "off", "disable", "disabled"}:
        return None
    win = float(s)
    if win <= 0:
        return None
    return win


def _parse_bool_text(value: Any, arg_name: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"{arg_name} must be true/false (got: {value}).")


def _parse_ms_lfp_distance_json(value: str | None) -> dict[str, dict[str, Any]]:
    txt = "{}" if value is None else str(value).strip()
    if not txt:
        return {}
    try:
        parsed = json.loads(txt)
    except Exception as exc:
        raise ValueError(f"--MS_LFP_D must be valid JSON: {exc}") from exc
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("--MS_LFP_D must be a JSON object.")

    out: dict[str, dict[str, Any]] = {}
    for shank_key, unit_map in parsed.items():
        if not isinstance(unit_map, dict):
            raise ValueError(
                f"--MS_LFP_D shank entry '{shank_key}' must be a JSON object of yy->distance."
            )
        out[str(shank_key)] = {str(unit_key): dist_val for unit_key, dist_val in unit_map.items()}
    return out


def _linspace_idx(n: int, m: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.int64)
    if m <= 0 or n <= m:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, num=m, dtype=np.int64)


def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]
    idx = _linspace_idx(x.size, max_points)
    return x[idx], y[idx]


def _downsample_heatmap_time(t: np.ndarray, z: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    tt = np.asarray(t, dtype=float).reshape(-1)
    zz = np.asarray(z, dtype=float)
    if zz.ndim != 2 or tt.size == 0:
        return tt, zz
    n = min(tt.size, zz.shape[1])
    tt = tt[:n]
    zz = zz[:, :n]
    idx = _linspace_idx(n, max_points)
    return tt[idx], zz[:, idx]


def _clip_signal_by_time_range(sig: np.ndarray, sr: float, time_range: tuple[float, float] | None) -> tuple[np.ndarray, np.ndarray] | None:
    n = int(sig.size)
    if n <= 1:
        return None
    t = np.arange(n, dtype=float) / float(sr)
    if time_range is None:
        return t, np.asarray(sig, dtype=float)

    t0, t1 = time_range
    t1_eff = min(float(t1), float(t[-1]))
    m = (t >= float(t0)) & (t <= float(t1_eff))
    if int(np.sum(m)) < 2:
        return None
    return t[m], np.asarray(sig, dtype=float)[m]


def _bandpass_zero_phase(sig: np.ndarray, sr: float, fmin: float, fmax: float, order: int = 4) -> np.ndarray | None:
    nyq = 0.5 * float(sr)
    if nyq <= 0:
        return None
    lo = float(fmin) / nyq
    hi = float(fmax) / nyq
    lo = max(lo, 1e-6)
    hi = min(hi, 0.999999)
    if lo >= hi:
        return None
    sos = butter(int(order), [lo, hi], btype="bandpass", output="sos")
    return sosfiltfilt(sos, np.asarray(sig, dtype=float))


def _prepare_bandpass_sos(
    theta_bands: list[tuple[str, tuple[float, float]]],
    sr: float,
    order: int = 4,
) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    nyq = 0.5 * float(sr)
    if nyq <= 0:
        return out
    for band_label, (fmin, fmax) in theta_bands:
        lo = float(fmin) / nyq
        hi = float(fmax) / nyq
        lo = max(lo, 1e-6)
        hi = min(hi, 0.999999)
        if lo >= hi:
            continue
        sos = butter(int(order), [lo, hi], btype="bandpass", output="sos")
        out.append((band_label, sos))
    return out


def _bandpass_zero_phase_with_sos(sig: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return sosfiltfilt(sos, np.asarray(sig, dtype=float))


def _causal_smooth_series(y: np.ndarray, win_points: int) -> np.ndarray:
    yy = np.asarray(y, dtype=float).reshape(-1)
    if yy.size == 0 or win_points <= 1:
        return yy

    win = int(max(1, win_points))
    val = np.nan_to_num(yy, nan=0.0, posinf=0.0, neginf=0.0)
    ok = np.isfinite(yy).astype(float)
    kernel = np.ones(win, dtype=float)
    num = np.convolve(val, kernel, mode="full")[: yy.size]
    den = np.convolve(ok, kernel, mode="full")[: yy.size]
    out = np.full_like(yy, np.nan, dtype=float)
    m = den > 0
    out[m] = num[m] / den[m]
    return out


def _smooth_by_time(y: np.ndarray, t: np.ndarray, smooth_win_sec: float | None) -> np.ndarray:
    if smooth_win_sec is None:
        return np.asarray(y, dtype=float).reshape(-1)
    tt = np.asarray(t, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    if tt.size != yy.size or tt.size < 2:
        return yy
    dt = np.diff(tt)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return yy
    step = float(np.median(dt))
    if not np.isfinite(step) or step <= 0:
        return yy
    win_points = int(np.ceil(float(smooth_win_sec) / step))
    return _causal_smooth_series(yy, win_points=win_points)


def _db_transform(y: np.ndarray, eps: float) -> np.ndarray:
    yy = np.asarray(y, dtype=float)
    return 10.0 * np.log10(np.maximum(yy, float(eps)))


def _require_mne_multitaper() -> Any:
    try:
        from mne.time_frequency import psd_array_multitaper
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Selected modes require mne.time_frequency.psd_array_multitaper, "
            "but mne is not installed. Please install mne first."
        ) from exc
    return psd_array_multitaper


def _compute_multitaper_psd_windows(
    sig: np.ndarray,
    sr: float,
    win_sec: float,
    step_sec: float,
    fmin: float,
    fmax: float,
    psd_array_multitaper: Any,
    mt_n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    x = np.asarray(sig, dtype=float).reshape(-1)
    if x.size < 8:
        return None

    win_samples = int(round(float(win_sec) * float(sr)))
    step_samples = int(round(float(step_sec) * float(sr)))
    if win_samples <= 1 or step_samples <= 0:
        return None
    if x.size < win_samples:
        return None

    n_windows = int(((x.size - win_samples) // step_samples) + 1)
    if n_windows <= 0:
        return None

    starts = np.arange(n_windows, dtype=np.int64) * int(step_samples)
    segs_view = np.lib.stride_tricks.sliding_window_view(x, win_samples)[::step_samples]
    segs = np.asarray(segs_view, dtype=float)

    try:
        psd, freqs = psd_array_multitaper(
            segs,
            sfreq=float(sr),
            fmin=float(fmin),
            fmax=float(fmax),
            normalization="full",
            n_jobs=int(mt_n_jobs),
            verbose=False,
        )
    except TypeError:
        psd, freqs = psd_array_multitaper(
            segs,
            sfreq=float(sr),
            fmin=float(fmin),
            fmax=float(fmax),
            normalization="full",
            verbose=False,
        )
    psd_arr = np.asarray(psd, dtype=float)
    if psd_arr.ndim == 3:
        psd_arr = psd_arr[:, 0, :]
    if psd_arr.ndim != 2:
        return None

    freqs_arr = np.asarray(freqs, dtype=float).reshape(-1)
    if freqs_arr.size == 0:
        return None
    t_right = (starts.astype(float) + float(win_samples)) / float(sr)
    return freqs_arr, psd_arr, t_right


def _compute_theta_band_power_from_psd(
    freqs: np.ndarray,
    psd_win_f: np.ndarray,
    bands: list[tuple[str, tuple[float, float]]],
) -> dict[str, np.ndarray]:
    f = np.asarray(freqs, dtype=float).reshape(-1)
    p = np.asarray(psd_win_f, dtype=float)
    if p.ndim != 2:
        return {}
    df = float(np.mean(np.diff(f))) if f.size > 1 else 1.0

    out: dict[str, np.ndarray] = {}
    for label, (lo, hi) in bands:
        m = (f >= float(lo)) & (f <= float(hi))
        if not np.any(m):
            continue
        out[label] = np.sum(p[:, m], axis=1) * df
    return out


def _compute_theta_delta_ratio_from_psd(
    freqs: np.ndarray,
    psd_win_f: np.ndarray,
    bands: list[tuple[str, tuple[float, float]]],
    delta_band: tuple[float, float],
    eps: float,
) -> dict[str, np.ndarray]:
    f = np.asarray(freqs, dtype=float).reshape(-1)
    p = np.asarray(psd_win_f, dtype=float)
    if p.ndim != 2:
        return {}
    df = float(np.mean(np.diff(f))) if f.size > 1 else 1.0
    m_delta = (f >= float(delta_band[0])) & (f <= float(delta_band[1]))
    if not np.any(m_delta):
        return {}
    p_delta = np.sum(p[:, m_delta], axis=1) * df
    out: dict[str, np.ndarray] = {}
    for label, (lo, hi) in bands:
        m = (f >= float(lo)) & (f <= float(hi))
        if not np.any(m):
            continue
        p_theta = np.sum(p[:, m], axis=1) * df
        out[label] = p_theta / (p_delta + float(eps))
    return out


def _compute_theta_entropy_from_psd(
    freqs: np.ndarray,
    psd_win_f: np.ndarray,
    bands: list[tuple[str, tuple[float, float]]],
    eps: float,
) -> dict[str, np.ndarray]:
    f = np.asarray(freqs, dtype=float).reshape(-1)
    p = np.asarray(psd_win_f, dtype=float)
    if p.ndim != 2:
        return {}

    out: dict[str, np.ndarray] = {}
    for label, (lo, hi) in bands:
        m = (f >= float(lo)) & (f <= float(hi))
        if not np.any(m):
            continue
        band_psd = np.maximum(p[:, m], 0.0)
        denom = np.sum(band_psd, axis=1, keepdims=True)
        prob = band_psd / (denom + float(eps))
        h = -np.sum(prob * np.log(prob + float(eps)), axis=1)
        n_f = int(np.sum(m))
        if n_f > 1:
            h = h / np.log(float(n_f))
        out[label] = h
    return out


def _compute_entropy_td_from_psd(
    freqs: np.ndarray,
    psd_win_f: np.ndarray,
    bands: list[tuple[str, tuple[float, float]]],
    delta_band: tuple[float, float],
    eps: float,
) -> dict[str, np.ndarray]:
    f = np.asarray(freqs, dtype=float).reshape(-1)
    p = np.asarray(psd_win_f, dtype=float)
    if p.ndim != 2:
        return {}
    df = float(np.mean(np.diff(f))) if f.size > 1 else 1.0
    m_delta = (f >= float(delta_band[0])) & (f <= float(delta_band[1]))
    if not np.any(m_delta):
        return {}
    p_delta = np.maximum(np.sum(p[:, m_delta], axis=1) * df, 0.0)

    out: dict[str, np.ndarray] = {}
    for label, (lo, hi) in bands:
        m = (f >= float(lo)) & (f <= float(hi))
        if not np.any(m):
            continue
        p_theta = np.maximum(np.sum(p[:, m], axis=1) * df, 0.0)
        total = p_theta + p_delta + float(eps)
        p_t = p_theta / total
        p_d = p_delta / total
        h = -(p_t * np.log2(p_t + float(eps)) + p_d * np.log2(p_d + float(eps)))
        out[label] = h
    return out


def _build_time_right(t_min: float, t_max: float, step_sec: float) -> np.ndarray:
    step = float(step_sec)
    if step <= 0:
        return np.array([], dtype=float)
    return np.arange(float(t_min), float(t_max) + 1e-6, step, dtype=float)


def _causal_window_indices(
    t_points: np.ndarray,
    time_centers: np.ndarray,
    win_sec: float,
) -> list[np.ndarray]:
    t = np.asarray(t_points, dtype=float).reshape(-1)
    c = np.asarray(time_centers, dtype=float).reshape(-1)
    if t.size == 0 or c.size == 0:
        return []

    # Fast path for monotonic time axes:
    # avoid O(len(t) * len(c)) boolean masking and build window indices
    # by boundary search.
    win = float(win_sec)
    if np.any(np.diff(t) < 0):
        out_fallback: list[np.ndarray] = []
        for tc in c:
            m = (t >= (float(tc) - win)) & (t <= float(tc))
            out_fallback.append(np.where(m)[0])
        return out_fallback
    left = np.searchsorted(t, c - win, side="left")
    right = np.searchsorted(t, c, side="right")

    out: list[np.ndarray] = []
    for l, r in zip(left, right):
        li = int(l)
        ri = int(r)
        if ri <= li:
            out.append(np.array([], dtype=np.int64))
            continue
        out.append(np.arange(li, ri, dtype=np.int64))
    return out


def _extract_hc_channel_number(hc_name: str) -> int | None:
    m = HC_FILE_RE.match(str(hc_name))
    label = m.group("label") if m else str(hc_name)
    mch = HC_LABEL_CH_RE.search(label)
    if mch is None:
        return None
    try:
        return int(mch.group("ch"))
    except Exception:
        return None


def _select_pseudo_lfp_row(
    lfp_rows: list[dict[str, Any]],
    hc_name: str,
) -> dict[str, Any] | None:
    if not lfp_rows:
        return None
    hc_ch = _extract_hc_channel_number(hc_name)
    if hc_ch is not None:
        for row in lfp_rows:
            try:
                if int(row.get("shank_int", -1)) == int(hc_ch):
                    return row
            except Exception:
                continue
    return lfp_rows[0]


def _merge_pseudo_lfp_rows_all_shanks(
    lfp_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray] | None:
    if not lfp_rows:
        return None

    t_ref: np.ndarray | None = None
    y_sum: np.ndarray | None = None

    for row in lfp_rows:
        t_r = np.asarray(row.get("time_sec", np.array([], dtype=float)), dtype=float).reshape(-1)
        y_r = np.asarray(row.get("lfp", np.array([], dtype=float)), dtype=float).reshape(-1)
        n = min(t_r.size, y_r.size)
        if n < 2:
            continue
        t_r = t_r[:n]
        y_r = y_r[:n]
        m = np.isfinite(t_r) & np.isfinite(y_r)
        if int(np.sum(m)) < 2:
            continue
        t_r = t_r[m]
        y_r = y_r[m]

        t_u, idx_u = np.unique(t_r, return_index=True)
        if t_u.size < 2:
            continue
        y_u = y_r[idx_u]

        if t_ref is None:
            t_ref = np.asarray(t_u, dtype=float)
            y_sum = np.asarray(y_u, dtype=float)
            continue

        assert y_sum is not None
        y_i = np.interp(t_ref, t_u, y_u, left=np.nan, right=np.nan)
        m_i = np.isfinite(y_i)
        if np.any(m_i):
            y_sum[m_i] += np.asarray(y_i[m_i], dtype=float)

    if t_ref is None or y_sum is None:
        return None
    m_out = np.isfinite(t_ref) & np.isfinite(y_sum)
    if int(np.sum(m_out)) < 2:
        return None
    if (y_out := zscore_1d(np.asarray(y_sum[m_out], dtype=float))) is None:
        return None
    return np.asarray(t_ref[m_out], dtype=float), np.asarray(y_out, dtype=float)


def _prepare_hc_pseudo_pair_meta(
    entry: EEGEntry,
    analysis_sampling_rate: float,
    spike_sampling_rate: float,
    time_range: tuple[float, float],
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None:
    clipped = _clip_signal_by_time_range(entry.eeg_z, analysis_sampling_rate, time_range)
    if clipped is None:
        return None
    t_hc, x_hc = clipped
    if t_hc.size < 2 or x_hc.size < 2:
        return None
    if not entry.ms_units:
        return None

    lfp_rows, lfp_warnings = synthesize_shank_lfp(
        units=entry.ms_units,
        sampling_rate=float(spike_sampling_rate),
        time_range_sec=(float(t_hc[0]), float(t_hc[-1])),
        sigma_sec=float(ms_lfp_sigma),
        amplitude=float(ms_lfp_a),
        gain_a0=float(ms_lfp_a0),
        distance_map=ms_lfp_distance_map,
        default_distance=float(ms_lfp_d_default),
        post_smooth_sec=float(ms_lfp_post_smooth_sec),
    )
    for w in lfp_warnings:
        print(f"\033[1;33m -- [{entry.subject} - {entry.session}] {w}\033[0m")

    merged = _merge_pseudo_lfp_rows_all_shanks(lfp_rows)
    if merged is None:
        return None
    t_p, y_p = merged

    t_p_uniq, uniq_idx = np.unique(t_p, return_index=True)
    if t_p_uniq.size < 2:
        return None
    y_p_uniq = y_p[uniq_idx]

    # Align pseudo-LFP preprocessing with HC preprocessing:
    # when analysis sampling is lower than spike sampling, apply
    # anti-alias lowpass + resample_poly before interpolation.
    t_p_proc = np.asarray(t_p_uniq, dtype=float)
    y_p_proc = np.asarray(y_p_uniq, dtype=float)
    sr_in = float(spike_sampling_rate)
    sr_out = float(analysis_sampling_rate)
    if sr_out > 0 and sr_in > 0 and (sr_out < (sr_in - 1e-9)):
        try:
            pseudo_down_cfg = _build_downsample_config(
                sampling_rate_in=sr_in,
                downsample_rate=sr_out,
                aa_order=6,
            )
        except Exception:
            pseudo_down_cfg = None
        if pseudo_down_cfg is not None:
            y_ds = _apply_aa_downsample(y_p_proc, pseudo_down_cfg)
            if y_ds.size >= 2:
                t0_ds = float(t_p_proc[0])
                fs_ds = float(pseudo_down_cfg.sampling_rate_out)
                t_ds = t0_ds + (np.arange(y_ds.size, dtype=float) / fs_ds)
                m_ds = np.isfinite(t_ds) & np.isfinite(y_ds)
                if int(np.sum(m_ds)) >= 2:
                    t_p_proc = np.asarray(t_ds[m_ds], dtype=float)
                    y_p_proc = np.asarray(y_ds[m_ds], dtype=float)

    y_interp = np.interp(t_hc, t_p_proc, y_p_proc, left=np.nan, right=np.nan)
    mm = np.isfinite(t_hc) & np.isfinite(x_hc) & np.isfinite(y_interp)
    if int(np.sum(mm)) < 2:
        return None
    meta = {
        "shank_int": -1,
        "shank_label": "ALL TT",
    }
    return (
        t_hc[mm],
        np.asarray(x_hc, dtype=float)[mm],
        np.asarray(y_interp, dtype=float)[mm],
        meta,
    )


def _prepare_hc_pseudo_pair(
    entry: EEGEntry,
    analysis_sampling_rate: float,
    spike_sampling_rate: float,
    time_range: tuple[float, float],
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    out = _prepare_hc_pseudo_pair_meta(
        entry=entry,
        analysis_sampling_rate=analysis_sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        time_range=time_range,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
    )
    if out is None:
        return None
    t_hc, x_hc, y_p, _ = out
    return t_hc, x_hc, y_p


def _compute_coherence_band_timeseries_pair(
    x_sig: np.ndarray,
    y_sig: np.ndarray,
    t_points: np.ndarray,
    sr: float,
    win_sec: float,
    step_sec: float,
    bands: list[tuple[str, tuple[float, float]]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    x = np.asarray(x_sig, dtype=float).reshape(-1)
    y = np.asarray(y_sig, dtype=float).reshape(-1)
    t = np.asarray(t_points, dtype=float).reshape(-1)
    n = min(x.size, y.size, t.size)
    if n < 2:
        return np.array([], dtype=np.float32), {}, {}
    x = x[:n]
    y = y[:n]
    t = t[:n]

    time_centers = _build_time_right(float(t[0]), float(t[-1]), step_sec)
    if time_centers.size == 0:
        return np.array([], dtype=np.float32), {}, {}
    indices = _causal_window_indices(t, time_centers, win_sec=win_sec)

    coh_out: dict[str, np.ndarray] = {}
    plv_out: dict[str, np.ndarray] = {}
    nyq = float(sr) / 2.0

    for label, (fmin, fmax) in bands:
        fmin_f = float(fmin)
        fmax_f = float(fmax)
        if fmin_f <= 0 or fmax_f <= 0 or fmin_f >= fmax_f or fmax_f >= nyq:
            raise ValueError(f"Invalid band [{fmin_f}, {fmax_f}] for sr={sr}.")

        sos = butter(4, [fmin_f, fmax_f], btype="bandpass", fs=float(sr), output="sos")
        x_f = sosfiltfilt(sos, x)
        y_f = sosfiltfilt(sos, y)
        x_h = hilbert(x_f)
        y_h = hilbert(y_f)

        coh_vals = np.full(time_centers.size, np.nan, dtype=np.float32)
        plv_vals = np.full(time_centers.size, np.nan, dtype=np.float32)

        for i, idx in enumerate(indices):
            if idx.size < 2:
                continue
            seg_x = x_h[idx]
            seg_y = y_h[idx]
            pxx = float(np.mean(np.abs(seg_x) ** 2))
            pyy = float(np.mean(np.abs(seg_y) ** 2))
            if pxx <= 0 or pyy <= 0:
                continue
            pxy = np.mean(seg_x * np.conjugate(seg_y))
            coh_vals[i] = float((np.abs(pxy) ** 2) / (pxx * pyy))

            dphi = np.angle(seg_x) - np.angle(seg_y)
            plv_vals[i] = float(np.abs(np.mean(np.exp(1j * dphi))))

        coh_out[label] = coh_vals
        plv_out[label] = plv_vals

    return time_centers.astype(np.float32), coh_out, plv_out


def _stable_seed_u32(*parts: Any) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), byteorder="little", signed=False) & 0xFFFFFFFF


def _windowed_sync_metrics_from_analytic(
    x_h: np.ndarray,
    y_h: np.ndarray,
    t_points: np.ndarray,
    time_centers: np.ndarray,
    win_sec: float,
    min_samples: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_h, dtype=np.complex128).reshape(-1)
    y = np.asarray(y_h, dtype=np.complex128).reshape(-1)
    t = np.asarray(t_points, dtype=float).reshape(-1)
    centers = np.asarray(time_centers, dtype=float).reshape(-1)
    n = min(x.size, y.size, t.size)
    coh = np.full(centers.size, np.nan, dtype=np.float32)
    plv = np.full(centers.size, np.nan, dtype=np.float32)
    if n < int(min_samples) or centers.size == 0:
        return coh, plv
    x = x[:n]
    y = y[:n]
    t = t[:n]

    amp_x = np.abs(x)
    amp_y = np.abs(y)
    cross = x * np.conjugate(y)
    pxx = amp_x * amp_x
    pyy = amp_y * amp_y
    ok = (
        np.isfinite(cross.real)
        & np.isfinite(cross.imag)
        & np.isfinite(pxx)
        & np.isfinite(pyy)
        & (pxx > 0.0)
        & (pyy > 0.0)
    )

    cross_clean = np.where(ok, cross, 0.0 + 0.0j)
    pxx_clean = np.where(ok, pxx, 0.0)
    pyy_clean = np.where(ok, pyy, 0.0)
    unit = np.zeros(n, dtype=np.complex128)
    unit[ok] = (x[ok] / amp_x[ok]) * np.conjugate(y[ok] / amp_y[ok])

    if not np.any(np.diff(t) < 0):
        left = np.searchsorted(t, centers - float(win_sec), side="left").astype(np.int64)
        right = np.searchsorted(t, centers, side="right").astype(np.int64)

        c_cross = np.empty(n + 1, dtype=np.complex128)
        c_cross[0] = 0.0 + 0.0j
        c_cross[1:] = np.cumsum(cross_clean, dtype=np.complex128)
        c_pxx = np.empty(n + 1, dtype=float)
        c_pxx[0] = 0.0
        c_pxx[1:] = np.cumsum(pxx_clean, dtype=float)
        c_pyy = np.empty(n + 1, dtype=float)
        c_pyy[0] = 0.0
        c_pyy[1:] = np.cumsum(pyy_clean, dtype=float)
        c_unit = np.empty(n + 1, dtype=np.complex128)
        c_unit[0] = 0.0 + 0.0j
        c_unit[1:] = np.cumsum(unit, dtype=np.complex128)
        c_count = np.empty(n + 1, dtype=np.int64)
        c_count[0] = 0
        c_count[1:] = np.cumsum(ok, dtype=np.int64)

        counts = c_count[right] - c_count[left]
        valid = counts >= int(min_samples)
        if np.any(valid):
            sum_cross = c_cross[right[valid]] - c_cross[left[valid]]
            sum_pxx = c_pxx[right[valid]] - c_pxx[left[valid]]
            sum_pyy = c_pyy[right[valid]] - c_pyy[left[valid]]
            denom = sum_pxx * sum_pyy
            coh_vals = np.full(sum_cross.size, np.nan, dtype=float)
            good = denom > 0.0
            coh_vals[good] = (np.abs(sum_cross[good]) ** 2) / denom[good]
            coh[valid] = np.asarray(np.clip(coh_vals, 0.0, 1.0), dtype=np.float32)

            sum_unit = c_unit[right[valid]] - c_unit[left[valid]]
            plv[valid] = np.asarray(
                np.clip(np.abs(sum_unit) / counts[valid], 0.0, 1.0),
                dtype=np.float32,
            )
        return coh, plv

    indices = _causal_window_indices(t, centers, win_sec=win_sec)
    for i, idx in enumerate(indices):
        if idx.size < int(min_samples):
            continue
        idx_ok = idx[ok[idx]]
        if idx_ok.size < int(min_samples):
            continue
        sx = float(np.sum(pxx_clean[idx_ok]))
        sy = float(np.sum(pyy_clean[idx_ok]))
        if sx > 0.0 and sy > 0.0:
            sc = np.sum(cross_clean[idx_ok])
            coh[i] = np.float32(np.clip((np.abs(sc) ** 2) / (sx * sy), 0.0, 1.0))
        su = np.sum(unit[idx_ok])
        plv[i] = np.float32(np.clip(np.abs(su) / float(idx_ok.size), 0.0, 1.0))
    return coh, plv


def _syncfc_surrogate_shifts(
    n_samples: int,
    n_surrogates: int,
    min_shift_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(n_samples)
    ns = int(max(0, n_surrogates))
    if ns <= 0 or n < 4:
        return np.zeros(0, dtype=np.int64)
    min_shift = int(max(1, min(int(min_shift_samples), n // 2)))
    max_shift = int(n - min_shift)
    if max_shift < min_shift:
        return np.full(ns, max(1, n // 2), dtype=np.int64)
    return rng.integers(min_shift, max_shift + 1, size=ns, dtype=np.int64)


def _syncfc_pair_metrics_from_indices(
    x_h: np.ndarray,
    y_h: np.ndarray,
    idx_x: np.ndarray,
    idx_y: np.ndarray,
    min_samples: int = 3,
) -> tuple[float, float]:
    x = np.asarray(x_h, dtype=np.complex128).reshape(-1)
    y = np.asarray(y_h, dtype=np.complex128).reshape(-1)
    ix = np.asarray(idx_x, dtype=np.int64).reshape(-1)
    iy = np.asarray(idx_y, dtype=np.int64).reshape(-1)
    m = int(min(ix.size, iy.size))
    if m < int(min_samples):
        return float("nan"), float("nan")
    sx = x[ix[:m]]
    sy = y[iy[:m]]
    ax = np.abs(sx)
    ay = np.abs(sy)
    ok = (
        np.isfinite(sx.real)
        & np.isfinite(sx.imag)
        & np.isfinite(sy.real)
        & np.isfinite(sy.imag)
        & np.isfinite(ax)
        & np.isfinite(ay)
        & (ax > 0.0)
        & (ay > 0.0)
    )
    if int(np.sum(ok)) < int(min_samples):
        return float("nan"), float("nan")
    sx = sx[ok]
    sy = sy[ok]
    ax = ax[ok]
    ay = ay[ok]
    pxx = float(np.sum(ax * ax))
    pyy = float(np.sum(ay * ay))
    coh = float("nan")
    if pxx > 0.0 and pyy > 0.0:
        cross = np.sum(sx * np.conjugate(sy))
        coh = float(np.clip((np.abs(cross) ** 2) / (pxx * pyy), 0.0, 1.0))
    unit = (sx / ax) * np.conjugate(sy / ay)
    plv = float(np.clip(np.abs(np.mean(unit)), 0.0, 1.0))
    return coh, plv


def _deranged_window_permutation(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base = np.asarray(indices, dtype=np.int64).reshape(-1)
    if base.size < 2:
        return np.zeros(0, dtype=np.int64)
    for _ in range(32):
        perm = rng.permutation(base)
        if not np.any(perm == base):
            return perm.astype(np.int64, copy=False)
    shift = int(rng.integers(1, int(base.size)))
    return np.roll(base, shift).astype(np.int64, copy=False)


def _sample_values_for_plot(
    values: np.ndarray,
    max_values: int,
    rng: np.random.Generator,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    vals = vals[np.isfinite(vals)]
    n_max = int(max(0, max_values))
    if n_max > 0 and vals.size > n_max:
        keep = rng.choice(vals.size, size=n_max, replace=False)
        vals = vals[np.sort(keep)]
    return vals.astype(np.float32, copy=False)


def _syncfc_window_shuffle_surrogates(
    x_h: np.ndarray,
    y_h: np.ndarray,
    t_points: np.ndarray,
    time_centers: np.ndarray,
    win_sec: float,
    n_surrogates: int,
    rng: np.random.Generator,
    valid_mask: np.ndarray | None = None,
    min_samples: int = 3,
    max_plot_values: int = 5000,
) -> dict[str, Any]:
    ns = int(max(0, n_surrogates))
    empty = {
        "coh_medians": np.zeros(0, dtype=np.float32),
        "plv_medians": np.zeros(0, dtype=np.float32),
        "coh_plot": np.zeros(0, dtype=np.float32),
        "plv_plot": np.zeros(0, dtype=np.float32),
        "n_pairs": 0,
    }
    if ns <= 0:
        return empty

    centers = np.asarray(time_centers, dtype=float).reshape(-1)
    indices = _causal_window_indices(np.asarray(t_points, dtype=float), centers, win_sec=float(win_sec))
    if not indices:
        return empty
    valid = np.asarray([idx.size >= int(min_samples) for idx in indices], dtype=bool)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        n_mask = min(valid.size, mask.size)
        valid[:n_mask] &= mask[:n_mask]
        if n_mask < valid.size:
            valid[n_mask:] = False
    valid_idx = np.where(valid)[0].astype(np.int64)
    if valid_idx.size < 2:
        return empty

    coh_medians: list[float] = []
    plv_medians: list[float] = []
    coh_plot_values: list[float] = []
    plv_plot_values: list[float] = []
    for _ in range(ns):
        shuffled_idx = _deranged_window_permutation(valid_idx, rng)
        if shuffled_idx.size != valid_idx.size:
            continue
        coh_perm = np.full(valid_idx.size, np.nan, dtype=np.float32)
        plv_perm = np.full(valid_idx.size, np.nan, dtype=np.float32)
        for k, (i_win, j_win) in enumerate(zip(valid_idx, shuffled_idx)):
            coh, plv = _syncfc_pair_metrics_from_indices(
                x_h=x_h,
                y_h=y_h,
                idx_x=indices[int(i_win)],
                idx_y=indices[int(j_win)],
                min_samples=min_samples,
            )
            coh_perm[k] = np.float32(coh)
            plv_perm[k] = np.float32(plv)
        coh_f = coh_perm[np.isfinite(coh_perm)]
        plv_f = plv_perm[np.isfinite(plv_perm)]
        if coh_f.size:
            coh_medians.append(float(np.nanmedian(coh_f)))
            coh_plot_values.extend(float(v) for v in coh_f)
        if plv_f.size:
            plv_medians.append(float(np.nanmedian(plv_f)))
            plv_plot_values.extend(float(v) for v in plv_f)

    return {
        "coh_medians": np.asarray(coh_medians, dtype=np.float32),
        "plv_medians": np.asarray(plv_medians, dtype=np.float32),
        "coh_plot": _sample_values_for_plot(np.asarray(coh_plot_values, dtype=np.float32), max_plot_values, rng),
        "plv_plot": _sample_values_for_plot(np.asarray(plv_plot_values, dtype=np.float32), max_plot_values, rng),
        "n_pairs": int(valid_idx.size),
    }


def _metric_surrogate_summary(real_values: np.ndarray, surrogate_medians: np.ndarray) -> dict[str, float]:
    real = np.asarray(real_values, dtype=float).reshape(-1)
    real = real[np.isfinite(real)]
    real_median = float(np.nanmedian(real)) if real.size else float("nan")
    surr = np.asarray(surrogate_medians, dtype=float).reshape(-1)
    surr = surr[np.isfinite(surr)]
    if surr.size == 0:
        return {
            "real_median": real_median,
            "surrogate_mean": float("nan"),
            "surrogate_sd": float("nan"),
            "surrogate_z": float("nan"),
            "surrogate_p": float("nan"),
        }
    s_mean = float(np.mean(surr))
    s_sd = float(np.std(surr, ddof=1)) if surr.size > 1 else float("nan")
    z = float("nan")
    if np.isfinite(real_median) and np.isfinite(s_sd) and s_sd > 0.0:
        z = float((real_median - s_mean) / s_sd)
    p = float((1 + int(np.sum(surr >= real_median))) / (surr.size + 1)) if np.isfinite(real_median) else float("nan")
    return {
        "real_median": real_median,
        "surrogate_mean": s_mean,
        "surrogate_sd": s_sd,
        "surrogate_z": z,
        "surrogate_p": p,
    }


def _compute_syncfc_band_timeseries_pair(
    x_sig: np.ndarray,
    y_sig: np.ndarray,
    t_points: np.ndarray,
    sr: float,
    win_sec: float,
    step_sec: float,
    bands: list[tuple[str, tuple[float, float]]],
    n_surrogates: int,
    min_shift_sec: float,
    seed: int,
    null_mode: str = "window_shuffle",
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    x = np.asarray(x_sig, dtype=float).reshape(-1)
    y = np.asarray(y_sig, dtype=float).reshape(-1)
    t = np.asarray(t_points, dtype=float).reshape(-1)
    n = min(x.size, y.size, t.size)
    if n < 2:
        return np.array([], dtype=np.float32), {}
    x = x[:n]
    y = y[:n]
    t = t[:n]

    time_centers = _build_time_right(float(t[0]), float(t[-1]), step_sec)
    if time_centers.size == 0:
        return np.array([], dtype=np.float32), {}

    out: dict[str, dict[str, Any]] = {}
    sr_f = float(sr)
    nyq = sr_f / 2.0
    min_shift_samples = int(round(max(0.0, float(min_shift_sec)) * sr_f))
    null_mode_s = str(null_mode).strip().lower()
    if null_mode_s not in {"window_shuffle", "circular_shift", "none"}:
        null_mode_s = "window_shuffle"

    for label, (fmin, fmax) in bands:
        fmin_f = float(fmin)
        fmax_f = float(fmax)
        if fmin_f <= 0 or fmax_f <= 0 or fmin_f >= fmax_f or fmax_f >= nyq:
            print(
                f"\033[1;33m -- SyncFC: skipping invalid band "
                f"{label} [{fmin_f:g}, {fmax_f:g}] for sr={sr_f:g}Hz\033[0m"
            )
            continue

        sos = butter(4, [fmin_f, fmax_f], btype="bandpass", fs=sr_f, output="sos")
        x_f = sosfiltfilt(sos, x)
        y_f = sosfiltfilt(sos, y)
        x_h = hilbert(x_f)
        y_h = hilbert(y_f)

        coh_vals, plv_vals = _windowed_sync_metrics_from_analytic(
            x_h=x_h,
            y_h=y_h,
            t_points=t,
            time_centers=time_centers,
            win_sec=float(win_sec),
        )
        valid = np.isfinite(coh_vals) | np.isfinite(plv_vals)
        if not np.any(valid):
            continue

        surr_coh_medians: list[float] = []
        surr_plv_medians: list[float] = []
        null_coh_plot = np.zeros(0, dtype=np.float32)
        null_plv_plot = np.zeros(0, dtype=np.float32)
        n_null = 0
        shifts = np.zeros(0, dtype=np.int64)
        if int(n_surrogates) > 0 and null_mode_s == "window_shuffle":
            rng = np.random.default_rng(_stable_seed_u32(seed, "window_shuffle", label, fmin_f, fmax_f))
            null = _syncfc_window_shuffle_surrogates(
                x_h=x_h,
                y_h=y_h,
                t_points=t,
                time_centers=time_centers,
                win_sec=float(win_sec),
                n_surrogates=int(n_surrogates),
                rng=rng,
                valid_mask=valid,
                max_plot_values=5000,
            )
            surr_coh_medians = [float(v) for v in np.asarray(null["coh_medians"], dtype=float)]
            surr_plv_medians = [float(v) for v in np.asarray(null["plv_medians"], dtype=float)]
            null_coh_plot = np.asarray(null["coh_plot"], dtype=np.float32)
            null_plv_plot = np.asarray(null["plv_plot"], dtype=np.float32)
            n_null = int(max(len(surr_coh_medians), len(surr_plv_medians)))
        elif int(n_surrogates) > 0 and null_mode_s == "circular_shift":
            rng = np.random.default_rng(_stable_seed_u32(seed, label, fmin_f, fmax_f))
            shifts = _syncfc_surrogate_shifts(
                n_samples=n,
                n_surrogates=int(n_surrogates),
                min_shift_samples=min_shift_samples,
                rng=rng,
            )
            coh_plot_values: list[float] = []
            plv_plot_values: list[float] = []
            for shift in shifts:
                y_shift = np.roll(y_h, int(shift))
                coh_s, plv_s = _windowed_sync_metrics_from_analytic(
                    x_h=x_h,
                    y_h=y_shift,
                    t_points=t,
                    time_centers=time_centers,
                    win_sec=float(win_sec),
                )
                coh_s_f = coh_s[np.isfinite(coh_s)]
                plv_s_f = plv_s[np.isfinite(plv_s)]
                if coh_s_f.size:
                    surr_coh_medians.append(float(np.nanmedian(coh_s_f)))
                    coh_plot_values.extend(float(v) for v in coh_s_f)
                if plv_s_f.size:
                    surr_plv_medians.append(float(np.nanmedian(plv_s_f)))
                    plv_plot_values.extend(float(v) for v in plv_s_f)
            null_coh_plot = _sample_values_for_plot(np.asarray(coh_plot_values, dtype=np.float32), 5000, rng)
            null_plv_plot = _sample_values_for_plot(np.asarray(plv_plot_values, dtype=np.float32), 5000, rng)
            n_null = int(shifts.size)

        coh_summary = _metric_surrogate_summary(coh_vals, np.asarray(surr_coh_medians, dtype=float))
        plv_summary = _metric_surrogate_summary(plv_vals, np.asarray(surr_plv_medians, dtype=float))
        median_coh = float(coh_summary["real_median"])
        median_plv = float(plv_summary["real_median"])
        surrogate_mean_coh = float(coh_summary["surrogate_mean"])
        surrogate_mean_plv = float(plv_summary["surrogate_mean"])
        out[str(label)] = {
            "band": (fmin_f, fmax_f),
            "coh": np.asarray(coh_vals, dtype=np.float32),
            "plv": np.asarray(plv_vals, dtype=np.float32),
            "null_coh": np.asarray(null_coh_plot, dtype=np.float32),
            "null_plv": np.asarray(null_plv_plot, dtype=np.float32),
            "n_windows": int(np.sum(valid)),
            "n_total_windows": int(time_centers.size),
            "n_surrogates": int(n_null),
            "null_mode": null_mode_s,
            "median_coh": median_coh,
            "median_plv": median_plv,
            "surrogate_mean_coh": surrogate_mean_coh,
            "surrogate_sd_coh": float(coh_summary["surrogate_sd"]),
            "z_coh": float(coh_summary["surrogate_z"]),
            "p_coh": float(coh_summary["surrogate_p"]),
            "surrogate_mean_plv": surrogate_mean_plv,
            "surrogate_sd_plv": float(plv_summary["surrogate_sd"]),
            "z_plv": float(plv_summary["surrogate_z"]),
            "p_plv": float(plv_summary["surrogate_p"]),
            "delta_coh": median_coh - surrogate_mean_coh if np.isfinite(median_coh) and np.isfinite(surrogate_mean_coh) else float("nan"),
            "delta_plv": median_plv - surrogate_mean_plv if np.isfinite(median_plv) and np.isfinite(surrogate_mean_plv) else float("nan"),
        }

    return time_centers.astype(np.float32), out


def _filter_phaselag_z_matrix(
    z_matrix: np.ndarray,
    count_matrix: np.ndarray,
    min_z: float,
    min_plv: float,
    min_peak_delta_z: float,
    min_peak_delta_frac: float,
    exclude_edge_peaks: bool,
    edge_guard_bins: int,
) -> dict[str, np.ndarray]:
    z = np.asarray(z_matrix, dtype=float)
    counts = np.asarray(count_matrix, dtype=float)
    if z.ndim != 2 or counts.shape != z.shape:
        return {
            "keep": np.zeros(0, dtype=bool),
            "best_idx": np.zeros(0, dtype=np.int64),
            "best_z": np.zeros(0, dtype=float),
            "best_plv": np.zeros(0, dtype=float),
            "second_z": np.zeros(0, dtype=float),
            "edge_hit": np.zeros(0, dtype=bool),
        }

    n_win = int(z.shape[0])
    n_lags = int(z.shape[1])
    keep = np.zeros(n_win, dtype=bool)
    best_idx = np.full(n_win, -1, dtype=np.int64)
    best_z = np.full(n_win, np.nan, dtype=float)
    best_plv = np.full(n_win, np.nan, dtype=float)
    second_z = np.full(n_win, np.nan, dtype=float)
    edge_hit = np.zeros(n_win, dtype=bool)

    min_z_f = float(max(0.0, min_z))
    min_plv_f = float(max(0.0, min_plv))
    min_peak_delta_z_f = float(max(0.0, min_peak_delta_z))
    min_peak_delta_frac_f = float(max(0.0, min_peak_delta_frac))
    edge_guard = int(max(0, edge_guard_bins))
    if n_lags > 0:
        edge_guard = int(min(edge_guard, max(0, (n_lags - 1) // 2)))

    for wi in range(n_win):
        row = np.asarray(z[wi], dtype=float)
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        z_fill = np.where(finite, row, -np.inf)
        bi = int(np.argmax(z_fill))
        bz = float(z_fill[bi])
        if not np.isfinite(bz):
            continue
        c = float(counts[wi, bi])
        if (not np.isfinite(c)) or c <= 0:
            continue
        row_second = z_fill.copy()
        row_second[bi] = -np.inf
        sz = float(np.max(row_second)) if np.any(np.isfinite(row_second)) else np.nan
        plv = float(np.sqrt(max(0.0, bz / c)))
        is_edge_hit = bool(bi <= edge_guard or bi >= (n_lags - 1 - edge_guard))

        good = True
        if bool(exclude_edge_peaks) and is_edge_hit:
            good = False
        if min_z_f > 0.0 and bz < min_z_f:
            good = False
        if good and min_plv_f > 0.0 and plv < min_plv_f:
            good = False
        if good and (min_peak_delta_z_f > 0.0 or min_peak_delta_frac_f > 0.0):
            delta_z = bz - sz if np.isfinite(sz) else np.inf
            if min_peak_delta_z_f > 0.0 and delta_z < min_peak_delta_z_f:
                good = False
            if good and min_peak_delta_frac_f > 0.0:
                delta_frac = delta_z / bz if bz > 0 else np.nan
                if (not np.isfinite(delta_frac)) or delta_frac < min_peak_delta_frac_f:
                    good = False

        best_idx[wi] = bi
        best_z[wi] = bz
        best_plv[wi] = plv
        second_z[wi] = sz
        edge_hit[wi] = is_edge_hit
        keep[wi] = bool(good)

    return {
        "keep": keep,
        "best_idx": best_idx,
        "best_z": best_z,
        "best_plv": best_plv,
        "second_z": second_z,
        "edge_hit": edge_hit,
    }


def _compute_phaselag_summary_pair(
    x_sig: np.ndarray,
    y_sig: np.ndarray,
    t_points: np.ndarray,
    sr: float,
    win_sec: float,
    step_sec: float,
    bands: list[tuple[str, tuple[float, float]]],
    min_z: float,
    min_plv: float,
    min_peak_delta_z: float,
    min_peak_delta_frac: float,
    min_valid_ratio: float,
    exclude_edge_peaks: bool,
    edge_guard_bins: int,
) -> dict[str, dict[str, Any]]:
    x = np.asarray(x_sig, dtype=float).reshape(-1)
    y = np.asarray(y_sig, dtype=float).reshape(-1)
    t = np.asarray(t_points, dtype=float).reshape(-1)
    n = min(x.size, y.size, t.size)
    if n < 2:
        return {}
    x = x[:n]
    y = y[:n]
    t = t[:n]

    time_centers = _build_time_right(float(t[0]), float(t[-1]), step_sec)
    if time_centers.size == 0:
        return {}
    is_monotonic_time = not np.any(np.diff(t) < 0)
    if is_monotonic_time:
        win_left = np.searchsorted(t, time_centers - float(win_sec), side="left")
        win_right = np.searchsorted(t, time_centers, side="right")
    else:
        indices = _causal_window_indices(t, time_centers, win_sec=win_sec)

    out: dict[str, dict[str, Any]] = {}
    sr_f = float(sr)
    nyq = sr_f / 2.0
    min_window_samples = int(max(3, round(float(win_sec) * sr_f)))

    for label, (fmin, fmax) in bands:
        fmin_f = float(fmin)
        fmax_f = float(fmax)
        if fmin_f <= 0 or fmax_f <= 0 or fmin_f >= fmax_f or fmax_f >= nyq:
            print(
                f"\033[1;33m -- PhaseLag: skipping invalid band "
                f"{label} [{fmin_f:g}, {fmax_f:g}] for sr={sr_f:g}Hz\033[0m"
            )
            continue

        sos = butter(4, [fmin_f, fmax_f], btype="bandpass", fs=sr_f, output="sos")
        x_f = sosfiltfilt(sos, x)
        y_f = sosfiltfilt(sos, y)
        phase_hc = np.angle(hilbert(x_f))
        phase_ms = np.angle(hilbert(y_f))

        center_hz = 0.5 * (fmin_f + fmax_f)
        max_lag_samples = int(round((0.5 / center_hz) * sr_f))
        max_lag_samples = int(max(0, max_lag_samples))
        lags = np.arange(-max_lag_samples, max_lag_samples + 1, dtype=np.int64)
        lag_ms = (lags.astype(float) / sr_f) * 1000.0
        min_common_samples = int(max(3, min_window_samples - (2 * max_lag_samples)))

        if is_monotonic_time:
            window_lengths = win_right - win_left
            base_start = np.maximum(win_left, max_lag_samples)
            base_end = np.minimum(win_right, n - max_lag_samples)
            base_counts_nominal = base_end - base_start
            valid_windows = (
                (window_lengths >= min_window_samples)
                & (base_counts_nominal >= min_common_samples)
            )
            valid_window_idx = np.where(valid_windows)[0]
            z_matrix = np.full((valid_window_idx.size, lags.size), np.nan, dtype=np.float32)
            count_matrix = np.zeros((valid_window_idx.size, lags.size), dtype=np.float32)

            base_offset = int(max_lag_samples)
            base_stop = int(n - max_lag_samples)
            if valid_window_idx.size > 0 and base_stop > base_offset:
                rel_start = (base_start[valid_window_idx] - base_offset).astype(np.int64)
                rel_end = (base_end[valid_window_idx] - base_offset).astype(np.int64)
                base_idx = np.arange(base_offset, base_stop, dtype=np.int64)
                e_ms = np.exp(1j * phase_ms)
                e_hc_conj = np.exp(-1j * phase_hc)

                for li, lag_samples in enumerate(lags):
                    q_raw = e_ms[base_idx] * e_hc_conj[base_idx + int(lag_samples)]
                    q_ok = np.isfinite(q_raw.real) & np.isfinite(q_raw.imag)
                    q = np.where(q_ok, q_raw, 0.0 + 0.0j)

                    csum_q = np.empty(q.size + 1, dtype=np.complex128)
                    csum_q[0] = 0.0 + 0.0j
                    csum_q[1:] = np.cumsum(q, dtype=np.complex128)
                    sums = csum_q[rel_end] - csum_q[rel_start]

                    csum_n = np.empty(q_ok.size + 1, dtype=np.int64)
                    csum_n[0] = 0
                    csum_n[1:] = np.cumsum(q_ok, dtype=np.int64)
                    counts = csum_n[rel_end] - csum_n[rel_start]

                    valid_counts = counts >= 3
                    if not np.any(valid_counts):
                        continue
                    abs_sum = np.abs(sums[valid_counts])
                    z_matrix[valid_counts, li] = np.asarray(
                        (abs_sum * abs_sum) / counts[valid_counts],
                        dtype=np.float32,
                    )
                    count_matrix[valid_counts, li] = counts[valid_counts].astype(np.float32)

            # Lag0 / zero-phase score concept kept for possible future use:
            #   C(tau) = mean(exp(1j * delta_phase(tau)))
            #   S0(tau) = n * max(real(C(tau)), 0)^2
            # It is intentionally not used in the current Z-shift output.
        else:
            z_rows: list[np.ndarray] = []
            count_rows: list[np.ndarray] = []
            for i, idx in enumerate(indices):
                if idx.size < min_window_samples:
                    continue
                idx_base = idx[(idx >= max_lag_samples) & (idx < (n - max_lag_samples))]
                if idx_base.size < min_common_samples:
                    continue

                z_row = np.full(lags.size, np.nan, dtype=np.float32)
                count_row = np.zeros(lags.size, dtype=np.float32)

                for li, lag_samples in enumerate(lags):
                    hc_idx = idx_base + int(lag_samples)
                    m_idx = (hc_idx >= 0) & (hc_idx < n)
                    if int(np.sum(m_idx)) < 3:
                        continue

                    ms_idx = idx_base[m_idx]
                    hc_idx = hc_idx[m_idx]
                    dphi = np.angle(np.exp(1j * (phase_ms[ms_idx] - phase_hc[hc_idx])))
                    dphi = dphi[np.isfinite(dphi)]
                    n_eff = int(dphi.size)
                    if n_eff < 3:
                        continue

                    mean_vec = np.mean(np.exp(1j * dphi))
                    plv = float(np.abs(mean_vec))
                    z_val = float(n_eff * (plv ** 2))
                    if not np.isfinite(z_val):
                        continue
                    z_row[li] = np.float32(z_val)
                    count_row[li] = np.float32(n_eff)
                if np.any(np.isfinite(z_row)):
                    z_rows.append(z_row)
                    count_rows.append(count_row)
            z_matrix = np.stack(z_rows, axis=0) if z_rows else np.zeros((0, lags.size), dtype=np.float32)
            count_matrix = np.stack(count_rows, axis=0) if count_rows else np.zeros((0, lags.size), dtype=np.float32)

        if z_matrix.size == 0:
            continue
        valid_any = np.any(np.isfinite(z_matrix), axis=1)
        n_total = int(np.sum(valid_any))
        if n_total <= 0:
            continue

        filt = _filter_phaselag_z_matrix(
            z_matrix=z_matrix,
            count_matrix=count_matrix,
            min_z=min_z,
            min_plv=min_plv,
            min_peak_delta_z=min_peak_delta_z,
            min_peak_delta_frac=min_peak_delta_frac,
            exclude_edge_peaks=exclude_edge_peaks,
            edge_guard_bins=edge_guard_bins,
        )
        keep = np.asarray(filt["keep"], dtype=bool)
        best_idx = np.asarray(filt["best_idx"], dtype=np.int64)
        best_z = np.asarray(filt["best_z"], dtype=float)
        best_plv = np.asarray(filt["best_plv"], dtype=float)
        edge_hit = np.asarray(filt["edge_hit"], dtype=bool)
        n_valid = int(np.sum(keep))
        if n_valid <= 0:
            continue
        valid_ratio = float(n_valid / max(1, n_total))
        if valid_ratio < float(min_valid_ratio):
            continue

        best_lags_ms = lag_ms[best_idx[keep]]
        best_z_keep = best_z[keep]
        best_plv_keep = best_plv[keep]
        z_keep = np.asarray(z_matrix[keep], dtype=float)
        row_max = np.nanmax(z_keep, axis=1)
        norm_z = np.full_like(z_keep, np.nan, dtype=float)
        np.divide(z_keep, row_max[:, None], out=norm_z, where=(row_max[:, None] > 0))
        norm_z[~np.isfinite(norm_z)] = np.nan
        mean_norm_z = np.nanmean(norm_z, axis=0)
        sem_norm_z = np.nanstd(norm_z, axis=0, ddof=1) / np.sqrt(float(n_valid)) if n_valid > 1 else np.zeros(lags.size, dtype=float)
        mean_z = np.nanmean(z_keep, axis=0)
        hist_counts = np.zeros(lags.size, dtype=float)
        for bi in best_idx[keep]:
            if 0 <= int(bi) < hist_counts.size:
                hist_counts[int(bi)] += 1.0
        hist_prob = hist_counts / float(n_valid)
        peak_lag_idx = int(np.nanargmax(mean_norm_z))
        mode_lag_idx = int(np.nanargmax(hist_counts))

        out[label] = {
            "band": (fmin_f, fmax_f),
            "lags_ms": np.asarray(lag_ms, dtype=np.float32),
            "best_lags_ms": np.asarray(best_lags_ms, dtype=np.float32),
            "best_z": np.asarray(best_z_keep, dtype=np.float32),
            "best_plv": np.asarray(best_plv_keep, dtype=np.float32),
            "mean_norm_z": np.asarray(mean_norm_z, dtype=np.float32),
            "sem_norm_z": np.asarray(sem_norm_z, dtype=np.float32),
            "mean_z": np.asarray(mean_z, dtype=np.float32),
            "hist_counts": np.asarray(hist_counts, dtype=np.float32),
            "hist_prob": np.asarray(hist_prob, dtype=np.float32),
            "n_total_windows": int(n_total),
            "n_valid_windows": int(n_valid),
            "n_edge_hit_windows": int(np.sum(edge_hit)),
            "valid_ratio": valid_ratio,
            "min_valid_ratio": float(min_valid_ratio),
            "exclude_edge_peaks": bool(exclude_edge_peaks),
            "edge_guard_bins": int(max(0, edge_guard_bins)),
            "lag_selection_score": "rayleigh_z",
            "peak_lag_ms": float(lag_ms[peak_lag_idx]),
            "mode_lag_ms": float(lag_ms[mode_lag_idx]),
            "median_lag_ms": float(np.nanmedian(best_lags_ms)),
            "mean_lag_ms": float(np.nanmean(best_lags_ms)),
            "ms_lead_ratio": float(np.mean(best_lags_ms > 0.0)),
            "hc_lead_ratio": float(np.mean(best_lags_ms < 0.0)),
            "zero_lag_ratio": float(np.mean(best_lags_ms == 0.0)),
            "median_z": float(np.nanmedian(best_z_keep)),
            "median_plv": float(np.nanmedian(best_plv_keep)),
        }

    return out


def _compute_pearson_band_timeseries_pair(
    x_sig: np.ndarray,
    y_sig: np.ndarray,
    t_points: np.ndarray,
    sr: float,
    win_sec: float,
    step_sec: float,
    bands: list[tuple[str, tuple[float, float]]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    x = np.asarray(x_sig, dtype=float).reshape(-1)
    y = np.asarray(y_sig, dtype=float).reshape(-1)
    t = np.asarray(t_points, dtype=float).reshape(-1)
    n = min(x.size, y.size, t.size)
    if n < 2:
        return np.array([], dtype=np.float32), {}, {}
    x = x[:n]
    y = y[:n]
    t = t[:n]

    time_centers = _build_time_right(float(t[0]), float(t[-1]), step_sec)
    if time_centers.size == 0:
        return np.array([], dtype=np.float32), {}, {}
    indices = _causal_window_indices(t, time_centers, win_sec=win_sec)

    r_out: dict[str, np.ndarray] = {}
    r2_out: dict[str, np.ndarray] = {}
    nyq = float(sr) / 2.0

    for label, (fmin, fmax) in bands:
        fmin_f = float(fmin)
        fmax_f = float(fmax)
        if fmin_f <= 0 or fmax_f <= 0 or fmin_f >= fmax_f or fmax_f >= nyq:
            raise ValueError(f"Invalid band [{fmin_f}, {fmax_f}] for sr={sr}.")

        sos = butter(4, [fmin_f, fmax_f], btype="bandpass", fs=float(sr), output="sos")
        x_f = sosfiltfilt(sos, x)
        y_f = sosfiltfilt(sos, y)

        r_vals = np.full(time_centers.size, np.nan, dtype=np.float32)
        r2_vals = np.full(time_centers.size, np.nan, dtype=np.float32)

        for i, idx in enumerate(indices):
            if idx.size < 3:
                continue
            sx = x_f[idx]
            sy = y_f[idx]
            m = np.isfinite(sx) & np.isfinite(sy)
            if int(np.sum(m)) < 3:
                continue
            sx = sx[m]
            sy = sy[m]

            sx = sx - float(np.mean(sx))
            sy = sy - float(np.mean(sy))
            vx = float(np.mean(sx * sx))
            vy = float(np.mean(sy * sy))
            if vx <= 0.0 or vy <= 0.0:
                continue

            cxy = float(np.mean(sx * sy))
            r = cxy / float(np.sqrt(vx * vy))
            if not np.isfinite(r):
                continue
            r = float(np.clip(r, -1.0, 1.0))
            r_vals[i] = np.float32(r)
            r2_vals[i] = np.float32(r * r)

        r_out[label] = r_vals
        r2_out[label] = r2_vals

    return time_centers.astype(np.float32), r_out, r2_out


def _compute_wavelet_coupling_maps_pair(
    x_sig: np.ndarray,
    y_sig: np.ndarray,
    t_points: np.ndarray,
    sr: float,
    win_sec: float,
    step_sec: float,
    fmin_plot: float,
    fmax_plot: float,
    fmin_calc: float,
    fmax_calc: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    x = np.asarray(x_sig, dtype=float).reshape(-1)
    y = np.asarray(y_sig, dtype=float).reshape(-1)
    t = np.asarray(t_points, dtype=float).reshape(-1)
    n = min(x.size, y.size, t.size)
    if n < 2:
        return None
    x = x[:n]
    y = y[:n]
    t = t[:n]

    sr_f = float(sr)
    if sr_f <= 0:
        return None

    fmin_wave = max(1e-6, float(fmin_calc))
    fmax_wave = float(fmax_calc)
    if fmax_wave <= fmin_wave:
        return None
    n_freqs = max(60, int(np.ceil(np.log2(fmax_wave / fmin_wave)) * 24))
    f_points = np.geomspace(fmin_wave, fmax_wave, num=n_freqs)

    w = 12.0
    widths = (w * sr_f) / (2.0 * np.pi * f_points)

    time_centers = _build_time_right(float(t[0]), float(t[-1]), step_sec)
    if time_centers.size == 0:
        return None
    win_indices = _causal_window_indices(t, time_centers, win_sec=win_sec)

    coh_full = np.full((n_freqs, time_centers.size), np.nan, dtype=np.float64)
    plv_full = np.full((n_freqs, time_centers.size), np.nan, dtype=np.float64)

    def _morlet2(M: int, s: float, w_: float) -> np.ndarray:
        tt = np.arange(-M // 2, M // 2 + M % 2, dtype=np.float64)
        return np.exp(1j * w_ * (tt / s)) * np.exp(-(tt**2) / (2.0 * (s**2)))

    for fi, s in enumerate(widths):
        M = int(np.ceil(10 * s)) * 2 + 1
        M = min(M, 2 * n + 1)
        if M < 3:
            M = 3
        try:
            psi = signal.morlet2(M, s, w=w)
        except Exception:
            psi = _morlet2(M, s, w)

        wx = signal.fftconvolve(x, psi, mode="same")
        wy = signal.fftconvolve(y, psi, mode="same")

        for ti, idx in enumerate(win_indices):
            if idx.size < 2:
                continue
            seg_x = wx[idx]
            seg_y = wy[idx]

            pxx = np.mean(np.abs(seg_x) ** 2)
            pyy = np.mean(np.abs(seg_y) ** 2)
            if pxx > 0 and pyy > 0:
                pxy = np.mean(seg_x * np.conjugate(seg_y))
                coh_val = (np.abs(pxy) ** 2) / (pxx * pyy)
                coh_full[fi, ti] = float(np.real(coh_val))

            dphi = np.angle(seg_x) - np.angle(seg_y)
            plv_full[fi, ti] = float(np.abs(np.mean(np.exp(1j * dphi))))

    m_band = (f_points >= float(fmin_plot)) & (f_points <= float(fmax_plot))
    if not np.any(m_band):
        return None

    f_sel = np.asarray(f_points[m_band], dtype=np.float32)
    coh_sel = np.asarray(np.clip(coh_full[m_band, :], 0.0, 1.0), dtype=np.float32)
    plv_sel = np.asarray(np.clip(plv_full[m_band, :], 0.0, 1.0), dtype=np.float32)
    t_sel = np.asarray(time_centers, dtype=np.float32)
    return f_sel, t_sel, coh_sel, plv_sel


def _remove_aperiodic_trend_tf(tf_map: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
    z = np.asarray(tf_map, dtype=float)
    f = np.asarray(freqs_hz, dtype=float).reshape(-1)
    if z.ndim != 2 or f.size != z.shape[0] or f.size < 3:
        return z
    m_f = np.isfinite(f) & (f > 0.0)
    if int(np.sum(m_f)) < 3:
        return z
    out = z.copy()
    log_f = np.log10(f[m_f])
    X = np.vstack([log_f, np.ones_like(log_f)]).T
    for ti in range(out.shape[1]):
        y = out[m_f, ti]
        m_y = np.isfinite(y)
        if int(np.sum(m_y)) < 3:
            continue
        beta, _, _, _ = np.linalg.lstsq(X[m_y], y[m_y], rcond=None)
        out[m_f, ti] = y - (X @ beta)
    return out


def _compute_wavelet_power_maps_pair(
    x_sig: np.ndarray,
    y_sig: np.ndarray,
    t_points: np.ndarray,
    sr: float,
    win_sec: float,
    step_sec: float,
    fmin_plot: float,
    fmax_plot: float,
    fmin_calc: float,
    fmax_calc: float,
    normalize: str,
    aperiodic_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]] | None:
    x = np.asarray(x_sig, dtype=float).reshape(-1)
    y = np.asarray(y_sig, dtype=float).reshape(-1)
    t = np.asarray(t_points, dtype=float).reshape(-1)
    n = min(x.size, y.size, t.size)
    if n < 8:
        return None
    x = x[:n]
    y = y[:n]
    t = t[:n]

    sr_f = float(sr)
    if sr_f <= 0:
        return None
    fmin_wave = max(1e-6, float(fmin_calc))
    fmax_wave = float(fmax_calc)
    if fmax_wave <= fmin_wave:
        return None

    n_freqs = max(60, int(np.ceil(np.log2(fmax_wave / fmin_wave)) * 24))
    f_points = np.geomspace(fmin_wave, fmax_wave, num=n_freqs)
    w = 12.0
    widths = (w * sr_f) / (2.0 * np.pi * f_points)

    time_centers = _build_time_right(float(t[0]), float(t[-1]), step_sec)
    if time_centers.size == 0:
        return None
    win_indices = _causal_window_indices(t, time_centers, win_sec=win_sec)
    hc_full = np.full((n_freqs, time_centers.size), np.nan, dtype=np.float32)
    ms_full = np.full((n_freqs, time_centers.size), np.nan, dtype=np.float32)

    norm_mode = str(normalize).strip().lower()
    if norm_mode not in {"mean", "none"}:
        norm_mode = "mean"

    def _morlet2(M: int, s: float, w_: float) -> np.ndarray:
        tt = np.arange(-M // 2, M // 2 + M % 2, dtype=np.float64)
        return np.exp(1j * w_ * (tt / s)) * np.exp(-(tt**2) / (2.0 * (s**2)))

    eps = 1e-8
    for fi, s in enumerate(widths):
        M = int(np.ceil(10 * s)) * 2 + 1
        M = min(M, 2 * n + 1)
        if M < 3:
            M = 3
        try:
            psi = signal.morlet2(M, s, w=w)
        except Exception:
            psi = _morlet2(M, s, w)

        wx = signal.fftconvolve(x, psi, mode="same")
        wy = signal.fftconvolve(y, psi, mode="same")
        px = np.asarray(np.abs(wx) ** 2, dtype=float)
        py = np.asarray(np.abs(wy) ** 2, dtype=float)
        if norm_mode == "mean":
            mx = float(np.nanmean(px))
            my = float(np.nanmean(py))
            px_db = 10.0 * (np.log10(px + eps) - np.log10(mx + eps))
            py_db = 10.0 * (np.log10(py + eps) - np.log10(my + eps))
        else:
            px_db = 10.0 * np.log10(px + eps)
            py_db = 10.0 * np.log10(py + eps)

        for ti, idx in enumerate(win_indices):
            if idx.size < 2:
                continue
            hc_full[fi, ti] = np.float32(np.nanmedian(px_db[idx]))
            ms_full[fi, ti] = np.float32(np.nanmedian(py_db[idx]))

    m_band = (f_points >= float(fmin_plot)) & (f_points <= float(fmax_plot))
    if not np.any(m_band):
        return None

    f_sel = np.asarray(f_points[m_band], dtype=np.float32)
    hc_sel = np.asarray(hc_full[m_band, :], dtype=np.float32)
    ms_sel = np.asarray(ms_full[m_band, :], dtype=np.float32)
    if str(aperiodic_mode).strip().lower() == "residual":
        hc_sel = np.asarray(_remove_aperiodic_trend_tf(hc_sel, f_sel), dtype=np.float32)
        ms_sel = np.asarray(_remove_aperiodic_trend_tf(ms_sel, f_sel), dtype=np.float32)
    diff_sel = np.asarray(ms_sel - hc_sel, dtype=np.float32)

    m = np.isfinite(hc_sel) & np.isfinite(ms_sel)
    corr = float("nan")
    mae = float("nan")
    if int(np.sum(m)) >= 3:
        a = hc_sel[m].astype(float)
        b = ms_sel[m].astype(float)
        if float(np.nanstd(a)) > 0.0 and float(np.nanstd(b)) > 0.0:
            corr = float(np.corrcoef(a, b)[0, 1])
        mae = float(np.nanmean(np.abs(b - a)))

    metrics = {
        "map_corr": corr,
        "mean_abs_diff_db": mae,
    }
    return f_sel, np.asarray(time_centers, dtype=np.float32), hc_sel, ms_sel, diff_sel, metrics


def _fit_bivariate_mvar_ols(
    sig_2xn: np.ndarray,
    order_p: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    x = np.asarray(sig_2xn, dtype=float)
    if x.ndim != 2 or x.shape[0] != 2:
        return None

    n = int(x.shape[1])
    p = int(order_p)
    if p < 1 or n <= (p + 2):
        return None

    n_eff = int(n - p)
    reg_dim = int(2 * p)
    if n_eff <= reg_dim:
        return None

    y = np.asarray(x[:, p:], dtype=float)
    z = np.empty((reg_dim, n_eff), dtype=float)
    for lag in range(1, p + 1):
        z[(2 * (lag - 1)):(2 * lag), :] = x[:, (p - lag):(n - lag)]

    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(z))):
        return None

    # Solve OLS with normal equations on a small (2p x 2p) system.
    # This is substantially faster than pinv/SVD for many windows.
    try:
        zz = z @ z.T
        yz = y @ z.T
        if (not np.all(np.isfinite(zz))) or (not np.all(np.isfinite(yz))):
            return None

        dim = int(zz.shape[0])
        scale = float(np.trace(zz) / max(dim, 1))
        base = float(max(scale, 1.0))
        eye = np.eye(dim, dtype=float)

        beta = None
        for lam_mul in (0.0, 1e-10, 1e-8, 1e-6):
            lam = float(lam_mul * base)
            try:
                if lam > 0.0:
                    beta_try = np.linalg.solve(zz + (lam * eye), yz.T).T
                else:
                    beta_try = np.linalg.solve(zz, yz.T).T
                if np.all(np.isfinite(beta_try)):
                    beta = np.asarray(beta_try, dtype=float)
                    break
            except Exception:
                continue

        if beta is None:
            beta = np.asarray(np.linalg.lstsq(z.T, y.T, rcond=None)[0].T, dtype=float)
    except Exception:
        return None

    resid = y - (beta @ z)
    cov = (resid @ resid.T) / float(max(n_eff, 1))
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None

    a_lags = np.zeros((2, 2, p), dtype=float)
    for lag in range(p):
        a_lags[:, :, lag] = beta[:, (2 * lag):(2 * (lag + 1))]
    return a_lags, cov


def _select_bivariate_mvar_order(
    sig_2xn: np.ndarray,
    p_max: int,
    criterion: str,
) -> int | None:
    x = np.asarray(sig_2xn, dtype=float)
    if x.ndim != 2 or x.shape[0] != 2:
        return None

    n = int(x.shape[1])
    if n < 8:
        return None

    p_cap = int(min(int(p_max), max(1, (n - 1) // 3)))
    if p_cap < 1:
        return None

    crit = str(criterion).strip().lower()
    if crit not in {"aic", "bic"}:
        crit = "bic"

    best_p: int | None = None
    best_score = np.inf

    for p in range(1, p_cap + 1):
        fit = _fit_bivariate_mvar_ols(x, p)
        if fit is None:
            continue
        _, cov = fit

        n_eff = int(n - p)
        if n_eff <= 1:
            continue
        n_params = int(4 * p)  # k*k*p, k=2
        if n_params >= n_eff:
            continue

        cov_reg = np.asarray(cov, dtype=float) + (1e-9 * np.eye(2, dtype=float))
        sign, logdet = np.linalg.slogdet(cov_reg)
        if sign <= 0 or (not np.isfinite(logdet)):
            continue

        if crit == "aic":
            penalty = (2.0 * float(n_params)) / float(n_eff)
        else:
            penalty = (np.log(float(n_eff)) * float(n_params)) / float(n_eff)
        score = float(logdet + penalty)
        if score < best_score:
            best_score = score
            best_p = p

    return best_p


def _compute_gpdc_from_mvar(
    a_lags: np.ndarray,
    resid_cov: np.ndarray,
    freq_hz: np.ndarray,
    sr: float,
) -> np.ndarray | None:
    a = np.asarray(a_lags, dtype=float)
    if a.ndim != 3 or a.shape[0] != 2 or a.shape[1] != 2 or a.shape[2] < 1:
        return None

    f = np.asarray(freq_hz, dtype=float).reshape(-1)
    if f.size == 0:
        return None
    sr_f = float(sr)
    if sr_f <= 0:
        return None

    cov = np.asarray(resid_cov, dtype=float)
    if cov.shape != (2, 2):
        cov = np.eye(2, dtype=float)
    var = np.diag(cov)
    var = np.where(np.isfinite(var) & (var > 0.0), var, 1.0)
    inv_std = 1.0 / np.sqrt(var)

    out = np.full((2, 2, f.size), np.nan, dtype=np.float32)
    ident = np.eye(2, dtype=np.complex128)
    p = int(a.shape[2])

    for fi, ff in enumerate(f):
        if not np.isfinite(ff):
            continue
        a_f = ident.copy()
        for lag in range(p):
            phase = np.exp(-2j * np.pi * float(ff) * float(lag + 1) / sr_f)
            a_f -= np.asarray(a[:, :, lag], dtype=np.complex128) * phase

        # Row-wise innovation scaling (generalized PDC style)
        a_tilde = inv_std[:, None] * a_f
        for src in range(2):
            col = a_tilde[:, src]
            denom = float(np.sqrt(np.sum(np.abs(col) ** 2)))
            if (not np.isfinite(denom)) or denom <= 0:
                continue
            out[:, src, fi] = np.asarray(np.abs(col) / denom, dtype=np.float32)

    return np.asarray(np.clip(out, 0.0, 1.0), dtype=np.float32)


def _compute_granger_gpdc_summary_pair(
    x_sig: np.ndarray,
    y_sig: np.ndarray,
    t_points: np.ndarray,
    sr: float,
    win_sec: float,
    step_sec: float,
    p_max: int,
    criterion: str,
    order_mode: str,
    fixed_order: int,
    fmin: float,
    fmax: float,
    n_freqs: int,
    epoch_jobs: int,
    progress_label: str | None = None,
    progress_epoch: bool = False,
    progress_callback: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    x = np.asarray(x_sig, dtype=float).reshape(-1)
    y = np.asarray(y_sig, dtype=float).reshape(-1)
    t = np.asarray(t_points, dtype=float).reshape(-1)
    n = min(x.size, y.size, t.size)
    if n < 16:
        return None
    x = x[:n]
    y = y[:n]
    t = t[:n]

    sr_f = float(sr)
    if sr_f <= 0:
        return None

    nyq = 0.5 * sr_f
    f_lo = max(0.0, float(fmin))
    f_hi = min(float(fmax), nyq * 0.999999)
    if f_hi <= f_lo:
        return None
    n_f = max(16, int(n_freqs))
    freq_hz = np.linspace(f_lo, f_hi, num=n_f, dtype=np.float64)

    time_right = np.arange(float(t[0]) + float(win_sec), float(t[-1]) + 1e-6, float(step_sec), dtype=float)
    if time_right.size == 0:
        return None
    indices = _causal_window_indices(t, time_right, win_sec=float(win_sec))

    epoch_n_jobs = int(max(1, int(epoch_jobs)))
    mode = str(order_mode).strip().lower()
    if mode not in {"median", "per_window"}:
        mode = "median"
    fixed_p = int(max(0, int(fixed_order)))
    if fixed_p > 0:
        mode = "fixed"
    n_windows_total = int(len(indices))
    use_epoch_progress = bool(progress_epoch) and (n_windows_total > 0)
    label = str(progress_label).strip() if progress_label is not None else ""
    phase_prefix = label if label else "Granger"

    def _iter_windows(it: Any, phase: str) -> Any:
        seq = it
        if use_epoch_progress:
            seq = tqdm(
                seq,
                total=n_windows_total,
                desc=f"{phase_prefix}:{phase}",
                unit="win",
                leave=False,
            )
        for item in seq:
            if progress_callback is not None:
                try:
                    progress_callback(1)
                except Exception:
                    pass
            yield item


    def _epoch_map(idxs: list[np.ndarray], worker_fn: Any) -> Any:
        if epoch_n_jobs <= 1:
            for idx in idxs:
                yield worker_fn(idx)
            return
        with ThreadPoolExecutor(max_workers=epoch_n_jobs) as ex:
            futures = [ex.submit(worker_fn, idx) for idx in idxs]
            for fut in as_completed(futures):
                yield fut.result()
    def _window_signal(idx: np.ndarray) -> np.ndarray | None:
        if idx.size < 16:
            return None
        sx = x[idx]
        sy = y[idx]
        m = np.isfinite(sx) & np.isfinite(sy)
        if int(np.sum(m)) < 16:
            return None
        sig_win = np.vstack([sx[m], sy[m]])
        sig_win = sig_win - np.mean(sig_win, axis=1, keepdims=True)
        return np.asarray(sig_win, dtype=float)

    def _select_order_worker(idx: np.ndarray) -> int | None:
        sig_win = _window_signal(idx)
        if sig_win is None:
            return None
        return _select_bivariate_mvar_order(
            sig_win,
            p_max=int(p_max),
            criterion=str(criterion),
        )

    def _gpdc_fixed_order_worker(idx: np.ndarray, p_use: int) -> np.ndarray | None:
        sig_win = _window_signal(idx)
        if sig_win is None:
            return None
        fit = _fit_bivariate_mvar_ols(sig_win, order_p=int(p_use))
        if fit is None:
            return None
        a_lags, cov = fit
        gpdc = _compute_gpdc_from_mvar(
            a_lags=a_lags,
            resid_cov=cov,
            freq_hz=freq_hz,
            sr=sr_f,
        )
        if gpdc is None or not np.any(np.isfinite(gpdc)):
            return None
        return np.asarray(gpdc, dtype=np.float32)

    gpdc_windows: list[np.ndarray] = []
    selected_orders: list[int] = []

    if mode == "fixed":
        p_fixed = int(fixed_p)

        def _fixed_worker(idx: np.ndarray) -> np.ndarray | None:
            return _gpdc_fixed_order_worker(idx, p_fixed)

        for gpdc_one in _iter_windows(
            _epoch_map(indices, _fixed_worker),
            f"fit_p={p_fixed}",
        ):
            if gpdc_one is None:
                continue
            gpdc_windows.append(np.asarray(gpdc_one, dtype=np.float32))
        if not gpdc_windows:
            return None
        selected_orders = [int(p_fixed)] * int(len(gpdc_windows))
    elif mode == "per_window":
        def _per_window_worker(idx: np.ndarray) -> tuple[np.ndarray, int] | None:
            p_sel = _select_order_worker(idx)
            if p_sel is None:
                return None
            gpdc_one = _gpdc_fixed_order_worker(idx, int(p_sel))
            if gpdc_one is None:
                return None
            return gpdc_one, int(p_sel)

        for res in _iter_windows(
            _epoch_map(indices, _per_window_worker),
            "order+fit",
        ):
            if res is None:
                continue
            gpdc_one, p_sel = res
            gpdc_windows.append(np.asarray(gpdc_one, dtype=np.float32))
            selected_orders.append(int(p_sel))
    else:
        p_candidates: list[int] = []
        for p_sel in _iter_windows(
            _epoch_map(indices, _select_order_worker),
            "order_select",
        ):
            if p_sel is None:
                continue
            p_candidates.append(int(p_sel))
        if not p_candidates:
            return None

        p_med = int(np.rint(float(np.median(np.asarray(p_candidates, dtype=float)))))
        p_med = int(np.clip(p_med, 1, int(max(1, p_max))))

        def _median_worker(idx: np.ndarray) -> np.ndarray | None:
            return _gpdc_fixed_order_worker(idx, p_med)

        for gpdc_one in _iter_windows(
            _epoch_map(indices, _median_worker),
            f"fit_p={p_med}",
        ):
            if gpdc_one is None:
                continue
            gpdc_windows.append(np.asarray(gpdc_one, dtype=np.float32))
        if not gpdc_windows:
            return None
        selected_orders = [int(p_med)] * int(len(gpdc_windows))

    if not gpdc_windows:
        return None

    stack = np.stack(gpdc_windows, axis=0)  # [W, 2, 2, F]
    mean_gpdc = np.nanmean(stack, axis=0)
    std_gpdc = np.nanstd(stack, axis=0, ddof=0)
    mean_gpdc = np.asarray(np.clip(mean_gpdc, 0.0, 1.0), dtype=np.float32)
    std_gpdc = np.asarray(np.clip(std_gpdc, 0.0, 1.0), dtype=np.float32)
    orders_arr = np.asarray(selected_orders, dtype=np.int32)

    return (
        np.asarray(freq_hz, dtype=np.float32),
        mean_gpdc,
        std_gpdc,
        orders_arr,
    )


def _compute_granger_cell_for_entry(
    entry: EEGEntry,
    analysis_sampling_rate: float,
    spike_sampling_rate: float,
    time_range: tuple[float, float],
    tf_win_sec: float,
    tf_step_sec: float,
    granger_order_max: int,
    granger_order_criterion: str,
    granger_order_mode: str,
    granger_fixed_order: int,
    granger_fmin: float,
    granger_fmax: float,
    granger_n_freqs: int,
    granger_epoch_jobs: int,
    granger_progress: str,
    progress_callback: Any | None,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
) -> ModeCell | None:
    t_start = perf_counter()
    pair = _prepare_hc_pseudo_pair(
        entry=entry,
        analysis_sampling_rate=analysis_sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        time_range=time_range,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
    )
    t_after_pair = perf_counter()
    if pair is None:
        return None
    t_sig, x_hc, y_ms = pair

    progress_mode = str(granger_progress).strip().lower()
    if progress_mode not in {"none", "entry", "epoch"}:
        progress_mode = "entry"

    summary = _compute_granger_gpdc_summary_pair(
        x_sig=x_hc,
        y_sig=y_ms,
        t_points=t_sig,
        sr=float(analysis_sampling_rate),
        win_sec=float(tf_win_sec),
        step_sec=float(tf_step_sec),
        p_max=int(granger_order_max),
        criterion=str(granger_order_criterion),
        order_mode=str(granger_order_mode),
        fixed_order=int(granger_fixed_order),
        fmin=float(granger_fmin),
        fmax=float(granger_fmax),
        n_freqs=int(granger_n_freqs),
        epoch_jobs=int(granger_epoch_jobs),
        progress_label=f"{entry.subject}/{entry.session}",
        progress_callback=progress_callback,
        progress_epoch=(progress_mode == "epoch"),
    )
    t_after_summary = perf_counter()
    if summary is None:
        return None

    freq_hz, mean_gpdc, std_gpdc, orders = summary
    if freq_hz.size < 2:
        return None
    if mean_gpdc.shape != (2, 2, freq_hz.size):
        return None
    if std_gpdc.shape != (2, 2, freq_hz.size):
        return None
    if orders.size == 0:
        return None

    payload = {
        "freq_hz": np.asarray(freq_hz, dtype=float),
        "gpdc_mean": np.asarray(mean_gpdc, dtype=float),
        "gpdc_std": np.asarray(std_gpdc, dtype=float),
        "n_windows": int(orders.size),
        "order_mean": float(np.mean(orders)),
        "order_std": float(np.std(orders)),
        "order_min": int(np.min(orders)),
        "order_max": int(np.max(orders)),
        "order_criterion": str(granger_order_criterion).strip().lower(),
        "order_mode": "fixed" if int(granger_fixed_order) > 0 else str(granger_order_mode).strip().lower(),
        "timing_sec": {
            "prepare_pair": float(t_after_pair - t_start),
            "granger_core": float(t_after_summary - t_after_pair),
            "total": float(t_after_summary - t_start),
        },
    }
    return ModeCell(subject=entry.subject, session=entry.session, traces=tuple(), payload=payload)


def _build_granger_mode_cells(
    eeg_entries: list[EEGEntry],
    analysis_sampling_rate: float,
    spike_sampling_rate: float,
    time_range: tuple[float, float],
    tf_win_sec: float,
    tf_step_sec: float,
    granger_order_max: int,
    granger_order_criterion: str,
    granger_order_mode: str,
    granger_fixed_order: int,
    granger_fmin: float,
    granger_epoch_jobs: int,
    granger_progress: str,
    granger_fmax: float,
    granger_n_freqs: int,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
    n_jobs: int,
) -> list[ModeCell]:
    out: list[ModeCell] = []
    progress_mode = str(granger_progress).strip().lower()
    if progress_mode not in {"none", "entry", "epoch"}:
        progress_mode = "entry"

    n_jobs_eff = int(max(1, int(n_jobs)))
    if progress_mode == "epoch" and n_jobs_eff > 1:
        print("-- granger progress=epoch: forcing entry-level n_jobs to 1 for readable sub-progress bars")
        n_jobs_eff = 1

    fixed_order_i = int(max(0, int(granger_fixed_order)))
    order_mode_l = "fixed" if fixed_order_i > 0 else str(granger_order_mode).strip().lower()
    pass_multiplier = 1 if order_mode_l == "per_window" else 2
    if order_mode_l == "fixed":
        pass_multiplier = 1

    def _estimate_entry_window_ops(entry: EEGEntry) -> int:
        if not entry.ms_units:
            return 0
        clipped = _clip_signal_by_time_range(
            sig=entry.eeg_z,
            sr=float(analysis_sampling_rate),
            time_range=time_range,
        )
        if clipped is None:
            return 0
        t_clip, _ = clipped
        if t_clip.size < 2:
            return 0
        time_right = np.arange(
            float(t_clip[0]) + float(tf_win_sec),
            float(t_clip[-1]) + 1e-6,
            float(tf_step_sec),
            dtype=float,
        )
        n_win = int(time_right.size)
        if n_win <= 0:
            return 0
        return int(n_win * pass_multiplier)

    estimated_ops = {id(entry): _estimate_entry_window_ops(entry) for entry in eeg_entries}
    total_ops = int(sum(estimated_ops.values()))
    if total_ops <= 0:
        total_ops = int(max(1, len(eeg_entries)))

    progress_lock = Lock()
    progress_done = 0

    bar = tqdm(total=total_ops, desc="Mode[granger]", unit="win")

    def _advance_progress(delta: int) -> None:
        nonlocal progress_done
        dn = int(delta)
        if dn <= 0:
            return
        with progress_lock:
            progress_done += dn
            if int(bar.n + dn) > int(bar.total):
                bar.total = int(bar.n + dn)
            bar.update(dn)

    worker = partial(
        _compute_granger_cell_for_entry,
        analysis_sampling_rate=analysis_sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        time_range=time_range,
        tf_win_sec=tf_win_sec,
        tf_step_sec=tf_step_sec,
        granger_order_max=granger_order_max,
        granger_order_mode=granger_order_mode,
        granger_fixed_order=fixed_order_i,
        granger_order_criterion=granger_order_criterion,
        granger_epoch_jobs=granger_epoch_jobs,
        granger_progress=progress_mode,
        progress_callback=_advance_progress,
        granger_fmin=granger_fmin,
        granger_fmax=granger_fmax,
        granger_n_freqs=granger_n_freqs,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
    )

    def _postfix(entry: EEGEntry, cell: ModeCell | None) -> str:
        if cell is None:
            return f"{entry.subject}/{entry.session} skipped"
        payload = cell.payload if isinstance(cell.payload, dict) else {}
        n_win = int(payload.get("n_windows", 0))
        p_min = payload.get("order_min", "?")
        p_max = payload.get("order_max", "?")
        timing = payload.get("timing_sec", {})
        total_sec = np.nan
        if isinstance(timing, dict):
            try:
                total_sec = float(timing.get("total", np.nan))
            except Exception:
                total_sec = np.nan

        if p_min == p_max:
            p_txt = f"p={p_min}"
        else:
            p_txt = f"p={p_min}-{p_max}"
        t_txt = f"{total_sec:.1f}s" if np.isfinite(total_sec) else "na"
        with progress_lock:
            done_now = int(progress_done)
            total_now = int(bar.total)
        return (
            f"{entry.subject}/{entry.session} win={n_win} {p_txt} "
            f"t={t_txt} progress={done_now}/{total_now}"
        )

    show_entry_progress = progress_mode in {"entry", "epoch"}
    try:
        if n_jobs_eff <= 1:
            for entry in eeg_entries:
                cell = worker(entry)
                if cell is not None:
                    out.append(cell)
                if cell is None:
                    est = int(estimated_ops.get(id(entry), 0))
                    if est > 0:
                        _advance_progress(est)
                if show_entry_progress:
                    bar.set_postfix_str(_postfix(entry, cell), refresh=False)
        else:
            with ThreadPoolExecutor(max_workers=n_jobs_eff) as ex:
                future_map = {ex.submit(worker, entry): entry for entry in eeg_entries}
                for fut in as_completed(future_map):
                    entry = future_map[fut]
                    cell = fut.result()
                    if cell is not None:
                        out.append(cell)
                    if cell is None:
                        est = int(estimated_ops.get(id(entry), 0))
                        if est > 0:
                            _advance_progress(est)
                    if show_entry_progress:
                        bar.set_postfix_str(_postfix(entry, cell), refresh=False)
    finally:
        with progress_lock:
            if int(bar.n) != int(bar.total):
                bar.total = int(bar.n)
                bar.refresh()
        bar.close()
    return out


def _compute_coherence_band_cell_for_entry(
    entry: EEGEntry,
    analysis_sampling_rate: float,
    spike_sampling_rate: float,
    theta_bands: list[tuple[str, tuple[float, float]]],
    time_range: tuple[float, float],
    tf_win_sec: float,
    tf_step_sec: float,
    max_points: int,
    smooth_win_sec: float | None,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
) -> ModeCell | None:
    pair = _prepare_hc_pseudo_pair(
        entry=entry,
        analysis_sampling_rate=analysis_sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        time_range=time_range,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
    )
    if pair is None:
        return None
    t_sig, x_sig, y_sig = pair

    t_out, coh_out, plv_out = _compute_coherence_band_timeseries_pair(
        x_sig=x_sig,
        y_sig=y_sig,
        t_points=t_sig,
        sr=float(analysis_sampling_rate),
        win_sec=float(tf_win_sec),
        step_sec=float(tf_step_sec),
        bands=theta_bands,
    )
    if t_out.size < 2:
        return None

    coh_traces: list[TraceSpec] = []
    plv_traces: list[TraceSpec] = []
    for bidx, (band_label, _) in enumerate(theta_bands):
        color = PLOTLY_COLORS[bidx % len(PLOTLY_COLORS)]

        y_coh = coh_out.get(band_label)
        if y_coh is not None:
            yy = _smooth_by_time(np.asarray(y_coh, dtype=float), t_out, smooth_win_sec)
            xx, yy = _downsample_xy(t_out, yy, max_points=max_points)
            coh_traces.append(
                TraceSpec(
                    name=band_label,
                    x=xx,
                    y=yy,
                    color=color,
                    width=1.2,
                    dash="solid",
                )
            )

        y_plv = plv_out.get(band_label)
        if y_plv is not None:
            yy = _smooth_by_time(np.asarray(y_plv, dtype=float), t_out, smooth_win_sec)
            xx, yy = _downsample_xy(t_out, yy, max_points=max_points)
            plv_traces.append(
                TraceSpec(
                    name=band_label,
                    x=xx,
                    y=yy,
                    color=color,
                    width=1.2,
                    dash="solid",
                )
            )

    if not coh_traces and not plv_traces:
        return None
    payload = {
        "primary_panel_label": "Coh",
        "secondary_panel_label": "PLV",
        "secondary_traces": tuple(plv_traces),
    }
    return ModeCell(subject=entry.subject, session=entry.session, traces=tuple(coh_traces), payload=payload)


def _build_coherence_band_mode_cells(
    eeg_entries: list[EEGEntry],
    analysis_sampling_rate: float,
    spike_sampling_rate: float,
    theta_bands: list[tuple[str, tuple[float, float]]],
    time_range: tuple[float, float],
    tf_win_sec: float,
    tf_step_sec: float,
    max_points: int,
    smooth_win_sec: float | None,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
    n_jobs: int,
) -> list[ModeCell]:
    out: list[ModeCell] = []
    worker = partial(
        _compute_coherence_band_cell_for_entry,
        analysis_sampling_rate=analysis_sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        theta_bands=theta_bands,
        time_range=time_range,
        tf_win_sec=tf_win_sec,
        tf_step_sec=tf_step_sec,
        max_points=max_points,
        smooth_win_sec=smooth_win_sec,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
    )
    for cell in tqdm(
        _parallel_map(eeg_entries, worker, n_jobs=n_jobs),
        total=len(eeg_entries),
        desc="Mode[coherence_band]",
        unit="eeg",
    ):
        if cell is not None:
            out.append(cell)
    return out


def _compute_syncfc_cell_for_entry(
    entry: EEGEntry,
    analysis_sampling_rate: float,
    spike_sampling_rate: float,
    theta_bands: list[tuple[str, tuple[float, float]]],
    time_range: tuple[float, float],
    tf_win_sec: float,
    tf_step_sec: float,
    max_points: int,
    smooth_win_sec: float | None,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
    syncfc_n_surrogates: int,
    syncfc_null: str,
    syncfc_min_shift_sec: float,
    syncfc_seed: int,
) -> ModeCell | None:
    pair = _prepare_hc_pseudo_pair(
        entry=entry,
        analysis_sampling_rate=analysis_sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        time_range=time_range,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
    )
    if pair is None:
        return None
    t_sig, x_sig, y_sig = pair

    t_out, band_stats = _compute_syncfc_band_timeseries_pair(
        x_sig=x_sig,
        y_sig=y_sig,
        t_points=t_sig,
        sr=float(analysis_sampling_rate),
        win_sec=float(tf_win_sec),
        step_sec=float(tf_step_sec),
        bands=theta_bands,
        n_surrogates=int(syncfc_n_surrogates),
        min_shift_sec=float(syncfc_min_shift_sec),
        seed=_stable_seed_u32(syncfc_seed, entry.subject, entry.session),
        null_mode=str(syncfc_null),
    )
    if t_out.size < 2 or not band_stats:
        return None

    coh_traces: list[TraceSpec] = []
    plv_traces: list[TraceSpec] = []
    for bidx, (band_label, _) in enumerate(theta_bands):
        st = band_stats.get(str(band_label))
        if not isinstance(st, dict):
            continue
        color = PLOTLY_COLORS[bidx % len(PLOTLY_COLORS)]

        y_coh = st.get("coh")
        if y_coh is not None:
            yy = _smooth_by_time(np.asarray(y_coh, dtype=float), t_out, smooth_win_sec)
            xx, yy = _downsample_xy(t_out, yy, max_points=max_points)
            coh_traces.append(
                TraceSpec(
                    name=band_label,
                    x=xx,
                    y=yy,
                    color=color,
                    width=1.2,
                    dash="solid",
                )
            )

        y_plv = st.get("plv")
        if y_plv is not None:
            yy = _smooth_by_time(np.asarray(y_plv, dtype=float), t_out, smooth_win_sec)
            xx, yy = _downsample_xy(t_out, yy, max_points=max_points)
            plv_traces.append(
                TraceSpec(
                    name=band_label,
                    x=xx,
                    y=yy,
                    color=color,
                    width=1.2,
                    dash="solid",
                )
            )

    if not coh_traces and not plv_traces:
        return None
    payload = {
        "primary_panel_label": "Coh",
        "secondary_panel_label": "PLV",
        "secondary_traces": tuple(plv_traces),
        "time_sec": np.asarray(t_out, dtype=np.float32),
        "band_stats": band_stats,
    }
    return ModeCell(subject=entry.subject, session=entry.session, traces=tuple(coh_traces), payload=payload)


def _build_syncfc_mode_cells(
    eeg_entries: list[EEGEntry],
    analysis_sampling_rate: float,
    spike_sampling_rate: float,
    theta_bands: list[tuple[str, tuple[float, float]]],
    time_range: tuple[float, float],
    tf_win_sec: float,
    tf_step_sec: float,
    max_points: int,
    smooth_win_sec: float | None,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
    syncfc_n_surrogates: int,
    syncfc_null: str,
    syncfc_min_shift_sec: float,
    syncfc_seed: int,
    n_jobs: int,
) -> list[ModeCell]:
    out: list[ModeCell] = []
    worker = partial(
        _compute_syncfc_cell_for_entry,
        analysis_sampling_rate=analysis_sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        theta_bands=theta_bands,
        time_range=time_range,
        tf_win_sec=tf_win_sec,
        tf_step_sec=tf_step_sec,
        max_points=max_points,
        smooth_win_sec=smooth_win_sec,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
        syncfc_n_surrogates=syncfc_n_surrogates,
        syncfc_null=syncfc_null,
        syncfc_min_shift_sec=syncfc_min_shift_sec,
        syncfc_seed=syncfc_seed,
    )
    for cell in tqdm(
        _parallel_map(eeg_entries, worker, n_jobs=n_jobs),
        total=len(eeg_entries),
        desc="Mode[syncfc]",
        unit="eeg",
    ):
        if cell is not None:
            out.append(cell)
    return out


def _compute_phaselag_cell_for_entry(
    entry: EEGEntry,
    analysis_sampling_rate: float,
    spike_sampling_rate: float,
    theta_bands: list[tuple[str, tuple[float, float]]],
    time_range: tuple[float, float],
    tf_win_sec: float,
    tf_step_sec: float,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
    phaselag_min_z: float,
    phaselag_min_plv: float,
    phaselag_min_peak_delta_z: float,
    phaselag_min_peak_delta_frac: float,
    phaselag_min_valid_ratio: float,
    phaselag_exclude_edge_peaks: bool,
    phaselag_edge_guard_bins: int,
) -> ModeCell | None:
    pair = _prepare_hc_pseudo_pair(
        entry=entry,
        analysis_sampling_rate=analysis_sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        time_range=time_range,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
    )
    if pair is None:
        return None
    t_sig, x_sig, y_sig = pair

    band_stats = _compute_phaselag_summary_pair(
        x_sig=x_sig,
        y_sig=y_sig,
        t_points=t_sig,
        sr=float(analysis_sampling_rate),
        win_sec=float(tf_win_sec),
        step_sec=float(tf_step_sec),
        bands=theta_bands,
        min_z=float(phaselag_min_z),
        min_plv=float(phaselag_min_plv),
        min_peak_delta_z=float(phaselag_min_peak_delta_z),
        min_peak_delta_frac=float(phaselag_min_peak_delta_frac),
        min_valid_ratio=float(phaselag_min_valid_ratio),
        exclude_edge_peaks=bool(phaselag_exclude_edge_peaks),
        edge_guard_bins=int(phaselag_edge_guard_bins),
    )
    if not band_stats:
        return None
    payload = {"band_stats": band_stats}
    return ModeCell(subject=entry.subject, session=entry.session, traces=tuple(), payload=payload)


def _build_phaselag_mode_cells(
    eeg_entries: list[EEGEntry],
    analysis_sampling_rate: float,
    spike_sampling_rate: float,
    theta_bands: list[tuple[str, tuple[float, float]]],
    time_range: tuple[float, float],
    tf_win_sec: float,
    tf_step_sec: float,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
    phaselag_min_z: float,
    phaselag_min_plv: float,
    phaselag_min_peak_delta_z: float,
    phaselag_min_peak_delta_frac: float,
    phaselag_min_valid_ratio: float,
    phaselag_exclude_edge_peaks: bool,
    phaselag_edge_guard_bins: int,
    n_jobs: int,
) -> list[ModeCell]:
    out: list[ModeCell] = []
    worker = partial(
        _compute_phaselag_cell_for_entry,
        analysis_sampling_rate=analysis_sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        theta_bands=theta_bands,
        time_range=time_range,
        tf_win_sec=tf_win_sec,
        tf_step_sec=tf_step_sec,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
        phaselag_min_z=phaselag_min_z,
        phaselag_min_plv=phaselag_min_plv,
        phaselag_min_peak_delta_z=phaselag_min_peak_delta_z,
        phaselag_min_peak_delta_frac=phaselag_min_peak_delta_frac,
        phaselag_min_valid_ratio=phaselag_min_valid_ratio,
        phaselag_exclude_edge_peaks=phaselag_exclude_edge_peaks,
        phaselag_edge_guard_bins=phaselag_edge_guard_bins,
    )
    for cell in tqdm(
        _parallel_map(eeg_entries, worker, n_jobs=n_jobs),
        total=len(eeg_entries),
        desc="Mode[phaselag]",
        unit="eeg",
    ):
        if cell is not None:
            out.append(cell)
    return out


def _compute_pearson_cell_for_entry(
    entry: EEGEntry,
    analysis_sampling_rate: float,
    spike_sampling_rate: float,
    theta_bands: list[tuple[str, tuple[float, float]]],
    time_range: tuple[float, float],
    tf_win_sec: float,
    tf_step_sec: float,
    max_points: int,
    smooth_win_sec: float | None,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
) -> ModeCell | None:
    pair = _prepare_hc_pseudo_pair(
        entry=entry,
        analysis_sampling_rate=analysis_sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        time_range=time_range,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
    )
    if pair is None:
        return None
    t_sig, x_sig, y_sig = pair

    t_out, r_out, r2_out = _compute_pearson_band_timeseries_pair(
        x_sig=x_sig,
        y_sig=y_sig,
        t_points=t_sig,
        sr=float(analysis_sampling_rate),
        win_sec=float(tf_win_sec),
        step_sec=float(tf_step_sec),
        bands=theta_bands,
    )
    if t_out.size < 2:
        return None

    r_traces: list[TraceSpec] = []
    r2_traces: list[TraceSpec] = []
    for bidx, (band_label, _) in enumerate(theta_bands):
        color = PLOTLY_COLORS[bidx % len(PLOTLY_COLORS)]

        y_r = r_out.get(band_label)
        if y_r is not None:
            yy = _smooth_by_time(np.asarray(y_r, dtype=float), t_out, smooth_win_sec)
            xx, yy = _downsample_xy(t_out, yy, max_points=max_points)
            r_traces.append(
                TraceSpec(
                    name=band_label,
                    x=xx,
                    y=yy,
                    color=color,
                    width=1.2,
                    dash="solid",
                )
            )

        y_r2 = r2_out.get(band_label)
        if y_r2 is not None:
            yy = _smooth_by_time(np.asarray(y_r2, dtype=float), t_out, smooth_win_sec)
            xx, yy = _downsample_xy(t_out, yy, max_points=max_points)
            r2_traces.append(
                TraceSpec(
                    name=band_label,
                    x=xx,
                    y=yy,
                    color=color,
                    width=1.2,
                    dash="solid",
                )
            )

    if not r_traces and not r2_traces:
        return None
    payload = {
        "primary_panel_label": "r",
        "secondary_panel_label": "r2",
        "secondary_traces": tuple(r2_traces),
    }
    return ModeCell(subject=entry.subject, session=entry.session, traces=tuple(r_traces), payload=payload)


def _build_pearson_mode_cells(
    eeg_entries: list[EEGEntry],
    analysis_sampling_rate: float,
    spike_sampling_rate: float,
    theta_bands: list[tuple[str, tuple[float, float]]],
    time_range: tuple[float, float],
    tf_win_sec: float,
    tf_step_sec: float,
    max_points: int,
    smooth_win_sec: float | None,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
    n_jobs: int,
) -> list[ModeCell]:
    out: list[ModeCell] = []
    worker = partial(
        _compute_pearson_cell_for_entry,
        analysis_sampling_rate=analysis_sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        theta_bands=theta_bands,
        time_range=time_range,
        tf_win_sec=tf_win_sec,
        tf_step_sec=tf_step_sec,
        max_points=max_points,
        smooth_win_sec=smooth_win_sec,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
    )
    for cell in tqdm(
        _parallel_map(eeg_entries, worker, n_jobs=n_jobs),
        total=len(eeg_entries),
        desc="Mode[pearson]",
        unit="eeg",
    ):
        if cell is not None:
            out.append(cell)
    return out


def _compute_spectrogram_cell_for_entry(
    entry: EEGEntry,
    analysis_sampling_rate: float,
    spike_sampling_rate: float,
    time_range: tuple[float, float],
    tf_win_sec: float,
    tf_step_sec: float,
    max_points: int,
    fmin_plot: float,
    fmax_plot: float,
    fmin_calc: float,
    fmax_calc: float,
    normalize: str,
    aperiodic_mode: str,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
) -> ModeCell | None:
    pair = _prepare_hc_pseudo_pair(
        entry=entry,
        analysis_sampling_rate=analysis_sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        time_range=time_range,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
    )
    if pair is None:
        return None
    t_sig, x_sig, y_sig = pair

    maps = _compute_wavelet_power_maps_pair(
        x_sig=x_sig,
        y_sig=y_sig,
        t_points=t_sig,
        sr=float(analysis_sampling_rate),
        win_sec=float(tf_win_sec),
        step_sec=float(tf_step_sec),
        fmin_plot=float(fmin_plot),
        fmax_plot=float(fmax_plot),
        fmin_calc=float(fmin_calc),
        fmax_calc=float(fmax_calc),
        normalize=str(normalize),
        aperiodic_mode=str(aperiodic_mode),
    )
    if maps is None:
        return None
    f_out, t_out, hc_map, ms_map, diff_map, metrics = maps
    if t_out.size < 2 or f_out.size < 2:
        return None

    payload = {
        "time_sec": np.asarray(t_out, dtype=float),
        "freq_hz": np.asarray(f_out, dtype=float),
        "hc_map": np.asarray(hc_map, dtype=float),
        "ms_map": np.asarray(ms_map, dtype=float),
        "diff_map": np.asarray(diff_map, dtype=float),
        "map_corr": float(metrics.get("map_corr", np.nan)),
        "mean_abs_diff_db": float(metrics.get("mean_abs_diff_db", np.nan)),
        "normalize": str(normalize),
        "aperiodic_mode": str(aperiodic_mode),
        "max_points": int(max_points),
    }
    return ModeCell(subject=entry.subject, session=entry.session, traces=tuple(), payload=payload)


def _build_spectrogram_mode_cells(
    eeg_entries: list[EEGEntry],
    analysis_sampling_rate: float,
    spike_sampling_rate: float,
    time_range: tuple[float, float],
    tf_win_sec: float,
    tf_step_sec: float,
    max_points: int,
    fmin_plot: float,
    fmax_plot: float,
    fmin_calc: float,
    fmax_calc: float,
    normalize: str,
    aperiodic_mode: str,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
    n_jobs: int,
) -> list[ModeCell]:
    out: list[ModeCell] = []
    worker = partial(
        _compute_spectrogram_cell_for_entry,
        analysis_sampling_rate=analysis_sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        time_range=time_range,
        tf_win_sec=tf_win_sec,
        tf_step_sec=tf_step_sec,
        max_points=max_points,
        fmin_plot=fmin_plot,
        fmax_plot=fmax_plot,
        fmin_calc=fmin_calc,
        fmax_calc=fmax_calc,
        normalize=normalize,
        aperiodic_mode=aperiodic_mode,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
    )
    for cell in tqdm(
        _parallel_map(eeg_entries, worker, n_jobs=n_jobs),
        total=len(eeg_entries),
        desc="Mode[spectrogram]",
        unit="eeg",
    ):
        if cell is not None:
            out.append(cell)
    return out


def _compute_coherence_cell_for_entry(
    entry: EEGEntry,
    analysis_sampling_rate: float,
    spike_sampling_rate: float,
    time_range: tuple[float, float],
    tf_win_sec: float,
    tf_step_sec: float,
    max_points: int,
    smooth_win_sec: float | None,
    fmin_plot: float,
    fmax_plot: float,
    fmin_calc: float,
    fmax_calc: float,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
) -> ModeCell | None:
    pair = _prepare_hc_pseudo_pair(
        entry=entry,
        analysis_sampling_rate=analysis_sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        time_range=time_range,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
    )
    if pair is None:
        return None
    t_sig, x_sig, y_sig = pair

    maps = _compute_wavelet_coupling_maps_pair(
        x_sig=x_sig,
        y_sig=y_sig,
        t_points=t_sig,
        sr=float(analysis_sampling_rate),
        win_sec=float(tf_win_sec),
        step_sec=float(tf_step_sec),
        fmin_plot=float(fmin_plot),
        fmax_plot=float(fmax_plot),
        fmin_calc=float(fmin_calc),
        fmax_calc=float(fmax_calc),
    )
    if maps is None:
        return None
    f_out, t_out, coh_map, plv_map = maps
    if t_out.size < 2:
        return None

    if coh_map.size == 0 or plv_map.size == 0:
        return None

    payload = {
        "time_sec": np.asarray(t_out, dtype=float),
        "freq_hz": np.asarray(f_out, dtype=float),
        "coh_map": np.asarray(coh_map, dtype=float),
        "plv_map": np.asarray(plv_map, dtype=float),
        "smooth_win_sec": None if smooth_win_sec is None else float(smooth_win_sec),
        "max_points": int(max_points),
    }
    return ModeCell(
        subject=entry.subject,
        session=entry.session,
        traces=tuple(),
        payload=payload,
    )


def _build_coherence_mode_cells(
    eeg_entries: list[EEGEntry],
    analysis_sampling_rate: float,
    spike_sampling_rate: float,
    time_range: tuple[float, float],
    tf_win_sec: float,
    tf_step_sec: float,
    max_points: int,
    smooth_win_sec: float | None,
    fmin_plot: float,
    fmax_plot: float,
    fmin_calc: float,
    fmax_calc: float,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
    n_jobs: int,
) -> list[ModeCell]:
    out: list[ModeCell] = []
    worker = partial(
        _compute_coherence_cell_for_entry,
        analysis_sampling_rate=analysis_sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        time_range=time_range,
        tf_win_sec=tf_win_sec,
        tf_step_sec=tf_step_sec,
        max_points=max_points,
        smooth_win_sec=smooth_win_sec,
        fmin_plot=fmin_plot,
        fmax_plot=fmax_plot,
        fmin_calc=fmin_calc,
        fmax_calc=fmax_calc,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
    )
    for cell in tqdm(
        _parallel_map(eeg_entries, worker, n_jobs=n_jobs),
        total=len(eeg_entries),
        desc="Mode[coherence]",
        unit="eeg",
    ):
        if cell is not None:
            out.append(cell)
    return out


def _grubbs_inlier_range(
    values: np.ndarray,
    alpha: float = 0.01,
    max_remove_frac: float = 0.02,
    min_n: int = 8,
) -> tuple[float, float, bool]:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return -1.0, 1.0, False
    if x.size < int(min_n):
        return float(np.min(x)), float(np.max(x)), False

    n0 = int(x.size)
    max_remove = int(max(1, np.floor(float(max_remove_frac) * float(n0))))
    keep = np.ones(n0, dtype=bool)
    removed = 0

    while removed < max_remove:
        idx_keep = np.flatnonzero(keep)
        cur = x[idx_keep]
        n = int(cur.size)
        if n < int(min_n):
            break

        mean = float(np.mean(cur))
        std = float(np.std(cur, ddof=1))
        if (not np.isfinite(std)) or std <= 0:
            break

        dev = np.abs(cur - mean)
        j = int(np.argmax(dev))
        g_stat = float(dev[j] / std)

        p = 1.0 - (float(alpha) / (2.0 * float(n)))
        t_crit = float(student_t.ppf(p, df=n - 2))
        if not np.isfinite(t_crit):
            break
        g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt((t_crit**2) / (n - 2 + t_crit**2))

        if g_stat > g_crit:
            keep[idx_keep[j]] = False
            removed += 1
            continue
        break

    if removed <= 0:
        return float(np.min(x)), float(np.max(x)), False
    inlier = x[keep]
    if inlier.size == 0:
        return float(np.min(x)), float(np.max(x)), False
    return float(np.min(inlier)), float(np.max(inlier)), True


def _build_subject_mode_figure(
    mode: str,
    subject: str,
    mode_cells: list[ModeCell],
    x_title: str,
    y_title: str,
    use_timeseries_outlier_rejection: bool,
    ms_units_by_session: dict[str, tuple[tuple[str, np.ndarray], ...]] | None,
    spike_sampling_rate: float,
    ms_lfp_enabled: bool,
    ms_lfp_overlay: bool,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
    ms_lfp_max_points: int,
) -> Any:
    _require_plotly()
    if not mode_cells:
        raise ValueError("No cells to plot.")

    include_spikes = (mode in TIME_SERIES_MODES) and bool(ms_units_by_session)
    split_metric_mode = mode in {"coherence_band", "syncfc", "pearson"}

    cells_sorted = sorted(mode_cells, key=lambda c: natural_key(c.session))
    row_specs: list[dict[str, Any]] = []
    session_groups: list[tuple[str, int, int]] = []

    for cell in cells_sorted:
        traces_primary = [t for t in cell.traces if t.x.size > 1 and t.y.size > 1]
        payload_cell = cell.payload if isinstance(cell.payload, dict) else None
        traces_secondary: list[TraceSpec] = []
        primary_panel_label = ""
        secondary_panel_label = ""
        if split_metric_mode and payload_cell is not None:
            st = payload_cell.get("secondary_traces")
            if isinstance(st, (list, tuple)):
                traces_secondary = [
                    t for t in st if isinstance(t, TraceSpec) and t.x.size > 1 and t.y.size > 1
                ]
            primary_panel_label = str(payload_cell.get("primary_panel_label", "")).strip()
            secondary_panel_label = str(payload_cell.get("secondary_panel_label", "")).strip()

        if not traces_primary:
            continue
        start_row = len(row_specs) + 1
        row_specs.append(
            {
                "kind": "hc",
                "session": cell.session,
                "subject": cell.subject,
                "traces": traces_primary,
                "panel_label": primary_panel_label,
            }
        )
        if split_metric_mode and traces_secondary:
            row_specs.append(
                {
                    "kind": "hc",
                    "session": cell.session,
                    "subject": cell.subject,
                    "traces": traces_secondary,
                    "panel_label": secondary_panel_label,
                }
            )

        session_x_min = np.inf
        session_x_max = -np.inf
        all_session_traces = list(traces_primary)
        if split_metric_mode and traces_secondary:
            all_session_traces.extend(traces_secondary)
        for tr in all_session_traces:
            x = np.asarray(tr.x, dtype=float).reshape(-1)
            y = np.asarray(tr.y, dtype=float).reshape(-1)
            n = min(x.size, y.size)
            if n < 2:
                continue
            x = x[:n]
            y = y[:n]
            m = np.isfinite(x) & np.isfinite(y)
            if int(np.sum(m)) < 2:
                continue
            x = x[m]
            session_x_min = min(session_x_min, float(np.min(x)))
            session_x_max = max(session_x_max, float(np.max(x)))

        if include_spikes and ms_units_by_session is not None:
            units = ms_units_by_session.get(cell.session, tuple())
            if ms_lfp_enabled:
                if np.isfinite(session_x_min) and np.isfinite(session_x_max) and session_x_max > session_x_min:
                    units_by_shank: dict[int, list[tuple[str, np.ndarray]]] = {}
                    for unit_name, spikes in units:
                        parsed_unit = parse_unit_name(str(unit_name))
                        if parsed_unit is None:
                            continue
                        shank_int, _ = parsed_unit
                        s_unit = np.asarray(spikes, dtype=float).reshape(-1)
                        s_unit = s_unit[np.isfinite(s_unit)]
                        if s_unit.size == 0:
                            continue
                        t_unit = (s_unit - 1.0) / float(spike_sampling_rate)
                        t_unit = t_unit[np.isfinite(t_unit)]
                        if t_unit.size == 0:
                            continue
                        m_unit = (t_unit >= float(session_x_min)) & (t_unit <= float(session_x_max))
                        t_unit = t_unit[m_unit]
                        if t_unit.size == 0:
                            continue
                        units_by_shank.setdefault(int(shank_int), []).append(
                            (str(unit_name), t_unit)
                        )

                    if split_metric_mode:
                        if bool(ms_lfp_overlay):
                            shank_units = []
                            for vv in units_by_shank.values():
                                shank_units.extend(vv)
                            for unit_name, t_unit in shank_units:
                                t_overlay = np.asarray(t_unit, dtype=float).reshape(-1)
                                max_spike_overlay = max(1000, int(ms_lfp_max_points) * 3)
                                if max_spike_overlay > 0 and t_overlay.size > max_spike_overlay:
                                    idx = _linspace_idx(t_overlay.size, max_spike_overlay)
                                    t_overlay = t_overlay[idx]
                                row_specs.append(
                                    {
                                        "kind": "ms_overlay_unit",
                                        "session": cell.session,
                                        "unit_name": str(unit_name),
                                        "spike_t": t_overlay,
                                    }
                                )
                    else:
                        payload = cell.payload if isinstance(cell.payload, dict) else None
                        pseudo_payload_traces = ()
                        pseudo_payload_shank_int = -1
                        pseudo_payload_shank_label = "shank"
                        if payload is not None:
                            pt = payload.get("pseudo_traces")
                            if isinstance(pt, (list, tuple)):
                                pseudo_payload_traces = tuple(
                                    t for t in pt if isinstance(t, TraceSpec) and t.x.size > 1 and t.y.size > 1
                                )
                            try:
                                pseudo_payload_shank_int = int(payload.get("pseudo_shank_int", -1))
                            except Exception:
                                pseudo_payload_shank_int = -1
                            pseudo_payload_shank_label = str(payload.get("pseudo_shank_label", "shank"))

                        if pseudo_payload_traces:
                            row_specs.append(
                                {
                                    "kind": "ms_metric",
                                    "session": cell.session,
                                    "shank_name": pseudo_payload_shank_label,
                                    "traces": pseudo_payload_traces,
                                }
                            )
                            if bool(ms_lfp_overlay):
                                if pseudo_payload_shank_int >= 0:
                                    shank_units = units_by_shank.get(int(pseudo_payload_shank_int), [])
                                else:
                                    shank_units = []
                                    for vv in units_by_shank.values():
                                        shank_units.extend(vv)
                                for unit_name, t_unit in shank_units:
                                    t_overlay = np.asarray(t_unit, dtype=float).reshape(-1)
                                    max_spike_overlay = max(1000, int(ms_lfp_max_points) * 3)
                                    if max_spike_overlay > 0 and t_overlay.size > max_spike_overlay:
                                        idx = _linspace_idx(t_overlay.size, max_spike_overlay)
                                        t_overlay = t_overlay[idx]
                                    row_specs.append(
                                        {
                                            "kind": "ms_overlay_unit",
                                            "session": cell.session,
                                            "unit_name": str(unit_name),
                                            "spike_t": t_overlay,
                                        }
                                    )
                        else:
                            lfp_rows, lfp_warnings = synthesize_shank_lfp(
                                units=units,
                                sampling_rate=float(spike_sampling_rate),
                                time_range_sec=(float(session_x_min), float(session_x_max)),
                                sigma_sec=float(ms_lfp_sigma),
                                amplitude=float(ms_lfp_a),
                                gain_a0=float(ms_lfp_a0),
                                distance_map=ms_lfp_distance_map,
                                default_distance=float(ms_lfp_d_default),
                                post_smooth_sec=float(ms_lfp_post_smooth_sec),
                            )
                            for w in lfp_warnings:
                                print(f"\033[1;33m -- [{subject} - {cell.session}] {w}\033[0m")

                            merged_lfp = _merge_pseudo_lfp_rows_all_shanks(lfp_rows)
                            if merged_lfp is not None:
                                t_lfp, y_lfp = merged_lfp
                                if ms_lfp_max_points > 0 and t_lfp.size > ms_lfp_max_points:
                                    t_lfp, y_lfp = _downsample_xy(t_lfp, y_lfp, ms_lfp_max_points)
                                row_specs.append(
                                    {
                                        "kind": "ms_lfp",
                                        "session": cell.session,
                                        "shank_name": "ALL TT",
                                        "lfp_t": t_lfp,
                                        "lfp_y": y_lfp,
                                    }
                                )
                                if bool(ms_lfp_overlay):
                                    shank_units = []
                                    for vv in units_by_shank.values():
                                        shank_units.extend(vv)
                                    for unit_name, t_unit in shank_units:
                                        t_overlay = np.asarray(t_unit, dtype=float).reshape(-1)
                                        max_spike_overlay = max(1000, int(ms_lfp_max_points) * 3)
                                        if max_spike_overlay > 0 and t_overlay.size > max_spike_overlay:
                                            idx = _linspace_idx(t_overlay.size, max_spike_overlay)
                                            t_overlay = t_overlay[idx]
                                        row_specs.append(
                                            {
                                                "kind": "ms_overlay_unit",
                                                "session": cell.session,
                                                "unit_name": str(unit_name),
                                                "spike_t": t_overlay,
                                            }
                                        )
            else:
                for unit_name, spikes in units:
                    s = np.asarray(spikes, dtype=float).reshape(-1)
                    s = s[np.isfinite(s)]
                    if s.size == 0:
                        continue
                    t = (s - 1.0) / float(spike_sampling_rate)
                    t = t[np.isfinite(t)]
                    if t.size == 0:
                        continue
                    if np.isfinite(session_x_min) and np.isfinite(session_x_max):
                        m = (t >= float(session_x_min)) & (t <= float(session_x_max))
                        t = t[m]
                        if t.size == 0:
                            continue
                    row_specs.append(
                        {
                            "kind": "ms",
                            "session": cell.session,
                            "unit_name": str(unit_name),
                            "spike_t": t,
                        }
                    )

        end_row = len(row_specs)
        session_groups.append((cell.session, start_row, end_row))

    if not row_specs:
        raise ValueError("No plottable rows for this subject/mode.")

    if len(session_groups) > 1:
        row_specs_spaced: list[dict[str, Any]] = []
        session_groups_spaced: list[tuple[str, int, int]] = []
        for gidx, (session, start_row, end_row) in enumerate(session_groups):
            row_specs_spaced.extend(row_specs[start_row - 1:end_row])
            new_start = len(row_specs_spaced) - (end_row - start_row)
            new_end = len(row_specs_spaced)
            session_groups_spaced.append((session, new_start, new_end))
            if gidx < len(session_groups) - 1:
                row_specs_spaced.append({"kind": "gap"})
        row_specs = row_specs_spaced
        session_groups = session_groups_spaced

    n_rows = len(row_specs)
    hc_row_height_px = 260.0
    ms_row_height_px = 34.0
    ms_lfp_row_height_px = 260.0
    session_gap_px = 18.0
    row_gap_px = 4.0
    margin_top = 54
    margin_bottom = 44
    row_heights_px = [
        (
            hc_row_height_px
            if spec["kind"] == "hc"
            else ms_row_height_px
            if spec["kind"] in {"ms", "ms_overlay_unit"}
            else ms_lfp_row_height_px
            if spec["kind"] in {"ms_lfp", "ms_metric"}
            else session_gap_px
        )
        for spec in row_specs
    ]
    inner_height_px = float(np.sum(row_heights_px)) + max(0.0, (n_rows - 1) * row_gap_px)
    vertical_spacing = (row_gap_px / inner_height_px) if n_rows > 1 and inner_height_px > 0 else 0.0

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=vertical_spacing,
        row_heights=row_heights_px,
    )

    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    y_values_all: list[np.ndarray] = []
    legend_seen: set[str] = set()
    hc_rows: list[int] = []
    ms_rows: list[int] = []
    ms_label_rows: list[tuple[int, str]] = []
    ms_lfp_ranges: dict[int, tuple[float, float]] = {}

    for row_idx, spec in enumerate(row_specs, start=1):
        if spec["kind"] != "hc":
            continue
        traces = spec["traces"]
        hc_rows.append(row_idx)
        for tr in traces:
            x = np.asarray(tr.x, dtype=float).reshape(-1)
            y = np.asarray(tr.y, dtype=float).reshape(-1)
            n = min(x.size, y.size)
            if n < 2:
                continue
            x = x[:n]
            y = y[:n]
            m = np.isfinite(x) & np.isfinite(y)
            if int(np.sum(m)) < 2:
                continue
            x = x[m]
            y = y[m]
            x_min = min(x_min, float(np.min(x)))
            x_max = max(x_max, float(np.max(x)))
            y_min = min(y_min, float(np.min(y)))
            y_max = max(y_max, float(np.max(y)))
            y_values_all.append(y)

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color=tr.color, width=float(tr.width), dash=str(tr.dash)),
                    name=tr.name,
                    legendgroup=tr.name,
                    showlegend=(tr.name not in legend_seen),
                    hovertemplate=f"{tr.name}<br>x=%{{x:.3f}}<br>y=%{{y:.6g}}<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )
            legend_seen.add(tr.name)

        xref = "x domain" if row_idx == 1 else f"x{row_idx} domain"
        yref = "y domain" if row_idx == 1 else f"y{row_idx} domain"
        panel_title = f"{subject} | {spec['session']}"
        panel_label = str(spec.get("panel_label", "")).strip()
        if panel_label:
            panel_title = f"{panel_title} | {panel_label}"
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref=xref,
            yref=yref,
            text=f"<b>{panel_title}</b>",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font={"size": 14, "color": "black"},
            bgcolor="rgba(255,255,255,0.65)",
            borderpad=1,
        )

    for row_idx, spec in enumerate(row_specs, start=1):
        row_kind = spec.get("kind")
        if row_kind in {"ms", "ms_overlay_unit"}:
            ms_rows.append(row_idx)
            ms_label_rows.append((row_idx, f"MS {spec['unit_name']}"))
            t = np.asarray(spec["spike_t"], dtype=float).reshape(-1)
            if np.isfinite(x_min) and np.isfinite(x_max):
                m = (t >= float(x_min)) & (t <= float(x_max))
                t = t[m]
            if t.size > 0:
                fig.add_trace(
                    go.Scattergl(
                        x=t,
                        y=np.zeros_like(t),
                        mode="markers",
                        marker=dict(
                            symbol="line-ns-open",
                            color="#111111",
                            size=8,
                            line=dict(color="#111111", width=1),
                        ),
                        name="MS spike" if row_kind == "ms" else "MS spike (overlay)",
                        legendgroup="MS spike" if row_kind == "ms" else "MS spike (overlay)",
                        showlegend=False,
                        hovertemplate=f"{spec['unit_name']}<br>t=%{{x:.3f}}s<extra></extra>",
                    ),
                    row=row_idx,
                    col=1,
                )
            continue

        if row_kind not in {"ms_lfp", "ms_metric"}:
            continue

        ms_rows.append(row_idx)
        shank_name = str(spec.get("shank_name", "shank"))
        ms_label_rows.append((row_idx, f"MS {shank_name}"))

        if row_kind == "ms_metric":
            metric_traces = spec.get("traces", tuple())
            y_l = np.inf
            y_h = -np.inf
            for tr in metric_traces:
                x_m = np.asarray(tr.x, dtype=float).reshape(-1)
                y_m = np.asarray(tr.y, dtype=float).reshape(-1)
                n_m = min(x_m.size, y_m.size)
                if n_m < 2:
                    continue
                x_m = x_m[:n_m]
                y_m = y_m[:n_m]
                m_m = np.isfinite(x_m) & np.isfinite(y_m)
                if int(np.sum(m_m)) < 2:
                    continue
                x_m = x_m[m_m]
                y_m = y_m[m_m]
                x_min = min(x_min, float(np.min(x_m)))
                x_max = max(x_max, float(np.max(x_m)))
                y_l = min(y_l, float(np.min(y_m)))
                y_h = max(y_h, float(np.max(y_m)))
                fig.add_trace(
                    go.Scatter(
                        x=x_m,
                        y=y_m,
                        mode="lines",
                        line=dict(color=tr.color, width=float(tr.width), dash=str(tr.dash)),
                        name=tr.name,
                        legendgroup=tr.name,
                        showlegend=(tr.name not in legend_seen),
                        hovertemplate=f"{tr.name}<br>x=%{{x:.3f}}<br>y=%{{y:.6g}}<extra></extra>",
                    ),
                    row=row_idx,
                    col=1,
                )
                legend_seen.add(tr.name)

            if np.isfinite(y_l) and np.isfinite(y_h):
                if y_h <= y_l:
                    y_l -= 1.0
                    y_h += 1.0
                y_pad_lfp = 0.08 * (y_h - y_l)
                ms_lfp_ranges[row_idx] = (float(y_l - y_pad_lfp), float(y_h + y_pad_lfp))
            else:
                ms_lfp_ranges[row_idx] = (-1.0, 1.0)
            continue

        t_lfp = np.asarray(spec.get("lfp_t", np.array([], dtype=float)), dtype=float).reshape(-1)
        y_lfp = np.asarray(spec.get("lfp_y", np.array([], dtype=float)), dtype=float).reshape(-1)
        n = min(t_lfp.size, y_lfp.size)
        if n >= 2:
            t_lfp = t_lfp[:n]
            y_lfp = y_lfp[:n]
            m = np.isfinite(t_lfp) & np.isfinite(y_lfp)
            if int(np.sum(m)) >= 2:
                t_lfp = t_lfp[m]
                y_lfp = y_lfp[m]
                fig.add_trace(
                    go.Scatter(
                        x=t_lfp,
                        y=y_lfp,
                        mode="lines",
                        line=dict(color="#0b5a8f", width=1.3),
                        name="MS pseudo-LFP",
                        legendgroup="MS pseudo-LFP",
                        showlegend=("MS pseudo-LFP" not in legend_seen),
                        hovertemplate=f"{shank_name} pseudo-LFP<br>t=%{{x:.3f}}s<br>v=%{{y:.6g}}<extra></extra>",
                    ),
                    row=row_idx,
                    col=1,
                )
                legend_seen.add("MS pseudo-LFP")

                y_l = float(np.min(y_lfp))
                y_h = float(np.max(y_lfp))
                y_l = min(y_l, 0.0)
                y_h = max(y_h, 0.0)
                if y_h <= y_l:
                    y_l -= 1.0
                    y_h += 1.0
                y_pad_lfp = 0.08 * (y_h - y_l)
                ms_lfp_ranges[row_idx] = (float(y_l - y_pad_lfp), float(y_h + y_pad_lfp))

        if row_idx not in ms_lfp_ranges:
            ms_lfp_ranges[row_idx] = (-1.0, 1.0)

    if not np.isfinite(x_min):
        x_min = 0.0
    if (not np.isfinite(x_max)) or x_max <= x_min:
        x_max = x_min + 1.0

    if not np.isfinite(y_min):
        y_min = -1.0
    if (not np.isfinite(y_max)) or y_max <= y_min:
        y_max = y_min + 1.0

    if use_timeseries_outlier_rejection and y_values_all:
        y_concat = np.concatenate(y_values_all)
        y_inlier_min, y_inlier_max, outlier_detected = _grubbs_inlier_range(
            y_concat,
            alpha=0.01,
            max_remove_frac=0.02,
            min_n=8,
        )
        if (
            outlier_detected
            and np.isfinite(y_inlier_min)
            and np.isfinite(y_inlier_max)
            and (y_inlier_max > y_inlier_min)
        ):
            y_min = y_inlier_min
            y_max = y_inlier_max

    y_pad = 0.03 * (y_max - y_min)
    y_lo = float(y_min - y_pad)
    y_hi = float(y_max + y_pad)

    y_anchor = hc_rows[0] if hc_rows else None
    y_anchor_match = "y" if y_anchor == 1 else f"y{y_anchor}" if y_anchor is not None else None
    row_kind_map = {idx + 1: spec["kind"] for idx, spec in enumerate(row_specs)}
    visible_rows = [r for r in range(1, n_rows + 1) if row_kind_map.get(r) != "gap"]
    bottom_visible_row = visible_rows[-1] if visible_rows else n_rows
    session_bottom_rows = {
        end_row
        for _, _, end_row in session_groups
        if row_kind_map.get(end_row) != "gap"
    }
    for row_idx in range(1, n_rows + 1):
        row_kind = row_kind_map.get(row_idx, "hc")
        if row_kind == "gap":
            fig.update_xaxes(visible=False, row=row_idx, col=1)
            fig.update_yaxes(visible=False, row=row_idx, col=1)
            continue
        fig.update_xaxes(
            range=[float(x_min), float(x_max)],
            matches="x",
            title_text=(x_title if row_idx == bottom_visible_row else None),
            showgrid=True,
            zeroline=False,
            showticklabels=(row_idx in session_bottom_rows),
            row=row_idx,
            col=1,
        )
        if row_kind == "hc":
            fig.update_yaxes(
                range=[y_lo, y_hi],
                matches=y_anchor_match,
                title_text=(y_title if row_idx == hc_rows[0] else None),
                showgrid=True,
                zeroline=False,
                row=row_idx,
                col=1,
            )
        elif row_kind in {"ms_lfp", "ms_metric"}:
            y_lfp_lo, y_lfp_hi = ms_lfp_ranges.get(row_idx, (-1.0, 1.0))
            fig.update_yaxes(
                range=[float(y_lfp_lo), float(y_lfp_hi)],
                showgrid=True,
                zeroline=True,
                showticklabels=True,
                ticks="outside",
                tickformat=".3g",
                nticks=4,
                row=row_idx,
                col=1,
            )
        else:
            fig.update_yaxes(
                range=[-1.0, 1.0],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                row=row_idx,
                col=1,
            )

    # MS labels: outside left of subplot frames, right-aligned.
    ms_label_x_default = -0.012
    ms_label_x_lfp = -0.06
    for row_idx, label_txt in ms_label_rows:
        axis_key = "yaxis" if row_idx == 1 else f"yaxis{row_idx}"
        axis_obj = getattr(fig.layout, axis_key, None)
        domain = getattr(axis_obj, "domain", None)
        if domain is None or len(domain) != 2:
            continue
        y_mid = 0.5 * (float(domain[0]) + float(domain[1]))
        row_kind = row_kind_map.get(row_idx, "ms")
        x_pos = ms_label_x_lfp if row_kind in {"ms_lfp", "ms_metric"} else ms_label_x_default
        fig.add_annotation(
            x=x_pos,
            y=y_mid,
            xref="paper",
            yref="paper",
            text=str(label_txt),
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            align="right",
            font={"size": 10, "color": "#2d3748"},
        )

    for session, start_row, end_row in session_groups:
        y_top_axis = fig.layout.yaxis if start_row == 1 else getattr(fig.layout, f"yaxis{start_row}")
        y_bot_axis = fig.layout.yaxis if end_row == 1 else getattr(fig.layout, f"yaxis{end_row}")
        y_top_domain = getattr(y_top_axis, "domain", None)
        y_bot_domain = getattr(y_bot_axis, "domain", None)
        if (
            y_top_domain is None
            or y_bot_domain is None
            or len(y_top_domain) != 2
            or len(y_bot_domain) != 2
        ):
            continue
        y0 = float(y_bot_domain[0])
        y1 = float(y_top_domain[1])
        fig.add_shape(
            type="rect",
            x0=0.0,
            x1=1.0,
            y0=y0,
            y1=y1,
            xref="paper",
            yref="paper",
            line={"color": "rgba(30,30,30,0.85)", "width": 1.0},
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )

    fig.update_layout(
        template="plotly_white",
        height=int(inner_height_px + margin_top + margin_bottom),
        margin={"l": 142, "r": 20, "t": (margin_top + 30 if legend_seen else margin_top), "b": margin_bottom},
        showlegend=bool(legend_seen),
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(255,255,255,0.75)",
            "bordercolor": "rgba(0,0,0,0.15)",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        dragmode="pan",
        hovermode="x",
    )
    return fig


def _build_subject_spectrogram_figure(
    subject: str,
    mode_cells: list[ModeCell],
    x_title: str,
    y_title: str,
) -> Any:
    _require_plotly()
    if not mode_cells:
        raise ValueError("No cells to plot.")

    cells_sorted = sorted(mode_cells, key=lambda c: natural_key(c.session))
    row_specs: list[dict[str, Any]] = []
    session_groups: list[tuple[str, int, int]] = []
    power_values: list[np.ndarray] = []
    diff_values: list[np.ndarray] = []

    for cell in cells_sorted:
        payload = cell.payload if isinstance(cell.payload, dict) else None
        if payload is None:
            continue
        t = np.asarray(payload.get("time_sec", np.array([], dtype=float)), dtype=float).reshape(-1)
        f = np.asarray(payload.get("freq_hz", np.array([], dtype=float)), dtype=float).reshape(-1)
        hc_map = np.asarray(payload.get("hc_map", np.array([], dtype=float)), dtype=float)
        ms_map = np.asarray(payload.get("ms_map", np.array([], dtype=float)), dtype=float)
        diff_map = np.asarray(payload.get("diff_map", np.array([], dtype=float)), dtype=float)
        if t.size < 2 or f.size < 2:
            continue
        if hc_map.shape != (f.size, t.size) or ms_map.shape != (f.size, t.size) or diff_map.shape != (f.size, t.size):
            continue

        start_row = len(row_specs) + 1
        row_specs.append(
            {
                "kind": "power",
                "session": cell.session,
                "subject": cell.subject,
                "t": t,
                "f": f,
                "z": hc_map,
                "label": "HC LFP",
                "metrics": payload,
            }
        )
        row_specs.append(
            {
                "kind": "power",
                "session": cell.session,
                "subject": cell.subject,
                "t": t,
                "f": f,
                "z": ms_map,
                "label": "MS pseudo-LFP",
                "metrics": payload,
            }
        )
        row_specs.append(
            {
                "kind": "diff",
                "session": cell.session,
                "subject": cell.subject,
                "t": t,
                "f": f,
                "z": diff_map,
                "label": "MS - HC",
                "metrics": payload,
            }
        )
        end_row = len(row_specs)
        session_groups.append((cell.session, start_row, end_row))

        for arr in (hc_map, ms_map):
            vv = np.asarray(arr, dtype=float).reshape(-1)
            vv = vv[np.isfinite(vv)]
            if vv.size:
                power_values.append(vv)
        dd = diff_map.reshape(-1)
        dd = dd[np.isfinite(dd)]
        if dd.size:
            diff_values.append(dd)

    if not row_specs:
        raise ValueError("No plottable spectrogram rows for this subject.")

    if len(session_groups) > 1:
        row_specs_spaced: list[dict[str, Any]] = []
        session_groups_spaced: list[tuple[str, int, int]] = []
        for gidx, (session, start_row, end_row) in enumerate(session_groups):
            row_specs_spaced.extend(row_specs[start_row - 1:end_row])
            new_start = len(row_specs_spaced) - (end_row - start_row)
            new_end = len(row_specs_spaced)
            session_groups_spaced.append((session, new_start, new_end))
            if gidx < len(session_groups) - 1:
                row_specs_spaced.append({"kind": "gap"})
        row_specs = row_specs_spaced
        session_groups = session_groups_spaced

    power_concat = np.concatenate(power_values) if power_values else np.array([], dtype=float)
    if power_concat.size:
        z_power_min, z_power_max = np.nanpercentile(power_concat, [2, 98])
        if not np.isfinite(z_power_min) or not np.isfinite(z_power_max) or z_power_max <= z_power_min:
            z_power_min, z_power_max = float(np.nanmin(power_concat)), float(np.nanmax(power_concat))
    else:
        z_power_min, z_power_max = -1.0, 1.0
    if z_power_max <= z_power_min:
        z_power_min -= 1.0
        z_power_max += 1.0

    diff_concat = np.concatenate(diff_values) if diff_values else np.array([], dtype=float)
    if diff_concat.size:
        z_diff_abs = float(np.nanpercentile(np.abs(diff_concat), 98))
        if not np.isfinite(z_diff_abs) or z_diff_abs <= 0.0:
            z_diff_abs = float(np.nanmax(np.abs(diff_concat)))
    else:
        z_diff_abs = 1.0
    if not np.isfinite(z_diff_abs) or z_diff_abs <= 0.0:
        z_diff_abs = 1.0

    n_rows = len(row_specs)
    map_row_height_px = 230.0
    session_gap_px = 18.0
    row_gap_px = 5.0
    margin_top = 58
    margin_bottom = 46
    row_heights_px = [
        (map_row_height_px if spec["kind"] != "gap" else session_gap_px)
        for spec in row_specs
    ]
    inner_height_px = float(np.sum(row_heights_px)) + max(0.0, (n_rows - 1) * row_gap_px)
    vertical_spacing = (row_gap_px / inner_height_px) if n_rows > 1 and inner_height_px > 0 else 0.0

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=vertical_spacing,
        row_heights=row_heights_px,
    )

    x_min = np.inf
    x_max = -np.inf
    f_min = np.inf
    f_max = -np.inf
    for spec in row_specs:
        if spec["kind"] == "gap":
            continue
        t = np.asarray(spec["t"], dtype=float).reshape(-1)
        f = np.asarray(spec["f"], dtype=float).reshape(-1)
        if t.size >= 2:
            x_min = min(x_min, float(np.nanmin(t)))
            x_max = max(x_max, float(np.nanmax(t)))
        if f.size >= 2:
            f_min = min(f_min, float(np.nanmin(f)))
            f_max = max(f_max, float(np.nanmax(f)))
    if not np.isfinite(x_min):
        x_min = 0.0
    if (not np.isfinite(x_max)) or x_max <= x_min:
        x_max = x_min + 1.0
    if not np.isfinite(f_min):
        f_min = 0.0
    if (not np.isfinite(f_max)) or f_max <= f_min:
        f_max = f_min + 1.0

    first_power_scale = True
    first_diff_scale = True
    row_kind_map = {idx + 1: spec["kind"] for idx, spec in enumerate(row_specs)}
    visible_rows = [r for r in range(1, n_rows + 1) if row_kind_map.get(r) != "gap"]
    bottom_visible_row = visible_rows[-1] if visible_rows else n_rows
    session_bottom_rows = {
        end_row
        for _, _, end_row in session_groups
        if row_kind_map.get(end_row) != "gap"
    }

    for row_idx, spec in enumerate(row_specs, start=1):
        if spec["kind"] == "gap":
            fig.update_xaxes(visible=False, row=row_idx, col=1)
            fig.update_yaxes(visible=False, row=row_idx, col=1)
            continue

        t = np.asarray(spec["t"], dtype=float).reshape(-1)
        f = np.asarray(spec["f"], dtype=float).reshape(-1)
        z = np.asarray(spec["z"], dtype=float)
        max_points = int(spec.get("metrics", {}).get("max_points", 0))
        t_plot, z_plot = _downsample_heatmap_time(t, z, max_points)

        is_diff = spec["kind"] == "diff"
        show_scale = first_diff_scale if is_diff else first_power_scale
        if is_diff:
            first_diff_scale = False
            colorscale = "RdBu_r"
            zmin = -z_diff_abs
            zmax = z_diff_abs
            ctitle = "MS-HC dB"
        else:
            first_power_scale = False
            colorscale = "Jet"
            zmin = float(z_power_min)
            zmax = float(z_power_max)
            ctitle = "Power (dB)"

        fig.add_trace(
            go.Heatmap(
                x=t_plot,
                y=f,
                z=z_plot,
                colorscale=colorscale,
                zmin=float(zmin),
                zmax=float(zmax),
                colorbar=(
                    {"title": ctitle, "len": 0.22, "thickness": 12}
                    if show_scale
                    else None
                ),
                showscale=show_scale,
                hovertemplate=f"{spec['label']}<br>t=%{{x:.3f}}s<br>f=%{{y:.3f}}Hz<br>v=%{{z:.6g}}<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )

        xref = "x domain" if row_idx == 1 else f"x{row_idx} domain"
        yref = "y domain" if row_idx == 1 else f"y{row_idx} domain"
        metrics = spec.get("metrics", {})
        extra = ""
        if spec["label"] == "HC LFP":
            corr = float(metrics.get("map_corr", np.nan))
            mae = float(metrics.get("mean_abs_diff_db", np.nan))
            corr_s = f"{corr:.3g}" if np.isfinite(corr) else "NA"
            mae_s = f"{mae:.3g}" if np.isfinite(mae) else "NA"
            extra = f"<br>map r={corr_s}, mean |MS-HC|={mae_s} dB"
        panel_title = f"{subject} | {spec['session']} | {spec['label']}{extra}"
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref=xref,
            yref=yref,
            text=f"<b>{panel_title}</b>",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font={"size": 12, "color": "white"},
            bgcolor="rgba(0,0,0,0.38)",
            borderpad=2,
        )

    for row_idx in range(1, n_rows + 1):
        row_kind = row_kind_map.get(row_idx, "power")
        if row_kind == "gap":
            continue
        fig.update_xaxes(
            range=[float(x_min), float(x_max)],
            matches="x",
            title_text=(x_title if row_idx == bottom_visible_row else None),
            showgrid=True,
            zeroline=False,
            showticklabels=(row_idx in session_bottom_rows),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(
            range=[float(f_min), float(f_max)],
            title_text=(y_title if row_idx == visible_rows[0] else None),
            showgrid=True,
            zeroline=False,
            row=row_idx,
            col=1,
        )

    for _, start_row, end_row in session_groups:
        y_top_axis = fig.layout.yaxis if start_row == 1 else getattr(fig.layout, f"yaxis{start_row}")
        y_bot_axis = fig.layout.yaxis if end_row == 1 else getattr(fig.layout, f"yaxis{end_row}")
        y_top_domain = getattr(y_top_axis, "domain", None)
        y_bot_domain = getattr(y_bot_axis, "domain", None)
        if (
            y_top_domain is None
            or y_bot_domain is None
            or len(y_top_domain) != 2
            or len(y_bot_domain) != 2
        ):
            continue
        fig.add_shape(
            type="rect",
            x0=0.0,
            x1=1.0,
            y0=float(y_bot_domain[0]),
            y1=float(y_top_domain[1]),
            xref="paper",
            yref="paper",
            line={"color": "rgba(30,30,30,0.85)", "width": 1.0},
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )

    fig.update_layout(
        template="plotly_white",
        height=int(inner_height_px + margin_top + margin_bottom),
        margin={"l": 86, "r": 28, "t": margin_top, "b": margin_bottom},
        showlegend=False,
        title={"text": f"{subject} | Spectrogram", "x": 0.01},
        dragmode="pan",
        hovermode="x",
    )
    return fig


def _build_subject_coherence_figure(
    subject: str,
    mode_cells: list[ModeCell],
    x_title: str,
    y_title: str,
) -> Any:
    _require_plotly()
    if not mode_cells:
        raise ValueError("No cells to plot.")

    cells_sorted = sorted(mode_cells, key=lambda c: natural_key(c.session))
    row_specs: list[dict[str, Any]] = []
    session_groups: list[tuple[str, int, int]] = []

    for cell in cells_sorted:
        payload = cell.payload if isinstance(cell.payload, dict) else None
        if payload is None:
            continue
        t = np.asarray(payload.get("time_sec", np.array([], dtype=float)), dtype=float).reshape(-1)
        f = np.asarray(payload.get("freq_hz", np.array([], dtype=float)), dtype=float).reshape(-1)
        coh = np.asarray(payload.get("coh_map", np.array([], dtype=float)), dtype=float)
        plv = np.asarray(payload.get("plv_map", np.array([], dtype=float)), dtype=float)
        if t.size < 2 or f.size < 2:
            continue
        if coh.ndim != 2 or plv.ndim != 2:
            continue
        if coh.shape != (f.size, t.size) or plv.shape != (f.size, t.size):
            continue

        start_row = len(row_specs) + 1
        row_specs.append(
            {
                "kind": "coh",
                "session": cell.session,
                "subject": cell.subject,
                "t": t,
                "f": f,
                "z": np.clip(coh, 0.0, 1.0),
                "label": "Coh",
            }
        )
        row_specs.append(
            {
                "kind": "plv",
                "session": cell.session,
                "subject": cell.subject,
                "t": t,
                "f": f,
                "z": np.clip(plv, 0.0, 1.0),
                "label": "PLV",
            }
        )
        end_row = len(row_specs)
        session_groups.append((cell.session, start_row, end_row))

    if not row_specs:
        raise ValueError("No plottable coherence rows for this subject.")

    if len(session_groups) > 1:
        row_specs_spaced: list[dict[str, Any]] = []
        session_groups_spaced: list[tuple[str, int, int]] = []
        for gidx, (session, start_row, end_row) in enumerate(session_groups):
            row_specs_spaced.extend(row_specs[start_row - 1:end_row])
            new_start = len(row_specs_spaced) - (end_row - start_row)
            new_end = len(row_specs_spaced)
            session_groups_spaced.append((session, new_start, new_end))
            if gidx < len(session_groups) - 1:
                row_specs_spaced.append({"kind": "gap"})
        row_specs = row_specs_spaced
        session_groups = session_groups_spaced

    n_rows = len(row_specs)
    map_row_height_px = 250.0
    session_gap_px = 18.0
    row_gap_px = 6.0
    margin_top = 56
    margin_bottom = 46
    row_heights_px = [
        (map_row_height_px if spec["kind"] != "gap" else session_gap_px)
        for spec in row_specs
    ]
    inner_height_px = float(np.sum(row_heights_px)) + max(0.0, (n_rows - 1) * row_gap_px)
    vertical_spacing = (row_gap_px / inner_height_px) if n_rows > 1 and inner_height_px > 0 else 0.0

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=vertical_spacing,
        row_heights=row_heights_px,
    )

    x_min = np.inf
    x_max = -np.inf
    f_min = np.inf
    f_max = -np.inf
    for spec in row_specs:
        if spec["kind"] == "gap":
            continue
        t = np.asarray(spec["t"], dtype=float).reshape(-1)
        f = np.asarray(spec["f"], dtype=float).reshape(-1)
        if t.size >= 2:
            x_min = min(x_min, float(np.nanmin(t)))
            x_max = max(x_max, float(np.nanmax(t)))
        if f.size >= 2:
            f_min = min(f_min, float(np.nanmin(f)))
            f_max = max(f_max, float(np.nanmax(f)))

    if not np.isfinite(x_min):
        x_min = 0.0
    if (not np.isfinite(x_max)) or x_max <= x_min:
        x_max = x_min + 1.0
    if not np.isfinite(f_min):
        f_min = 0.0
    if (not np.isfinite(f_max)) or f_max <= f_min:
        f_max = f_min + 1.0

    first_scale_seen: set[str] = set()
    row_kind_map = {idx + 1: spec["kind"] for idx, spec in enumerate(row_specs)}
    visible_rows = [r for r in range(1, n_rows + 1) if row_kind_map.get(r) != "gap"]
    bottom_visible_row = visible_rows[-1] if visible_rows else n_rows
    session_bottom_rows = {
        end_row
        for _, _, end_row in session_groups
        if row_kind_map.get(end_row) != "gap"
    }

    for row_idx, spec in enumerate(row_specs, start=1):
        if spec["kind"] == "gap":
            fig.update_xaxes(visible=False, row=row_idx, col=1)
            fig.update_yaxes(visible=False, row=row_idx, col=1)
            continue

        metric = str(spec.get("label", "")).strip() or "Metric"
        t = np.asarray(spec["t"], dtype=float).reshape(-1)
        f = np.asarray(spec["f"], dtype=float).reshape(-1)
        z = np.asarray(spec["z"], dtype=float)
        z = np.where(np.isfinite(z), z, np.nan)

        show_scale = metric not in first_scale_seen
        first_scale_seen.add(metric)

        fig.add_trace(
            go.Heatmap(
                x=t,
                y=f,
                z=z,
                colorscale="Jet",
                zmin=0.0,
                zmax=1.0,
                colorbar=(
                    {
                        "title": metric,
                        "len": 0.24,
                        "thickness": 12,
                    }
                    if show_scale
                    else None
                ),
                showscale=show_scale,
                hovertemplate=f"{metric}<br>t=%{{x:.3f}}s<br>f=%{{y:.3f}}Hz<br>v=%{{z:.6g}}<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )

        xref = "x domain" if row_idx == 1 else f"x{row_idx} domain"
        yref = "y domain" if row_idx == 1 else f"y{row_idx} domain"
        panel_title = f"{subject} | {spec['session']} | {metric}"
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref=xref,
            yref=yref,
            text=f"<b>{panel_title}</b>",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font={"size": 13, "color": "white"},
            bgcolor="rgba(0,0,0,0.35)",
            borderpad=2,
        )

    for row_idx in range(1, n_rows + 1):
        row_kind = row_kind_map.get(row_idx, "coh")
        if row_kind == "gap":
            continue
        fig.update_xaxes(
            range=[float(x_min), float(x_max)],
            matches="x",
            title_text=(x_title if row_idx == bottom_visible_row else None),
            showgrid=True,
            zeroline=False,
            showticklabels=(row_idx in session_bottom_rows),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(
            range=[float(f_min), float(f_max)],
            title_text=(y_title if row_idx == visible_rows[0] else None),
            showgrid=True,
            zeroline=False,
            row=row_idx,
            col=1,
        )

    for _, start_row, end_row in session_groups:
        y_top_axis = fig.layout.yaxis if start_row == 1 else getattr(fig.layout, f"yaxis{start_row}")
        y_bot_axis = fig.layout.yaxis if end_row == 1 else getattr(fig.layout, f"yaxis{end_row}")
        y_top_domain = getattr(y_top_axis, "domain", None)
        y_bot_domain = getattr(y_bot_axis, "domain", None)
        if (
            y_top_domain is None
            or y_bot_domain is None
            or len(y_top_domain) != 2
            or len(y_bot_domain) != 2
        ):
            continue
        y0 = float(y_bot_domain[0])
        y1 = float(y_top_domain[1])
        fig.add_shape(
            type="rect",
            x0=0.0,
            x1=1.0,
            y0=y0,
            y1=y1,
            xref="paper",
            yref="paper",
            line={"color": "rgba(30,30,30,0.85)", "width": 1.0},
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )

    fig.update_layout(
        template="plotly_white",
        height=int(inner_height_px + margin_top + margin_bottom),
        margin={"l": 95, "r": 72, "t": margin_top, "b": margin_bottom},
        showlegend=False,
        dragmode="pan",
        hovermode="x",
    )
    return fig


def _build_subject_granger_figure(
    subject: str,
    mode_cells: list[ModeCell],
    x_title: str,
    y_title: str,
) -> Any:
    _require_plotly()
    if not mode_cells:
        raise ValueError("No cells to plot.")

    cells_sorted = sorted(mode_cells, key=lambda c: natural_key(c.session))
    row_specs: list[dict[str, Any]] = []
    for cell in cells_sorted:
        payload = cell.payload if isinstance(cell.payload, dict) else None
        if payload is None:
            continue
        freq = np.asarray(payload.get("freq_hz", np.array([], dtype=float)), dtype=float).reshape(-1)
        gpdc_mean = np.asarray(payload.get("gpdc_mean", np.array([], dtype=float)), dtype=float)
        gpdc_std = np.asarray(payload.get("gpdc_std", np.array([], dtype=float)), dtype=float)
        if freq.size < 2:
            continue
        if gpdc_mean.shape != (2, 2, freq.size):
            continue
        if gpdc_std.shape != (2, 2, freq.size):
            continue

        row_specs.append(
            {
                "session": cell.session,
                "freq": freq,
                "gpdc_mean": np.clip(gpdc_mean, 0.0, 1.0),
                "gpdc_std": np.clip(gpdc_std, 0.0, 1.0),
                "n_windows": int(payload.get("n_windows", 0)),
                "order_mean": float(payload.get("order_mean", np.nan)),
                "order_std": float(payload.get("order_std", np.nan)),
                "order_min": int(payload.get("order_min", -1)),
                "order_max": int(payload.get("order_max", -1)),
                "order_criterion": str(payload.get("order_criterion", "bic")),
                "order_mode": str(payload.get("order_mode", "per_window")),
            }
        )

    if not row_specs:
        raise ValueError("No plottable granger rows for this subject.")

    n_sessions = len(row_specs)
    n_rows = n_sessions
    vertical_spacing = (0.06 / float(max(1, n_sessions))) if n_rows > 1 else 0.0
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.08,
        vertical_spacing=vertical_spacing,
    )

    x_min = np.inf
    x_max = -np.inf
    for spec in row_specs:
        f = np.asarray(spec["freq"], dtype=float).reshape(-1)
        if f.size >= 2:
            x_min = min(x_min, float(np.nanmin(f)))
            x_max = max(x_max, float(np.nanmax(f)))
    if not np.isfinite(x_min):
        x_min = 0.0
    if (not np.isfinite(x_max)) or x_max <= x_min:
        x_max = x_min + 1.0

    panel_defs = [
        (1, 0, 1, 1, "HC->MS", "#2ca02c", "rgba(44,160,44,0.22)"),
        (0, 1, 1, 2, "MS->HC", "#d62728", "rgba(214,39,40,0.22)"),
    ]

    def _axis_domain_ref(row: int, col: int, axis: str) -> str:
        idx = ((int(row) - 1) * 2) + int(col)
        if idx == 1:
            return f"{axis} domain"
        return f"{axis}{idx} domain"

    for sess_idx, spec in enumerate(row_specs):
        base_row = sess_idx + 1
        freq = np.asarray(spec["freq"], dtype=float).reshape(-1)
        gpdc_mean = np.asarray(spec["gpdc_mean"], dtype=float)
        gpdc_std = np.asarray(spec["gpdc_std"], dtype=float)
        ord_mode = str(spec.get("order_mode", "per_window")).lower()
        crit = str(spec["order_criterion"]).upper()
        info_tail = (
            f"windows={int(spec['n_windows'])}, "
            f"mode={ord_mode}, p({crit})={spec['order_mean']:.2f}+/-{spec['order_std']:.2f} "
            f"[{int(spec['order_min'])}-{int(spec['order_max'])}]"
        )

        for i, j, row_off, col_idx, label, line_color, fill_color in panel_defs:
            row_idx = base_row + (row_off - 1)
            mean_curve = np.asarray(gpdc_mean[i, j, :], dtype=float).reshape(-1)
            std_curve = np.asarray(gpdc_std[i, j, :], dtype=float).reshape(-1)
            n = min(freq.size, mean_curve.size, std_curve.size)
            if n < 2:
                continue
            xf = freq[:n]
            ym = mean_curve[:n]
            ys = std_curve[:n]
            m = np.isfinite(xf) & np.isfinite(ym) & np.isfinite(ys)
            if int(np.sum(m)) < 2:
                continue
            xf = xf[m]
            ym = ym[m]
            ys = ys[m]

            y_lo = np.clip(ym - ys, 0.0, 1.0)
            y_hi = np.clip(ym + ys, 0.0, 1.0)

            fig.add_trace(
                go.Scatter(
                    x=xf,
                    y=y_hi,
                    mode="lines",
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row_idx,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=xf,
                    y=y_lo,
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=fill_color,
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row_idx,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=xf,
                    y=np.clip(ym, 0.0, 1.0),
                    mode="lines",
                    line={"color": line_color, "width": 1.8},
                    name=label,
                    showlegend=False,
                    hovertemplate=f"{label}<br>f=%{{x:.3f}}Hz<br>gPDC=%{{y:.6g}}<extra></extra>",
                ),
                row=row_idx,
                col=col_idx,
            )

            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref=_axis_domain_ref(row_idx, col_idx, "x"),
                yref=_axis_domain_ref(row_idx, col_idx, "y"),
                text=f"<b>{subject} | {spec['session']} | {label}</b><br>{info_tail}",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                align="left",
                font={"size": 12, "color": "black"},
                bgcolor="rgba(255,255,255,0.72)",
                borderpad=1,
            )

    for row_idx in range(1, n_rows + 1):
        for col_idx in (1, 2):
            is_bottom = (row_idx == n_rows)
            fig.update_xaxes(
                range=[float(x_min), float(x_max)],
                title_text=(x_title if is_bottom else None),
                showgrid=True,
                minor={
                    "dtick": 10,
                    "showgrid": True,
                    "gridcolor": "rgba(120,120,120,0.18)",
                    "gridwidth": 0.5,
                },
                zeroline=False,
                showticklabels=True,
                row=row_idx,
                col=col_idx,
            )
            fig.update_yaxes(
                range=[0.0, 1.0],
                title_text=(y_title if col_idx == 1 else None),
                showgrid=True,
                zeroline=False,
                row=row_idx,
                col=col_idx,
            )

    for sess_idx in range(n_sessions):
        row = sess_idx + 1
        y_idx = ((row - 1) * 2) + 1
        key = "yaxis" if y_idx == 1 else f"yaxis{y_idx}"
        axis = getattr(fig.layout, key, None)
        domain = getattr(axis, "domain", None)
        if domain is None or len(domain) != 2:
            continue
        fig.add_shape(
            type="rect",
            x0=0.0,
            x1=1.0,
            y0=float(domain[0]),
            y1=float(domain[1]),
            xref="paper",
            yref="paper",
            line={"color": "rgba(30,30,30,0.85)", "width": 1.0},
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )

    fig.update_layout(
        template="plotly_white",
        height=int((300 * n_sessions) + 80),
        margin={"l": 96, "r": 24, "t": 58, "b": 44},
        showlegend=False,
        dragmode="pan",
        hovermode="x",
    )
    return fig


def _build_subject_phaselag_figure(
    subject: str,
    mode_cells: list[ModeCell],
) -> Any:
    _require_plotly()
    cells_sorted = sorted(mode_cells, key=lambda c: natural_key(c.session))
    row_specs: list[dict[str, Any]] = []
    x_min = np.inf
    x_max = -np.inf

    for cell in cells_sorted:
        payload = cell.payload if isinstance(cell.payload, dict) else None
        if payload is None:
            continue
        band_stats = payload.get("band_stats")
        if not isinstance(band_stats, dict) or not band_stats:
            continue
        clean_stats: dict[str, dict[str, Any]] = {}
        for band_label, st in band_stats.items():
            if not isinstance(st, dict):
                continue
            lags = np.asarray(st.get("lags_ms", np.array([], dtype=float)), dtype=float).reshape(-1)
            hist_prob = np.asarray(st.get("hist_prob", np.array([], dtype=float)), dtype=float).reshape(-1)
            mean_norm_z = np.asarray(st.get("mean_norm_z", np.array([], dtype=float)), dtype=float).reshape(-1)
            n = min(lags.size, hist_prob.size, mean_norm_z.size)
            if n < 2:
                continue
            lags = lags[:n]
            hist_prob = hist_prob[:n]
            mean_norm_z = mean_norm_z[:n]
            m = np.isfinite(lags) & (np.isfinite(hist_prob) | np.isfinite(mean_norm_z))
            if int(np.sum(m)) < 2:
                continue
            x_min = min(x_min, float(np.nanmin(lags[m])))
            x_max = max(x_max, float(np.nanmax(lags[m])))
            clean_stats[str(band_label)] = st
        if clean_stats:
            row_specs.append({"session": cell.session, "band_stats": clean_stats})

    if not row_specs:
        raise ValueError("No plottable PhaseLag rows for this subject.")
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        x_min, x_max = -1.0, 1.0

    fig = make_subplots(
        rows=len(row_specs),
        cols=2,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.08,
        vertical_spacing=(0.06 / float(max(1, len(row_specs)))) if len(row_specs) > 1 else 0.0,
        subplot_titles=None,
    )

    for row_idx, spec in enumerate(row_specs, start=1):
        band_stats = spec["band_stats"]
        summary_lines: list[str] = []
        for bidx, (band_label, st) in enumerate(band_stats.items()):
            color = PLOTLY_COLORS[bidx % len(PLOTLY_COLORS)]
            lags = np.asarray(st.get("lags_ms", np.array([], dtype=float)), dtype=float).reshape(-1)
            hist_prob = np.asarray(st.get("hist_prob", np.array([], dtype=float)), dtype=float).reshape(-1)
            mean_norm_z = np.asarray(st.get("mean_norm_z", np.array([], dtype=float)), dtype=float).reshape(-1)
            sem_norm_z = np.asarray(st.get("sem_norm_z", np.zeros_like(mean_norm_z)), dtype=float).reshape(-1)
            n = min(lags.size, hist_prob.size, mean_norm_z.size, sem_norm_z.size)
            if n < 2:
                continue
            lags = lags[:n]
            hist_prob = hist_prob[:n]
            mean_norm_z = mean_norm_z[:n]
            sem_norm_z = sem_norm_z[:n]

            fig.add_trace(
                go.Scatter(
                    x=lags,
                    y=hist_prob,
                    mode="lines+markers",
                    line={"color": color, "width": 1.4},
                    marker={"size": 4},
                    name=band_label,
                    legendgroup=band_label,
                    showlegend=(row_idx == 1),
                    hovertemplate=(
                        f"{band_label}<br>lag=%{{x:.3g}} ms"
                        "<br>best-lag probability=%{y:.3g}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )

            y_hi = np.clip(mean_norm_z + sem_norm_z, 0.0, np.inf)
            y_lo = np.clip(mean_norm_z - sem_norm_z, 0.0, np.inf)
            fig.add_trace(
                go.Scatter(
                    x=lags,
                    y=y_hi,
                    mode="lines",
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=band_label,
                ),
                row=row_idx,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=lags,
                    y=y_lo,
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(120,120,120,0.12)",
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=band_label,
                ),
                row=row_idx,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=lags,
                    y=mean_norm_z,
                    mode="lines",
                    line={"color": color, "width": 1.8},
                    name=band_label,
                    legendgroup=band_label,
                    showlegend=False,
                    hovertemplate=(
                        f"{band_label}<br>lag=%{{x:.3g}} ms"
                        "<br>mean normalized Rayleigh Z=%{y:.3g}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=2,
            )

            try:
                summary_lines.append(
                    f"{band_label}: peak={float(st.get('peak_lag_ms', np.nan)):.3g}ms, "
                    f"mode={float(st.get('mode_lag_ms', np.nan)):.3g}ms, "
                    f"MSlead={float(st.get('ms_lead_ratio', np.nan)):.2f}, "
                    f"valid={int(st.get('n_valid_windows', 0))}/{int(st.get('n_total_windows', 0))}, "
                    f"edge={int(st.get('n_edge_hit_windows', 0))}"
                )
            except Exception:
                pass

        for col_idx in (1, 2):
            fig.add_vline(
                x=0.0,
                line={"color": "rgba(40,40,40,0.55)", "width": 1.0, "dash": "dot"},
                row=row_idx,
                col=col_idx,
            )
            fig.update_xaxes(
                range=[float(x_min), float(x_max)],
                title_text=("Lag (ms)" if row_idx == len(row_specs) else None),
                showgrid=True,
                zeroline=False,
                row=row_idx,
                col=col_idx,
            )
        fig.update_yaxes(
            title_text="Best-lag probability",
            showgrid=True,
            zeroline=False,
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(
            title_text="Mean norm. Rayleigh Z",
            showgrid=True,
            zeroline=False,
            row=row_idx,
            col=2,
        )
        annotation_text = f"<b>{subject} | {spec['session']}</b>"
        if summary_lines:
            annotation_text += "<br>" + "<br>".join(summary_lines[:8])
        fig.add_annotation(
            x=0.01,
            y=0.99,
            xref=f"x{(row_idx - 1) * 2 + 1 if row_idx > 1 else ''} domain",
            yref=f"y{(row_idx - 1) * 2 + 1 if row_idx > 1 else ''} domain",
            text=annotation_text,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font={"size": 10, "color": "black"},
            bgcolor="rgba(255,255,255,0.74)",
            borderpad=1,
        )

    fig.update_layout(
        template="plotly_white",
        height=int((330 * len(row_specs)) + 90),
        margin={"l": 86, "r": 24, "t": 58, "b": 48},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.0},
        title={"text": f"{subject} | PhaseLag Z-shift lag distribution", "x": 0.01},
        dragmode="pan",
        hovermode="x",
    )
    return fig


def _build_subject_syncfc_figure(
    subject: str,
    mode_cells: list[ModeCell],
) -> Any:
    _require_plotly()
    cells_sorted = sorted(mode_cells, key=lambda c: natural_key(c.session))
    row_specs: list[dict[str, Any]] = []
    band_order: list[str] = []

    for cell in cells_sorted:
        payload = cell.payload if isinstance(cell.payload, dict) else None
        if payload is None:
            continue
        band_stats = payload.get("band_stats")
        if not isinstance(band_stats, dict) or not band_stats:
            continue
        clean_stats: dict[str, dict[str, Any]] = {}
        for band_label, st in band_stats.items():
            if not isinstance(st, dict):
                continue
            plv = np.asarray(st.get("plv", np.array([], dtype=float)), dtype=float).reshape(-1)
            coh = np.asarray(st.get("coh", np.array([], dtype=float)), dtype=float).reshape(-1)
            if int(np.sum(np.isfinite(plv))) < 2 and int(np.sum(np.isfinite(coh))) < 2:
                continue
            band = str(band_label)
            clean_stats[band] = st
            if band not in band_order:
                band_order.append(band)
        if clean_stats:
            row_specs.append({"session": cell.session, "band_stats": clean_stats})

    if not row_specs:
        raise ValueError("No plottable SyncFC rows for this subject.")

    fig = make_subplots(
        rows=len(row_specs),
        cols=2,
        shared_xaxes=False,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=(0.06 / float(max(1, len(row_specs)))) if len(row_specs) > 1 else 0.0,
        subplot_titles=None,
    )

    legend_seen: set[str] = set()
    for row_idx, spec in enumerate(row_specs, start=1):
        band_stats = spec["band_stats"]
        summary_lines: list[str] = []
        for bidx, band_label in enumerate(band_order):
            st = band_stats.get(band_label)
            if not isinstance(st, dict):
                continue
            color = PLOTLY_COLORS[bidx % len(PLOTLY_COLORS)]
            for col_idx, metric_key, metric_label in (
                (1, "plv", "PLV"),
                (2, "coh", "Coherence"),
            ):
                vals = np.asarray(st.get(metric_key, np.array([], dtype=float)), dtype=float).reshape(-1)
                vals = vals[np.isfinite(vals)]
                if vals.size >= 2:
                    legend_key = "syncfc_real"
                    show_legend = legend_key not in legend_seen
                    legend_seen.add(legend_key)
                    fig.add_trace(
                        go.Box(
                            x=[band_label] * int(vals.size),
                            y=vals,
                            name="real",
                            legendgroup="real",
                            showlegend=show_legend,
                            marker={"color": color, "size": 2.4, "opacity": 0.22},
                            line={"color": color, "width": 1.9},
                            fillcolor=color,
                            opacity=0.74,
                            boxmean=True,
                            boxpoints="suspectedoutliers",
                            jitter=0.16,
                            pointpos=0.0,
                            width=0.58,
                            hovertemplate=(
                                f"{band_label}<br>{metric_label} real=%{{y:.4g}}"
                                "<extra></extra>"
                            ),
                        ),
                        row=row_idx,
                        col=col_idx,
                    )

                null_vals = np.asarray(
                    st.get(f"null_{metric_key}", np.array([], dtype=float)),
                    dtype=float,
                ).reshape(-1)
                null_vals = null_vals[np.isfinite(null_vals)]
                if null_vals.size >= 2:
                    legend_key = "syncfc_null"
                    show_legend = legend_key not in legend_seen
                    legend_seen.add(legend_key)
                    fig.add_trace(
                        go.Box(
                            x=[band_label] * int(null_vals.size),
                            y=null_vals,
                            name="permutation",
                            legendgroup="permutation",
                            showlegend=show_legend,
                            marker={"color": "rgba(90,90,90,0.35)", "size": 2.0, "opacity": 0.12},
                            line={"color": "rgba(90,90,90,0.95)", "width": 1.6},
                            fillcolor="rgba(150,150,150,0.45)",
                            opacity=0.62,
                            boxmean=True,
                            boxpoints=False,
                            width=0.58,
                            hovertemplate=(
                                f"{band_label}<br>{metric_label} permutation=%{{y:.4g}}"
                                "<extra></extra>"
                            ),
                        ),
                        row=row_idx,
                        col=col_idx,
                    )

            try:
                null_label = "perm" if str(st.get("null_mode", "")) == "window_shuffle" else "null"
                summary_lines.append(
                    f"{band_label}: "
                    f"PLV={float(st.get('median_plv', np.nan)):.3g}/{null_label} "
                    f"{float(st.get('surrogate_mean_plv', np.nan)):.3g}, "
                    f"p={float(st.get('p_plv', np.nan)):.3g}; "
                    f"Coh={float(st.get('median_coh', np.nan)):.3g}/{null_label} "
                    f"{float(st.get('surrogate_mean_coh', np.nan)):.3g}, "
                    f"p={float(st.get('p_coh', np.nan)):.3g}; "
                    f"n={int(st.get('n_windows', 0))}, null={int(st.get('n_surrogates', 0))}"
                )
            except Exception:
                pass

        for col_idx, title in ((1, "Window PLV real/permutation distribution"), (2, "Window coherence real/permutation distribution")):
            fig.update_xaxes(
                title_text=("Band" if row_idx == len(row_specs) else None),
                showgrid=True,
                zeroline=False,
                categoryorder="array",
                categoryarray=band_order,
                row=row_idx,
                col=col_idx,
            )
            fig.update_yaxes(
                range=[-0.02, 1.02],
                title_text=(title if row_idx == 1 else None),
                showgrid=True,
                zeroline=False,
                row=row_idx,
                col=col_idx,
            )

        annotation_text = f"<b>{subject} | {spec['session']}</b>"
        if summary_lines:
            annotation_text += "<br>" + "<br>".join(summary_lines[:8])
        axis_num = (row_idx - 1) * 2 + 1
        axis_suffix = "" if axis_num == 1 else str(axis_num)
        fig.add_annotation(
            x=0.01,
            y=0.98,
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            text=annotation_text,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font={"size": 10, "color": "black"},
            bgcolor="rgba(255,255,255,0.72)",
            borderpad=2,
        )

    fig.update_layout(
        template="plotly_white",
        height=int((330 * len(row_specs)) + 90),
        margin={"l": 86, "r": 24, "t": 58, "b": 48},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.0},
        title={"text": f"{subject} | SyncFC window statistics", "x": 0.01},
        dragmode="pan",
        hovermode="closest",
        boxmode="group",
        boxgap=0.18,
        boxgroupgap=0.08,
    )
    return fig


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _wilcoxon_greater_p(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2 or np.allclose(vals, 0.0):
        return float("nan")
    try:
        return float(wilcoxon(vals, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def _wilcoxon_two_sided_p(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2 or np.allclose(vals, 0.0):
        return float("nan")
    try:
        return float(wilcoxon(vals, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def _bh_fdr(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float).reshape(-1)
    q = np.full(p.size, np.nan, dtype=float)
    finite_idx = np.where(np.isfinite(p))[0]
    if finite_idx.size == 0:
        return [float(x) for x in q]
    p_f = p[finite_idx]
    order = np.argsort(p_f)
    ranked = p_f[order]
    m = float(ranked.size)
    q_ranked = ranked * m / np.arange(1, ranked.size + 1, dtype=float)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q_f = np.empty_like(q_ranked)
    q_f[order] = q_ranked
    q[finite_idx] = q_f
    return [float(x) for x in q]


def _interp_to_ref(freq: np.ndarray, y: np.ndarray, freq_ref: np.ndarray) -> np.ndarray:
    f = np.asarray(freq, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    fr = np.asarray(freq_ref, dtype=float).reshape(-1)
    n = min(f.size, yy.size)
    if n < 2 or fr.size == 0:
        return np.full(fr.size, np.nan, dtype=float)
    f = f[:n]
    yy = yy[:n]
    m = np.isfinite(f) & np.isfinite(yy)
    if int(np.sum(m)) < 2:
        return np.full(fr.size, np.nan, dtype=float)
    f = f[m]
    yy = yy[m]
    order = np.argsort(f)
    f = f[order]
    yy = yy[order]
    f_u, idx_u = np.unique(f, return_index=True)
    yy_u = yy[idx_u]
    if f_u.size < 2:
        return np.full(fr.size, np.nan, dtype=float)
    return np.interp(fr, f_u, yy_u, left=np.nan, right=np.nan)


def _finite_metric_values(rows: list[dict[str, Any]], band: str, key: str) -> np.ndarray:
    vals = np.asarray(
        [
            float(r.get(key, np.nan))
            for r in rows
            if r.get("band") == band and np.isfinite(float(r.get(key, np.nan)))
        ],
        dtype=float,
    )
    return vals[np.isfinite(vals)]


def _format_stats_value(value: Any, fmt: str = ".3g") -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return "NA"
    return format(v, fmt)


def _plotly_table_trace(headers: list[str], rows: list[list[str]]) -> Any:
    columns = [[row[i] if i < len(row) else "" for row in rows] for i in range(len(headers))]
    return go.Table(
        header={
            "values": [f"<b>{h}</b>" for h in headers],
            "fill_color": "#e8f2ff",
            "align": "center",
            "font": {"size": 12, "color": "#1f2937"},
        },
        cells={
            "values": columns,
            "fill_color": "#ffffff",
            "align": "center",
            "font": {"size": 11, "color": "#1f2937"},
            "height": 24,
        },
    )


def _nan_mean_sem(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(stack, dtype=float)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    finite = np.isfinite(arr)
    n_eff = np.sum(finite, axis=0)
    sums = np.nansum(np.where(finite, arr, 0.0), axis=0)
    mean = np.full(arr.shape[1], np.nan, dtype=float)
    np.divide(sums, n_eff, out=mean, where=n_eff > 0)
    sem = np.full(arr.shape[1], np.nan, dtype=float)
    ok = n_eff > 1
    if np.any(ok):
        sem[ok] = np.nanstd(arr[:, ok], axis=0, ddof=1) / np.sqrt(n_eff[ok])
    return mean, sem


def _add_mean_sem_traces(
    fig: Any,
    x: np.ndarray,
    stack: np.ndarray,
    *,
    row: int,
    col: int,
    color: str,
    name: str,
    show_subjects: bool = True,
    showlegend: bool = True,
    subject_alpha: float = 0.28,
    subject_width: float = 0.85,
) -> None:
    xx = np.asarray(x, dtype=float).reshape(-1)
    arr = np.asarray(stack, dtype=float)
    if arr.ndim != 2 or xx.size == 0:
        return
    if show_subjects:
        for curve in arr:
            fig.add_trace(
                go.Scatter(
                    x=xx,
                    y=curve,
                    mode="lines",
                    line={"color": _hex_to_rgba(color, float(subject_alpha)), "width": float(subject_width)},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
    mean, sem = _nan_mean_sem(arr)
    if mean.size != xx.size:
        return
    lo = mean - sem
    hi = mean + sem
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([xx, xx[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            mode="lines",
            line={"width": 0},
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.16),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=xx,
            y=mean,
            mode="lines",
            line={"color": color, "width": 2.3},
            name=name,
            showlegend=showlegend,
            hovertemplate="%{x:.4g}, %{y:.4g}<extra>" + name + "</extra>",
        ),
        row=row,
        col=col,
    )


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = str(hex_color).lstrip("#")
    if len(color) != 6:
        return f"rgba(0,0,0,{float(alpha):.3g})"
    try:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    except ValueError:
        return f"rgba(0,0,0,{float(alpha):.3g})"
    return f"rgba({r},{g},{b},{float(alpha):.3g})"


def _collect_granger_group_stats(
    mode_cells: list[ModeCell],
    bands: list[tuple[str, tuple[float, float]]],
) -> dict[str, Any] | None:
    records: list[dict[str, Any]] = []
    curve_by_subject: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    freq_ref: np.ndarray | None = None

    for cell in mode_cells:
        payload = cell.payload if isinstance(cell.payload, dict) else None
        if payload is None:
            continue
        freq = np.asarray(payload.get("freq_hz", np.array([], dtype=float)), dtype=float).reshape(-1)
        gpdc = np.asarray(payload.get("gpdc_mean", np.array([], dtype=float)), dtype=float)
        if freq.size < 2 or gpdc.shape != (2, 2, freq.size):
            continue
        ms_to_hc = np.asarray(gpdc[0, 1, :], dtype=float).reshape(-1)
        hc_to_ms = np.asarray(gpdc[1, 0, :], dtype=float).reshape(-1)
        if freq_ref is None:
            freq_ref = freq.copy()
        ms_curve = _interp_to_ref(freq, ms_to_hc, freq_ref)
        hc_curve = _interp_to_ref(freq, hc_to_ms, freq_ref)
        curve_by_subject.setdefault(cell.subject, []).append((ms_curve, hc_curve))

        for band_label, (fmin, fmax) in bands:
            m_band = (freq >= float(fmin)) & (freq <= float(fmax))
            m_band &= np.isfinite(ms_to_hc) & np.isfinite(hc_to_ms)
            if int(np.sum(m_band)) < 1:
                continue
            f_b = freq[m_band]
            ms_b = ms_to_hc[m_band]
            hc_b = hc_to_ms[m_band]
            ms_mean = float(np.nanmean(ms_b))
            hc_mean = float(np.nanmean(hc_b))
            denom = ms_mean + hc_mean + 1e-12
            di = float((ms_mean - hc_mean) / denom)
            if f_b.size >= 2:
                ms_auc = float(np.trapz(ms_b, f_b))
                hc_auc = float(np.trapz(hc_b, f_b))
            else:
                width = float(fmax) - float(fmin)
                ms_auc = float(ms_mean * width)
                hc_auc = float(hc_mean * width)
            peak_idx = int(np.nanargmax(ms_b))
            records.append(
                {
                    "subject": cell.subject,
                    "session": cell.session,
                    "band": band_label,
                    "ms_to_hc_mean": ms_mean,
                    "hc_to_ms_mean": hc_mean,
                    "ms_to_hc_auc": ms_auc,
                    "hc_to_ms_auc": hc_auc,
                    "di": di,
                    "ms_to_hc_peak_freq": float(f_b[peak_idx]),
                    "ms_to_hc_peak_value": float(ms_b[peak_idx]),
                }
            )

    if not records or freq_ref is None:
        return None

    subjects = sorted({r["subject"] for r in records}, key=natural_key)
    subject_band_rows: list[dict[str, Any]] = []
    for subject in subjects:
        for band_label, _ in bands:
            rr = [r for r in records if r["subject"] == subject and r["band"] == band_label]
            if not rr:
                continue
            sessions = sorted({str(r["session"]) for r in rr}, key=natural_key)
            subject_band_rows.append(
                {
                    "subject": subject,
                    "band": band_label,
                    "n_sessions": len(sessions),
                    "ms_to_hc_mean": _finite_mean([float(r["ms_to_hc_mean"]) for r in rr]),
                    "hc_to_ms_mean": _finite_mean([float(r["hc_to_ms_mean"]) for r in rr]),
                    "ms_to_hc_auc": _finite_mean([float(r["ms_to_hc_auc"]) for r in rr]),
                    "hc_to_ms_auc": _finite_mean([float(r["hc_to_ms_auc"]) for r in rr]),
                    "di": _finite_mean([float(r["di"]) for r in rr]),
                    "ms_to_hc_peak_freq": _finite_mean([float(r["ms_to_hc_peak_freq"]) for r in rr]),
                    "ms_to_hc_peak_value": _finite_mean([float(r["ms_to_hc_peak_value"]) for r in rr]),
                }
            )

    subject_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for subject, curves in curve_by_subject.items():
        if not curves:
            continue
        ms_stack = np.stack([c[0] for c in curves], axis=0)
        hc_stack = np.stack([c[1] for c in curves], axis=0)
        subject_curves[subject] = (np.nanmean(ms_stack, axis=0), np.nanmean(hc_stack, axis=0))

    summary_rows: list[dict[str, Any]] = []
    for band_label, _ in bands:
        subj_rows = [r for r in subject_band_rows if r["band"] == band_label]
        sess_rows = [r for r in records if r["band"] == band_label]
        if not subj_rows:
            continue
        di_vals = np.asarray([float(r["di"]) for r in subj_rows], dtype=float)
        di_vals = di_vals[np.isfinite(di_vals)]
        peak_vals = np.asarray([float(r["ms_to_hc_peak_freq"]) for r in subj_rows], dtype=float)
        peak_vals = peak_vals[np.isfinite(peak_vals)]
        if di_vals.size == 0:
            continue
        q1, med, q3 = np.percentile(di_vals, [25, 50, 75])
        p_two_sided = _wilcoxon_two_sided_p(di_vals)
        summary_rows.append(
            {
                "band": band_label,
                "n_subjects": int(di_vals.size),
                "n_sessions": int(len(sess_rows)),
                "median_di": float(med),
                "iqr_low": float(q1),
                "iqr_high": float(q3),
                "p_di_ne_0": p_two_sided,
                "n_di_positive": int(np.sum(di_vals > 0)),
                "median_peak_freq": float(np.nanmedian(peak_vals)) if peak_vals.size else float("nan"),
            }
        )

    return {
        "records": records,
        "subject_band_rows": subject_band_rows,
        "subject_curves": subject_curves,
        "freq_ref": freq_ref,
        "summary_rows": summary_rows,
    }


def _write_granger_stats_pdf(
    mode_cells: list[ModeCell],
    bands: list[tuple[str, tuple[float, float]]],
    out_path: Path,
    run_name: str,
) -> bool:
    stats = _collect_granger_group_stats(mode_cells, bands)
    if stats is None:
        print("\033[1;33m -- Granger stats PDF skipped: no plottable granger data.\033[0m")
        return False

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_path.parent.mkdir(parents=True, exist_ok=True)
    freq_ref = np.asarray(stats["freq_ref"], dtype=float)
    subject_curves = stats["subject_curves"]
    subject_band_rows = stats["subject_band_rows"]
    summary_rows = stats["summary_rows"]

    with PdfPages(out_path) as pdf:
        if subject_curves:
            subjects = sorted(subject_curves.keys(), key=natural_key)
            ms_stack = np.stack([subject_curves[s][0] for s in subjects], axis=0)
            hc_stack = np.stack([subject_curves[s][1] for s in subjects], axis=0)
            fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
            for ax, stack, title, color in (
                (axes[0], ms_stack, "MS -> HC", "#d62728"),
                (axes[1], hc_stack, "HC -> MS", "#2ca02c"),
            ):
                for row in stack:
                    ax.plot(freq_ref, row, color="0.75", linewidth=0.7, alpha=0.45)
                mean = np.nanmean(stack, axis=0)
                n_eff = np.sum(np.isfinite(stack), axis=0)
                sem = np.full(mean.size, np.nan, dtype=float)
                ok = n_eff > 1
                sem[ok] = np.nanstd(stack[:, ok], axis=0, ddof=1) / np.sqrt(n_eff[ok])
                ax.plot(freq_ref, mean, color=color, linewidth=2.0)
                ax.fill_between(freq_ref, mean - sem, mean + sem, color=color, alpha=0.18, linewidth=0)
                ax.set_ylabel("gPDC")
                ax.set_title(title)
                ax.set_ylim(0, max(1.0, float(np.nanmax(mean + np.nan_to_num(sem, nan=0.0))) * 1.1))
                ax.grid(True, which="major", alpha=0.35)
                ax.grid(True, which="minor", alpha=0.18)
                ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
            axes[-1].set_xlabel("Frequency (Hz)")
            fig.suptitle(f"{run_name} | Granger gPDC subject-averaged curves", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

        band_labels = [row["band"] for row in summary_rows]
        if band_labels:
            di_data = [
                np.asarray(
                    [
                        float(r["di"])
                        for r in subject_band_rows
                        if r["band"] == band and np.isfinite(float(r["di"]))
                    ],
                    dtype=float,
                )
                for band in band_labels
            ]
            fig, ax = plt.subplots(figsize=(11, 6.5))
            ax.axhline(0, color="0.25", linewidth=1.0)
            _boxplot_with_labels(ax, di_data, band_labels, showfliers=False)
            rng = np.random.default_rng(0)
            for i, vals in enumerate(di_data, start=1):
                if vals.size == 0:
                    continue
                x = i + rng.uniform(-0.12, 0.12, size=vals.size)
                ax.scatter(x, vals, s=28, color="#d62728", alpha=0.75, edgecolor="white", linewidth=0.4)
            p_txt = []
            for row in summary_rows:
                p = float(row["p_di_ne_0"])
                if np.isfinite(p):
                    p_txt.append(f"{row['band']}: p={p:.3g}")
            ax.set_title("Band directionality index across subjects\nDI=(MS->HC - HC->MS)/(MS->HC + HC->MS)")
            ax.set_ylabel("Subject-mean DI")
            ax.grid(True, axis="y", alpha=0.3)
            ax.text(
                0.01,
                0.99,
                "Wilcoxon signed-rank, two-sided DI != 0\n" + "\n".join(p_txt),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.85"},
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            peak_data = [
                np.asarray(
                    [
                        float(r["ms_to_hc_peak_freq"])
                        for r in subject_band_rows
                        if r["band"] == band and np.isfinite(float(r["ms_to_hc_peak_freq"]))
                    ],
                    dtype=float,
                )
                for band in band_labels
            ]
            fig, ax = plt.subplots(figsize=(11, 6.5))
            _boxplot_with_labels(ax, peak_data, band_labels, showfliers=False)
            for i, vals in enumerate(peak_data, start=1):
                if vals.size == 0:
                    continue
                x = i + rng.uniform(-0.12, 0.12, size=vals.size)
                ax.scatter(x, vals, s=28, color="#1f77b4", alpha=0.75, edgecolor="white", linewidth=0.4)
            ax.set_title("MS -> HC peak frequency by band, subject-averaged across sessions")
            ax.set_ylabel("Peak frequency (Hz)")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            table_rows = []
            for row in summary_rows:
                p = float(row["p_di_ne_0"])
                table_rows.append(
                    [
                        row["band"],
                        str(int(row["n_subjects"])),
                        str(int(row["n_sessions"])),
                        f"{float(row['median_di']):.3f}",
                        f"{float(row['iqr_low']):.3f}..{float(row['iqr_high']):.3f}",
                        f"{p:.3g}" if np.isfinite(p) else "NA",
                        f"{int(row['n_di_positive'])}/{int(row['n_subjects'])}",
                        f"{float(row['median_peak_freq']):.1f}" if np.isfinite(float(row["median_peak_freq"])) else "NA",
                    ]
                )
            fig, ax = plt.subplots(figsize=(11, 6.5))
            ax.axis("off")
            ax.set_title(
                f"{run_name} | Granger band statistics\n"
                "Session metrics are averaged within subject before inference.",
                fontsize=12,
                pad=16,
            )
            table = ax.table(
                cellText=table_rows,
                colLabels=[
                    "Band",
                    "N subj",
                    "N sess",
                    "Median DI",
                    "IQR DI",
                    "p DI!=0",
                    "DI>0",
                    "Median peak Hz",
                ],
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.45)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"-- saved Granger stats PDF: {out_path}")
    return True


def _collect_syncfc_group_stats(mode_cells: list[ModeCell]) -> dict[str, Any] | None:
    records: list[dict[str, Any]] = []
    for cell in mode_cells:
        payload = cell.payload if isinstance(cell.payload, dict) else None
        if payload is None:
            continue
        band_stats = payload.get("band_stats")
        if not isinstance(band_stats, dict):
            continue
        for band_label, st in band_stats.items():
            if not isinstance(st, dict):
                continue
            records.append(
                {
                    "subject": cell.subject,
                    "session": cell.session,
                    "band": str(band_label),
                    "median_coh": float(st.get("median_coh", np.nan)),
                    "median_plv": float(st.get("median_plv", np.nan)),
                    "surrogate_mean_coh": float(st.get("surrogate_mean_coh", np.nan)),
                    "surrogate_mean_plv": float(st.get("surrogate_mean_plv", np.nan)),
                    "z_coh": float(st.get("z_coh", np.nan)),
                    "z_plv": float(st.get("z_plv", np.nan)),
                    "p_coh": float(st.get("p_coh", np.nan)),
                    "p_plv": float(st.get("p_plv", np.nan)),
                    "n_windows": int(st.get("n_windows", 0)),
                    "n_total_windows": int(st.get("n_total_windows", 0)),
                    "n_surrogates": int(st.get("n_surrogates", 0)),
                    "null_mode": str(st.get("null_mode", "")),
                }
            )

    if not records:
        return None

    subjects = sorted({r["subject"] for r in records}, key=natural_key)
    bands = sorted({r["band"] for r in records}, key=natural_key)
    subject_band_rows: list[dict[str, Any]] = []
    for subject in subjects:
        for band in bands:
            rr = [r for r in records if r["subject"] == subject and r["band"] == band]
            if not rr:
                continue
            median_coh = _finite_mean([float(r["median_coh"]) for r in rr])
            median_plv = _finite_mean([float(r["median_plv"]) for r in rr])
            surr_coh = _finite_mean([float(r["surrogate_mean_coh"]) for r in rr])
            surr_plv = _finite_mean([float(r["surrogate_mean_plv"]) for r in rr])
            subject_band_rows.append(
                {
                    "subject": subject,
                    "band": band,
                    "n_sessions": len({str(r["session"]) for r in rr}),
                    "median_coh": median_coh,
                    "median_plv": median_plv,
                    "surrogate_mean_coh": surr_coh,
                    "surrogate_mean_plv": surr_plv,
                    "delta_coh": median_coh - surr_coh if np.isfinite(median_coh) and np.isfinite(surr_coh) else float("nan"),
                    "delta_plv": median_plv - surr_plv if np.isfinite(median_plv) and np.isfinite(surr_plv) else float("nan"),
                    "z_coh": _finite_mean([float(r["z_coh"]) for r in rr]),
                    "z_plv": _finite_mean([float(r["z_plv"]) for r in rr]),
                    "p_coh": _finite_mean([float(r["p_coh"]) for r in rr]),
                    "p_plv": _finite_mean([float(r["p_plv"]) for r in rr]),
                    "n_windows": int(np.sum([int(r["n_windows"]) for r in rr])),
                    "n_total_windows": int(np.sum([int(r["n_total_windows"]) for r in rr])),
                    "n_surrogates": int(np.nanmax([int(r["n_surrogates"]) for r in rr])),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for band in bands:
        subj_rows = [r for r in subject_band_rows if r["band"] == band]
        sess_rows = [r for r in records if r["band"] == band]
        if not subj_rows:
            continue
        plv_vals = np.asarray([float(r["median_plv"]) for r in subj_rows], dtype=float)
        coh_vals = np.asarray([float(r["median_coh"]) for r in subj_rows], dtype=float)
        z_plv_vals = np.asarray([float(r["z_plv"]) for r in subj_rows], dtype=float)
        z_coh_vals = np.asarray([float(r["z_coh"]) for r in subj_rows], dtype=float)
        d_plv_vals = np.asarray([float(r["delta_plv"]) for r in subj_rows], dtype=float)
        d_coh_vals = np.asarray([float(r["delta_coh"]) for r in subj_rows], dtype=float)
        plv_vals = plv_vals[np.isfinite(plv_vals)]
        coh_vals = coh_vals[np.isfinite(coh_vals)]
        z_plv_vals = z_plv_vals[np.isfinite(z_plv_vals)]
        z_coh_vals = z_coh_vals[np.isfinite(z_coh_vals)]
        d_plv_vals = d_plv_vals[np.isfinite(d_plv_vals)]
        d_coh_vals = d_coh_vals[np.isfinite(d_coh_vals)]
        if plv_vals.size == 0 and coh_vals.size == 0:
            continue
        summary_rows.append(
            {
                "band": band,
                "n_subjects": int(len({r["subject"] for r in subj_rows})),
                "n_sessions": int(len(sess_rows)),
                "median_plv": float(np.nanmedian(plv_vals)) if plv_vals.size else float("nan"),
                "iqr_plv_low": float(np.nanpercentile(plv_vals, 25)) if plv_vals.size else float("nan"),
                "iqr_plv_high": float(np.nanpercentile(plv_vals, 75)) if plv_vals.size else float("nan"),
                "median_coh": float(np.nanmedian(coh_vals)) if coh_vals.size else float("nan"),
                "iqr_coh_low": float(np.nanpercentile(coh_vals, 25)) if coh_vals.size else float("nan"),
                "iqr_coh_high": float(np.nanpercentile(coh_vals, 75)) if coh_vals.size else float("nan"),
                "median_z_plv": float(np.nanmedian(z_plv_vals)) if z_plv_vals.size else float("nan"),
                "median_z_coh": float(np.nanmedian(z_coh_vals)) if z_coh_vals.size else float("nan"),
                "median_delta_plv": float(np.nanmedian(d_plv_vals)) if d_plv_vals.size else float("nan"),
                "median_delta_coh": float(np.nanmedian(d_coh_vals)) if d_coh_vals.size else float("nan"),
                "p_z_plv_gt_0": _wilcoxon_greater_p(z_plv_vals),
                "p_z_coh_gt_0": _wilcoxon_greater_p(z_coh_vals),
                "p_delta_plv_gt_0": _wilcoxon_greater_p(d_plv_vals),
                "p_delta_coh_gt_0": _wilcoxon_greater_p(d_coh_vals),
                "n_z_plv_positive": int(np.sum(z_plv_vals > 0)),
                "n_z_coh_positive": int(np.sum(z_coh_vals > 0)),
            }
        )

    q_plv = _bh_fdr([float(r["p_z_plv_gt_0"]) for r in summary_rows])
    q_coh = _bh_fdr([float(r["p_z_coh_gt_0"]) for r in summary_rows])
    q_delta_plv = _bh_fdr([float(r["p_delta_plv_gt_0"]) for r in summary_rows])
    q_delta_coh = _bh_fdr([float(r["p_delta_coh_gt_0"]) for r in summary_rows])
    for row, qp, qc, qdp, qdc in zip(summary_rows, q_plv, q_coh, q_delta_plv, q_delta_coh):
        row["q_z_plv_gt_0"] = float(qp)
        row["q_z_coh_gt_0"] = float(qc)
        row["q_delta_plv_gt_0"] = float(qdp)
        row["q_delta_coh_gt_0"] = float(qdc)

    return {
        "records": records,
        "subject_band_rows": subject_band_rows,
        "summary_rows": summary_rows,
        "bands": bands,
    }


def _boxplot_with_labels(ax: Any, data: list[np.ndarray], labels: list[str], **kwargs: Any) -> Any:
    try:
        return ax.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError as exc:
        if "tick_labels" not in str(exc):
            raise
        return ax.boxplot(data, labels=labels, **kwargs)


def _write_syncfc_stats_pdf(
    mode_cells: list[ModeCell],
    out_path: Path,
    run_name: str,
) -> bool:
    stats = _collect_syncfc_group_stats(mode_cells)
    if stats is None:
        print("\033[1;33m -- SyncFC stats PDF skipped: no plottable syncfc data.\033[0m")
        return False

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = stats["subject_band_rows"]
    summary_rows = stats["summary_rows"]
    bands = [row["band"] for row in summary_rows]
    rng = np.random.default_rng(0)

    with PdfPages(out_path) as pdf:
        if bands:
            fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
            band_x = np.arange(1, len(bands) + 1, dtype=float)
            for ax, real_metric, null_metric, color, title, ylabel in (
                (axes[0], "median_plv", "surrogate_mean_plv", "#1f77b4", "Band-limited PLV", "PLV"),
                (axes[1], "median_coh", "surrogate_mean_coh", "#d62728", "Band-limited coherence", "Coherence"),
            ):
                data: list[np.ndarray] = []
                positions: list[float] = []
                box_colors: list[str] = []
                scatter_specs: list[tuple[float, np.ndarray, str]] = []
                for i, band in enumerate(bands, start=1):
                    real = np.asarray(
                        [
                            float(r[real_metric])
                            for r in rows
                            if r["band"] == band and np.isfinite(float(r[real_metric]))
                        ],
                        dtype=float,
                    )
                    null = np.asarray(
                        [
                            float(r[null_metric])
                            for r in rows
                            if r["band"] == band and np.isfinite(float(r[null_metric]))
                        ],
                        dtype=float,
                    )
                    for pos, vals, c in ((i - 0.18, real, color), (i + 0.18, null, "0.55")):
                        if vals.size == 0:
                            continue
                        data.append(vals)
                        positions.append(float(pos))
                        box_colors.append(c)
                        scatter_specs.append((float(pos), vals, c))
                if data:
                    bp = ax.boxplot(
                        data,
                        positions=positions,
                        widths=0.28,
                        patch_artist=True,
                        showfliers=False,
                    )
                    for patch, c in zip(bp["boxes"], box_colors):
                        patch.set_facecolor(c)
                        patch.set_alpha(0.42 if c == "0.55" else 0.55)
                        patch.set_linewidth(1.4)
                    for key in ("whiskers", "caps", "medians"):
                        for artist in bp[key]:
                            artist.set_linewidth(1.2)
                for pos, vals, c in scatter_specs:
                    ax.scatter(
                        pos + rng.uniform(-0.045, 0.045, size=vals.size),
                        vals,
                        s=26,
                        color=c,
                        alpha=0.72 if c != "0.55" else 0.48,
                        edgecolor="white",
                        linewidth=0.4,
                    )
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.set_ylim(-0.02, 1.02)
                ax.set_xlim(0.4, len(bands) + 0.6)
                ax.set_xticks(band_x)
                ax.set_xticklabels(bands, rotation=0)
                ax.grid(True, axis="y", alpha=0.3)
                ax.scatter([], [], s=32, color=color, alpha=0.72, label="real")
                ax.scatter([], [], s=32, color="0.55", alpha=0.48, label="null")
                ax.legend(loc="upper right")
            fig.suptitle(f"{run_name} | SyncFC subject-level real-null synchrony", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

            has_z = any(np.isfinite(float(r.get("z_plv", np.nan))) or np.isfinite(float(r.get("z_coh", np.nan))) for r in rows)
            if has_z:
                fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
                for ax, metric, color, title, p_key, q_key in (
                    (axes[0], "z_plv", "#1f77b4", "Null-normalized PLV", "p_z_plv_gt_0", "q_z_plv_gt_0"),
                    (axes[1], "z_coh", "#d62728", "Null-normalized coherence", "p_z_coh_gt_0", "q_z_coh_gt_0"),
                ):
                    data = [
                        np.asarray(
                            [
                                float(r[metric])
                                for r in rows
                                if r["band"] == band and np.isfinite(float(r[metric]))
                            ],
                            dtype=float,
                        )
                        for band in bands
                    ]
                    ax.axhline(0, color="0.25", linewidth=1.0)
                    _boxplot_with_labels(ax, data, bands, showfliers=False)
                    for i, vals in enumerate(data, start=1):
                        if vals.size == 0:
                            continue
                        ax.scatter(
                            i + rng.uniform(-0.12, 0.12, size=vals.size),
                            vals,
                            s=26,
                            color=color,
                            alpha=0.75,
                            edgecolor="white",
                            linewidth=0.4,
                        )
                    p_txt = []
                    for row in summary_rows:
                        p = float(row.get(p_key, np.nan))
                        q = float(row.get(q_key, np.nan))
                        if np.isfinite(p):
                            q_part = f", q={q:.3g}" if np.isfinite(q) else ""
                            p_txt.append(f"{row['band']}: p={p:.3g}{q_part}")
                    ax.set_title(title)
                    ax.set_ylabel("Zsync")
                    ax.grid(True, axis="y", alpha=0.3)
                    ax.text(
                        0.01,
                        0.99,
                        "Wilcoxon signed-rank, one-sided Zsync > 0\n" + "\n".join(p_txt),
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.85"},
                    )
                fig.suptitle(f"{run_name} | SyncFC null-normalized statistics", fontsize=13)
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig)
                plt.close(fig)

                fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
                x = np.arange(len(bands), dtype=float)
                width = 0.36
                for ax, real_key, surr_key, color, title in (
                    (axes[0], "median_plv", "surrogate_mean_plv", "#1f77b4", "PLV real vs null"),
                    (axes[1], "median_coh", "surrogate_mean_coh", "#d62728", "Coherence real vs null"),
                ):
                    real_means = []
                    surr_means = []
                    real_sem = []
                    surr_sem = []
                    for band in bands:
                        real = np.asarray(
                            [
                                float(r[real_key])
                                for r in rows
                                if r["band"] == band and np.isfinite(float(r[real_key]))
                            ],
                            dtype=float,
                        )
                        surr = np.asarray(
                            [
                                float(r[surr_key])
                                for r in rows
                                if r["band"] == band and np.isfinite(float(r[surr_key]))
                            ],
                            dtype=float,
                        )
                        real_means.append(float(np.nanmean(real)) if real.size else np.nan)
                        surr_means.append(float(np.nanmean(surr)) if surr.size else np.nan)
                        real_sem.append(float(np.nanstd(real, ddof=1) / np.sqrt(real.size)) if real.size > 1 else 0.0)
                        surr_sem.append(float(np.nanstd(surr, ddof=1) / np.sqrt(surr.size)) if surr.size > 1 else 0.0)
                    ax.bar(x - width / 2, real_means, width, yerr=real_sem, label="real", color=color, alpha=0.78)
                    ax.bar(x + width / 2, surr_means, width, yerr=surr_sem, label="null", color="0.6", alpha=0.65)
                    ax.set_title(title)
                    ax.set_ylabel(real_key.replace("median_", "").upper())
                    ax.set_ylim(-0.02, 1.02)
                    ax.grid(True, axis="y", alpha=0.3)
                    ax.legend(loc="upper right")
                axes[-1].set_xticks(x)
                axes[-1].set_xticklabels(bands, rotation=0)
                fig.suptitle(f"{run_name} | SyncFC real-null comparison", fontsize=13)
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig)
                plt.close(fig)

            table_rows = []
            for row in summary_rows:
                p_plv = float(row.get("p_z_plv_gt_0", np.nan))
                q_plv = float(row.get("q_z_plv_gt_0", np.nan))
                p_coh = float(row.get("p_z_coh_gt_0", np.nan))
                q_coh = float(row.get("q_z_coh_gt_0", np.nan))
                table_rows.append(
                    [
                        row["band"],
                        str(int(row["n_subjects"])),
                        str(int(row["n_sessions"])),
                        f"{float(row['median_plv']):.3f}",
                        f"{float(row['median_z_plv']):.3g}" if np.isfinite(float(row["median_z_plv"])) else "NA",
                        f"{p_plv:.3g}" if np.isfinite(p_plv) else "NA",
                        f"{q_plv:.3g}" if np.isfinite(q_plv) else "NA",
                        f"{float(row['median_coh']):.3f}",
                        f"{float(row['median_z_coh']):.3g}" if np.isfinite(float(row["median_z_coh"])) else "NA",
                        f"{p_coh:.3g}" if np.isfinite(p_coh) else "NA",
                        f"{q_coh:.3g}" if np.isfinite(q_coh) else "NA",
                    ]
                )
            fig, ax = plt.subplots(figsize=(11, 6.5))
            ax.axis("off")
            ax.set_title(
                f"{run_name} | SyncFC band statistics\n"
                "Session metrics are averaged within subject before inference.",
                fontsize=12,
                pad=16,
            )
            table = ax.table(
                cellText=table_rows,
                colLabels=[
                    "Band",
                    "N subj",
                    "N sess",
                    "Median PLV",
                    "Median Z PLV",
                    "p ZPLV>0",
                    "q ZPLV",
                    "Median Coh",
                    "Median Z Coh",
                    "p ZCoh>0",
                    "q ZCoh",
                ],
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1.0, 1.42)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"-- saved SyncFC stats PDF: {out_path}")
    return True


def _collect_phaselag_group_stats(mode_cells: list[ModeCell]) -> dict[str, Any] | None:
    records: list[dict[str, Any]] = []
    curve_by_subject_band: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    lag_ref_by_band: dict[str, np.ndarray] = {}

    for cell in mode_cells:
        payload = cell.payload if isinstance(cell.payload, dict) else None
        if payload is None:
            continue
        band_stats = payload.get("band_stats")
        if not isinstance(band_stats, dict):
            continue
        for band_label, st in band_stats.items():
            if not isinstance(st, dict):
                continue
            band = str(band_label)
            lags = np.asarray(st.get("lags_ms", np.array([], dtype=float)), dtype=float).reshape(-1)
            mean_norm_z = np.asarray(st.get("mean_norm_z", np.array([], dtype=float)), dtype=float).reshape(-1)
            n = min(lags.size, mean_norm_z.size)
            if n < 2:
                continue
            lags = lags[:n]
            mean_norm_z = mean_norm_z[:n]
            if band not in lag_ref_by_band:
                lag_ref_by_band[band] = lags.copy()
            curve_by_subject_band.setdefault((cell.subject, band), []).append(
                (lag_ref_by_band[band], _interp_to_ref(lags, mean_norm_z, lag_ref_by_band[band]))
            )
            records.append(
                {
                    "subject": cell.subject,
                    "session": cell.session,
                    "band": band,
                    "peak_lag_ms": float(st.get("peak_lag_ms", np.nan)),
                    "mode_lag_ms": float(st.get("mode_lag_ms", np.nan)),
                    "median_lag_ms": float(st.get("median_lag_ms", np.nan)),
                    "mean_lag_ms": float(st.get("mean_lag_ms", np.nan)),
                    "ms_lead_ratio": float(st.get("ms_lead_ratio", np.nan)),
                    "hc_lead_ratio": float(st.get("hc_lead_ratio", np.nan)),
                    "valid_ratio": float(st.get("valid_ratio", np.nan)),
                    "median_z": float(st.get("median_z", np.nan)),
                    "median_plv": float(st.get("median_plv", np.nan)),
                    "n_edge_hit_windows": int(st.get("n_edge_hit_windows", 0)),
                    "n_valid_windows": int(st.get("n_valid_windows", 0)),
                    "n_total_windows": int(st.get("n_total_windows", 0)),
                }
            )

    if not records:
        return None

    subjects = sorted({r["subject"] for r in records}, key=natural_key)
    bands = sorted({r["band"] for r in records}, key=natural_key)
    subject_band_rows: list[dict[str, Any]] = []
    for subject in subjects:
        for band in bands:
            rr = [r for r in records if r["subject"] == subject and r["band"] == band]
            if not rr:
                continue
            subject_band_rows.append(
                {
                    "subject": subject,
                    "band": band,
                    "n_sessions": len({str(r["session"]) for r in rr}),
                    "peak_lag_ms": _finite_mean([float(r["peak_lag_ms"]) for r in rr]),
                    "mode_lag_ms": _finite_mean([float(r["mode_lag_ms"]) for r in rr]),
                    "median_lag_ms": _finite_mean([float(r["median_lag_ms"]) for r in rr]),
                    "mean_lag_ms": _finite_mean([float(r["mean_lag_ms"]) for r in rr]),
                    "ms_lead_ratio": _finite_mean([float(r["ms_lead_ratio"]) for r in rr]),
                    "valid_ratio": _finite_mean([float(r["valid_ratio"]) for r in rr]),
                    "median_z": _finite_mean([float(r["median_z"]) for r in rr]),
                    "median_plv": _finite_mean([float(r["median_plv"]) for r in rr]),
                    "n_edge_hit_windows": int(np.sum([int(r["n_edge_hit_windows"]) for r in rr])),
                    "n_valid_windows": int(np.sum([int(r["n_valid_windows"]) for r in rr])),
                    "n_total_windows": int(np.sum([int(r["n_total_windows"]) for r in rr])),
                }
            )

    subject_curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key, curves in curve_by_subject_band.items():
        if not curves:
            continue
        lag_ref = np.asarray(curves[0][0], dtype=float)
        stack = np.stack([np.asarray(c[1], dtype=float) for c in curves], axis=0)
        subject_curves[key] = (lag_ref, np.nanmean(stack, axis=0))

    summary_rows: list[dict[str, Any]] = []
    for band in bands:
        subj_rows = [r for r in subject_band_rows if r["band"] == band]
        sess_rows = [r for r in records if r["band"] == band]
        if not subj_rows:
            continue
        peak_vals = np.asarray([float(r["peak_lag_ms"]) for r in subj_rows], dtype=float)
        peak_vals = peak_vals[np.isfinite(peak_vals)]
        lead_vals = np.asarray([float(r["ms_lead_ratio"]) for r in subj_rows], dtype=float)
        lead_vals = lead_vals[np.isfinite(lead_vals)]
        valid_vals = np.asarray([float(r["valid_ratio"]) for r in subj_rows], dtype=float)
        valid_vals = valid_vals[np.isfinite(valid_vals)]
        if peak_vals.size == 0:
            continue
        q1, med, q3 = np.percentile(peak_vals, [25, 50, 75])
        p_lag_gt_0 = _wilcoxon_greater_p(peak_vals)
        p_lead_gt_half = _wilcoxon_greater_p(lead_vals - 0.5) if lead_vals.size >= 2 else float("nan")
        summary_rows.append(
            {
                "band": band,
                "n_subjects": int(peak_vals.size),
                "n_sessions": int(len(sess_rows)),
                "median_peak_lag_ms": float(med),
                "iqr_peak_lag_low": float(q1),
                "iqr_peak_lag_high": float(q3),
                "p_lag_gt_0": p_lag_gt_0,
                "median_ms_lead_ratio": float(np.nanmedian(lead_vals)) if lead_vals.size else float("nan"),
                "p_lead_ratio_gt_0_5": p_lead_gt_half,
                "median_valid_ratio": float(np.nanmedian(valid_vals)) if valid_vals.size else float("nan"),
                "n_peak_lag_positive": int(np.sum(peak_vals > 0)),
            }
        )

    return {
        "records": records,
        "subject_band_rows": subject_band_rows,
        "subject_curves": subject_curves,
        "summary_rows": summary_rows,
        "bands": bands,
    }


def _write_phaselag_stats_pdf(
    mode_cells: list[ModeCell],
    out_path: Path,
    run_name: str,
) -> bool:
    stats = _collect_phaselag_group_stats(mode_cells)
    if stats is None:
        print("\033[1;33m -- PhaseLag stats PDF skipped: no plottable phaselag data.\033[0m")
        return False

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = stats["subject_band_rows"]
    summary_rows = stats["summary_rows"]
    bands = [row["band"] for row in summary_rows]
    rng = np.random.default_rng(0)

    with PdfPages(out_path) as pdf:
        if bands:
            peak_data = [
                np.asarray(
                    [
                        float(r["peak_lag_ms"])
                        for r in rows
                        if r["band"] == band and np.isfinite(float(r["peak_lag_ms"]))
                    ],
                    dtype=float,
                )
                for band in bands
            ]
            fig, ax = plt.subplots(figsize=(11, 6.5))
            ax.axhline(0, color="0.25", linewidth=1.0)
            _boxplot_with_labels(ax, peak_data, bands, showfliers=False)
            for i, vals in enumerate(peak_data, start=1):
                if vals.size == 0:
                    continue
                ax.scatter(
                    i + rng.uniform(-0.12, 0.12, size=vals.size),
                    vals,
                    s=28,
                    color="#d62728",
                    alpha=0.75,
                    edgecolor="white",
                    linewidth=0.4,
                )
            p_txt = []
            for row in summary_rows:
                p = float(row["p_lag_gt_0"])
                if np.isfinite(p):
                    p_txt.append(f"{row['band']}: p={p:.3g}")
            ax.set_title("PhaseLag representative lag across subjects\nPositive lag means MS leads HC")
            ax.set_ylabel("Peak lag from mean normalized Rayleigh Z (ms)")
            ax.grid(True, axis="y", alpha=0.3)
            ax.text(
                0.01,
                0.99,
                "Wilcoxon signed-rank, one-sided lag > 0\n" + "\n".join(p_txt),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.85"},
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            lead_data = [
                np.asarray(
                    [
                        float(r["ms_lead_ratio"])
                        for r in rows
                        if r["band"] == band and np.isfinite(float(r["ms_lead_ratio"]))
                    ],
                    dtype=float,
                )
                for band in bands
            ]
            fig, ax = plt.subplots(figsize=(11, 6.5))
            ax.axhline(0.5, color="0.25", linewidth=1.0)
            _boxplot_with_labels(ax, lead_data, bands, showfliers=False)
            for i, vals in enumerate(lead_data, start=1):
                if vals.size == 0:
                    continue
                ax.scatter(
                    i + rng.uniform(-0.12, 0.12, size=vals.size),
                    vals,
                    s=28,
                    color="#1f77b4",
                    alpha=0.75,
                    edgecolor="white",
                    linewidth=0.4,
                )
            ax.set_title("Fraction of valid windows with MS-leading lag")
            ax.set_ylabel("MS-leading window ratio")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            subject_curves = stats["subject_curves"]
            n_cols = 2
            n_rows = int(np.ceil(len(bands) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, max(4.0, 3.0 * n_rows)), squeeze=False)
            for ax in axes.ravel():
                ax.axis("off")
            for bi, band in enumerate(bands):
                ax = axes[bi // n_cols][bi % n_cols]
                ax.axis("on")
                curves = [
                    subject_curves[key]
                    for key in sorted(subject_curves.keys(), key=lambda k: (natural_key(k[0]), natural_key(k[1])))
                    if key[1] == band
                ]
                if not curves:
                    continue
                lag_ref = np.asarray(curves[0][0], dtype=float)
                stack = np.stack([np.asarray(c[1], dtype=float) for c in curves], axis=0)
                for row in stack:
                    ax.plot(lag_ref, row, color="0.78", linewidth=0.7, alpha=0.45)
                mean = np.nanmean(stack, axis=0)
                n_eff = np.sum(np.isfinite(stack), axis=0)
                sem = np.full(mean.size, np.nan, dtype=float)
                ok = n_eff > 1
                sem[ok] = np.nanstd(stack[:, ok], axis=0, ddof=1) / np.sqrt(n_eff[ok])
                ax.plot(lag_ref, mean, color="#d62728", linewidth=2.0)
                ax.fill_between(lag_ref, mean - sem, mean + sem, color="#d62728", alpha=0.16, linewidth=0)
                ax.axvline(0, color="0.25", linewidth=0.9, linestyle=":")
                ax.set_title(band)
                ax.set_xlabel("Lag (ms)")
                ax.set_ylabel("Mean norm. Z")
                ax.grid(True, alpha=0.28)
            fig.suptitle(f"{run_name} | PhaseLag mean normalized Rayleigh Z curves", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

            table_rows = []
            for row in summary_rows:
                p_lag = float(row["p_lag_gt_0"])
                p_lead = float(row["p_lead_ratio_gt_0_5"])
                table_rows.append(
                    [
                        row["band"],
                        str(int(row["n_subjects"])),
                        str(int(row["n_sessions"])),
                        f"{float(row['median_peak_lag_ms']):.3g}",
                        f"{float(row['iqr_peak_lag_low']):.3g}..{float(row['iqr_peak_lag_high']):.3g}",
                        f"{p_lag:.3g}" if np.isfinite(p_lag) else "NA",
                        f"{float(row['median_ms_lead_ratio']):.2f}" if np.isfinite(float(row["median_ms_lead_ratio"])) else "NA",
                        f"{p_lead:.3g}" if np.isfinite(p_lead) else "NA",
                        f"{float(row['median_valid_ratio']):.2f}" if np.isfinite(float(row["median_valid_ratio"])) else "NA",
                    ]
                )
            fig, ax = plt.subplots(figsize=(11, 6.5))
            ax.axis("off")
            ax.set_title(
                f"{run_name} | PhaseLag band statistics\n"
                "Session metrics are averaged within subject before inference.",
                fontsize=12,
                pad=16,
            )
            table = ax.table(
                cellText=table_rows,
                colLabels=[
                    "Band",
                    "N subj",
                    "N sess",
                    "Median lag ms",
                    "IQR lag",
                    "p lag>0",
                    "Median MSlead",
                    "p MSlead>0.5",
                    "Valid ratio",
                ],
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.45)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"-- saved PhaseLag stats PDF: {out_path}")
    return True


def _build_granger_stats_figure(
    mode_cells: list[ModeCell],
    bands: list[tuple[str, tuple[float, float]]],
    run_name: str,
) -> Any:
    _require_plotly()
    stats = _collect_granger_group_stats(mode_cells, bands)
    if stats is None:
        raise ValueError("No plottable Granger statistics.")

    freq_ref = np.asarray(stats["freq_ref"], dtype=float)
    subject_curves = stats["subject_curves"]
    rows = stats["subject_band_rows"]
    summary_rows = stats["summary_rows"]
    band_labels = [row["band"] for row in summary_rows]

    fig = make_subplots(
        rows=4,
        cols=1,
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
        row_heights=[0.36, 0.21, 0.21, 0.22],
        vertical_spacing=0.06,
        subplot_titles=(
            "Subject-averaged gPDC",
            "Band directionality index",
            "MS -> HC peak frequency",
            "Band statistics",
        ),
    )

    subjects = sorted(subject_curves.keys(), key=natural_key)
    if subjects:
        ms_stack = np.stack([subject_curves[s][0] for s in subjects], axis=0)
        hc_stack = np.stack([subject_curves[s][1] for s in subjects], axis=0)
        _add_mean_sem_traces(
            fig,
            freq_ref,
            ms_stack,
            row=1,
            col=1,
            color="#d62728",
            name="MS -> HC mean",
            showlegend=True,
            subject_alpha=0.24,
        )
        _add_mean_sem_traces(
            fig,
            freq_ref,
            hc_stack,
            row=1,
            col=1,
            color="#2ca02c",
            name="HC -> MS mean",
            showlegend=True,
            subject_alpha=0.24,
        )

    if band_labels:
        di_x: list[str] = []
        di_y: list[float] = []
        peak_x: list[str] = []
        peak_y: list[float] = []
        for band in band_labels:
            di_vals = _finite_metric_values(rows, band, "di")
            peak_vals = _finite_metric_values(rows, band, "ms_to_hc_peak_freq")
            di_x.extend([band] * int(di_vals.size))
            di_y.extend([float(v) for v in di_vals])
            peak_x.extend([band] * int(peak_vals.size))
            peak_y.extend([float(v) for v in peak_vals])
        fig.add_trace(
            go.Box(
                x=di_x,
                y=di_y,
                name="DI",
                marker={"color": "#d62728", "size": 5, "opacity": 0.72},
                line={"color": "#d62728", "width": 1.8},
                fillcolor="rgba(214,39,40,0.38)",
                boxpoints="all",
                jitter=0.22,
                pointpos=0,
                width=0.58,
                showlegend=False,
                hovertemplate="band=%{x}<br>DI=%{y:.4g}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0.0, line={"color": "rgba(40,40,40,0.75)", "width": 1.0}, row=2, col=1)
        fig.add_trace(
            go.Box(
                x=peak_x,
                y=peak_y,
                name="MS -> HC peak frequency",
                marker={"color": "#1f77b4", "size": 5, "opacity": 0.72},
                line={"color": "#1f77b4", "width": 1.8},
                fillcolor="rgba(31,119,180,0.35)",
                boxpoints="all",
                jitter=0.22,
                pointpos=0,
                width=0.58,
                showlegend=False,
                hovertemplate="band=%{x}<br>peak=%{y:.4g} Hz<extra></extra>",
            ),
            row=3,
            col=1,
        )

    table_rows: list[list[str]] = []
    for row in summary_rows:
        table_rows.append(
            [
                str(row["band"]),
                str(int(row["n_subjects"])),
                str(int(row["n_sessions"])),
                _format_stats_value(row["median_di"], ".3f"),
                f"{_format_stats_value(row['iqr_low'], '.3f')}..{_format_stats_value(row['iqr_high'], '.3f')}",
                _format_stats_value(row["p_di_ne_0"], ".3g"),
                f"{int(row['n_di_positive'])}/{int(row['n_subjects'])}",
                _format_stats_value(row["median_peak_freq"], ".1f"),
            ]
        )
    fig.add_trace(
        _plotly_table_trace(
            [
                "Band",
                "N subj",
                "N sess",
                "Median DI",
                "IQR DI",
                "p DI!=0",
                "DI>0",
                "Median peak Hz",
            ],
            table_rows,
        ),
        row=4,
        col=1,
    )

    fig.update_xaxes(title_text="Frequency (Hz)", showgrid=True, zeroline=False, minor={"dtick": 10, "showgrid": True}, row=1, col=1)
    fig.update_yaxes(title_text="gPDC", showgrid=True, zeroline=False, row=1, col=1)
    fig.update_xaxes(categoryorder="array", categoryarray=band_labels, title_text="Band", row=2, col=1)
    fig.update_yaxes(title_text="Subject-mean DI", showgrid=True, zeroline=False, row=2, col=1)
    fig.update_xaxes(categoryorder="array", categoryarray=band_labels, title_text="Band", row=3, col=1)
    fig.update_yaxes(title_text="Peak frequency (Hz)", showgrid=True, zeroline=False, row=3, col=1)
    fig.update_layout(
        template="plotly_white",
        height=1550,
        margin={"l": 82, "r": 28, "t": 78, "b": 44},
        title={
            "text": (
                f"{run_name} | Granger statistics<br>"
                "<sup>Session metrics are averaged within subject before inference. "
                "Wilcoxon signed-rank tests are two-sided for DI != 0.</sup>"
            ),
            "x": 0.01,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.0},
        dragmode="pan",
        hovermode="closest",
        boxmode="overlay",
    )
    return fig


def _build_phaselag_stats_figure(mode_cells: list[ModeCell], run_name: str) -> Any:
    _require_plotly()
    stats = _collect_phaselag_group_stats(mode_cells)
    if stats is None:
        raise ValueError("No plottable PhaseLag statistics.")

    rows = stats["subject_band_rows"]
    summary_rows = stats["summary_rows"]
    subject_curves = stats["subject_curves"]
    bands = [row["band"] for row in summary_rows]
    total_rows = 4
    specs = [[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]]
    row_heights = [0.22, 0.22, 0.34, 0.22]
    subplot_titles = (
        "Representative lag across subjects",
        "MS-leading window ratio",
        "Mean normalized Rayleigh Z by lag",
        "Band statistics",
    )
    fig = make_subplots(
        rows=total_rows,
        cols=1,
        specs=specs,
        row_heights=row_heights,
        vertical_spacing=0.045,
        subplot_titles=tuple(subplot_titles),
    )

    peak_x: list[str] = []
    peak_y: list[float] = []
    lead_x: list[str] = []
    lead_y: list[float] = []
    for band in bands:
        peak_vals = _finite_metric_values(rows, band, "peak_lag_ms")
        lead_vals = _finite_metric_values(rows, band, "ms_lead_ratio")
        peak_x.extend([band] * int(peak_vals.size))
        peak_y.extend([float(v) for v in peak_vals])
        lead_x.extend([band] * int(lead_vals.size))
        lead_y.extend([float(v) for v in lead_vals])

    fig.add_trace(
        go.Box(
            x=peak_x,
            y=peak_y,
            name="peak lag",
            marker={"color": "#d62728", "size": 5, "opacity": 0.72},
            line={"color": "#d62728", "width": 1.8},
            fillcolor="rgba(214,39,40,0.38)",
            boxpoints="all",
            jitter=0.22,
            pointpos=0,
            width=0.58,
            showlegend=False,
            hovertemplate="band=%{x}<br>peak lag=%{y:.4g} ms<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0.0, line={"color": "rgba(40,40,40,0.75)", "width": 1.0}, row=1, col=1)
    fig.add_trace(
        go.Box(
            x=lead_x,
            y=lead_y,
            name="MS lead ratio",
            marker={"color": "#1f77b4", "size": 5, "opacity": 0.72},
            line={"color": "#1f77b4", "width": 1.8},
            fillcolor="rgba(31,119,180,0.35)",
            boxpoints="all",
            jitter=0.22,
            pointpos=0,
            width=0.58,
            showlegend=False,
            hovertemplate="band=%{x}<br>MS lead ratio=%{y:.4g}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.5, line={"color": "rgba(40,40,40,0.75)", "width": 1.0}, row=2, col=1)

    for bi, band in enumerate(bands):
        curves = [
            subject_curves[key]
            for key in sorted(subject_curves.keys(), key=lambda k: (natural_key(k[0]), natural_key(k[1])))
            if key[1] == band
        ]
        if curves:
            lag_ref = np.asarray(curves[0][0], dtype=float)
            stack = np.stack([np.asarray(c[1], dtype=float) for c in curves], axis=0)
            _add_mean_sem_traces(
                fig,
                lag_ref,
                stack,
                row=3,
                col=1,
                color=PLOTLY_COLORS[bi % len(PLOTLY_COLORS)],
                name=band,
                showlegend=True,
                subject_alpha=0.22,
            )
    fig.add_vline(x=0.0, line={"color": "rgba(40,40,40,0.65)", "width": 1.0, "dash": "dot"}, row=3, col=1)
    fig.update_xaxes(title_text="Lag (ms)", showgrid=True, zeroline=False, row=3, col=1)
    fig.update_yaxes(title_text="Mean norm. Z", showgrid=True, zeroline=False, row=3, col=1)

    table_rows: list[list[str]] = []
    for row in summary_rows:
        table_rows.append(
            [
                str(row["band"]),
                str(int(row["n_subjects"])),
                str(int(row["n_sessions"])),
                _format_stats_value(row["median_peak_lag_ms"], ".3g"),
                f"{_format_stats_value(row['iqr_peak_lag_low'], '.3g')}..{_format_stats_value(row['iqr_peak_lag_high'], '.3g')}",
                _format_stats_value(row["p_lag_gt_0"], ".3g"),
                _format_stats_value(row["median_ms_lead_ratio"], ".2f"),
                _format_stats_value(row["p_lead_ratio_gt_0_5"], ".3g"),
                _format_stats_value(row["median_valid_ratio"], ".2f"),
            ]
        )
    fig.add_trace(
        _plotly_table_trace(
            [
                "Band",
                "N subj",
                "N sess",
                "Median lag ms",
                "IQR lag",
                "p lag>0",
                "Median MSlead",
                "p MSlead>0.5",
                "Valid ratio",
            ],
            table_rows,
        ),
        row=4,
        col=1,
    )

    for r in (1, 2):
        fig.update_xaxes(categoryorder="array", categoryarray=bands, title_text="Band", showgrid=True, row=r, col=1)
    fig.update_yaxes(title_text="Peak lag from mean normalized Rayleigh Z (ms)", showgrid=True, zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="MS-leading window ratio", range=[-0.02, 1.02], showgrid=True, zeroline=False, row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=1500,
        margin={"l": 96, "r": 28, "t": 82, "b": 44},
        title={
            "text": (
                f"{run_name} | PhaseLag statistics<br>"
                "<sup>Positive lag means MS leads HC. Session metrics are averaged within subject before inference.</sup>"
            ),
            "x": 0.01,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.0},
        dragmode="pan",
        hovermode="closest",
        boxmode="overlay",
    )
    return fig


def _build_syncfc_stats_figure(mode_cells: list[ModeCell], run_name: str) -> Any:
    _require_plotly()
    stats = _collect_syncfc_group_stats(mode_cells)
    if stats is None:
        raise ValueError("No plottable SyncFC statistics.")

    rows = stats["subject_band_rows"]
    summary_rows = stats["summary_rows"]
    bands = [row["band"] for row in summary_rows]
    fig = make_subplots(
        rows=7,
        cols=1,
        specs=[
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "table"}],
        ],
        row_heights=[0.14, 0.14, 0.13, 0.13, 0.13, 0.13, 0.2],
        vertical_spacing=0.045,
        subplot_titles=(
            "Band-limited PLV: real vs permutation",
            "Band-limited coherence: real vs permutation",
            "Null-normalized PLV",
            "Null-normalized coherence",
            "PLV real-null mean comparison",
            "Coherence real-null mean comparison",
            "Band statistics",
        ),
    )

    def add_real_null_boxes(plot_row: int, real_key: str, null_key: str, ylabel: str, color: str) -> None:
        real_x: list[str] = []
        real_y: list[float] = []
        null_x: list[str] = []
        null_y: list[float] = []
        for band in bands:
            real_vals = _finite_metric_values(rows, band, real_key)
            null_vals = _finite_metric_values(rows, band, null_key)
            real_x.extend([band] * int(real_vals.size))
            real_y.extend([float(v) for v in real_vals])
            null_x.extend([band] * int(null_vals.size))
            null_y.extend([float(v) for v in null_vals])
        fig.add_trace(
            go.Box(
                x=null_x,
                y=null_y,
                name=f"permutation {ylabel}",
                legendgroup=f"null-{ylabel}",
                marker={"color": "rgba(100,100,100,0.35)", "size": 4, "opacity": 0.28},
                line={"color": "rgba(90,90,90,0.95)", "width": 1.6},
                fillcolor="rgba(145,145,145,0.28)",
                boxpoints="all",
                jitter=0.11,
                pointpos=0,
                width=0.72,
                showlegend=(plot_row == 1),
                hovertemplate="band=%{x}<br>permutation=%{y:.4g}<extra></extra>",
            ),
            row=plot_row,
            col=1,
        )
        fig.add_trace(
            go.Box(
                x=real_x,
                y=real_y,
                name=f"real {ylabel}",
                legendgroup=f"real-{ylabel}",
                marker={"color": color, "size": 5, "opacity": 0.7},
                line={"color": color, "width": 1.9},
                fillcolor=_hex_to_rgba(color, 0.42),
                boxpoints="all",
                jitter=0.16,
                pointpos=0,
                width=0.48,
                showlegend=(plot_row == 1),
                hovertemplate="band=%{x}<br>real=%{y:.4g}<extra></extra>",
            ),
            row=plot_row,
            col=1,
        )
        fig.update_yaxes(title_text=ylabel, range=[-0.02, 1.02], showgrid=True, zeroline=False, row=plot_row, col=1)

    add_real_null_boxes(1, "median_plv", "surrogate_mean_plv", "PLV", "#1f77b4")
    add_real_null_boxes(2, "median_coh", "surrogate_mean_coh", "Coherence", "#d62728")

    for plot_row, metric, ylabel, color in (
        (3, "z_plv", "Zsync PLV", "#1f77b4"),
        (4, "z_coh", "Zsync coherence", "#d62728"),
    ):
        x_vals: list[str] = []
        y_vals: list[float] = []
        for band in bands:
            vals = _finite_metric_values(rows, band, metric)
            x_vals.extend([band] * int(vals.size))
            y_vals.extend([float(v) for v in vals])
        fig.add_trace(
            go.Box(
                x=x_vals,
                y=y_vals,
                name=ylabel,
                marker={"color": color, "size": 5, "opacity": 0.72},
                line={"color": color, "width": 1.8},
                fillcolor=_hex_to_rgba(color, 0.35),
                boxpoints="all",
                jitter=0.2,
                pointpos=0,
                showlegend=False,
                hovertemplate="band=%{x}<br>%{y:.4g}<extra></extra>",
            ),
            row=plot_row,
            col=1,
        )
        fig.add_hline(y=0.0, line={"color": "rgba(40,40,40,0.75)", "width": 1.0}, row=plot_row, col=1)
        fig.update_yaxes(title_text=ylabel, showgrid=True, zeroline=False, row=plot_row, col=1)

    def add_real_null_summary_boxes(plot_row: int, real_key: str, null_key: str, ylabel: str, color: str) -> None:
        real_x: list[str] = []
        real_y: list[float] = []
        null_x: list[str] = []
        null_y: list[float] = []
        for band in bands:
            real = _finite_metric_values(rows, band, real_key)
            null = _finite_metric_values(rows, band, null_key)
            real_x.extend([band] * int(real.size))
            real_y.extend([float(v) for v in real])
            null_x.extend([band] * int(null.size))
            null_y.extend([float(v) for v in null])
        fig.add_trace(
            go.Box(
                x=null_x,
                y=null_y,
                name=f"permutation {ylabel}",
                legendgroup=f"summary-null-{ylabel}",
                marker={"color": "rgba(100,100,100,0.35)", "size": 4, "opacity": 0.28},
                line={"color": "rgba(90,90,90,0.95)", "width": 1.6},
                fillcolor="rgba(145,145,145,0.28)",
                boxmean=True,
                boxpoints="all",
                jitter=0.11,
                pointpos=0,
                width=0.72,
                showlegend=False,
                hovertemplate="band=%{x}<br>permutation=%{y:.4g}<extra></extra>",
            ),
            row=plot_row,
            col=1,
        )
        fig.add_trace(
            go.Box(
                x=real_x,
                y=real_y,
                name=f"real {ylabel}",
                legendgroup=f"summary-real-{ylabel}",
                marker={"color": color, "size": 5, "opacity": 0.7},
                line={"color": color, "width": 1.9},
                fillcolor=_hex_to_rgba(color, 0.42),
                boxmean=True,
                boxpoints="all",
                jitter=0.16,
                pointpos=0,
                width=0.48,
                showlegend=False,
                hovertemplate="band=%{x}<br>real=%{y:.4g}<extra></extra>",
            ),
            row=plot_row,
            col=1,
        )
        fig.update_yaxes(title_text=ylabel, range=[-0.02, 1.02], showgrid=True, zeroline=False, row=plot_row, col=1)

    add_real_null_summary_boxes(5, "median_plv", "surrogate_mean_plv", "PLV", "#1f77b4")
    add_real_null_summary_boxes(6, "median_coh", "surrogate_mean_coh", "Coherence", "#d62728")

    table_rows: list[list[str]] = []
    for row in summary_rows:
        table_rows.append(
            [
                str(row["band"]),
                str(int(row["n_subjects"])),
                str(int(row["n_sessions"])),
                _format_stats_value(row["median_plv"], ".3f"),
                _format_stats_value(row["median_z_plv"], ".3g"),
                _format_stats_value(row["p_z_plv_gt_0"], ".3g"),
                _format_stats_value(row["q_z_plv_gt_0"], ".3g"),
                _format_stats_value(row["median_coh"], ".3f"),
                _format_stats_value(row["median_z_coh"], ".3g"),
                _format_stats_value(row["p_z_coh_gt_0"], ".3g"),
                _format_stats_value(row["q_z_coh_gt_0"], ".3g"),
            ]
        )
    fig.add_trace(
        _plotly_table_trace(
            [
                "Band",
                "N subj",
                "N sess",
                "Median PLV",
                "Median Z PLV",
                "p ZPLV>0",
                "q ZPLV",
                "Median Coh",
                "Median Z Coh",
                "p ZCoh>0",
                "q ZCoh",
            ],
            table_rows,
        ),
        row=7,
        col=1,
    )

    for r in range(1, 7):
        fig.update_xaxes(categoryorder="array", categoryarray=bands, title_text="Band", showgrid=True, zeroline=False, row=r, col=1)
    fig.update_layout(
        template="plotly_white",
        height=2200,
        margin={"l": 86, "r": 28, "t": 82, "b": 44},
        title={
            "text": (
                f"{run_name} | SyncFC statistics<br>"
                "<sup>Real and permutation values are subject-level session averages. "
                "Wilcoxon signed-rank tests are one-sided for Zsync &gt; 0.</sup>"
            ),
            "x": 0.01,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.0},
        dragmode="pan",
        hovermode="closest",
        boxmode="overlay",
    )
    return fig


def _inject_reset_listener(html_text: str) -> str:
    idx = html_text.rfind("</body>")
    if idx >= 0:
        return html_text[:idx] + _PLOTLY_RESET_LISTENER + html_text[idx:]
    return html_text + _PLOTLY_RESET_LISTENER


def write_plotly_html(fig: Any, out_path: Path, include_plotlyjs: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    include_mode: Any = "cdn" if include_plotlyjs == "cdn" else True
    html = fig.to_html(include_plotlyjs=include_mode, full_html=True)
    html = _inject_reset_listener(html)
    out_path.write_text(html, encoding="utf-8")


def _build_mode_index_html(
    mode_pages: list[ModePage],
    run_name: str,
    subject_order: list[str],
    mode_order: list[str],
) -> str:
    items = [
        {
            "subject": m.subject,
            "mode": m.mode,
            "mode_title": m.mode_title,
            "href": m.href,
        }
        for m in mode_pages
    ]
    mode_defs = [
        {"mode": mode, "title": MODE_SHORT_TITLE_MAP.get(mode, mode)}
        for mode in mode_order
    ]
    items_json = json.dumps(items, ensure_ascii=False)
    run_name_json = json.dumps(run_name, ensure_ascii=False)
    subjects_json = json.dumps(subject_order, ensure_ascii=False)
    mode_defs_json = json.dumps(mode_defs, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Interactive Analysis Index</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --line: #d9dde6;
      --txt: #212734;
      --muted: #5e6575;
      --accent: #1f6feb;
      --head-bg: #f8fbff;
      --row-col-w: 140px;
      --col-col-w: 61px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Calibri, "Segoe UI", Arial, sans-serif;
      color: var(--txt);
      background: var(--bg);
      height: 100dvh;
      overflow: hidden;
    }}
    .root {{
      display: grid;
      grid-template-columns: clamp(300px, 32vw, 440px) 1fr;
      height: 100dvh;
      min-height: 0;
    }}
    .side {{
      border-right: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      flex-direction: column;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
    }}
    .side-head {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      font-weight: 700;
      line-height: 1.25;
      background: var(--panel);
      flex: 0 0 auto;
    }}
    .side-sub {{
      display: block;
      margin-top: 4px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 400;
    }}
    .matrix-wrap {{
      flex: 1 1 auto;
      min-height: 0;
      min-width: 0;
      overflow: auto;
      padding: 8px;
    }}
    .matrix {{
      border-collapse: separate;
      border-spacing: 0;
      width: max-content;
      table-layout: fixed;
      font-size: 12px;
    }}
    .matrix th, .matrix td {{
      border: 1px solid var(--line);
      padding: 0;
      vertical-align: middle;
      background: #fff;
    }}
    .matrix .corner {{
      position: sticky;
      top: 0;
      left: 0;
      z-index: 5;
      min-width: var(--row-col-w);
      width: var(--row-col-w);
      background: var(--head-bg);
      text-align: left;
      padding: 7px 8px;
      font-weight: 700;
    }}
    .matrix .colhead {{
      position: sticky;
      top: 0;
      z-index: 4;
      min-width: var(--col-col-w);
      width: var(--col-col-w);
      background: var(--head-bg);
      text-align: left;
      padding: 6px 8px;
      font-weight: 700;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .matrix .rowhead {{
      position: sticky;
      left: 0;
      z-index: 3;
      min-width: var(--row-col-w);
      width: var(--row-col-w);
      background: var(--head-bg);
      text-align: left;
      padding: 6px 8px;
      font-weight: 700;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .matrix .rowhead.active, .matrix .colhead.active {{
      background: #e8f2ff;
    }}
    .matrix .cell-wrap {{
      width: var(--col-col-w);
      min-width: var(--col-col-w);
      height: 36px;
      background: #fff;
    }}
    .matrix .cell {{
      display: block;
      width: 100%;
      height: 100%;
      border: 0;
      background: transparent;
      cursor: pointer;
      text-align: center;
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
    }}
    .matrix .cell:hover {{
      background: #f1f6ff;
      color: var(--accent);
    }}
    .matrix .cell.active {{
      background: #dcecff;
      color: var(--accent);
      box-shadow: inset 0 0 0 2px var(--accent);
    }}
    .matrix .empty {{
      width: var(--col-col-w);
      min-width: var(--col-col-w);
      height: 36px;
      background: #f1f3f8;
    }}
    .main {{
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-width: 0;
      min-height: 0;
      overflow: hidden;
    }}
    .top {{
      display: grid;
      grid-template-columns: auto auto auto auto 1fr;
      gap: 8px;
      align-items: center;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      padding: 8px 10px;
    }}
    .btn {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 13px;
    }}
    .btn:hover {{ background: #f7f9fc; }}
    .sync-switch {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 32px;
      padding: 5px 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      cursor: pointer;
      user-select: none;
      white-space: nowrap;
    }}
    .sync-switch input {{
      position: absolute;
      width: 1px;
      height: 1px;
      margin: -1px;
      padding: 0;
      border: 0;
      overflow: hidden;
      clip: rect(0 0 0 0);
      clip-path: inset(50%);
    }}
    .sync-slider {{
      position: relative;
      width: 38px;
      height: 22px;
      border-radius: 999px;
      background: #c6cedb;
      transition: background 0.2s ease;
      flex: 0 0 auto;
    }}
    .sync-slider::after {{
      content: "";
      position: absolute;
      top: 3px;
      left: 3px;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #fff;
      box-shadow: 0 1px 2px rgba(0,0,0,0.25);
      transition: transform 0.2s ease;
    }}
    .sync-label {{
      font-size: 12px;
      color: var(--txt);
      font-weight: 700;
    }}
    .sync-switch.active {{
      border-color: #9dc0ff;
      background: #f2f7ff;
    }}
    .sync-switch.active .sync-slider {{
      background: var(--accent);
    }}
    .sync-switch.active .sync-slider::after {{
      transform: translateX(16px);
    }}
    .modebar-host {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 2px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      flex: 0 0 auto;
    }}
    .modebar-proxy-btn {{
      border: 1px solid transparent;
      background: transparent;
      border-radius: 6px;
      min-width: 46px;
      height: 28px;
      padding: 0 8px;
      cursor: pointer;
      font-size: 12px;
      color: var(--txt);
      font-weight: 700;
      line-height: 1;
    }}
    .modebar-proxy-btn:hover {{
      background: #f1f6ff;
      color: var(--accent);
    }}
    .modebar-proxy-btn.active {{
      background: #e8f2ff;
      border-color: #b6d1ff;
      color: var(--accent);
    }}
    .meta {{
      min-width: 0;
      font-size: 13px;
      color: var(--muted);
      padding-left: 8px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .meta-wrap {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }}
    .viewer {{
      position: relative;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      background: #fff;
    }}
    iframe {{
      position: absolute;
      inset: 0;
      display: block;
      width: 100%;
      height: 100%;
      border: 0;
      background: #fff;
    }}
  </style>
</head>
<body>
  <div class="root">
    <aside class="side">
      <div class="side-head">
        Interactive Analysis Browser
        <span class="side-sub" id="runName"></span>
        <span class="side-sub">Grid: rows=Subject, columns=Mode</span>
        <span class="side-sub">Keys: ←/→ mode, ↑/↓ subject, R reset zoom</span>
      </div>
      <div class="matrix-wrap">
        <table class="matrix" id="matrix"></table>
      </div>
    </aside>
    <main class="main">
      <div class="top">
        <button class="btn" id="prevBtn">Prev</button>
        <button class="btn" id="nextBtn">Next</button>
        <button class="btn" id="resetBtn" title="Reset zoom in current page">Reset Zoom</button>
        <label class="sync-switch" id="syncSwitchWrap" title="Sync zoom/pan across all subplots in current plot">
          <input type="checkbox" id="syncToggle" />
          <span class="sync-slider" aria-hidden="true"></span>
          <span class="sync-label">Sync Subplots</span>
        </label>
        <div class="meta-wrap">
          <div class="meta" id="meta"></div>
          <div class="modebar-host" id="modebarHost"></div>
        </div>
      </div>
      <div class="viewer">
        <iframe id="frame" title="interactive-analysis-plot"></iframe>
      </div>
    </main>
  </div>
  <script>
    const RUN_NAME = {run_name_json};
    const entries = {items_json};
    const subjects = {subjects_json};
    const modeDefs = {mode_defs_json};

    const runNameEl = document.getElementById("runName");
    const matrixEl = document.getElementById("matrix");
    const metaEl = document.getElementById("meta");
    const modebarHostEl = document.getElementById("modebarHost");
    const frameEl = document.getElementById("frame");
    const syncToggleEl = document.getElementById("syncToggle");
    const syncSwitchWrapEl = document.getElementById("syncSwitchWrap");
    runNameEl.textContent = "Run: " + RUN_NAME;

    const modeList = modeDefs.map(m => m.mode);
    const modeTitleMap = new Map(modeDefs.map(m => [m.mode, m.title]));
    const entryMap = new Map(entries.map(e => [`${{e.subject}}|||${{e.mode}}`, e]));
    let curRow = 0;
    let curCol = 0;
    let syncAllSubplots = true;
    let currentDragMode = "zoom";

    const MODEBAR_ACTIONS = [
      {{ id: "zoom", label: "Zoom", title: "Zoom drag mode" }},
      {{ id: "pan", label: "Pan", title: "Pan drag mode" }},
      {{ id: "reset", label: "Reset", title: "Reset zoom/pan" }},
    ];

    function keyOf(r, c) {{
      return `${{subjects[r]}}|||${{modeList[c]}}`;
    }}

    function getEntry(r, c) {{
      if (r < 0 || r >= subjects.length || c < 0 || c >= modeList.length) return null;
      return entryMap.get(keyOf(r, c)) || null;
    }}

    function hasEntry(r, c) {{
      return !!getEntry(r, c);
    }}

    function findFirstAvailable() {{
      for (let r = 0; r < subjects.length; r++) {{
        for (let c = 0; c < modeList.length; c++) {{
          if (hasEntry(r, c)) return [r, c];
        }}
      }}
      return null;
    }}

    function buildMatrix() {{
      matrixEl.innerHTML = "";
      const headTr = document.createElement("tr");
      const corner = document.createElement("th");
      corner.className = "corner";
      corner.textContent = "Subject \\\\ Mode";
      headTr.appendChild(corner);
      for (let c = 0; c < modeList.length; c++) {{
        const mode = modeList[c];
        const th = document.createElement("th");
        th.className = "colhead";
        th.dataset.col = String(c);
        th.textContent = modeTitleMap.get(mode) || mode;
        headTr.appendChild(th);
      }}
      matrixEl.appendChild(headTr);

      for (let r = 0; r < subjects.length; r++) {{
        const tr = document.createElement("tr");
        const rh = document.createElement("th");
        rh.className = "rowhead";
        rh.dataset.row = String(r);
        rh.textContent = subjects[r];
        tr.appendChild(rh);

        for (let c = 0; c < modeList.length; c++) {{
          const e = getEntry(r, c);
          const td = document.createElement("td");
          if (!e) {{
            td.className = "empty";
            tr.appendChild(td);
            continue;
          }}
          td.className = "cell-wrap";
          const btn = document.createElement("button");
          btn.className = "cell";
          btn.textContent = "●";
          btn.dataset.row = String(r);
          btn.dataset.col = String(c);
          btn.title = `${{subjects[r]}} | ${{modeTitleMap.get(modeList[c]) || modeList[c]}}`;
          btn.addEventListener("click", () => go(r, c));
          td.appendChild(btn);
          tr.appendChild(td);
        }}
        matrixEl.appendChild(tr);
      }}
    }}

    function updateActive() {{
      for (const el of matrixEl.querySelectorAll(".rowhead")) {{
        el.classList.toggle("active", Number(el.dataset.row) === curRow);
      }}
      for (const el of matrixEl.querySelectorAll(".colhead")) {{
        el.classList.toggle("active", Number(el.dataset.col) === curCol);
      }}
      for (const el of matrixEl.querySelectorAll(".cell")) {{
        const r = Number(el.dataset.row);
        const c = Number(el.dataset.col);
        el.classList.toggle("active", r === curRow && c === curCol);
      }}
    }}

    function go(r, c) {{
      if (!subjects.length || !modeList.length) return false;
      if (!hasEntry(r, c)) return false;
      curRow = r;
      curCol = c;
      const e = getEntry(r, c);
      if (!e) return false;
      frameEl.src = e.href;
      metaEl.textContent = `subject=${{e.subject}} | mode=${{modeTitleMap.get(e.mode) || e.mode}}`;
      updateActive();
      return true;
    }}

    function moveLinear(step) {{
      if (!subjects.length || !modeList.length) return;
      const total = subjects.length * modeList.length;
      let idx = curRow * modeList.length + curCol;
      for (let n = 0; n < total; n++) {{
        idx = (idx + step + total) % total;
        const r = Math.floor(idx / modeList.length);
        const c = idx % modeList.length;
        if (go(r, c)) return;
      }}
    }}

    function moveGrid(dRow, dCol) {{
      if (!subjects.length || !modeList.length) return;
      const maxSteps = subjects.length * modeList.length;
      let r = curRow;
      let c = curCol;
      for (let i = 0; i < maxSteps; i++) {{
        r = (r + dRow + subjects.length) % subjects.length;
        c = (c + dCol + modeList.length) % modeList.length;
        if (go(r, c)) return;
      }}
    }}

    function resetZoom() {{
      try {{
        frameEl.contentWindow.postMessage({{ type: "reset-plotly-view" }}, "*");
      }} catch (err) {{}}
    }}

    function postPlotlyAction(action) {{
      try {{
        if (frameEl.contentWindow && typeof frameEl.contentWindow.postMessage === "function") {{
          frameEl.contentWindow.postMessage({{ type: "plotly-modebar-action", action: action }}, "*");
        }}
      }} catch (err) {{}}
    }}

    function setActiveDragButton(action) {{
      if (!modebarHostEl) return;
      const btns = modebarHostEl.querySelectorAll(".modebar-proxy-btn");
      for (const b of btns) {{
        const a = b.dataset.action || "";
        const isDrag = (a === "zoom" || a === "pan");
        b.classList.toggle("active", isDrag && a === action);
      }}
    }}

    function dispatchPlotlyAction(action) {{
      if (!action) return;
      if (action === "reset") {{
        resetZoom();
        return;
      }}
      postPlotlyAction(action);
      if (action === "zoom" || action === "pan") {{
        currentDragMode = action;
        setActiveDragButton(action);
      }}
    }}

    function buildModebarHost() {{
      if (!modebarHostEl) return;
      modebarHostEl.innerHTML = "";
      for (const item of MODEBAR_ACTIONS) {{
        const b = document.createElement("button");
        b.type = "button";
        b.className = "modebar-proxy-btn";
        b.dataset.action = item.id;
        b.title = item.title;
        b.textContent = item.label;
        b.addEventListener("click", (ev) => {{
          ev.preventDefault();
          dispatchPlotlyAction(item.id);
        }});
        modebarHostEl.appendChild(b);
      }}
      setActiveDragButton(currentDragMode);
    }}

    function dispatchSyncSetting(enabled) {{
      try {{
        if (frameEl.contentWindow && typeof frameEl.contentWindow.postMessage === "function") {{
          frameEl.contentWindow.postMessage(
            {{ type: "plotly-sync-all-subplots", enabled: !!enabled }},
            "*"
          );
        }}
      }} catch (err) {{}}
    }}

    function updateSyncToggleUI() {{
      if (syncToggleEl) syncToggleEl.checked = !!syncAllSubplots;
      if (syncSwitchWrapEl) syncSwitchWrapEl.classList.toggle("active", !!syncAllSubplots);
    }}

    function setSyncAllSubplots(enabled) {{
      syncAllSubplots = !!enabled;
      updateSyncToggleUI();
      dispatchSyncSetting(syncAllSubplots);
    }}

    document.getElementById("prevBtn").addEventListener("click", () => moveLinear(-1));
    document.getElementById("nextBtn").addEventListener("click", () => moveLinear(1));
    document.getElementById("resetBtn").addEventListener("click", resetZoom);
    if (syncToggleEl) {{
      syncToggleEl.addEventListener("change", () => {{
        setSyncAllSubplots(!!syncToggleEl.checked);
      }});
    }}
    frameEl.addEventListener("load", () => {{
      window.setTimeout(() => {{
        dispatchSyncSetting(syncAllSubplots);
        postPlotlyAction(currentDragMode);
      }}, 120);
    }});
    document.addEventListener("keydown", (ev) => {{
      if (ev.key === "ArrowUp") {{ ev.preventDefault(); moveGrid(-1, 0); }}
      else if (ev.key === "ArrowDown") {{ ev.preventDefault(); moveGrid(1, 0); }}
      else if (ev.key === "ArrowLeft") {{ ev.preventDefault(); moveGrid(0, -1); }}
      else if (ev.key === "ArrowRight") {{ ev.preventDefault(); moveGrid(0, 1); }}
      else if (ev.key === "r" || ev.key === "R") {{ ev.preventDefault(); resetZoom(); }}
    }});

    buildMatrix();
    buildModebarHost();
    updateSyncToggleUI();
    const first = findFirstAvailable();
    if (first) go(first[0], first[1]);
    else metaEl.textContent = "No subject/mode pages found.";
  </script>
</body>
</html>
"""


def _build_stats_index_html(stats_pages: list[StatsPage], run_name: str) -> str:
    items = [
        {
            "key": p.key,
            "title": p.title,
            "href": p.href,
        }
        for p in stats_pages
    ]
    items_json = json.dumps(items, ensure_ascii=False)
    run_name_json = json.dumps(run_name, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Group Statistics</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --line: #d9dde6;
      --txt: #212734;
      --muted: #5e6575;
      --accent: #1f6feb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      height: 100dvh;
      overflow: hidden;
      font-family: Calibri, "Segoe UI", Arial, sans-serif;
      color: var(--txt);
      background: var(--bg);
    }}
    .root {{
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      height: 100dvh;
      min-height: 0;
    }}
    .top {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 10px 12px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    .title {{
      min-width: 0;
      font-size: 14px;
      font-weight: 700;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .subtitle {{
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 400;
    }}
    .tabs {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-width: 0;
    }}
    .tab, .tool {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--txt);
      border-radius: 7px;
      min-height: 32px;
      padding: 0 11px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 700;
    }}
    .tab:hover, .tool:hover {{ background: #f1f6ff; color: var(--accent); }}
    .tab.active {{
      background: #e8f2ff;
      border-color: #b6d1ff;
      color: var(--accent);
      box-shadow: inset 0 0 0 1px rgba(31,111,235,0.2);
    }}
    .tools {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding-left: 6px;
      border-left: 1px solid var(--line);
    }}
    .viewer {{
      position: relative;
      min-height: 0;
      min-width: 0;
      background: #fff;
      overflow: hidden;
    }}
    iframe {{
      position: absolute;
      inset: 0;
      display: block;
      width: 100%;
      height: 100%;
      border: 0;
      background: #fff;
    }}
  </style>
</head>
<body>
  <div class="root">
    <header class="top">
      <div class="title">
        Group Statistics Browser
        <span class="subtitle" id="runName"></span>
      </div>
      <div class="tabs" id="tabs"></div>
    </header>
    <main class="viewer">
      <iframe id="frame" title="group-statistics-plot"></iframe>
    </main>
  </div>
  <script>
    const RUN_NAME = {run_name_json};
    const pages = {items_json};
    const tabsEl = document.getElementById("tabs");
    const frameEl = document.getElementById("frame");
    document.getElementById("runName").textContent = "Run: " + RUN_NAME;
    let cur = 0;
    function post(action) {{
      try {{
        frameEl.contentWindow.postMessage({{ type: "plotly-modebar-action", action }}, "*");
      }} catch (err) {{}}
    }}
    function resetZoom() {{
      try {{
        frameEl.contentWindow.postMessage({{ type: "reset-plotly-view" }}, "*");
      }} catch (err) {{}}
    }}
    function renderTabs() {{
      tabsEl.innerHTML = "";
      pages.forEach((p, idx) => {{
        const b = document.createElement("button");
        b.className = "tab" + (idx === cur ? " active" : "");
        b.textContent = p.title;
        b.title = p.title;
        b.addEventListener("click", () => go(idx));
        tabsEl.appendChild(b);
      }});
      const tools = document.createElement("span");
      tools.className = "tools";
      for (const item of [
        {{ label: "Zoom", action: "zoom" }},
        {{ label: "Pan", action: "pan" }},
        {{ label: "Reset", action: "reset" }},
      ]) {{
        const b = document.createElement("button");
        b.className = "tool";
        b.textContent = item.label;
        b.addEventListener("click", () => item.action === "reset" ? resetZoom() : post(item.action));
        tools.appendChild(b);
      }}
      tabsEl.appendChild(tools);
    }}
    function go(idx) {{
      if (!pages.length) return;
      cur = (idx + pages.length) % pages.length;
      frameEl.src = pages[cur].href;
      renderTabs();
    }}
    document.addEventListener("keydown", (ev) => {{
      if (ev.key === "ArrowLeft") {{ ev.preventDefault(); go(cur - 1); }}
      else if (ev.key === "ArrowRight") {{ ev.preventDefault(); go(cur + 1); }}
      else if (ev.key === "r" || ev.key === "R") {{ ev.preventDefault(); resetZoom(); }}
    }});
    if (pages.length) go(0);
  </script>
</body>
</html>
"""


def _compute_psd_cell(
    entry: EEGEntry,
    sampling_rate: float,
    spike_sampling_rate: float,
    nperseg: int,
    max_freq: float,
    aperiodic_mode: str,
    fit_range_hz: tuple[float, float] | None,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
) -> ModeCell | None:
    def _postprocess_psd(
        f_in: np.ndarray,
        p_in: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        f = np.asarray(f_in, dtype=float).reshape(-1)
        p = np.asarray(p_in, dtype=float).reshape(-1)
        n = min(f.size, p.size)
        if n < 2:
            return None
        f = f[:n]
        p = p[:n]
        m = np.isfinite(f) & np.isfinite(p)
        if int(np.sum(m)) < 2:
            return None
        f = f[m]
        p = p[m]
        if max_freq > 0:
            mmax = f <= float(max_freq)
            if np.any(mmax):
                f = f[mmax]
                p = p[mmax]
        if aperiodic_mode == "residual":
            if fit_range_hz is None:
                return None
            p_res = _psd_remove_1f_residual(f, p, fit_range_hz=fit_range_hz)
            if p_res is not None:
                p = p_res
        if f.size < 2:
            return None
        return f, p

    hc_psd = compute_psd_db(entry.eeg_z, sampling_rate=sampling_rate, nperseg=nperseg)
    if hc_psd is None:
        return None
    hc_post = _postprocess_psd(hc_psd[0], hc_psd[1])
    if hc_post is None:
        return None
    f_hc, p_hc = hc_post
    traces: list[TraceSpec] = [
        TraceSpec(name="HC PSD", x=f_hc, y=p_hc, color="#1f77b4", width=1.2)
    ]

    full_tmax = float(max(0.0, (int(entry.eeg_z.size) - 1) / float(sampling_rate)))
    pair = _prepare_hc_pseudo_pair_meta(
        entry=entry,
        analysis_sampling_rate=float(sampling_rate),
        spike_sampling_rate=float(spike_sampling_rate),
        time_range=(0.0, full_tmax),
        ms_lfp_sigma=float(ms_lfp_sigma),
        ms_lfp_a=float(ms_lfp_a),
        ms_lfp_a0=float(ms_lfp_a0),
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=float(ms_lfp_d_default),
        ms_lfp_post_smooth_sec=float(ms_lfp_post_smooth_sec),
    )
    if pair is not None:
        _, _, y_pseudo, _ = pair
        psd_pseudo = compute_psd_db(y_pseudo, sampling_rate=sampling_rate, nperseg=nperseg)
        if psd_pseudo is not None:
            pseudo_post = _postprocess_psd(psd_pseudo[0], psd_pseudo[1])
            if pseudo_post is not None:
                f_ms, p_ms = pseudo_post
                traces.append(
                    TraceSpec(
                        name="MS pseudo PSD",
                        x=f_ms,
                        y=p_ms,
                        color="#ff7f0e",
                        width=1.2,
                        dash="solid",
                    )
                )

    return ModeCell(subject=entry.subject, session=entry.session, traces=tuple(traces))


def _build_psd_mode_cells(
    eeg_entries: list[EEGEntry],
    sampling_rate: float,
    spike_sampling_rate: float,
    nperseg: int,
    max_freq: float,
    aperiodic_mode: str,
    fit_range_hz: tuple[float, float] | None,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
    n_jobs: int,
) -> list[ModeCell]:
    out: list[ModeCell] = []
    worker = partial(
        _compute_psd_cell,
        sampling_rate=sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        nperseg=nperseg,
        max_freq=max_freq,
        aperiodic_mode=aperiodic_mode,
        fit_range_hz=fit_range_hz,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
    )
    for cell in tqdm(
        _parallel_map(eeg_entries, worker, n_jobs=n_jobs),
        total=len(eeg_entries),
        desc="Mode[psd]",
        unit="eeg",
    ):
        if cell is not None:
            out.append(cell)
    return out


def _compute_signal_hilbert_cells_for_entry(
    entry: EEGEntry,
    sampling_rate: float,
    spike_sampling_rate: float,
    band_sos: list[tuple[str, np.ndarray]],
    time_range: tuple[float, float],
    max_points: int,
    apply_db: bool,
    db_eps: float,
    smooth_win_sec: float | None,
    run_signal: bool,
    run_hilbert: bool,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
) -> tuple[ModeCell | None, ModeCell | None]:
    clipped = _clip_signal_by_time_range(entry.eeg_z, sampling_rate, time_range)
    if clipped is None:
        return None, None
    t_clip, raw_clip = clipped
    if t_clip.size < 2:
        return None, None

    ds_idx = _linspace_idx(t_clip.size, max_points)
    t_ds = t_clip[ds_idx]

    sig_traces: list[TraceSpec] = []
    if run_signal:
        sig_traces.append(
            TraceSpec(name="raw", x=t_ds, y=np.asarray(raw_clip, dtype=float)[ds_idx], color="#111111", width=1.3)
        )

    hil_traces: list[TraceSpec] = []
    if run_hilbert:
        raw_hpow = np.abs(hilbert(np.asarray(raw_clip, dtype=float))) ** 2
        if apply_db:
            raw_hpow = _db_transform(raw_hpow, eps=db_eps)
        raw_hpow = _smooth_by_time(raw_hpow, t_clip, smooth_win_sec)
        hil_traces.append(
            TraceSpec(name="raw", x=t_ds, y=np.asarray(raw_hpow, dtype=float)[ds_idx], color="#111111", width=1.3)
        )

    for bidx, (band_label, sos) in enumerate(band_sos):
        try:
            bp = _bandpass_zero_phase_with_sos(entry.eeg_z, sos)
        except Exception:
            continue
        bp_clip = _clip_signal_by_time_range(bp, sampling_rate, time_range)
        if bp_clip is None:
            continue
        _, y_bp = bp_clip
        y_bp_arr = np.asarray(y_bp, dtype=float).reshape(-1)
        t_loc = t_clip
        idx_loc = ds_idx
        t_loc_ds = t_ds
        if y_bp_arr.size != t_clip.size:
            n = min(y_bp_arr.size, t_clip.size)
            if n < 2:
                continue
            y_bp_arr = y_bp_arr[:n]
            t_loc = t_clip[:n]
            idx_loc = _linspace_idx(n, max_points)
            t_loc_ds = t_loc[idx_loc]

        color = PLOTLY_COLORS[bidx % len(PLOTLY_COLORS)]
        if run_signal:
            sig_traces.append(
                TraceSpec(name=band_label, x=t_loc_ds, y=y_bp_arr[idx_loc], color=color, width=1.0)
            )

        if run_hilbert:
            h_bp = np.abs(hilbert(y_bp_arr)) ** 2
            if apply_db:
                h_bp = _db_transform(h_bp, eps=db_eps)
            h_bp = _smooth_by_time(h_bp, t_loc, smooth_win_sec)
            hil_traces.append(
                TraceSpec(name=band_label, x=t_loc_ds, y=np.asarray(h_bp, dtype=float)[idx_loc], color=color, width=1.0)
            )

    pseudo_sig_traces: list[TraceSpec] = []
    pseudo_hil_traces: list[TraceSpec] = []
    pseudo_meta: dict[str, Any] | None = None
    pseudo_pair = _prepare_hc_pseudo_pair_meta(
        entry=entry,
        analysis_sampling_rate=float(sampling_rate),
        spike_sampling_rate=float(spike_sampling_rate),
        time_range=time_range,
        ms_lfp_sigma=float(ms_lfp_sigma),
        ms_lfp_a=float(ms_lfp_a),
        ms_lfp_a0=float(ms_lfp_a0),
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=float(ms_lfp_d_default),
        ms_lfp_post_smooth_sec=float(ms_lfp_post_smooth_sec),
    )
    if pseudo_pair is not None:
        t_p, _, y_p, pseudo_meta = pseudo_pair
        if t_p.size >= 2 and y_p.size >= 2:
            ds_idx_p = _linspace_idx(t_p.size, max_points)
            t_p_ds = t_p[ds_idx_p]
            y_p_arr = np.asarray(y_p, dtype=float).reshape(-1)

            if run_signal:
                pseudo_sig_traces.append(
                    TraceSpec(
                        name="raw",
                        x=t_p_ds,
                        y=y_p_arr[ds_idx_p],
                        color="#111111",
                        width=1.3,
                    )
                )

            if run_hilbert:
                p_raw_h = np.abs(hilbert(y_p_arr)) ** 2
                if apply_db:
                    p_raw_h = _db_transform(p_raw_h, eps=db_eps)
                p_raw_h = _smooth_by_time(p_raw_h, t_p, smooth_win_sec)
                pseudo_hil_traces.append(
                    TraceSpec(
                        name="raw",
                        x=t_p_ds,
                        y=np.asarray(p_raw_h, dtype=float)[ds_idx_p],
                        color="#111111",
                        width=1.3,
                    )
                )

            for bidx, (band_label, sos) in enumerate(band_sos):
                try:
                    y_bp_p = _bandpass_zero_phase_with_sos(y_p_arr, sos)
                except Exception:
                    continue
                y_bp_p = np.asarray(y_bp_p, dtype=float).reshape(-1)
                n_p = min(y_bp_p.size, t_p.size)
                if n_p < 2:
                    continue
                t_loc = t_p[:n_p]
                y_loc = y_bp_p[:n_p]
                idx_loc = _linspace_idx(n_p, max_points)
                t_loc_ds = t_loc[idx_loc]
                color = PLOTLY_COLORS[bidx % len(PLOTLY_COLORS)]

                if run_signal:
                    pseudo_sig_traces.append(
                        TraceSpec(
                            name=band_label,
                            x=t_loc_ds,
                            y=y_loc[idx_loc],
                            color=color,
                            width=1.0,
                        )
                    )

                if run_hilbert:
                    p_h_bp = np.abs(hilbert(y_loc)) ** 2
                    if apply_db:
                        p_h_bp = _db_transform(p_h_bp, eps=db_eps)
                    p_h_bp = _smooth_by_time(p_h_bp, t_loc, smooth_win_sec)
                    pseudo_hil_traces.append(
                        TraceSpec(
                            name=band_label,
                            x=t_loc_ds,
                            y=np.asarray(p_h_bp, dtype=float)[idx_loc],
                            color=color,
                            width=1.0,
                        )
                    )

    sig_payload: dict[str, Any] | None = None
    if pseudo_sig_traces:
        sig_payload = {
            "pseudo_traces": tuple(pseudo_sig_traces),
            "pseudo_shank_int": int((pseudo_meta or {}).get("shank_int", -1)),
            "pseudo_shank_label": str((pseudo_meta or {}).get("shank_label", "shank")),
        }
    hil_payload: dict[str, Any] | None = None
    if pseudo_hil_traces:
        hil_payload = {
            "pseudo_traces": tuple(pseudo_hil_traces),
            "pseudo_shank_int": int((pseudo_meta or {}).get("shank_int", -1)),
            "pseudo_shank_label": str((pseudo_meta or {}).get("shank_label", "shank")),
        }

    sig_cell = (
        ModeCell(subject=entry.subject, session=entry.session, traces=tuple(sig_traces), payload=sig_payload)
        if run_signal and sig_traces
        else None
    )
    hil_cell = (
        ModeCell(subject=entry.subject, session=entry.session, traces=tuple(hil_traces), payload=hil_payload)
        if run_hilbert and hil_traces
        else None
    )
    return sig_cell, hil_cell


def _build_signal_hilbert_mode_cells(
    eeg_entries: list[EEGEntry],
    sampling_rate: float,
    spike_sampling_rate: float,
    theta_bands: list[tuple[str, tuple[float, float]]],
    time_range: tuple[float, float],
    max_points: int,
    apply_db: bool,
    db_eps: float,
    smooth_win_sec: float | None,
    run_signal: bool,
    run_hilbert: bool,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
    n_jobs: int,
) -> tuple[list[ModeCell], list[ModeCell]]:
    signal_cells: list[ModeCell] = []
    hilbert_cells: list[ModeCell] = []

    band_sos = _prepare_bandpass_sos(theta_bands, sr=sampling_rate, order=4)
    worker = partial(
        _compute_signal_hilbert_cells_for_entry,
        sampling_rate=sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        band_sos=band_sos,
        time_range=time_range,
        max_points=max_points,
        apply_db=apply_db,
        db_eps=db_eps,
        smooth_win_sec=smooth_win_sec,
        run_signal=run_signal,
        run_hilbert=run_hilbert,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
    )

    bars: list[Any] = []
    if run_signal:
        bars.append(tqdm(total=len(eeg_entries), desc="Mode[signal]", unit="eeg"))
    if run_hilbert:
        bars.append(tqdm(total=len(eeg_entries), desc="Mode[hilbert]", unit="eeg"))

    try:
        for sig_cell, hil_cell in _parallel_map(eeg_entries, worker, n_jobs=n_jobs):
            if sig_cell is not None:
                signal_cells.append(sig_cell)
            if hil_cell is not None:
                hilbert_cells.append(hil_cell)
            for b in bars:
                b.update(1)
    finally:
        for b in bars:
            b.close()

    return signal_cells, hilbert_cells


def _compute_theta_family_cells_for_entry(
    entry: EEGEntry,
    sampling_rate: float,
    spike_sampling_rate: float,
    time_range: tuple[float, float],
    tf_win_sec: float,
    tf_step_sec: float,
    fmin_all: float,
    fmax_all: float,
    psd_array_multitaper: Any,
    theta_bands: list[tuple[str, tuple[float, float]]],
    active_set: set[str],
    max_points: int,
    apply_db: bool,
    db_eps: float,
    smooth_win_sec: float | None,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
) -> dict[str, ModeCell]:
    clipped = _clip_signal_by_time_range(entry.eeg_z, sampling_rate, time_range)
    if clipped is None:
        return {}
    _, x_clip = clipped
    pseudo_pair = _prepare_hc_pseudo_pair_meta(
        entry=entry,
        analysis_sampling_rate=float(sampling_rate),
        spike_sampling_rate=float(spike_sampling_rate),
        time_range=time_range,
        ms_lfp_sigma=float(ms_lfp_sigma),
        ms_lfp_a=float(ms_lfp_a),
        ms_lfp_a0=float(ms_lfp_a0),
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=float(ms_lfp_d_default),
        ms_lfp_post_smooth_sec=float(ms_lfp_post_smooth_sec),
    )

    mt_out = _compute_multitaper_psd_windows(
        sig=x_clip,
        sr=sampling_rate,
        win_sec=tf_win_sec,
        step_sec=tf_step_sec,
        fmin=fmin_all,
        fmax=fmax_all,
        psd_array_multitaper=psd_array_multitaper,
        mt_n_jobs=1,
    )
    if mt_out is None:
        return {}
    freqs_mt, psd_mt, t_right = mt_out
    if t_right.size < 2:
        return {}

    theta_pow_hc = (
        _compute_theta_band_power_from_psd(freqs_mt, psd_mt, theta_bands)
        if "theta_power" in active_set
        else {}
    )
    theta_ratio_hc = (
        _compute_theta_delta_ratio_from_psd(
            freqs_mt, psd_mt, theta_bands, delta_band=DELTA_BAND, eps=1e-12
        )
        if "theta_delta_ratio" in active_set
        else {}
    )
    ent_hc = (
        _compute_theta_entropy_from_psd(freqs_mt, psd_mt, theta_bands, eps=1e-12)
        if "entropy" in active_set
        else {}
    )
    ent_td_hc = (
        _compute_entropy_td_from_psd(
            freqs_mt, psd_mt, theta_bands, delta_band=DELTA_BAND, eps=1e-12
        )
        if "entropy_td" in active_set
        else {}
    )

    theta_pow_ms: dict[str, np.ndarray] = {}
    theta_ratio_ms: dict[str, np.ndarray] = {}
    ent_ms: dict[str, np.ndarray] = {}
    ent_td_ms: dict[str, np.ndarray] = {}
    t_right_ms: np.ndarray | None = None
    pseudo_meta: dict[str, Any] | None = None
    if pseudo_pair is not None:
        t_p, _, y_p, p_meta = pseudo_pair
        pseudo_meta = p_meta
        mt_out_ms = _compute_multitaper_psd_windows(
            sig=np.asarray(y_p, dtype=float),
            sr=sampling_rate,
            win_sec=tf_win_sec,
            step_sec=tf_step_sec,
            fmin=fmin_all,
            fmax=fmax_all,
            psd_array_multitaper=psd_array_multitaper,
            mt_n_jobs=1,
        )
        if mt_out_ms is not None:
            freqs_ms, psd_ms, t_right_ms_loc = mt_out_ms
            if freqs_ms.shape == freqs_mt.shape and np.allclose(freqs_ms, freqs_mt):
                t_right_ms = t_right_ms_loc
                if "theta_power" in active_set:
                    theta_pow_ms = _compute_theta_band_power_from_psd(freqs_ms, psd_ms, theta_bands)
                if "theta_delta_ratio" in active_set:
                    theta_ratio_ms = _compute_theta_delta_ratio_from_psd(
                        freqs_ms, psd_ms, theta_bands, delta_band=DELTA_BAND, eps=1e-12
                    )
                if "entropy" in active_set:
                    ent_ms = _compute_theta_entropy_from_psd(freqs_ms, psd_ms, theta_bands, eps=1e-12)
                if "entropy_td" in active_set:
                    ent_td_ms = _compute_entropy_td_from_psd(
                        freqs_ms, psd_ms, theta_bands, delta_band=DELTA_BAND, eps=1e-12
                    )

    def _build_traces(
        metric: dict[str, np.ndarray],
        t_axis: np.ndarray,
        do_db: bool,
        name_prefix: str = "",
        line_dash: str = "solid",
    ) -> tuple[TraceSpec, ...]:
        traces: list[TraceSpec] = []
        for bidx, (band_label, _) in enumerate(theta_bands):
            y = metric.get(band_label)
            if y is None:
                continue
            yy = np.asarray(y, dtype=float).reshape(-1)
            if yy.size != t_axis.size:
                continue
            if do_db:
                yy = _db_transform(yy, eps=db_eps)
            yy = _smooth_by_time(yy, t_axis, smooth_win_sec)
            xx, yy = _downsample_xy(t_axis, yy, max_points)
            traces.append(
                TraceSpec(
                    name=(f"{name_prefix}{band_label}" if name_prefix else band_label),
                    x=xx,
                    y=yy,
                    color=PLOTLY_COLORS[bidx % len(PLOTLY_COLORS)],
                    width=1.1,
                    dash=line_dash,
                )
            )
        return tuple(traces)

    out: dict[str, ModeCell] = {}
    if "theta_power" in active_set:
        tp_traces = _build_traces(theta_pow_hc, t_right, do_db=apply_db)
        tp_ms_traces = (
            _build_traces(theta_pow_ms, t_right_ms, do_db=apply_db)
            if t_right_ms is not None
            else tuple()
        )
        if tp_traces or tp_ms_traces:
            payload = None
            if tp_ms_traces:
                payload = {
                    "pseudo_traces": tp_ms_traces,
                    "pseudo_shank_int": int((pseudo_meta or {}).get("shank_int", -1)),
                    "pseudo_shank_label": str((pseudo_meta or {}).get("shank_label", "shank")),
                }
            out["theta_power"] = ModeCell(
                subject=entry.subject,
                session=entry.session,
                traces=tp_traces,
                payload=payload,
            )
    if "theta_delta_ratio" in active_set:
        tr_traces = _build_traces(theta_ratio_hc, t_right, do_db=False)
        tr_ms_traces = (
            _build_traces(theta_ratio_ms, t_right_ms, do_db=False)
            if t_right_ms is not None
            else tuple()
        )
        if tr_traces or tr_ms_traces:
            payload = None
            if tr_ms_traces:
                payload = {
                    "pseudo_traces": tr_ms_traces,
                    "pseudo_shank_int": int((pseudo_meta or {}).get("shank_int", -1)),
                    "pseudo_shank_label": str((pseudo_meta or {}).get("shank_label", "shank")),
                }
            out["theta_delta_ratio"] = ModeCell(
                subject=entry.subject,
                session=entry.session,
                traces=tr_traces,
                payload=payload,
            )
    if "entropy" in active_set:
        en_traces = _build_traces(ent_hc, t_right, do_db=False)
        en_ms_traces = (
            _build_traces(ent_ms, t_right_ms, do_db=False)
            if t_right_ms is not None
            else tuple()
        )
        if en_traces or en_ms_traces:
            payload = None
            if en_ms_traces:
                payload = {
                    "pseudo_traces": en_ms_traces,
                    "pseudo_shank_int": int((pseudo_meta or {}).get("shank_int", -1)),
                    "pseudo_shank_label": str((pseudo_meta or {}).get("shank_label", "shank")),
                }
            out["entropy"] = ModeCell(
                subject=entry.subject,
                session=entry.session,
                traces=en_traces,
                payload=payload,
            )
    if "entropy_td" in active_set:
        etd_traces = _build_traces(ent_td_hc, t_right, do_db=False)
        etd_ms_traces = (
            _build_traces(ent_td_ms, t_right_ms, do_db=False)
            if t_right_ms is not None
            else tuple()
        )
        if etd_traces or etd_ms_traces:
            payload = None
            if etd_ms_traces:
                payload = {
                    "pseudo_traces": etd_ms_traces,
                    "pseudo_shank_int": int((pseudo_meta or {}).get("shank_int", -1)),
                    "pseudo_shank_label": str((pseudo_meta or {}).get("shank_label", "shank")),
                }
            out["entropy_td"] = ModeCell(
                subject=entry.subject,
                session=entry.session,
                traces=etd_traces,
                payload=payload,
            )
    return out


def _build_theta_family_mode_cells(
    eeg_entries: list[EEGEntry],
    sampling_rate: float,
    spike_sampling_rate: float,
    theta_bands: list[tuple[str, tuple[float, float]]],
    time_range: tuple[float, float],
    tf_win_sec: float,
    tf_step_sec: float,
    max_points: int,
    apply_db: bool,
    db_eps: float,
    smooth_win_sec: float | None,
    active_modes: list[str],
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_distance_map: dict[str, dict[str, Any]] | None,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
    n_jobs: int,
) -> dict[str, list[ModeCell]]:
    psd_array_multitaper = _require_mne_multitaper()
    active_set = {m for m in active_modes if m in THETA_FAMILY_MODES}
    mode_order = ["theta_power", "theta_delta_ratio", "entropy", "entropy_td"]
    out: dict[str, list[ModeCell]] = {
        "theta_power": [],
        "theta_delta_ratio": [],
        "entropy": [],
        "entropy_td": [],
    }
    if not active_set:
        return out

    fmin_all = min(float(DELTA_BAND[0]), min(b[1][0] for b in theta_bands))
    fmax_all = max(float(DELTA_BAND[1]), max(b[1][1] for b in theta_bands))

    worker = partial(
        _compute_theta_family_cells_for_entry,
        sampling_rate=sampling_rate,
        spike_sampling_rate=spike_sampling_rate,
        time_range=time_range,
        tf_win_sec=tf_win_sec,
        tf_step_sec=tf_step_sec,
        fmin_all=fmin_all,
        fmax_all=fmax_all,
        psd_array_multitaper=psd_array_multitaper,
        theta_bands=theta_bands,
        active_set=active_set,
        max_points=max_points,
        apply_db=apply_db,
        db_eps=db_eps,
        smooth_win_sec=smooth_win_sec,
        ms_lfp_sigma=ms_lfp_sigma,
        ms_lfp_a=ms_lfp_a,
        ms_lfp_a0=ms_lfp_a0,
        ms_lfp_distance_map=ms_lfp_distance_map,
        ms_lfp_d_default=ms_lfp_d_default,
        ms_lfp_post_smooth_sec=ms_lfp_post_smooth_sec,
    )

    bars: list[Any] = [
        tqdm(total=len(eeg_entries), desc=f"Mode[{m}]", unit="eeg")
        for m in mode_order
        if m in active_set
    ]
    try:
        for mode_cells in _parallel_map(eeg_entries, worker, n_jobs=n_jobs):
            for mode in active_set:
                cell = mode_cells.get(mode)
                if cell is not None:
                    out[mode].append(cell)
            for b in bars:
                b.update(1)
    finally:
        for b in bars:
            b.close()

    return out


def _mode_axis_titles(
    mode: str,
    apply_db: bool,
    psd_aperiodic_mode: str = "none",
) -> tuple[str, str]:
    if mode == "psd":
        if str(psd_aperiodic_mode).strip().lower() == "residual":
            return "Frequency (Hz)", "PSD (dB, resid.)"
        return "Frequency (Hz)", "PSD (dB/Hz)"
    if mode == "signal":
        return "Time (s)", "Amplitude (z-score)"
    if mode == "hilbert":
        return "Time (s)", "Hilbert Power (dB)" if apply_db else "Hilbert Power"
    if mode == "spectrogram":
        return "Time (s)", "Frequency (Hz)"
    if mode == "theta_power":
        return "Time (s)", "Power (dB)" if apply_db else "Power"
    if mode == "theta_delta_ratio":
        return "Time (s)", "Theta/Delta Ratio"
    if mode == "entropy":
        return "Time (s)", "Entropy"
    if mode == "entropy_td":
        return "Time (s)", "Entropy TD Balance"
    if mode == "coherence_band":
        return "Time (s)", "Coh / PLV"
    if mode == "syncfc":
        return "Time (s)", "SyncFC Coh / PLV"
    if mode == "phaselag":
        return "Lag (ms)", "Rayleigh-Z lag distribution"
    if mode == "pearson":
        return "Time (s)", "Pearson r / r2"
    if mode == "coherence":
        return "Time (s)", "Frequency (Hz)"
    if mode == "granger":
        return "Frequency (Hz)", "gPDC"
    return "x", "y"


def main_code(args: argparse.Namespace) -> None:
    if args.sampling_rate <= 0:
        raise ValueError("--sampling-rate must be > 0.")
    if float(args.downsample_rate) < 0:
        raise ValueError("--downsample-rate must be >= 0.")
    if args.psd_nperseg < 8:
        raise ValueError("--psd-nperseg must be >= 8.")
    if args.tf_win_sec <= 0:
        raise ValueError("--tf_win_sec must be > 0.")
    if args.tf_step_sec <= 0:
        raise ValueError("--tf_step_sec must be > 0.")
    if args.interactive_max_points <= 0:
        raise ValueError("--interactive_max_points must be > 0.")
    if float(args.phaselag_min_z) < 0:
        raise ValueError("--phaselag_min_z must be >= 0.")
    if float(args.phaselag_min_plv) < 0 or float(args.phaselag_min_plv) > 1:
        raise ValueError("--phaselag_min_plv must be between 0 and 1.")
    if float(args.phaselag_min_peak_delta_z) < 0:
        raise ValueError("--phaselag_min_peak_delta_z must be >= 0.")
    if float(args.phaselag_min_peak_delta_frac) < 0:
        raise ValueError("--phaselag_min_peak_delta_frac must be >= 0.")
    if float(args.phaselag_min_valid_ratio) < 0 or float(args.phaselag_min_valid_ratio) > 1:
        raise ValueError("--phaselag_min_valid_ratio must be between 0 and 1.")
    if int(args.phaselag_edge_guard_bins) < 0:
        raise ValueError("--phaselag_edge_guard_bins must be >= 0.")
    if int(args.syncfc_n_surrogates) < 0:
        raise ValueError("--syncfc_n_surrogates must be >= 0.")
    syncfc_null = str(args.syncfc_null).strip().lower()
    if syncfc_null not in {"window_shuffle", "circular_shift", "none"}:
        raise ValueError("--syncfc_null must be one of: window_shuffle, circular_shift, none.")
    if float(args.syncfc_min_shift_sec) < 0:
        raise ValueError("--syncfc_min_shift_sec must be >= 0.")
    if (not np.isfinite(args.db_eps)) or float(args.db_eps) <= 0:
        raise ValueError("--db_eps must be a positive finite value.")
    if (
        not isinstance(args.freq_plot, (list, tuple))
        or len(args.freq_plot) != 2
    ):
        raise ValueError("--freq_plot must be [FMIN FMAX].")
    if (
        not isinstance(args.freq_calc, (list, tuple))
        or len(args.freq_calc) != 2
    ):
        raise ValueError("--freq_calc must be [FMIN FMAX].")
    if (
        not isinstance(args.granger_freq, (list, tuple))
        or len(args.granger_freq) != 2
    ):
        raise ValueError("--granger_freq must be [FMIN FMAX].")
    if int(args.granger_n_freqs) < 16:
        raise ValueError("--granger_n_freqs must be >= 16.")
    if int(args.granger_order_max) < 1:
        raise ValueError("--granger_order_max must be >= 1.")
    if int(args.granger_fixed_order) < 0:
        raise ValueError("--granger_fixed_order must be >= 0.")

    apply_line_noise_removal = _parse_bool_text(
        getattr(args, "APPLY_LINE_NOISE_REMOVAL", "true"),
        "--APPLY_LINE_NOISE_REMOVAL",
    )
    line_noise_hz = float(getattr(args, "LINE_NOISE_HZ", 50.0))
    line_noise_q = float(getattr(args, "LINE_NOISE_Q", 30.0))
    if apply_line_noise_removal:
        if (not np.isfinite(line_noise_hz)) or line_noise_hz <= 0:
            raise ValueError("--LINE_NOISE_HZ must be a positive finite value.")
        if (not np.isfinite(line_noise_q)) or line_noise_q <= 0:
            raise ValueError("--LINE_NOISE_Q must be a positive finite value.")

    ms_lfp_enabled = _parse_bool_text(getattr(args, "MS_LFP", "false"), "--MS_LFP")
    ms_lfp_overlay = _parse_bool_text(getattr(args, "MS_LFP_OVERLAY", "true"), "--MS_LFP_OVERLAY")
    ms_lfp_sigma = float(getattr(args, "MS_LFP_SIGMA", 0.004))
    ms_lfp_a = float(getattr(args, "MS_LFP_A", 0.2))
    ms_lfp_a0 = float(getattr(args, "MS_LFP_A0", 1.0))
    ms_lfp_distance_map = _parse_ms_lfp_distance_json(getattr(args, "MS_LFP_D", "{}"))
    ms_lfp_d_default = float(getattr(args, "MS_LFP_D_DEFAULT", 3.0))
    ms_lfp_post_smooth_sec = float(getattr(args, "MS_LFP_POST_SMOOTH_SEC", 0.012))

    if (not np.isfinite(ms_lfp_sigma)) or ms_lfp_sigma <= 0:
        raise ValueError("--MS_LFP_SIGMA must be a positive finite value.")
    if not np.isfinite(ms_lfp_a):
        raise ValueError("--MS_LFP_A must be finite.")
    if not np.isfinite(ms_lfp_a0):
        raise ValueError("--MS_LFP_A0 must be finite.")
    if (not np.isfinite(ms_lfp_d_default)) or ms_lfp_d_default <= 0:
        raise ValueError("--MS_LFP_D_DEFAULT must be a positive finite value.")
    if (not np.isfinite(ms_lfp_post_smooth_sec)) or ms_lfp_post_smooth_sec < 0:
        raise ValueError("--MS_LFP_POST_SMOOTH_SEC must be a finite value >= 0.")

    modes = _normalize_modes(args.modes)
    if not modes:
        print("No analysis modes specified; nothing to do.")
        return

    theta_bands = _parse_theta_bands(args.theta_bands)
    granger_stats_bands = _parse_freq_bands(args.granger_stats_bands, "--granger_stats_bands")
    ts_range = _parse_time_range_for_timeseries(modes, args.time_range)
    smooth_win_sec = _parse_smooth_window_sec(args.ts_smooth_win_sec)
    include_plotlyjs = _resolve_plotly_js_mode(args)
    n_jobs = _resolve_n_jobs(getattr(args, "n_jobs", 0))
    granger_epoch_jobs = _resolve_n_jobs(getattr(args, "granger_epoch_jobs", 1))
    down_cfg = _build_downsample_config(
        sampling_rate_in=float(args.sampling_rate),
        downsample_rate=float(args.downsample_rate),
        aa_order=6,
    )
    analysis_sampling_rate = float(
        down_cfg.sampling_rate_out if down_cfg is not None else float(args.sampling_rate)
    )
    psd_nyquist = 0.5 * analysis_sampling_rate
    req_max_freq = float(args.max_freq)
    if req_max_freq <= 0:
        psd_max_freq = float(psd_nyquist)
    else:
        psd_max_freq = float(min(req_max_freq, psd_nyquist))
    psd_aperiodic_mode = str(args.psd_aperiodic_mode).strip().lower()
    if psd_aperiodic_mode not in {"residual", "none"}:
        raise ValueError("--psd_aperiodic_mode must be one of: residual, none.")
    psd_fit_range_hz: tuple[float, float] | None = None
    if psd_aperiodic_mode == "residual":
        fit_lo = float(args.psd_1f_fit_range[0])
        fit_hi_req = float(args.psd_1f_fit_range[1])
        if (not np.isfinite(fit_lo)) or (not np.isfinite(fit_hi_req)):
            raise ValueError("--psd_1f_fit_range must contain finite values.")
        if fit_lo <= 0 or fit_lo >= fit_hi_req:
            raise ValueError("--psd_1f_fit_range must satisfy 0 < FMIN < FMAX.")
        fit_hi = float(min(fit_hi_req, psd_max_freq))
        if fit_hi <= fit_lo:
            raise ValueError(
                f"Invalid effective PSD 1/f fit range after clamping to PSD max frequency: "
                f"[{fit_lo:g}, {fit_hi:g}] Hz. Increase --max-freq or lower --psd_1f_fit_range."
            )
        psd_fit_range_hz = (fit_lo, fit_hi)

    freq_plot_min_req = float(args.freq_plot[0])
    freq_plot_max_req = float(args.freq_plot[1])
    freq_calc_min_req = float(args.freq_calc[0])
    freq_calc_max_req = float(args.freq_calc[1])
    granger_freq_min_req = float(args.granger_freq[0])
    granger_freq_max_req = float(args.granger_freq[1])
    granger_order_criterion = str(args.granger_order_criterion).strip().lower()
    granger_order_mode = str(args.granger_order_mode).strip().lower()
    granger_progress = str(args.granger_progress).strip().lower()
    if granger_progress not in {"none", "entry", "epoch"}:
        raise ValueError("--granger_progress must be one of: none, entry, epoch.")
    if granger_order_mode not in {"median", "per_window"}:
        raise ValueError("--granger_order_mode must be one of: median, per_window.")
    if granger_order_criterion not in {"bic", "aic"}:
        raise ValueError("--granger_order_criterion must be one of: bic, aic.")
    if (
        (not np.isfinite(freq_plot_min_req))
        or (not np.isfinite(freq_plot_max_req))
        or freq_plot_min_req <= 0
        or freq_plot_min_req >= freq_plot_max_req
    ):
        raise ValueError("--freq_plot must satisfy 0 < FMIN < FMAX.")
    if (
        (not np.isfinite(freq_calc_min_req))
        or (not np.isfinite(freq_calc_max_req))
        or freq_calc_min_req <= 0
        or freq_calc_min_req >= freq_calc_max_req
    ):
        raise ValueError("--freq_calc must satisfy 0 < FMIN < FMAX.")
    if (
        (not np.isfinite(granger_freq_min_req))
        or (not np.isfinite(granger_freq_max_req))
        or granger_freq_min_req < 0
        or granger_freq_min_req >= granger_freq_max_req
    ):
        raise ValueError("--granger_freq must satisfy 0 <= FMIN < FMAX.")

    nyq_analysis = 0.5 * analysis_sampling_rate
    freq_calc_max_eff = float(min(freq_calc_max_req, nyq_analysis * 0.999999))
    if freq_calc_max_eff <= freq_calc_min_req:
        raise ValueError(
            f"Invalid effective --freq_calc after Nyquist clamp: "
            f"[{freq_calc_min_req:g}, {freq_calc_max_eff:g}] Hz. "
            "Lower --freq_calc FMIN or raise --downsample-rate."
        )
    freq_plot_max_eff = float(min(freq_plot_max_req, freq_calc_max_eff))
    if freq_plot_max_eff <= freq_plot_min_req:
        raise ValueError(
            f"Invalid effective --freq_plot after clamp: "
            f"[{freq_plot_min_req:g}, {freq_plot_max_eff:g}] Hz. "
            "Adjust --freq_plot/--freq_calc."
        )
    granger_freq_max_eff = float(min(granger_freq_max_req, nyq_analysis * 0.999999))
    if granger_freq_max_eff <= granger_freq_min_req:
        raise ValueError(
            f"Invalid effective --granger_freq after Nyquist clamp: "
            f"[{granger_freq_min_req:g}, {granger_freq_max_eff:g}] Hz. "
            "Lower --granger_freq FMIN or raise --downsample-rate."
        )

    _, dataset_dir_str = initialization(args.folder)
    dataset_dir = Path(dataset_dir_str)
    items = build_work_items(dataset_dir)

    print(f"-- dataset: {dataset_dir.name}")
    print(f"-- subject-session EEG targets: {len(items)}")
    print(f"-- selected modes: {', '.join(modes)}")
    print(f"-- parallel workers (n_jobs): {n_jobs}")
    if apply_line_noise_removal:
        print(
            "-- HC preprocessing: harmonic line-noise notch "
            f"(line_hz={line_noise_hz:g}, Q={line_noise_q:g})"
        )
    else:
        print("-- HC preprocessing: harmonic line-noise notch disabled")
    if ms_lfp_enabled:
        d_mode = "explicit-map" if bool(ms_lfp_distance_map) else f"default-scalar({ms_lfp_d_default:g})"
        print(
            "-- MS render mode: pseudo-LFP "
            f"(overlay={ms_lfp_overlay}, sigma={ms_lfp_sigma:g}s, "
            f"A={ms_lfp_a:g}, A0={ms_lfp_a0:g}, "
            f"post_smooth={ms_lfp_post_smooth_sec:g}s, d_mode={d_mode})"
        )
    else:
        print("-- MS render mode: legacy spike raster")
    if down_cfg is None:
        print("-- preprocessing: AA lowpass + downsampling disabled (downsample-rate=0)")
    else:
        print(
            "-- preprocessing: AA lowpass + downsampling enabled "
            f"(in={float(args.sampling_rate):g}Hz -> out={analysis_sampling_rate:g}Hz, "
            f"AA cutoff={down_cfg.aa_cutoff_hz:g}Hz, ratio={down_cfg.up}/{down_cfg.down})"
        )
    if apply_line_noise_removal:
        print("-- preprocessing order: line-noise notch -> anti-alias/downsample -> z-score")
    else:
        print("-- preprocessing order: anti-alias/downsample -> z-score")
    if req_max_freq <= 0:
        print(f"-- PSD max frequency: auto -> Nyquist = {psd_max_freq:g} Hz")
    elif req_max_freq > psd_nyquist:
        print(
            f"-- PSD max frequency clamped: requested={req_max_freq:g} Hz, "
            f"Nyquist={psd_nyquist:g} Hz, effective={psd_max_freq:g} Hz"
        )
    else:
        print(f"-- PSD max frequency: {psd_max_freq:g} Hz")
    if psd_aperiodic_mode == "residual":
        assert psd_fit_range_hz is not None
        print(
            "-- PSD aperiodic mode: residual "
            f"(1/f fit range={psd_fit_range_hz[0]:g}-{psd_fit_range_hz[1]:g} Hz)"
        )
    else:
        print("-- PSD aperiodic mode: none")
    if "spectrogram" in modes:
        print(
            "-- spectrogram frequency: "
            f"plot={freq_plot_min_req:g}-{freq_plot_max_eff:g}Hz, "
            f"calc={freq_calc_min_req:g}-{freq_calc_max_eff:g}Hz, "
            f"normalize={args.spectrogram_normalize}, "
            f"aperiodic={args.spectrogram_aperiodic_mode}"
        )
    if "coherence" in modes:
        print(
            "-- coherence frequency: "
            f"plot={freq_plot_min_req:g}-{freq_plot_max_eff:g}Hz, "
            f"calc={freq_calc_min_req:g}-{freq_calc_max_eff:g}Hz"
        )
    if "granger" in modes:
        print(
            "-- granger gPDC: "
            f"freq={granger_freq_min_req:g}-{granger_freq_max_eff:g}Hz, "
            f"n_freqs={int(args.granger_n_freqs)}, "
            f"order_max={int(args.granger_order_max)}, "
            f"fixed_order={int(args.granger_fixed_order)}, "
            f"criterion={granger_order_criterion}, "
            f"order_mode={granger_order_mode}, "
            f"epoch_jobs={granger_epoch_jobs}, progress={granger_progress}"
        )
    if not items:
        return

    if args.output_html:
        output_html = Path(args.output_html).expanduser().resolve()
    else:
        output_html = dataset_dir / f"interactive_analysis_index_{dataset_dir.name}.html"

    pages_dir = output_html.parent / f"interactive_analysis_pages_{sanitize_token(dataset_dir.name)}"

    eeg_entries: list[EEGEntry] = []
    skipped_missing_key = 0
    skipped_invalid = 0
    ms_load_errors = 0
    need_spike_rows = any(m in TIME_SERIES_MODES for m in modes)

    for subject, session, hc_path, ms_path in tqdm(items, desc="Load EEG by subject/session"):
        if args.dry_run:
            print(
                f"[dry-run] subject={subject} session={session} "
                f"HC={hc_path.name} MS={ms_path.name if ms_path else '-'}"
            )
            continue

        eeg_loaded = matFileLoad(str(hc_path.parent), hc_path.name)
        try:
            eeg_mat = eeg_loaded["eegx"]
        except KeyError:
            keys = list(eeg_loaded.keys()) if isinstance(eeg_loaded, dict) else type(eeg_loaded).__name__
            print(
                f"\033[1;31m -- 'eegx' key not found in {hc_path.name}. "
                f"Available keys: {keys}\033[0m"
            )
            skipped_missing_key += 1
            continue

        eeg_vec = to_numeric_1d(eeg_mat)
        if apply_line_noise_removal:
            eeg_vec = _apply_harmonic_notch(
                eeg_vec,
                fs=float(args.sampling_rate),
                line_hz=float(line_noise_hz),
                q=float(line_noise_q),
            )
        eeg_vec = _apply_aa_downsample(eeg_vec, down_cfg)
        eeg_z = zscore_1d(eeg_vec)
        if eeg_z is None:
            skipped_invalid += 1
            print(
                f"\033[1;33m -- skipped invalid/constant EEG in {hc_path.name} "
                f"(subject={subject}, session={session})\033[0m"
            )
            continue

        ms_units: tuple[tuple[str, np.ndarray], ...] = tuple()
        if need_spike_rows and (ms_path is not None):
            try:
                loaded_units = load_ms_spike_units(ms_path)
                ms_units = tuple(
                    (str(unit_name), np.asarray(spike_idx, dtype=float).reshape(-1))
                    for unit_name, spike_idx in loaded_units
                )
            except Exception as exc:
                ms_load_errors += 1
                print(
                    f"\033[1;33m -- failed to load MS spikes: {ms_path.name} "
                    f"(subject={subject}, session={session}): {exc}\033[0m"
                )

        eeg_entries.append(
            EEGEntry(
                subject=subject,
                session=session,
                hc_name=hc_path.name,
                eeg_z=eeg_z,
                ms_units=ms_units,
            )
        )

    print(f"-- loaded EEG entries: {len(eeg_entries)}")
    print(f"-- skipped (missing eegx): {skipped_missing_key}")
    print(f"-- skipped (invalid EEG): {skipped_invalid}")
    print(f"-- MS spike load errors: {ms_load_errors}")

    if args.dry_run:
        print(f"-- dry-run complete (no HTML written): {output_html}")
        return

    if not eeg_entries:
        print("\033[1;31m -- no valid EEG entries to analyze; HTML not written.\033[0m")
        return

    # Analysis Phase
    mode_cells: dict[str, list[ModeCell]] = {}

    if "psd" in modes:
        mode_cells["psd"] = _build_psd_mode_cells(
            eeg_entries,
            sampling_rate=analysis_sampling_rate,
            spike_sampling_rate=float(args.sampling_rate),
            nperseg=int(args.psd_nperseg),
            max_freq=psd_max_freq,
            aperiodic_mode=psd_aperiodic_mode,
            fit_range_hz=psd_fit_range_hz,
            ms_lfp_sigma=float(ms_lfp_sigma),
            ms_lfp_a=float(ms_lfp_a),
            ms_lfp_a0=float(ms_lfp_a0),
            ms_lfp_distance_map=ms_lfp_distance_map,
            ms_lfp_d_default=float(ms_lfp_d_default),
            ms_lfp_post_smooth_sec=float(ms_lfp_post_smooth_sec),
            n_jobs=n_jobs,
        )

    if any(m in modes for m in ("signal", "hilbert")):
        assert ts_range is not None
        run_signal = "signal" in modes
        run_hilbert = "hilbert" in modes
        sig_cells, hil_cells = _build_signal_hilbert_mode_cells(
            eeg_entries=eeg_entries,
            sampling_rate=analysis_sampling_rate,
            spike_sampling_rate=float(args.sampling_rate),
            theta_bands=theta_bands,
            time_range=ts_range,
            max_points=int(args.interactive_max_points),
            apply_db=bool(args.apply_db),
            db_eps=float(args.db_eps),
            smooth_win_sec=smooth_win_sec,
            run_signal=run_signal,
            run_hilbert=run_hilbert,
            ms_lfp_sigma=float(ms_lfp_sigma),
            ms_lfp_a=float(ms_lfp_a),
            ms_lfp_a0=float(ms_lfp_a0),
            ms_lfp_distance_map=ms_lfp_distance_map,
            ms_lfp_d_default=float(ms_lfp_d_default),
            ms_lfp_post_smooth_sec=float(ms_lfp_post_smooth_sec),
            n_jobs=n_jobs,
        )
        if run_signal:
            mode_cells["signal"] = sig_cells
        if run_hilbert:
            mode_cells["hilbert"] = hil_cells

    if any(m in modes for m in THETA_FAMILY_MODES):
        assert ts_range is not None
        theta_active_modes = [m for m in modes if m in THETA_FAMILY_MODES]
        theta_cells = _build_theta_family_mode_cells(
            eeg_entries=eeg_entries,
            sampling_rate=analysis_sampling_rate,
            spike_sampling_rate=float(args.sampling_rate),
            theta_bands=theta_bands,
            time_range=ts_range,
            tf_win_sec=float(args.tf_win_sec),
            tf_step_sec=float(args.tf_step_sec),
            max_points=int(args.interactive_max_points),
            apply_db=bool(args.apply_db),
            db_eps=float(args.db_eps),
            smooth_win_sec=smooth_win_sec,
            active_modes=theta_active_modes,
            ms_lfp_sigma=float(ms_lfp_sigma),
            ms_lfp_a=float(ms_lfp_a),
            ms_lfp_a0=float(ms_lfp_a0),
            ms_lfp_distance_map=ms_lfp_distance_map,
            ms_lfp_d_default=float(ms_lfp_d_default),
            ms_lfp_post_smooth_sec=float(ms_lfp_post_smooth_sec),
            n_jobs=n_jobs,
        )
        mode_cells.update(theta_cells)

    if "spectrogram" in modes:
        assert ts_range is not None
        mode_cells["spectrogram"] = _build_spectrogram_mode_cells(
            eeg_entries=eeg_entries,
            analysis_sampling_rate=analysis_sampling_rate,
            spike_sampling_rate=float(args.sampling_rate),
            time_range=ts_range,
            tf_win_sec=float(args.tf_win_sec),
            tf_step_sec=float(args.tf_step_sec),
            max_points=int(args.interactive_max_points),
            fmin_plot=float(freq_plot_min_req),
            fmax_plot=float(freq_plot_max_eff),
            fmin_calc=float(freq_calc_min_req),
            fmax_calc=float(freq_calc_max_eff),
            normalize=str(args.spectrogram_normalize),
            aperiodic_mode=str(args.spectrogram_aperiodic_mode),
            ms_lfp_sigma=float(ms_lfp_sigma),
            ms_lfp_a=float(ms_lfp_a),
            ms_lfp_a0=float(ms_lfp_a0),
            ms_lfp_distance_map=ms_lfp_distance_map,
            ms_lfp_d_default=float(ms_lfp_d_default),
            ms_lfp_post_smooth_sec=float(ms_lfp_post_smooth_sec),
            n_jobs=n_jobs,
        )

    if "coherence_band" in modes:
        assert ts_range is not None
        mode_cells["coherence_band"] = _build_coherence_band_mode_cells(
            eeg_entries=eeg_entries,
            analysis_sampling_rate=analysis_sampling_rate,
            spike_sampling_rate=float(args.sampling_rate),
            theta_bands=theta_bands,
            time_range=ts_range,
            tf_win_sec=float(args.tf_win_sec),
            tf_step_sec=float(args.tf_step_sec),
            max_points=int(args.interactive_max_points),
            smooth_win_sec=smooth_win_sec,
            ms_lfp_sigma=float(ms_lfp_sigma),
            ms_lfp_a=float(ms_lfp_a),
            ms_lfp_a0=float(ms_lfp_a0),
            ms_lfp_distance_map=ms_lfp_distance_map,
            ms_lfp_d_default=float(ms_lfp_d_default),
            ms_lfp_post_smooth_sec=float(ms_lfp_post_smooth_sec),
            n_jobs=n_jobs,
        )

    if "syncfc" in modes:
        print(
            "-- SyncFC: "
            f"null={syncfc_null}, "
            f"n_null={int(args.syncfc_n_surrogates)}, "
            f"min_shift_sec={float(args.syncfc_min_shift_sec):g}"
        )
        assert ts_range is not None
        mode_cells["syncfc"] = _build_syncfc_mode_cells(
            eeg_entries=eeg_entries,
            analysis_sampling_rate=analysis_sampling_rate,
            spike_sampling_rate=float(args.sampling_rate),
            theta_bands=theta_bands,
            time_range=ts_range,
            tf_win_sec=float(args.tf_win_sec),
            tf_step_sec=float(args.tf_step_sec),
            max_points=int(args.interactive_max_points),
            smooth_win_sec=smooth_win_sec,
            ms_lfp_sigma=float(ms_lfp_sigma),
            ms_lfp_a=float(ms_lfp_a),
            ms_lfp_a0=float(ms_lfp_a0),
            ms_lfp_distance_map=ms_lfp_distance_map,
            ms_lfp_d_default=float(ms_lfp_d_default),
            ms_lfp_post_smooth_sec=float(ms_lfp_post_smooth_sec),
            syncfc_n_surrogates=int(args.syncfc_n_surrogates),
            syncfc_null=syncfc_null,
            syncfc_min_shift_sec=float(args.syncfc_min_shift_sec),
            syncfc_seed=int(args.syncfc_seed),
            n_jobs=n_jobs,
        )

    if "phaselag" in modes:
        print(
            "-- PhaseLag filters: "
            f"min_z={float(args.phaselag_min_z):g}, "
            f"min_plv={float(args.phaselag_min_plv):g}, "
            f"min_peak_delta_z={float(args.phaselag_min_peak_delta_z):g}, "
            f"min_peak_delta_frac={float(args.phaselag_min_peak_delta_frac):g}, "
            f"min_valid_ratio={float(args.phaselag_min_valid_ratio):g}, "
            f"exclude_edge_peaks={not bool(args.phaselag_include_edge_peaks)}, "
            f"edge_guard_bins={int(args.phaselag_edge_guard_bins)}, "
            "score=rayleigh_z"
        )
        assert ts_range is not None
        mode_cells["phaselag"] = _build_phaselag_mode_cells(
            eeg_entries=eeg_entries,
            analysis_sampling_rate=analysis_sampling_rate,
            spike_sampling_rate=float(args.sampling_rate),
            theta_bands=theta_bands,
            time_range=ts_range,
            tf_win_sec=float(args.tf_win_sec),
            tf_step_sec=float(args.tf_step_sec),
            ms_lfp_sigma=float(ms_lfp_sigma),
            ms_lfp_a=float(ms_lfp_a),
            ms_lfp_a0=float(ms_lfp_a0),
            ms_lfp_distance_map=ms_lfp_distance_map,
            ms_lfp_d_default=float(ms_lfp_d_default),
            ms_lfp_post_smooth_sec=float(ms_lfp_post_smooth_sec),
            phaselag_min_z=float(args.phaselag_min_z),
            phaselag_min_plv=float(args.phaselag_min_plv),
            phaselag_min_peak_delta_z=float(args.phaselag_min_peak_delta_z),
            phaselag_min_peak_delta_frac=float(args.phaselag_min_peak_delta_frac),
            phaselag_min_valid_ratio=float(args.phaselag_min_valid_ratio),
            phaselag_exclude_edge_peaks=(not bool(args.phaselag_include_edge_peaks)),
            phaselag_edge_guard_bins=int(args.phaselag_edge_guard_bins),
            n_jobs=n_jobs,
        )

    if "pearson" in modes:
        assert ts_range is not None
        mode_cells["pearson"] = _build_pearson_mode_cells(
            eeg_entries=eeg_entries,
            analysis_sampling_rate=analysis_sampling_rate,
            spike_sampling_rate=float(args.sampling_rate),
            theta_bands=theta_bands,
            time_range=ts_range,
            tf_win_sec=float(args.tf_win_sec),
            tf_step_sec=float(args.tf_step_sec),
            max_points=int(args.interactive_max_points),
            smooth_win_sec=smooth_win_sec,
            ms_lfp_sigma=float(ms_lfp_sigma),
            ms_lfp_a=float(ms_lfp_a),
            ms_lfp_a0=float(ms_lfp_a0),
            ms_lfp_distance_map=ms_lfp_distance_map,
            ms_lfp_d_default=float(ms_lfp_d_default),
            ms_lfp_post_smooth_sec=float(ms_lfp_post_smooth_sec),
            n_jobs=n_jobs,
        )

    if "coherence" in modes:
        assert ts_range is not None
        mode_cells["coherence"] = _build_coherence_mode_cells(
            eeg_entries=eeg_entries,
            analysis_sampling_rate=analysis_sampling_rate,
            spike_sampling_rate=float(args.sampling_rate),
            time_range=ts_range,
            tf_win_sec=float(args.tf_win_sec),
            tf_step_sec=float(args.tf_step_sec),
            max_points=int(args.interactive_max_points),
            smooth_win_sec=smooth_win_sec,
            fmin_plot=float(freq_plot_min_req),
            fmax_plot=float(freq_plot_max_eff),
            fmin_calc=float(freq_calc_min_req),
            fmax_calc=float(freq_calc_max_eff),
            ms_lfp_sigma=float(ms_lfp_sigma),
            ms_lfp_a=float(ms_lfp_a),
            ms_lfp_a0=float(ms_lfp_a0),
            ms_lfp_distance_map=ms_lfp_distance_map,
            ms_lfp_d_default=float(ms_lfp_d_default),
            ms_lfp_post_smooth_sec=float(ms_lfp_post_smooth_sec),
            n_jobs=n_jobs,
        )

    if "granger" in modes:
        assert ts_range is not None
        mode_cells["granger"] = _build_granger_mode_cells(
            eeg_entries=eeg_entries,
            analysis_sampling_rate=analysis_sampling_rate,
            spike_sampling_rate=float(args.sampling_rate),
            granger_order_mode=granger_order_mode,
            granger_fixed_order=int(args.granger_fixed_order),
            time_range=ts_range,
            tf_win_sec=float(args.tf_win_sec),
            tf_step_sec=float(args.tf_step_sec),
            granger_order_max=int(args.granger_order_max),
            granger_order_criterion=granger_order_criterion,
            granger_fmin=float(granger_freq_min_req),
            granger_epoch_jobs=int(granger_epoch_jobs),
            granger_progress=granger_progress,
            granger_fmax=float(granger_freq_max_eff),
            granger_n_freqs=int(args.granger_n_freqs),
            ms_lfp_sigma=float(ms_lfp_sigma),
            ms_lfp_a=float(ms_lfp_a),
            ms_lfp_a0=float(ms_lfp_a0),
            ms_lfp_distance_map=ms_lfp_distance_map,
            ms_lfp_d_default=float(ms_lfp_d_default),
            ms_lfp_post_smooth_sec=float(ms_lfp_post_smooth_sec),
            n_jobs=n_jobs,
        )

    mode_cell_lookup: dict[str, dict[tuple[str, str], ModeCell]] = {}
    for mode_name, cells in mode_cells.items():
        mode_cell_lookup[mode_name] = {
            (cell.subject, cell.session): cell
            for cell in cells
        }

    subject_order = sorted({e.subject for e in eeg_entries}, key=natural_key)
    sessions_by_subject: dict[str, list[str]] = {}
    ms_units_by_key: dict[tuple[str, str], tuple[tuple[str, np.ndarray], ...]] = {}
    for entry in eeg_entries:
        sessions_by_subject.setdefault(entry.subject, []).append(entry.session)
        ms_units_by_key[(entry.subject, entry.session)] = entry.ms_units
    for subject in subject_order:
        uniq_sessions = sorted(set(sessions_by_subject.get(subject, [])), key=natural_key)
        sessions_by_subject[subject] = uniq_sessions

    mode_pages: list[ModePage] = []
    mode_failures: list[str] = []
    for subject in subject_order:
        subject_sessions = sessions_by_subject.get(subject, [])
        for mode in modes:
            lookup = mode_cell_lookup.get(mode, {})
            subject_cells: list[ModeCell] = []
            for session in subject_sessions:
                cell = lookup.get((subject, session))
                if cell is not None:
                    subject_cells.append(cell)

            if not subject_cells:
                continue

            try:
                x_title, y_title = _mode_axis_titles(
                    mode,
                    apply_db=bool(args.apply_db),
                    psd_aperiodic_mode=psd_aperiodic_mode,
                )
                ms_by_session: dict[str, tuple[tuple[str, np.ndarray], ...]] | None = None
                if mode in TIME_SERIES_MODES:
                    ms_by_session = {
                        session: ms_units_by_key.get((subject, session), tuple())
                        for session in subject_sessions
                    }
                if mode == "coherence":
                    fig = _build_subject_coherence_figure(
                        subject=subject,
                        mode_cells=subject_cells,
                        x_title=x_title,
                        y_title=y_title,
                    )
                elif mode == "spectrogram":
                    fig = _build_subject_spectrogram_figure(
                        subject=subject,
                        mode_cells=subject_cells,
                        x_title=x_title,
                        y_title=y_title,
                    )
                elif mode == "granger":
                    fig = _build_subject_granger_figure(
                        subject=subject,
                        mode_cells=subject_cells,
                        x_title=x_title,
                        y_title=y_title,
                    )
                elif mode == "phaselag":
                    fig = _build_subject_phaselag_figure(
                        subject=subject,
                        mode_cells=subject_cells,
                    )
                elif mode == "syncfc":
                    fig = _build_subject_syncfc_figure(
                        subject=subject,
                        mode_cells=subject_cells,
                    )
                else:
                    fig = _build_subject_mode_figure(
                        mode=mode,
                        subject=subject,
                        mode_cells=subject_cells,
                        x_title=x_title,
                        y_title=y_title,
                        use_timeseries_outlier_rejection=(mode in TIME_SERIES_MODES),
                        ms_units_by_session=ms_by_session,
                        spike_sampling_rate=float(args.sampling_rate),
                        ms_lfp_enabled=bool(ms_lfp_enabled),
                        ms_lfp_overlay=bool(ms_lfp_overlay),
                        ms_lfp_sigma=float(ms_lfp_sigma),
                        ms_lfp_a=float(ms_lfp_a),
                        ms_lfp_a0=float(ms_lfp_a0),
                        ms_lfp_distance_map=ms_lfp_distance_map,
                        ms_lfp_d_default=float(ms_lfp_d_default),
                        ms_lfp_post_smooth_sec=float(ms_lfp_post_smooth_sec),
                        ms_lfp_max_points=int(args.interactive_max_points),
                    )
                child_path = pages_dir / f"{sanitize_token(subject)}__{sanitize_token(mode)}.html"
                write_plotly_html(fig, out_path=child_path, include_plotlyjs=include_plotlyjs)
                rel = child_path.relative_to(output_html.parent).as_posix()
                mode_pages.append(
                    ModePage(
                        subject=subject,
                        mode=mode,
                        mode_title=MODE_TITLE_MAP.get(mode, mode),
                        href=rel,
                    )
                )
                print(f"-- saved subject/mode page: {subject} | {mode} -> {child_path}")
            except Exception as exc:
                msg = f"{subject}/{mode}: {exc}"
                mode_failures.append(msg)
                print(f"\033[1;33m -- subject/mode skipped ({msg})\033[0m")

    if mode_failures:
        print("-- mode failures:")
        for m in mode_failures:
            print(f"   - {m}")

    if "granger" in modes and mode_cells.get("granger"):
        raw_stats_pdf = str(args.granger_stats_pdf).strip()
        if raw_stats_pdf.lower() not in {"", "none", "false", "off", "0"}:
            if raw_stats_pdf.upper() == "AUTO":
                granger_stats_pdf = output_html.parent / f"granger_stats_{sanitize_token(dataset_dir.name)}.pdf"
            else:
                granger_stats_pdf = Path(raw_stats_pdf).expanduser().resolve()
            try:
                _write_granger_stats_pdf(
                    mode_cells=mode_cells["granger"],
                    bands=granger_stats_bands,
                    out_path=granger_stats_pdf,
                    run_name=dataset_dir.name,
                )
            except Exception as exc:
                print(f"\033[1;33m -- Granger stats PDF skipped: {exc}\033[0m")

    if "phaselag" in modes and mode_cells.get("phaselag"):
        raw_stats_pdf = str(args.phaselag_stats_pdf).strip()
        if raw_stats_pdf.lower() not in {"", "none", "false", "off", "0"}:
            if raw_stats_pdf.upper() == "AUTO":
                phaselag_stats_pdf = output_html.parent / f"phaselag_stats_{sanitize_token(dataset_dir.name)}.pdf"
            else:
                phaselag_stats_pdf = Path(raw_stats_pdf).expanduser().resolve()
            try:
                _write_phaselag_stats_pdf(
                    mode_cells=mode_cells["phaselag"],
                    out_path=phaselag_stats_pdf,
                    run_name=dataset_dir.name,
                )
            except Exception as exc:
                print(f"\033[1;33m -- PhaseLag stats PDF skipped: {exc}\033[0m")

    if "syncfc" in modes and mode_cells.get("syncfc"):
        raw_stats_pdf = str(args.syncfc_stats_pdf).strip()
        if raw_stats_pdf.lower() not in {"", "none", "false", "off", "0"}:
            if raw_stats_pdf.upper() == "AUTO":
                syncfc_stats_pdf = output_html.parent / f"syncfc_stats_{sanitize_token(dataset_dir.name)}.pdf"
            else:
                syncfc_stats_pdf = Path(raw_stats_pdf).expanduser().resolve()
            try:
                _write_syncfc_stats_pdf(
                    mode_cells=mode_cells["syncfc"],
                    out_path=syncfc_stats_pdf,
                    run_name=dataset_dir.name,
                )
            except Exception as exc:
                print(f"\033[1;33m -- SyncFC stats PDF skipped: {exc}\033[0m")

    raw_stats_html = str(getattr(args, "stats_html", "AUTO")).strip()
    if raw_stats_html.lower() not in {"", "none", "false", "off", "0"}:
        if raw_stats_html.upper() == "AUTO":
            stats_html = output_html.parent / f"stats_summary_{sanitize_token(dataset_dir.name)}.html"
        else:
            stats_html = Path(raw_stats_html).expanduser().resolve()
        stats_pages_dir = stats_html.parent / f"stats_summary_pages_{sanitize_token(dataset_dir.name)}"
        stats_pages: list[StatsPage] = []

        def add_stats_page(key: str, title: str, fig: Any) -> None:
            child_path = stats_pages_dir / f"{sanitize_token(key)}.html"
            write_plotly_html(fig, out_path=child_path, include_plotlyjs=include_plotlyjs)
            rel = child_path.relative_to(stats_html.parent).as_posix()
            stats_pages.append(StatsPage(key=key, title=title, href=rel))
            print(f"-- saved stats page: {title} -> {child_path}")

        if "granger" in modes and mode_cells.get("granger"):
            try:
                add_stats_page(
                    "granger",
                    "Granger",
                    _build_granger_stats_figure(
                        mode_cells=mode_cells["granger"],
                        bands=granger_stats_bands,
                        run_name=dataset_dir.name,
                    ),
                )
            except Exception as exc:
                print(f"\033[1;33m -- Granger stats HTML skipped: {exc}\033[0m")
        if "phaselag" in modes and mode_cells.get("phaselag"):
            try:
                add_stats_page(
                    "phaselag",
                    "PhaseLag",
                    _build_phaselag_stats_figure(
                        mode_cells=mode_cells["phaselag"],
                        run_name=dataset_dir.name,
                    ),
                )
            except Exception as exc:
                print(f"\033[1;33m -- PhaseLag stats HTML skipped: {exc}\033[0m")
        if "syncfc" in modes and mode_cells.get("syncfc"):
            try:
                add_stats_page(
                    "syncfc",
                    "SyncFC",
                    _build_syncfc_stats_figure(
                        mode_cells=mode_cells["syncfc"],
                        run_name=dataset_dir.name,
                    ),
                )
            except Exception as exc:
                print(f"\033[1;33m -- SyncFC stats HTML skipped: {exc}\033[0m")

        if stats_pages:
            stats_html.parent.mkdir(parents=True, exist_ok=True)
            stats_html.write_text(
                _build_stats_index_html(stats_pages=stats_pages, run_name=dataset_dir.name),
                encoding="utf-8",
            )
            print(f"-- saved group stats HTML: {stats_html}")
        else:
            print("\033[1;33m -- group stats HTML skipped: no plottable stats pages.\033[0m")

    if not mode_pages:
        print("\033[1;31m -- no mode pages generated; parent HTML not written.\033[0m")
        return

    parent_html = _build_mode_index_html(
        mode_pages=mode_pages,
        run_name=dataset_dir.name,
        subject_order=subject_order,
        mode_order=modes,
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(parent_html, encoding="utf-8")
    print(f"-- saved parent mode index HTML: {output_html}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main_code(args)
