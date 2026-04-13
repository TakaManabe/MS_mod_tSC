#!/usr/bin/env python3
"""
Temporary ds003374 pipeline:
- Inspect BIDS + NIX layout.
- Build pseudo-LFP from sorted curated spike units in NIX.
- Compare pseudo-LFP vs recorded LFP with Plotly using main_PreP rendering logic.

This script does NOT modify python_tools/main_PreP.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    from scipy.signal import filtfilt, iirnotch
except Exception:
    filtfilt = None
    iirnotch = None
try:
    from scipy import signal as sp_signal
except Exception:
    sp_signal = None

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
PYTHON_TOOLS = REPO_ROOT / "python_tools"
if str(PYTHON_TOOLS) not in sys.path:
    sys.path.insert(0, str(PYTHON_TOOLS))

from main_PreP import PLOTLY_COLORS, write_plotly_html  # noqa: E402
from ms_lfp_transform import parse_unit_name, synthesize_shank_lfp  # noqa: E402

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as exc:  # pragma: no cover
    raise RuntimeError("plotly is required for HTML output.") from exc


NIX_FILE_RE = re.compile(r"^Data_Subject_(?P<subject>\d+)_Session_(?P<session>\d+)\.h5$", re.IGNORECASE)
IEEG_TRIAL_RE = re.compile(r"^iEEG_Data_Trial_(?P<trial>\d+)$", re.IGNORECASE)
SPIKE_ARRAY_RE = re.compile(
    r"^Spike_Times_Unit_(?P<unit>\d+)_(?P<wireprefix>[A-Za-z]+)_(?P<wirenum>\d+)_Trial_(?P<trial>\d+)$",
    re.IGNORECASE,
)
THETA_BAND_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*$")
DEFAULT_THETA_BAND_TOKENS = [
    "1-4",
    "4-6",
    "4-8",
    "4-10",
    "4-12",
    "6-8",
    "6-10",
    "6-12",
    "8-10",
    "8-12",
    "10-12",
]


@dataclass
class SubjectRun:
    subject: str
    session: str
    fs: float
    trial_count: int
    channel_labels: list[str]
    time_sec: np.ndarray
    channel_data: dict[str, np.ndarray]
    ms_units: tuple[tuple[str, np.ndarray], ...]
    unit_inventory_rows: list[dict[str, Any]]
    source_inventory_rows: list[dict[str, Any]]


def _natural_key(text: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(text))
    out: list[Any] = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p.lower())
    return tuple(out)


def _decode_text(x: Any) -> str:
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        try:
            return bytes(x).decode("utf-8")
        except Exception:
            return str(x)
    return str(x)


def _parse_bool(text: str) -> bool:
    s = str(text).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {text}")


def _passes_anatomy_filter(anatomy: str, anatomy_substring: str | None) -> bool:
    if anatomy_substring is None:
        return True
    anat = str(anatomy or "").strip().lower()
    if not anat:
        return True
    return str(anatomy_substring).strip().lower() in anat


def _linspace_idx(n: int, m: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.int64)
    if m <= 0 or n <= m:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, num=m, dtype=np.int64)


def _macro_channel_to_shank(label: str) -> int | None:
    low = str(label).strip().lower()
    if "mal" in low:
        return 1
    if "mar" in low:
        return 2
    return None


def _clean_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    xx = np.asarray(x, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    n = min(xx.size, yy.size)
    if n < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    xx = xx[:n]
    yy = yy[:n]
    m = np.isfinite(xx) & np.isfinite(yy)
    if int(np.sum(m)) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    xx = xx[m]
    yy = yy[m]
    if max_points > 0 and xx.size > max_points:
        idx = _linspace_idx(xx.size, max_points)
        xx = xx[idx]
        yy = yy[idx]
    return xx, yy


def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    xx = np.asarray(x, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    n = min(xx.size, yy.size)
    if n < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    xx = xx[:n]
    yy = yy[:n]
    if max_points > 0 and xx.size > max_points:
        idx = _linspace_idx(xx.size, max_points)
        xx = xx[idx]
        yy = yy[idx]
    return xx, yy


def _zscore_1d(y: np.ndarray) -> np.ndarray:
    yy = np.asarray(y, dtype=float).reshape(-1)
    if yy.size == 0:
        return yy
    mu = float(np.mean(yy))
    sd = float(np.std(yy, ddof=0))
    if (not np.isfinite(sd)) or sd <= 0:
        return yy - mu
    return (yy - mu) / sd


def _apply_harmonic_notch(y: np.ndarray, fs: float, line_hz: float, q: float) -> np.ndarray:
    yy = np.asarray(y, dtype=float).reshape(-1)
    if yy.size < 16:
        return yy
    if (not np.isfinite(fs)) or fs <= 0:
        return yy
    if (not np.isfinite(line_hz)) or line_hz <= 0:
        return yy
    if (not np.isfinite(q)) or q <= 0:
        return yy
    if iirnotch is None or filtfilt is None:
        return yy

    nyq = 0.5 * float(fs)
    out = yy.copy()
    f = float(line_hz)
    while f < (nyq - 1e-9):
        w0 = f / nyq
        if not (0.0 < w0 < 1.0):
            break
        try:
            b, a = iirnotch(w0=float(w0), Q=float(q))
            out = filtfilt(b, a, out)
        except Exception:
            break
        f += float(line_hz)
    return out


def _compute_psd_fullrange(y: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    yy = np.asarray(y, dtype=float).reshape(-1)
    yy = yy[np.isfinite(yy)]
    if yy.size < 8:
        return np.array([], dtype=float), np.array([], dtype=float)
    if (not np.isfinite(fs)) or fs <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    yy = yy - float(np.mean(yy))
    n = int(yy.size)

    # Full-range periodogram (one FFT over the entire time span).
    win = np.hanning(n).astype(float)
    u = float(np.sum(win * win))
    if (not np.isfinite(u)) or u <= 0:
        win = np.ones(n, dtype=float)
        u = float(n)

    y_fft = np.fft.rfft(yy * win)
    psd = (np.abs(y_fft) ** 2) / (float(fs) * u)
    if n % 2 == 0:
        if psd.size > 2:
            psd[1:-1] *= 2.0
    else:
        if psd.size > 1:
            psd[1:] *= 2.0

    freq = np.fft.rfftfreq(n, d=(1.0 / float(fs)))
    return np.asarray(freq, dtype=float), np.asarray(psd, dtype=float)


def _fmt_freq(x: float) -> str:
    if abs(float(x) - round(float(x))) < 1e-9:
        return str(int(round(float(x))))
    return f"{float(x):g}"


def _parse_theta_bands(tokens: list[str] | tuple[str, ...]) -> list[tuple[str, tuple[float, float]]]:
    if not tokens:
        raise ValueError("--POWER_TIME_THETA_BANDS must not be empty.")
    out: list[tuple[str, tuple[float, float]]] = []
    seen: set[tuple[float, float]] = set()
    flat_vals: list[float] = []

    for tok in tokens:
        s = str(tok).strip()
        if not s:
            continue
        m = THETA_BAND_RE.match(s)
        if m is not None:
            lo = float(m.group(1))
            hi = float(m.group(2))
            if lo >= hi or lo < 0:
                raise ValueError(f"Invalid theta band: {s}")
            key = (lo, hi)
            if key not in seen:
                seen.add(key)
                out.append((f"{_fmt_freq(lo)}-{_fmt_freq(hi)}", (lo, hi)))
            continue

        for part in s.replace(",", " ").split():
            flat_vals.append(float(part))

    if flat_vals:
        if len(flat_vals) % 2 != 0:
            raise ValueError("Invalid flat POWER_TIME_THETA_BANDS list; must be low/high pairs.")
        for i in range(0, len(flat_vals), 2):
            lo = float(flat_vals[i])
            hi = float(flat_vals[i + 1])
            if lo >= hi or lo < 0:
                raise ValueError(f"Invalid theta band pair: {lo} {hi}")
            key = (lo, hi)
            if key in seen:
                continue
            seen.add(key)
            out.append((f"{_fmt_freq(lo)}-{_fmt_freq(hi)}", (lo, hi)))

    if not out:
        raise ValueError("No valid POWER_TIME_THETA_BANDS provided.")
    return out


def _parse_smooth_window_sec(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.lower() in {"none", "off", "disable", "disabled"}:
        return None
    v = float(s)
    if (not np.isfinite(v)) or v <= 0:
        return None
    return float(v)


def _causal_smooth_series(y: np.ndarray, win_points: int) -> np.ndarray:
    yy = np.asarray(y, dtype=float).reshape(-1)
    if yy.size == 0 or int(win_points) <= 1:
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
    yy = np.asarray(y, dtype=float).reshape(-1)
    tt = np.asarray(t, dtype=float).reshape(-1)
    if smooth_win_sec is None:
        return yy
    if yy.size != tt.size or yy.size < 2:
        return yy
    dt = np.diff(tt)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return yy
    step = float(np.median(dt))
    if (not np.isfinite(step)) or step <= 0:
        return yy
    win_points = int(np.ceil(float(smooth_win_sec) / step))
    return _causal_smooth_series(yy, win_points=win_points)


def _db_transform(y: np.ndarray, eps: float) -> np.ndarray:
    yy = np.asarray(y, dtype=float).reshape(-1)
    return 10.0 * np.log10(np.maximum(yy, float(eps)))


def _require_mne_multitaper() -> Any | None:
    try:
        from mne.time_frequency import psd_array_multitaper
    except Exception as exc:
        print(
            f"\033[1;33m -- [PowerTime] mne not found ({exc}); "
            "falling back to Welch-window mode.\033[0m"
        )
        return None
    return psd_array_multitaper


def _compute_multitaper_psd_windows(
    sig: np.ndarray,
    sr: float,
    win_sec: float,
    step_sec: float,
    fmin: float,
    fmax: float,
    psd_array_multitaper: Any,
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

    if psd_array_multitaper is not None:
        try:
            psd, freqs = psd_array_multitaper(
                segs,
                sfreq=float(sr),
                fmin=float(fmin),
                fmax=float(fmax),
                normalization="full",
                n_jobs=1,
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

        p = np.asarray(psd, dtype=float)
        if p.ndim == 3:
            p = p[:, 0, :]
        if p.ndim != 2:
            return None

        f = np.asarray(freqs, dtype=float).reshape(-1)
        if f.size == 0:
            return None
    else:
        if sp_signal is None:
            return None
        p_list: list[np.ndarray] = []
        f_sel: np.ndarray | None = None
        nperseg = int(win_samples)
        noverlap = int(max(0, nperseg // 2))
        for seg in segs:
            f_w, p_w = sp_signal.welch(
                np.asarray(seg, dtype=float),
                fs=float(sr),
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
            )
            f_w = np.asarray(f_w, dtype=float).reshape(-1)
            p_w = np.asarray(p_w, dtype=float).reshape(-1)
            m = (f_w >= float(fmin)) & (f_w <= float(fmax))
            if not np.any(m):
                continue
            if f_sel is None:
                f_sel = np.asarray(f_w[m], dtype=float)
            p_list.append(np.asarray(p_w[m], dtype=float))
        if not p_list or f_sel is None:
            return None
        p = np.vstack(p_list)
        f = f_sel

    t_right = (starts.astype(float) + float(win_samples)) / float(sr)
    return f, p, t_right


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


def _compute_theta_power_timeseries(
    y: np.ndarray,
    fs: float,
    theta_bands: list[tuple[str, tuple[float, float]]],
    tf_win_sec: float,
    tf_step_sec: float,
    apply_db: bool,
    db_eps: float,
    smooth_win_sec: float | None,
    max_points: int,
    psd_array_multitaper: Any,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    yy = np.asarray(y, dtype=float).reshape(-1)
    yy = yy[np.isfinite(yy)]
    if yy.size < 8:
        return {}

    fmin_all = min(float(lo) for _, (lo, _) in theta_bands)
    fmax_all = max(float(hi) for _, (_, hi) in theta_bands)
    mt = _compute_multitaper_psd_windows(
        sig=yy,
        sr=float(fs),
        win_sec=float(tf_win_sec),
        step_sec=float(tf_step_sec),
        fmin=float(fmin_all),
        fmax=float(fmax_all),
        psd_array_multitaper=psd_array_multitaper,
    )
    if mt is None:
        return {}

    freqs_mt, psd_mt, t_right = mt
    if t_right.size < 2:
        return {}

    band_power = _compute_theta_band_power_from_psd(freqs_mt, psd_mt, theta_bands)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, _ in theta_bands:
        arr = band_power.get(label)
        if arr is None:
            continue
        yv = np.asarray(arr, dtype=float).reshape(-1)
        if yv.size != t_right.size:
            continue
        if bool(apply_db):
            yv = _db_transform(yv, eps=float(db_eps))
        yv = _smooth_by_time(yv, t_right, smooth_win_sec=smooth_win_sec)
        xx, yy_ds = _downsample_xy(np.asarray(t_right, dtype=float), yv, max_points=max_points)
        if xx.size < 2:
            continue
        out[label] = (xx, yy_ds)
    return out


def _load_powerline_frequency(
    ds_root: Path,
    subject: str,
    session: str,
) -> tuple[float | None, str]:
    ieeg_dir = ds_root / subject / session / "ieeg"
    if not ieeg_dir.is_dir():
        return None, ""
    json_files = sorted(ieeg_dir.glob("*_ieeg.json"), key=lambda p: p.name)
    if not json_files:
        return None, ""
    meta_path = json_files[0]
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None, meta_path.name
    raw = meta.get("PowerLineFrequency", None)
    try:
        hz = float(raw)
    except Exception:
        return None, meta_path.name
    if (not np.isfinite(hz)) or hz <= 0:
        return None, meta_path.name
    return float(hz), meta_path.name


def _extract_nix_source_index(base: h5py.Group) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    """Return source map by UUID and a flat inventory table."""
    out: dict[str, dict[str, str]] = {}
    rows: list[dict[str, str]] = []

    src_root = base["sources"]
    for src_key in src_root.keys():
        src = src_root[src_key]
        src_name = _decode_text(src.attrs.get("name", src_key))
        src_uuid = _decode_text(src.attrs.get("entity_id", src_key))

        macro_label = ""
        anatomy_label = ""
        soz_label = ""

        nested = src.get("sources", None)
        if isinstance(nested, h5py.Group):
            for nk in nested.keys():
                nobj = nested[nk]
                nname = _decode_text(nobj.attrs.get("name", nk))
                nlow = nname.lower()
                if ("-" in nname) and nname.startswith("m"):
                    macro_label = nname
                elif "amyg" in nlow or "hipp" in nlow:
                    anatomy_label = nname
                elif "soz" in nname.upper():
                    soz_label = nname

        meta = {
            "source_key": str(src_key),
            "source_uuid": src_uuid,
            "source_name": src_name,
            "macro_label": macro_label,
            "anatomy": anatomy_label,
            "soz": soz_label,
        }
        out[str(src_key)] = meta
        out[str(src_uuid)] = meta
        rows.append(meta)

    return out, rows


def _canonical_shank_from_wire(wire_prefix: str, fallback_map: dict[str, int]) -> int:
    up = str(wire_prefix).upper()
    if "AL" in up:
        return 1
    if "AR" in up:
        return 2
    if up not in fallback_map:
        fallback_map[up] = max([2, *fallback_map.values()]) + 1
    return int(fallback_map[up])


def _extract_subject_run(
    nix_path: Path,
    trial_gap_sec: float = 0.0,
    anatomy_substring: str | None = None,
) -> SubjectRun:
    m = NIX_FILE_RE.match(nix_path.name)
    if m is None:
        raise ValueError(f"Unexpected NIX filename format: {nix_path.name}")

    subject = f"sub-{int(m.group('subject')):02d}"
    session = f"ses-{int(m.group('session')):02d}"

    with h5py.File(nix_path, "r") as f:
        base = f[f"/data/{nix_path.stem}"]
        data_arrays = base["data_arrays"]

        source_index, source_rows = _extract_nix_source_index(base)

        trial_names = [k for k in data_arrays.keys() if IEEG_TRIAL_RE.match(k)]
        if not trial_names:
            raise RuntimeError(f"No iEEG trials found in {nix_path.name}")
        trial_names = sorted(trial_names, key=_natural_key)

        labels_ref: list[str] | None = None
        fs_ref: float | None = None

        trial_offsets: dict[int, float] = {}
        trial_t0: dict[int, float] = {}
        t_segments: list[np.ndarray] = []
        channel_segments: dict[str, list[np.ndarray]] = {}

        running_offset = 0.0

        for trial_name in trial_names:
            tm = IEEG_TRIAL_RE.match(trial_name)
            if tm is None:
                continue
            trial_no = int(tm.group("trial"))

            tg = data_arrays[trial_name]
            arr = np.asarray(tg["data"][()], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2:
                raise RuntimeError(
                    f"Unexpected iEEG shape in {nix_path.name}::{trial_name}: {arr.shape}"
                )

            dim1 = tg["dimensions"]["1"]["labels"][()]
            labels = [_decode_text(x) for x in np.asarray(dim1).reshape(-1)]
            if not labels:
                raise RuntimeError(f"No channel labels in {nix_path.name}::{trial_name}")

            dim2 = tg["dimensions"]["2"]
            sampling_interval = float(dim2.attrs["sampling_interval"])
            t0 = float(dim2.attrs["offset"])
            fs = 1.0 / sampling_interval

            if arr.shape[0] == len(labels):
                data_ch_by_time = arr
            elif arr.shape[1] == len(labels):
                data_ch_by_time = arr.T
            else:
                raise RuntimeError(
                    f"Cannot align data shape {arr.shape} with labels {len(labels)} "
                    f"in {nix_path.name}::{trial_name}"
                )

            n_time = data_ch_by_time.shape[1]
            t_rel = (np.arange(n_time, dtype=np.float64) / fs) + t0
            t_rel0 = t_rel - float(t_rel[0])
            t_session = running_offset + t_rel0

            trial_offsets[trial_no] = running_offset
            trial_t0[trial_no] = t0
            t_segments.append(t_session)

            for ch_idx, ch_label in enumerate(labels):
                channel_segments.setdefault(ch_label, []).append(data_ch_by_time[ch_idx, :].astype(np.float64))

            running_offset = float(t_session[-1]) + (1.0 / fs) + float(trial_gap_sec)

            if labels_ref is None:
                labels_ref = labels
                fs_ref = fs
            else:
                if labels != labels_ref:
                    raise RuntimeError(
                        f"Channel labels changed across trials in {nix_path.name}: "
                        f"{labels_ref} vs {labels}"
                    )
                if abs(fs - float(fs_ref)) > 1e-12:
                    raise RuntimeError(
                        f"Sampling rate changed across trials in {nix_path.name}: "
                        f"{fs_ref} vs {fs}"
                    )

        assert labels_ref is not None
        assert fs_ref is not None

        t_concat = np.concatenate(t_segments).astype(np.float64)

        selected_labels = list(labels_ref)
        allowed_macro_labels = {
            str(r.get("macro_label", "")).strip()
            for r in source_rows
            if str(r.get("macro_label", "")).strip()
            and _passes_anatomy_filter(str(r.get("anatomy", "")), anatomy_substring)
        }
        if anatomy_substring is not None:
            selected_labels = [lab for lab in labels_ref if lab in allowed_macro_labels]
            if not selected_labels:
                raise RuntimeError(
                    f"No iEEG channels passed anatomy filter '{anatomy_substring}' "
                    f"in {nix_path.name}"
                )

        channel_data = {
            label: np.concatenate(channel_segments[label]).astype(np.float64)
            for label in selected_labels
        }

        spike_names = [k for k in data_arrays.keys() if k.startswith("Spike_Times_Unit_")]
        spike_names = sorted(spike_names, key=_natural_key)

        fallback_shank_map: dict[str, int] = {}
        unit_alias: dict[tuple[int, str, str, str], str] = {}
        shank_unit_counter: dict[int, int] = {}
        unit_spikes: dict[str, list[np.ndarray]] = {}
        unit_rows: list[dict[str, Any]] = []

        for spike_name in spike_names:
            sm = SPIKE_ARRAY_RE.match(spike_name)
            if sm is None:
                continue

            orig_unit_id = sm.group("unit")
            wire_prefix = sm.group("wireprefix")
            wire_num = sm.group("wirenum")
            trial_no = int(sm.group("trial"))

            if trial_no not in trial_offsets or trial_no not in trial_t0:
                continue

            shank = _canonical_shank_from_wire(wire_prefix, fallback_shank_map)
            unit_key = (shank, wire_prefix.upper(), wire_num, orig_unit_id)
            if unit_key not in unit_alias:
                next_idx = shank_unit_counter.get(shank, 0) + 1
                shank_unit_counter[shank] = next_idx
                unit_alias[unit_key] = f"T{shank}_{next_idx}"
            mapped_unit = unit_alias[unit_key]

            sg = data_arrays[spike_name]
            spike_t = np.asarray(sg["data"][()], dtype=np.float64).reshape(-1)
            spike_t = spike_t[np.isfinite(spike_t)]
            if spike_t.size == 0:
                continue

            t_session = float(trial_offsets[trial_no]) + (spike_t - float(trial_t0[trial_no]))
            sample_idx = np.rint(t_session * float(fs_ref)).astype(np.int64) + 1
            if sample_idx.size == 0:
                continue

            src_uuid = ""
            src_name = ""
            macro_label = ""
            anatomy = ""
            soz = ""
            src_grp = sg.get("sources", None)
            if isinstance(src_grp, h5py.Group) and len(src_grp.keys()) > 0:
                src_uuid = next(iter(src_grp.keys()))
                src_meta = source_index.get(src_uuid, {})
                src_name = src_meta.get("source_name", "")
                macro_label = src_meta.get("macro_label", "")
                anatomy = src_meta.get("anatomy", "")
                soz = src_meta.get("soz", "")

            if not _passes_anatomy_filter(anatomy=anatomy, anatomy_substring=anatomy_substring):
                continue
            unit_spikes.setdefault(mapped_unit, []).append(sample_idx)

            unit_rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "nix_file": nix_path.name,
                    "spike_array": spike_name,
                    "mapped_unit": mapped_unit,
                    "shank": shank,
                    "orig_unit_id": orig_unit_id,
                    "wire_prefix": wire_prefix,
                    "wire_num": wire_num,
                    "trial": trial_no,
                    "n_spikes": int(sample_idx.size),
                    "source_uuid": src_uuid,
                    "source_name": src_name,
                    "macro_label": macro_label,
                    "anatomy": anatomy,
                    "soz": soz,
                }
            )

        ms_units: list[tuple[str, np.ndarray]] = []
        for mapped_unit in sorted(unit_spikes.keys(), key=_natural_key):
            chunks = unit_spikes[mapped_unit]
            if not chunks:
                continue
            merged = np.concatenate(chunks).astype(np.float64)
            merged = merged[np.isfinite(merged)]
            if merged.size == 0:
                continue
            merged = np.rint(merged).astype(np.int64)
            if np.any(np.diff(merged) < 0):
                merged = np.sort(merged)
            ms_units.append((mapped_unit, merged))

        return SubjectRun(
            subject=subject,
            session=session,
            fs=float(fs_ref),
            trial_count=len(trial_names),
            channel_labels=list(selected_labels),
            time_sec=t_concat,
            channel_data=channel_data,
            ms_units=tuple(ms_units),
            unit_inventory_rows=unit_rows,
            source_inventory_rows=source_rows,
        )


def _build_subject_plot(
    run: SubjectRun,
    out_html: Path,
    include_plotlyjs: str,
    ms_lfp_overlay: bool,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
    max_points: int,
    apply_zscore: bool,
    macro_line_noise_hz: float | None,
    macro_line_noise_q: float,
) -> None:
    if max_points <= 0:
        raise ValueError("max_points must be > 0.")

    # Keep macro channels as separate rows (no overlay in one panel).
    macro_rows: list[dict[str, Any]] = []
    x_min = np.inf
    x_max = -np.inf
    for i, label in enumerate(run.channel_labels, start=1):
        y = np.asarray(run.channel_data[label], dtype=float).reshape(-1)
        x = np.asarray(run.time_sec, dtype=float).reshape(-1)
        xx, yy = _clean_xy(x, y, max_points=0)
        if yy.size < 2:
            continue
        if macro_line_noise_hz is not None:
            yy = _apply_harmonic_notch(
                y=yy,
                fs=float(run.fs),
                line_hz=float(macro_line_noise_hz),
                q=float(macro_line_noise_q),
            )
        if bool(apply_zscore):
            yy = _zscore_1d(yy)
        xx, yy = _downsample_xy(xx, yy, max_points=max_points)
        if xx.size < 2:
            continue
        x_min = min(x_min, float(np.min(xx)))
        x_max = max(x_max, float(np.max(xx)))
        macro_rows.append(
            {
                "pair_idx": i,
                "label": str(label),
                "x": xx,
                "y": yy,
                "color": PLOTLY_COLORS[(i - 1) % len(PLOTLY_COLORS)],
                "shank": _macro_channel_to_shank(str(label)),
            }
        )

    if not macro_rows:
        raise RuntimeError(f"No plottable LFP traces for {run.subject}")

    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        raise RuntimeError(f"Invalid x-range for {run.subject}: {x_min}, {x_max}")

    units_by_shank: dict[int, list[tuple[str, np.ndarray]]] = {}
    for unit_name, spikes in run.ms_units:
        parsed = parse_unit_name(str(unit_name))
        if parsed is None:
            continue
        shank_int, _ = parsed
        units_by_shank.setdefault(int(shank_int), []).append(
            (str(unit_name), np.asarray(spikes, dtype=float).reshape(-1))
        )

    for shank in list(units_by_shank.keys()):
        units_by_shank[shank] = sorted(units_by_shank[shank], key=lambda x: _natural_key(x[0]))

    row_specs: list[dict[str, Any]] = []
    for macro in macro_rows:
        row_specs.append({"kind": "macro", **macro})

        shank = macro.get("shank")
        pseudo_x = np.array([], dtype=float)
        pseudo_y = np.array([], dtype=float)
        pseudo_note = "no paired micro units"
        shank_units: list[tuple[str, np.ndarray]] = []

        if shank is not None:
            shank_units = units_by_shank.get(int(shank), [])
            if shank_units:
                lfp_rows, lfp_warnings = synthesize_shank_lfp(
                    units=shank_units,
                    sampling_rate=float(run.fs),
                    time_range_sec=(float(x_min), float(x_max)),
                    sigma_sec=float(ms_lfp_sigma),
                    amplitude=float(ms_lfp_a),
                    gain_a0=float(ms_lfp_a0),
                    distance_map={},
                    default_distance=float(ms_lfp_d_default),
                    post_smooth_sec=float(ms_lfp_post_smooth_sec),
                )
                for w in lfp_warnings:
                    print(
                        f"\033[1;33m -- [{run.subject} {run.session} {macro['label']}] {w}\033[0m"
                    )

                chosen_row: dict[str, Any] | None = None
                for r in lfp_rows:
                    if int(r.get("shank_int", -1)) == int(shank):
                        chosen_row = r
                        break
                if chosen_row is None and lfp_rows:
                    chosen_row = lfp_rows[0]

                if chosen_row is not None:
                    pseudo_x, pseudo_y = _clean_xy(
                        np.asarray(chosen_row.get("time_sec", np.array([], dtype=float))),
                        np.asarray(chosen_row.get("lfp", np.array([], dtype=float))),
                        max_points=0,
                    )
                    if bool(apply_zscore) and pseudo_y.size >= 2:
                        pseudo_y = _zscore_1d(pseudo_y)
                    pseudo_x, pseudo_y = _downsample_xy(pseudo_x, pseudo_y, max_points=max_points)

                if pseudo_x.size >= 2:
                    pseudo_note = f"shank {int(shank)} pseudo-LFP"

        row_specs.append(
            {
                "kind": "pseudo",
                "pair_idx": macro["pair_idx"],
                "label": macro["label"],
                "shank": shank,
                "x": pseudo_x,
                "y": pseudo_y,
                "note": pseudo_note,
            }
        )

        # Restore legacy layout style: one spike-unit row per unit beneath pseudo-LFP.
        if bool(ms_lfp_overlay) and shank_units:
            cap = max(1000, int(max_points) * 3)
            for unit_idx, (unit_name, s_raw) in enumerate(shank_units, start=1):
                s = np.asarray(s_raw, dtype=float).reshape(-1)
                s = s[np.isfinite(s)]
                if s.size == 0:
                    continue
                t = (s - 1.0) / float(run.fs)
                t = t[np.isfinite(t)]
                if t.size == 0:
                    continue
                m = (t >= float(x_min)) & (t <= float(x_max))
                t = t[m]
                if t.size == 0:
                    continue
                if t.size > cap:
                    t = t[_linspace_idx(t.size, cap)]
                row_specs.append(
                    {
                        "kind": "spike",
                        "pair_idx": macro["pair_idx"],
                        "label": macro["label"],
                        "unit_name": str(unit_name),
                        "unit_idx": int(unit_idx),
                        "x": np.asarray(t, dtype=float),
                    }
                )

    n_rows = len(row_specs)
    row_heights = [1.0 if spec["kind"] in {"macro", "pseudo"} else 0.28 for spec in row_specs]
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.012,
        row_heights=row_heights,
    )

    legend_seen: set[str] = set()
    y_ranges: dict[int, tuple[float, float]] = {}

    for row_idx, spec in enumerate(row_specs, start=1):
        kind = spec["kind"]
        pair_idx = int(spec["pair_idx"])

        if kind == "macro":
            x = np.asarray(spec["x"], dtype=float)
            y = np.asarray(spec["y"], dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color=spec["color"], width=1.1),
                    name="Macro LFP",
                    legendgroup="Macro LFP",
                    showlegend=("Macro LFP" not in legend_seen),
                    hovertemplate=(
                        f"Macro {spec['label']}<br>t=%{{x:.3f}}s<br>v=%{{y:.6g}}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )
            legend_seen.add("Macro LFP")

            y_lo = float(np.min(y))
            y_hi = float(np.max(y))
            if y_hi <= y_lo:
                y_lo -= 1.0
                y_hi += 1.0
            pad = 0.06 * (y_hi - y_lo)
            y_ranges[row_idx] = (y_lo - pad, y_hi + pad)

            title_text = f"<b>Macro LFP {pair_idx}</b>: {spec['label']}"
        elif kind == "pseudo":
            x = np.asarray(spec["x"], dtype=float)
            y = np.asarray(spec["y"], dtype=float)
            if x.size >= 2 and y.size >= 2:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        line=dict(color="#0b5a8f", width=1.3),
                        name="Pseudo LFP",
                        legendgroup="Pseudo LFP",
                        showlegend=("Pseudo LFP" not in legend_seen),
                        hovertemplate=(
                            f"Pseudo (pair {pair_idx})<br>t=%{{x:.3f}}s<br>v=%{{y:.6g}}<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=1,
                )
                legend_seen.add("Pseudo LFP")
                y_lo = float(np.min(y))
                y_hi = float(np.max(y))
                if y_hi <= y_lo:
                    y_lo -= 1.0
                    y_hi += 1.0
                pad = 0.08 * (y_hi - y_lo)
                y_ranges[row_idx] = (y_lo - pad, y_hi + pad)
            else:
                fig.add_trace(
                    go.Scatter(
                        x=np.array([x_min, x_max], dtype=float),
                        y=np.array([0.0, 0.0], dtype=float),
                        mode="lines",
                        line=dict(color="#9aa3b2", width=1.0, dash="dot"),
                        name="Pseudo LFP (none)",
                        legendgroup="Pseudo LFP (none)",
                        showlegend=("Pseudo LFP (none)" not in legend_seen),
                        hoverinfo="skip",
                    ),
                    row=row_idx,
                    col=1,
                )
                legend_seen.add("Pseudo LFP (none)")
                y_ranges[row_idx] = (-1.0, 1.0)

            title_text = f"<b>Pseudo {pair_idx}</b>: {spec.get('note', '')}"
        else:
            t_spike = np.asarray(spec.get("x", np.array([], dtype=float)), dtype=float)
            if t_spike.size > 0:
                fig.add_trace(
                    go.Scattergl(
                        x=t_spike,
                        y=np.zeros_like(t_spike),
                        mode="markers",
                        marker=dict(
                            symbol="line-ns-open",
                            color="#111111",
                            size=8,
                            line=dict(color="#111111", width=1),
                        ),
                        name="MS spike",
                        legendgroup="MS spike",
                        showlegend=("MS spike" not in legend_seen),
                        hovertemplate=f"{spec['unit_name']}<br>t=%{{x:.3f}}s<extra></extra>",
                    ),
                    row=row_idx,
                    col=1,
                )
                legend_seen.add("MS spike")

            y_ranges[row_idx] = (-1.0, 1.0)
            title_text = f"<b>Spike {pair_idx}.{spec.get('unit_idx', 0)}</b>: {spec.get('unit_name', '')}"

        xref = "x domain" if row_idx == 1 else f"x{row_idx} domain"
        yref = "y domain" if row_idx == 1 else f"y{row_idx} domain"
        fig.add_annotation(
            x=0.01,
            y=0.98,
            xref=xref,
            yref=yref,
            text=title_text,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font={"size": 12, "color": "black"},
            bgcolor="rgba(255,255,255,0.75)",
            borderpad=1,
        )

    for row_idx in range(1, n_rows + 1):
        kind = str(row_specs[row_idx - 1].get("kind", "macro"))
        fig.update_xaxes(
            range=[float(x_min), float(x_max)],
            showgrid=(kind != "spike"),
            zeroline=False,
            showticklabels=(row_idx == n_rows),
            title_text=("Time [sec]" if row_idx == n_rows else None),
            row=row_idx,
            col=1,
        )
        y_lo, y_hi = y_ranges.get(row_idx, (-1.0, 1.0))
        fig.update_yaxes(
            range=[float(y_lo), float(y_hi)],
            showgrid=(kind != "spike"),
            zeroline=False,
            showticklabels=(kind != "spike"),
            row=row_idx,
            col=1,
        )

    n_macro = sum(1 for s in row_specs if s["kind"] == "macro")
    n_pseudo = sum(1 for s in row_specs if s["kind"] == "pseudo")
    n_spike = sum(1 for s in row_specs if s["kind"] == "spike")
    fig.update_layout(
        template="plotly_white",
        height=int((220 * n_macro) + (220 * n_pseudo) + (56 * n_spike) + 70),
        margin={"l": 95, "r": 20, "t": 48, "b": 52},
        title={"text": f"{run.subject} {run.session} | Macro/Pseudo/Spike Paired Rows", "x": 0.01},
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(255,255,255,0.72)",
            "bordercolor": "rgba(0,0,0,0.15)",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        dragmode="pan",
        hovermode="x",
    )

    write_plotly_html(fig, out_path=out_html, include_plotlyjs=include_plotlyjs)


def _build_subject_psd_plot(
    run: SubjectRun,
    out_html: Path,
    include_plotlyjs: str,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
    max_points: int,
    apply_zscore: bool,
    macro_line_noise_hz: float | None,
    macro_line_noise_q: float,
) -> None:
    if max_points <= 0:
        raise ValueError("max_points must be > 0.")

    macro_rows: list[dict[str, Any]] = []
    x_min = np.inf
    x_max = -np.inf
    for i, label in enumerate(run.channel_labels, start=1):
        y = np.asarray(run.channel_data[label], dtype=float).reshape(-1)
        x = np.asarray(run.time_sec, dtype=float).reshape(-1)
        xx, yy = _clean_xy(x, y, max_points=0)
        if xx.size < 2 or yy.size < 2:
            continue
        if macro_line_noise_hz is not None:
            yy = _apply_harmonic_notch(
                y=yy,
                fs=float(run.fs),
                line_hz=float(macro_line_noise_hz),
                q=float(macro_line_noise_q),
            )
        if bool(apply_zscore):
            yy = _zscore_1d(yy)

        x_min = min(x_min, float(np.min(xx)))
        x_max = max(x_max, float(np.max(xx)))
        macro_rows.append(
            {
                "pair_idx": i,
                "label": str(label),
                "x": xx,
                "y": yy,
                "color": PLOTLY_COLORS[(i - 1) % len(PLOTLY_COLORS)],
                "shank": _macro_channel_to_shank(str(label)),
            }
        )

    if not macro_rows:
        raise RuntimeError(f"No plottable macro traces for PSD in {run.subject}")
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        raise RuntimeError(f"Invalid x-range for PSD in {run.subject}: {x_min}, {x_max}")

    units_by_shank: dict[int, list[tuple[str, np.ndarray]]] = {}
    for unit_name, spikes in run.ms_units:
        parsed = parse_unit_name(str(unit_name))
        if parsed is None:
            continue
        shank_int, _ = parsed
        units_by_shank.setdefault(int(shank_int), []).append(
            (str(unit_name), np.asarray(spikes, dtype=float).reshape(-1))
        )

    row_specs: list[dict[str, Any]] = []
    for macro in macro_rows:
        pair_idx = int(macro["pair_idx"])

        f_macro, p_macro = _compute_psd_fullrange(macro["y"], fs=float(run.fs))
        f_macro, p_macro = _downsample_xy(f_macro, p_macro, max_points=max_points)
        p_macro_db = 10.0 * np.log10(np.maximum(p_macro, 1e-18)) if p_macro.size else np.array([], dtype=float)

        shank = macro.get("shank")
        pseudo_note = "no paired micro units"
        f_pseudo = np.array([], dtype=float)
        p_pseudo_db = np.array([], dtype=float)

        if shank is not None:
            shank_units = units_by_shank.get(int(shank), [])
            if shank_units:
                lfp_rows, lfp_warnings = synthesize_shank_lfp(
                    units=shank_units,
                    sampling_rate=float(run.fs),
                    time_range_sec=(float(x_min), float(x_max)),
                    sigma_sec=float(ms_lfp_sigma),
                    amplitude=float(ms_lfp_a),
                    gain_a0=float(ms_lfp_a0),
                    distance_map={},
                    default_distance=float(ms_lfp_d_default),
                    post_smooth_sec=float(ms_lfp_post_smooth_sec),
                )
                for w in lfp_warnings:
                    print(
                        f"\033[1;33m -- [PSD {run.subject} {run.session} {macro['label']}] {w}\033[0m"
                    )
                chosen_row: dict[str, Any] | None = None
                for r in lfp_rows:
                    if int(r.get("shank_int", -1)) == int(shank):
                        chosen_row = r
                        break
                if chosen_row is None and lfp_rows:
                    chosen_row = lfp_rows[0]

                if chosen_row is not None:
                    _, pseudo_y = _clean_xy(
                        np.asarray(chosen_row.get("time_sec", np.array([], dtype=float))),
                        np.asarray(chosen_row.get("lfp", np.array([], dtype=float))),
                        max_points=0,
                    )
                    if bool(apply_zscore) and pseudo_y.size >= 2:
                        pseudo_y = _zscore_1d(pseudo_y)
                    f_pseudo, p_pseudo = _compute_psd_fullrange(pseudo_y, fs=float(run.fs))
                    f_pseudo, p_pseudo = _downsample_xy(
                        f_pseudo, p_pseudo, max_points=max_points
                    )
                    if p_pseudo.size > 0:
                        p_pseudo_db = 10.0 * np.log10(np.maximum(p_pseudo, 1e-18))
                        pseudo_note = f"shank {int(shank)} pseudo-LFP"

        row_specs.append(
            {
                "pair_idx": pair_idx,
                "label": str(macro["label"]),
                "macro_color": str(macro["color"]),
                "f_macro": np.asarray(f_macro, dtype=float),
                "p_macro_db": np.asarray(p_macro_db, dtype=float),
                "f_pseudo": np.asarray(f_pseudo, dtype=float),
                "p_pseudo_db": np.asarray(p_pseudo_db, dtype=float),
                "pseudo_note": pseudo_note,
            }
        )

    n_rows = len(row_specs)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.014,
        row_heights=[1.0 for _ in row_specs],
    )

    legend_seen: set[str] = set()
    y_ranges: dict[int, tuple[float, float]] = {}
    f_max_global = 0.0

    for row_idx, spec in enumerate(row_specs, start=1):
        pair_idx = int(spec["pair_idx"])

        f_macro = np.asarray(spec["f_macro"], dtype=float)
        p_macro_db = np.asarray(spec["p_macro_db"], dtype=float)
        f_pseudo = np.asarray(spec["f_pseudo"], dtype=float)
        p_pseudo_db = np.asarray(spec["p_pseudo_db"], dtype=float)

        y_blocks: list[np.ndarray] = []

        if f_macro.size >= 2 and p_macro_db.size >= 2:
            fig.add_trace(
                go.Scatter(
                    x=f_macro,
                    y=p_macro_db,
                    mode="lines",
                    line=dict(color=spec["macro_color"], width=1.2),
                    name="Macro PSD",
                    legendgroup="Macro PSD",
                    showlegend=("Macro PSD" not in legend_seen),
                    hovertemplate=(
                        f"Macro {spec['label']}<br>f=%{{x:.3f}} Hz<br>PSD=%{{y:.3f}} dB/Hz<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )
            legend_seen.add("Macro PSD")
            y_blocks.append(p_macro_db)
            f_max_global = max(f_max_global, float(np.max(f_macro)))

        if f_pseudo.size >= 2 and p_pseudo_db.size >= 2:
            fig.add_trace(
                go.Scatter(
                    x=f_pseudo,
                    y=p_pseudo_db,
                    mode="lines",
                    line=dict(color="#0b5a8f", width=1.2),
                    name="Pseudo PSD",
                    legendgroup="Pseudo PSD",
                    showlegend=("Pseudo PSD" not in legend_seen),
                    hovertemplate=(
                        f"Pseudo pair {pair_idx}<br>f=%{{x:.3f}} Hz<br>PSD=%{{y:.3f}} dB/Hz<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )
            legend_seen.add("Pseudo PSD")
            y_blocks.append(p_pseudo_db)
            f_max_global = max(f_max_global, float(np.max(f_pseudo)))

        if y_blocks:
            y_cat = np.concatenate(y_blocks)
            y_lo = float(np.min(y_cat))
            y_hi = float(np.max(y_cat))
            if y_hi <= y_lo:
                y_lo -= 1.0
                y_hi += 1.0
            pad = 0.08 * (y_hi - y_lo)
            y_ranges[row_idx] = (y_lo - pad, y_hi + pad)
        else:
            fig.add_trace(
                go.Scatter(
                    x=np.array([0.0, float(run.fs) * 0.5], dtype=float),
                    y=np.array([0.0, 0.0], dtype=float),
                    mode="lines",
                    line=dict(color="#9aa3b2", width=1.0, dash="dot"),
                    name="PSD (none)",
                    legendgroup="PSD (none)",
                    showlegend=("PSD (none)" not in legend_seen),
                    hoverinfo="skip",
                ),
                row=row_idx,
                col=1,
            )
            legend_seen.add("PSD (none)")
            y_ranges[row_idx] = (-1.0, 1.0)

        title_text = f"<b>PSD {pair_idx}</b>: {spec['label']} | {spec.get('pseudo_note', '')}"
        xref = "x domain" if row_idx == 1 else f"x{row_idx} domain"
        yref = "y domain" if row_idx == 1 else f"y{row_idx} domain"
        fig.add_annotation(
            x=0.01,
            y=0.98,
            xref=xref,
            yref=yref,
            text=title_text,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font={"size": 12, "color": "black"},
            bgcolor="rgba(255,255,255,0.75)",
            borderpad=1,
        )

    f_hi = float(f_max_global) if np.isfinite(f_max_global) and f_max_global > 0 else (float(run.fs) * 0.5)
    if f_hi <= 0:
        f_hi = 1.0

    for row_idx in range(1, n_rows + 1):
        fig.update_xaxes(
            range=[0.0, f_hi],
            showgrid=True,
            zeroline=False,
            showticklabels=(row_idx == n_rows),
            title_text=("Frequency [Hz]" if row_idx == n_rows else None),
            row=row_idx,
            col=1,
        )
        y_lo, y_hi = y_ranges.get(row_idx, (-1.0, 1.0))
        fig.update_yaxes(
            range=[float(y_lo), float(y_hi)],
            showgrid=True,
            zeroline=False,
            title_text=("PSD [dB/Hz]" if row_idx == 1 else None),
            row=row_idx,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        height=int(260 * n_rows + 70),
        margin={"l": 95, "r": 20, "t": 48, "b": 52},
        title={"text": f"{run.subject} {run.session} | Macro vs Pseudo PSD (full time range)", "x": 0.01},
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(255,255,255,0.72)",
            "bordercolor": "rgba(0,0,0,0.15)",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        dragmode="pan",
        hovermode="x",
    )

    write_plotly_html(fig, out_path=out_html, include_plotlyjs=include_plotlyjs)


def _build_subject_power_time_plot(
    run: SubjectRun,
    out_html: Path,
    include_plotlyjs: str,
    ms_lfp_overlay: bool,
    ms_lfp_sigma: float,
    ms_lfp_a: float,
    ms_lfp_a0: float,
    ms_lfp_d_default: float,
    ms_lfp_post_smooth_sec: float,
    max_points: int,
    apply_zscore: bool,
    macro_line_noise_hz: float | None,
    macro_line_noise_q: float,
    theta_bands: list[tuple[str, tuple[float, float]]],
    tf_win_sec: float,
    tf_step_sec: float,
    power_apply_db: bool,
    power_db_eps: float,
    power_smooth_win_sec: float | None,
) -> None:
    if max_points <= 0:
        raise ValueError("max_points must be > 0.")

    psd_array_multitaper = _require_mne_multitaper()

    macro_rows: list[dict[str, Any]] = []
    x_min = np.inf
    x_max = -np.inf
    for i, label in enumerate(run.channel_labels, start=1):
        y = np.asarray(run.channel_data[label], dtype=float).reshape(-1)
        x = np.asarray(run.time_sec, dtype=float).reshape(-1)
        xx, yy = _clean_xy(x, y, max_points=0)
        if xx.size < 2 or yy.size < 2:
            continue
        if macro_line_noise_hz is not None:
            yy = _apply_harmonic_notch(
                y=yy,
                fs=float(run.fs),
                line_hz=float(macro_line_noise_hz),
                q=float(macro_line_noise_q),
            )
        if bool(apply_zscore):
            yy = _zscore_1d(yy)

        x_min = min(x_min, float(np.min(xx)))
        x_max = max(x_max, float(np.max(xx)))
        macro_rows.append(
            {
                "pair_idx": i,
                "label": str(label),
                "x": xx,
                "y": yy,
                "shank": _macro_channel_to_shank(str(label)),
            }
        )

    if not macro_rows:
        raise RuntimeError(f"No plottable macro traces for Power Time in {run.subject}")
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        raise RuntimeError(f"Invalid x-range for Power Time in {run.subject}: {x_min}, {x_max}")

    units_by_shank: dict[int, list[tuple[str, np.ndarray]]] = {}
    for unit_name, spikes in run.ms_units:
        parsed = parse_unit_name(str(unit_name))
        if parsed is None:
            continue
        shank_int, _ = parsed
        units_by_shank.setdefault(int(shank_int), []).append(
            (str(unit_name), np.asarray(spikes, dtype=float).reshape(-1))
        )
    for shank in list(units_by_shank.keys()):
        units_by_shank[shank] = sorted(units_by_shank[shank], key=lambda x: _natural_key(x[0]))

    row_specs: list[dict[str, Any]] = []
    for macro in macro_rows:
        pair_idx = int(macro["pair_idx"])
        macro_theta = _compute_theta_power_timeseries(
            y=np.asarray(macro["y"], dtype=float),
            fs=float(run.fs),
            theta_bands=theta_bands,
            tf_win_sec=float(tf_win_sec),
            tf_step_sec=float(tf_step_sec),
            apply_db=bool(power_apply_db),
            db_eps=float(power_db_eps),
            smooth_win_sec=power_smooth_win_sec,
            max_points=max_points,
            psd_array_multitaper=psd_array_multitaper,
        )
        row_specs.append(
            {
                "kind": "macro_power",
                "pair_idx": pair_idx,
                "label": str(macro["label"]),
                "theta": macro_theta,
            }
        )

        shank = macro.get("shank")
        pseudo_theta: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        pseudo_note = "no paired micro units"
        shank_units: list[tuple[str, np.ndarray]] = []
        if shank is not None:
            shank_units = units_by_shank.get(int(shank), [])
            if shank_units:
                lfp_rows, lfp_warnings = synthesize_shank_lfp(
                    units=shank_units,
                    sampling_rate=float(run.fs),
                    time_range_sec=(float(x_min), float(x_max)),
                    sigma_sec=float(ms_lfp_sigma),
                    amplitude=float(ms_lfp_a),
                    gain_a0=float(ms_lfp_a0),
                    distance_map={},
                    default_distance=float(ms_lfp_d_default),
                    post_smooth_sec=float(ms_lfp_post_smooth_sec),
                )
                for w in lfp_warnings:
                    print(
                        f"\033[1;33m -- [PowerTime {run.subject} {run.session} {macro['label']}] {w}\033[0m"
                    )

                chosen_row: dict[str, Any] | None = None
                for r in lfp_rows:
                    if int(r.get("shank_int", -1)) == int(shank):
                        chosen_row = r
                        break
                if chosen_row is None and lfp_rows:
                    chosen_row = lfp_rows[0]

                if chosen_row is not None:
                    _, pseudo_y = _clean_xy(
                        np.asarray(chosen_row.get("time_sec", np.array([], dtype=float))),
                        np.asarray(chosen_row.get("lfp", np.array([], dtype=float))),
                        max_points=0,
                    )
                    if bool(apply_zscore) and pseudo_y.size >= 2:
                        pseudo_y = _zscore_1d(pseudo_y)
                    pseudo_theta = _compute_theta_power_timeseries(
                        y=pseudo_y,
                        fs=float(run.fs),
                        theta_bands=theta_bands,
                        tf_win_sec=float(tf_win_sec),
                        tf_step_sec=float(tf_step_sec),
                        apply_db=bool(power_apply_db),
                        db_eps=float(power_db_eps),
                        smooth_win_sec=power_smooth_win_sec,
                        max_points=max_points,
                        psd_array_multitaper=psd_array_multitaper,
                    )
                    if pseudo_theta:
                        pseudo_note = f"shank {int(shank)} pseudo-LFP"

        row_specs.append(
            {
                "kind": "pseudo_power",
                "pair_idx": pair_idx,
                "label": str(macro["label"]),
                "theta": pseudo_theta,
                "note": pseudo_note,
            }
        )

        if bool(ms_lfp_overlay) and shank_units:
            cap = max(1000, int(max_points) * 3)
            for unit_idx, (unit_name, s_raw) in enumerate(shank_units, start=1):
                s = np.asarray(s_raw, dtype=float).reshape(-1)
                s = s[np.isfinite(s)]
                if s.size == 0:
                    continue
                t = (s - 1.0) / float(run.fs)
                t = t[np.isfinite(t)]
                if t.size == 0:
                    continue
                m = (t >= float(x_min)) & (t <= float(x_max))
                t = t[m]
                if t.size == 0:
                    continue
                if t.size > cap:
                    t = t[_linspace_idx(t.size, cap)]
                row_specs.append(
                    {
                        "kind": "spike",
                        "pair_idx": pair_idx,
                        "unit_name": str(unit_name),
                        "unit_idx": int(unit_idx),
                        "x": np.asarray(t, dtype=float),
                    }
                )

    n_rows = len(row_specs)
    row_heights = [1.0 if spec["kind"] in {"macro_power", "pseudo_power"} else 0.28 for spec in row_specs]
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.012,
        row_heights=row_heights,
    )

    legend_seen: set[str] = set()
    y_ranges: dict[int, tuple[float, float]] = {}

    for row_idx, spec in enumerate(row_specs, start=1):
        kind = str(spec.get("kind", "macro_power"))
        pair_idx = int(spec["pair_idx"])

        if kind in {"macro_power", "pseudo_power"}:
            theta_map = spec.get("theta", {}) or {}
            y_blocks: list[np.ndarray] = []
            added = 0
            for bidx, (band_label, _) in enumerate(theta_bands):
                xy = theta_map.get(band_label)
                if xy is None:
                    continue
                x, y = xy
                xx = np.asarray(x, dtype=float).reshape(-1)
                yy = np.asarray(y, dtype=float).reshape(-1)
                n = min(xx.size, yy.size)
                if n < 2:
                    continue
                xx = xx[:n]
                yy = yy[:n]
                m = np.isfinite(xx) & np.isfinite(yy)
                if int(np.sum(m)) < 2:
                    continue
                xx = xx[m]
                yy = yy[m]
                line_dash = "solid" if kind == "macro_power" else "dash"
                trace_name = (
                    f"Macro {band_label}"
                    if kind == "macro_power"
                    else f"Pseudo {band_label}"
                )
                fig.add_trace(
                    go.Scatter(
                        x=xx,
                        y=yy,
                        mode="lines",
                        line=dict(
                            color=PLOTLY_COLORS[bidx % len(PLOTLY_COLORS)],
                            width=1.1,
                            dash=line_dash,
                        ),
                        name=trace_name,
                        legendgroup=trace_name,
                        showlegend=(trace_name not in legend_seen),
                        hovertemplate=f"{trace_name}<br>t=%{{x:.3f}}s<br>v=%{{y:.6g}}<extra></extra>",
                    ),
                    row=row_idx,
                    col=1,
                )
                legend_seen.add(trace_name)
                y_blocks.append(yy)
                added += 1

            if y_blocks:
                y_cat = np.concatenate(y_blocks)
                y_lo = float(np.min(y_cat))
                y_hi = float(np.max(y_cat))
                if y_hi <= y_lo:
                    y_lo -= 1.0
                    y_hi += 1.0
                pad = 0.08 * (y_hi - y_lo)
                y_ranges[row_idx] = (y_lo - pad, y_hi + pad)
            else:
                fig.add_trace(
                    go.Scatter(
                        x=np.array([x_min, x_max], dtype=float),
                        y=np.array([0.0, 0.0], dtype=float),
                        mode="lines",
                        line=dict(color="#9aa3b2", width=1.0, dash="dot"),
                        name="PowerTime (none)",
                        legendgroup="PowerTime (none)",
                        showlegend=("PowerTime (none)" not in legend_seen),
                        hoverinfo="skip",
                    ),
                    row=row_idx,
                    col=1,
                )
                legend_seen.add("PowerTime (none)")
                y_ranges[row_idx] = (-1.0, 1.0)

            if kind == "macro_power":
                title_text = f"<b>Macro Power {pair_idx}</b>: {spec.get('label', '')}"
            else:
                title_text = f"<b>Pseudo Power {pair_idx}</b>: {spec.get('note', '')}"
        else:
            t_spike = np.asarray(spec.get("x", np.array([], dtype=float)), dtype=float)
            if t_spike.size > 0:
                fig.add_trace(
                    go.Scattergl(
                        x=t_spike,
                        y=np.zeros_like(t_spike),
                        mode="markers",
                        marker=dict(
                            symbol="line-ns-open",
                            color="#111111",
                            size=8,
                            line=dict(color="#111111", width=1),
                        ),
                        name="MS spike",
                        legendgroup="MS spike",
                        showlegend=("MS spike" not in legend_seen),
                        hovertemplate=f"{spec['unit_name']}<br>t=%{{x:.3f}}s<extra></extra>",
                    ),
                    row=row_idx,
                    col=1,
                )
                legend_seen.add("MS spike")
            y_ranges[row_idx] = (-1.0, 1.0)
            title_text = f"<b>Spike {pair_idx}.{spec.get('unit_idx', 0)}</b>: {spec.get('unit_name', '')}"

        xref = "x domain" if row_idx == 1 else f"x{row_idx} domain"
        yref = "y domain" if row_idx == 1 else f"y{row_idx} domain"
        fig.add_annotation(
            x=0.01,
            y=0.98,
            xref=xref,
            yref=yref,
            text=title_text,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font={"size": 12, "color": "black"},
            bgcolor="rgba(255,255,255,0.75)",
            borderpad=1,
        )

    for row_idx in range(1, n_rows + 1):
        kind = str(row_specs[row_idx - 1].get("kind", "macro_power"))
        fig.update_xaxes(
            range=[float(x_min), float(x_max)],
            showgrid=(kind != "spike"),
            zeroline=False,
            showticklabels=(row_idx == n_rows),
            title_text=("Time [sec]" if row_idx == n_rows else None),
            row=row_idx,
            col=1,
        )
        y_lo, y_hi = y_ranges.get(row_idx, (-1.0, 1.0))
        fig.update_yaxes(
            range=[float(y_lo), float(y_hi)],
            showgrid=(kind != "spike"),
            zeroline=False,
            showticklabels=(kind != "spike"),
            row=row_idx,
            col=1,
        )

    n_macro = sum(1 for s in row_specs if s["kind"] == "macro_power")
    n_pseudo = sum(1 for s in row_specs if s["kind"] == "pseudo_power")
    n_spike = sum(1 for s in row_specs if s["kind"] == "spike")
    y_label = "Theta Power [dB]" if bool(power_apply_db) else "Theta Power"
    fig.update_layout(
        template="plotly_white",
        height=int((220 * n_macro) + (220 * n_pseudo) + (56 * n_spike) + 70),
        margin={"l": 95, "r": 20, "t": 48, "b": 52},
        title={"text": f"{run.subject} {run.session} | Power Time (ThetaPower-like)", "x": 0.01},
        yaxis={"title": {"text": y_label}},
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(255,255,255,0.72)",
            "bordercolor": "rgba(0,0,0,0.15)",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        dragmode="pan",
        hovermode="x",
    )

    write_plotly_html(fig, out_path=out_html, include_plotlyjs=include_plotlyjs)


def _write_tsv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def _load_bids_channel_summary(ds_root: Path, subject: str) -> dict[str, Any]:
    ieeg_dir = ds_root / subject / "ses-01" / "ieeg"
    ch_files = sorted(ieeg_dir.glob("*_channels.tsv"), key=lambda p: p.name)
    el_files = sorted(ieeg_dir.glob("*_electrodes.tsv"), key=lambda p: p.name)

    out: dict[str, Any] = {
        "subject": subject,
        "bids_channels_tsv": ch_files[0].name if ch_files else "",
        "bids_electrodes_tsv": el_files[0].name if el_files else "",
        "bids_channel_names": "",
        "bids_channel_groups": "",
        "bids_channel_count": 0,
        "bids_electrode_count": 0,
    }

    if ch_files:
        with ch_files[0].open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        out["bids_channel_count"] = len(rows)
        out["bids_channel_names"] = ",".join([str(r.get("name", "")) for r in rows])
        out["bids_channel_groups"] = ",".join(sorted({str(r.get("group", "")) for r in rows if r.get("group", "")}))

    if el_files:
        with el_files[0].open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        out["bids_electrode_count"] = len(rows)

    return out


def _write_index_html(path: Path, rows: list[dict[str, Any]]) -> None:
    cards: list[str] = []
    for r in rows:
        href = r.get("plot_html", "")
        psd_href = r.get("psd_html", "")
        pow_href = r.get("powertime_html", "")
        subject = r.get("subject", "")
        lfp = r.get("lfp_channels", "")
        n_units = r.get("n_mapped_units", 0)
        n_trials = r.get("trial_count", 0)
        if href:
            link = f'<a href="{href}" target="_blank">Open Plotly</a>'
        else:
            link = "(not generated)"
        if psd_href:
            psd_link = f'<a href="{psd_href}" target="_blank">Open PSD</a>'
        else:
            psd_link = "(not generated)"
        if pow_href:
            pow_link = f'<a href="{pow_href}" target="_blank">Open Power Time</a>'
        else:
            pow_link = "(not generated)"
        cards.append(
            f"""
            <tr>
              <td>{subject}</td>
              <td>{n_trials}</td>
              <td>{n_units}</td>
              <td>{lfp}</td>
              <td>{link}</td>
              <td>{psd_link}</td>
              <td>{pow_link}</td>
            </tr>
            """
        )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>ds003374 pseudo-LFP vs LFP index</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; }}
    h1 {{ margin: 0 0 12px 0; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 1200px; }}
    th, td {{ border: 1px solid #cfcfcf; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f6fb; }}
  </style>
</head>
<body>
  <h1>ds003374: pseudo-LFP vs recorded LFP</h1>
  <p>Generated from NIX spike times + iEEG channels using my_PreP rendering helpers (main_PreP unchanged).</p>
  <table>
    <thead>
      <tr>
        <th>Subject</th>
        <th>Trials</th>
        <th>Mapped Units</th>
        <th>LFP Channels</th>
        <th>Plot</th>
        <th>PSD</th>
        <th>Power Time</th>
      </tr>
    </thead>
    <tbody>
      {''.join(cards)}
    </tbody>
  </table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build ds003374 pseudo-LFP vs LFP Plotly pages without editing main_PreP.py."
        )
    )
    p.add_argument(
        "--ds-root",
        type=str,
        default="/Users/takamanabe/Documents/Git/ds003374",
        help="Path to ds003374 root.",
    )
    p.add_argument(
        "--nix-dir",
        type=str,
        default="",
        help="Optional override path to NIX files. Default: <ds-root>/bidsignore/data_NIX",
    )
    p.add_argument(
        "--anatomy-substring",
        type=str,
        default="amyg",
        help=(
            "Keep channels/units whose source anatomy contains this substring "
            "(case-insensitive). Empty string disables filtering. "
            "Default: 'amyg'."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(SCRIPT_PATH.parent / "output_ds003374"),
        help="Output directory for plots and summaries.",
    )
    p.add_argument(
        "--subject",
        action="append",
        default=[],
        help="Subject token to include (e.g., sub-01). Repeatable. Default: all available.",
    )
    p.add_argument("--trial-gap-sec", type=float, default=0.0, help="Gap inserted between trials when concatenating.")
    p.add_argument("--MS_LFP_OVERLAY", type=str, default="true", help="Overlay spike rasters on pseudo-LFP rows.")
    p.add_argument("--MS_LFP_SIGMA", type=float, default=0.004, help="Pseudo-LFP AP kernel sigma [sec].")
    p.add_argument("--MS_LFP_A", type=float, default=0.2, help="Pseudo-LFP AP kernel amplitude A.")
    p.add_argument("--MS_LFP_A0", type=float, default=1.0, help="Pseudo-LFP gain A0.")
    p.add_argument("--MS_LFP_D_DEFAULT", type=float, default=3.0, help="Pseudo-LFP default distance d.")
    p.add_argument(
        "--MS_LFP_POST_SMOOTH_SEC",
        type=float,
        default=0.012,
        help="Pseudo-LFP Gaussian post smoothing sigma [sec].",
    )
    p.add_argument(
        "--APPLY_MACRO_LINE_NOISE",
        type=str,
        default="true",
        help="Apply notch at JSON PowerLineFrequency harmonics to macro LFP before plotting.",
    )
    p.add_argument(
        "--MACRO_LINE_NOISE_Q",
        type=float,
        default=30.0,
        help="Q-factor for macro line-noise notch filters.",
    )
    p.add_argument(
        "--APPLY_ZSCORE",
        type=str,
        default="true",
        help="Apply z-score to both macro LFP and pseudo-LFP traces.",
    )
    p.add_argument(
        "--POWER_TIME_THETA_BANDS",
        nargs="+",
        type=str,
        default=DEFAULT_THETA_BAND_TOKENS,
        help=(
            "Theta bands for Power Time mode. "
            "Formats: '4-8 4-10' or flat list '4 8 4 10'."
        ),
    )
    p.add_argument(
        "--POWER_TIME_TF_WIN_SEC",
        type=float,
        default=10.0,
        help="Sliding window length (sec) for Power Time.",
    )
    p.add_argument(
        "--POWER_TIME_TF_STEP_SEC",
        type=float,
        default=1.0,
        help="Sliding window step (sec) for Power Time.",
    )
    p.add_argument(
        "--POWER_TIME_APPLY_DB",
        type=str,
        default="false",
        help="Apply dB transform to Power Time traces.",
    )
    p.add_argument(
        "--POWER_TIME_DB_EPS",
        type=float,
        default=float(np.finfo(np.float32).eps),
        help="Epsilon for Power Time dB transform.",
    )
    p.add_argument(
        "--POWER_TIME_SMOOTH_WIN_SEC",
        type=str,
        default="10",
        help="Causal smoothing window (sec) for Power Time. Use NONE to disable.",
    )
    p.add_argument("--interactive_max_points", type=int, default=20000, help="Max points per pseudo-LFP trace.")
    p.add_argument("--plotly-js", choices=("cdn", "inline"), default="cdn", help="Plotly JS embedding mode.")
    p.add_argument("--skip-no-spike", action="store_true", help="Skip subjects with no spike units.")
    p.add_argument("--verbose", action="store_true", help="Verbose logs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ds_root = Path(args.ds_root).expanduser().resolve()
    if not ds_root.is_dir():
        raise FileNotFoundError(f"ds-root does not exist: {ds_root}")

    nix_dir = Path(args.nix_dir).expanduser().resolve() if args.nix_dir else (ds_root / "bidsignore" / "data_NIX")
    if not nix_dir.is_dir():
        raise FileNotFoundError(f"NIX dir does not exist: {nix_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay = _parse_bool(args.MS_LFP_OVERLAY)
    apply_macro_line_noise = _parse_bool(args.APPLY_MACRO_LINE_NOISE)
    apply_zscore = _parse_bool(args.APPLY_ZSCORE)
    power_time_apply_db = _parse_bool(args.POWER_TIME_APPLY_DB)
    power_time_theta_bands = _parse_theta_bands(args.POWER_TIME_THETA_BANDS)
    power_time_smooth_win_sec = _parse_smooth_window_sec(args.POWER_TIME_SMOOTH_WIN_SEC)
    anatomy_filter = str(args.anatomy_substring).strip().lower() or None
    if args.interactive_max_points <= 0:
        raise ValueError("--interactive_max_points must be > 0")
    if (not np.isfinite(float(args.MACRO_LINE_NOISE_Q))) or float(args.MACRO_LINE_NOISE_Q) <= 0:
        raise ValueError("--MACRO_LINE_NOISE_Q must be > 0")
    if (not np.isfinite(float(args.POWER_TIME_TF_WIN_SEC))) or float(args.POWER_TIME_TF_WIN_SEC) <= 0:
        raise ValueError("--POWER_TIME_TF_WIN_SEC must be > 0")
    if (not np.isfinite(float(args.POWER_TIME_TF_STEP_SEC))) or float(args.POWER_TIME_TF_STEP_SEC) <= 0:
        raise ValueError("--POWER_TIME_TF_STEP_SEC must be > 0")
    if (not np.isfinite(float(args.POWER_TIME_DB_EPS))) or float(args.POWER_TIME_DB_EPS) <= 0:
        raise ValueError("--POWER_TIME_DB_EPS must be > 0")

    nix_files = sorted([p for p in nix_dir.glob("Data_Subject_*_Session_*.h5") if NIX_FILE_RE.match(p.name)], key=lambda p: _natural_key(p.name))
    if not nix_files:
        raise RuntimeError(f"No NIX files found in: {nix_dir}")

    requested = {s.strip() for s in args.subject if s.strip()}

    subject_rows: list[dict[str, Any]] = []
    unit_rows_all: list[dict[str, Any]] = []
    source_rows_all: list[dict[str, Any]] = []

    for nix_path in nix_files:
        m = NIX_FILE_RE.match(nix_path.name)
        assert m is not None
        subject = f"sub-{int(m.group('subject')):02d}"

        if requested and subject not in requested:
            continue

        try:
            run = _extract_subject_run(
                nix_path=nix_path,
                trial_gap_sec=float(args.trial_gap_sec),
                anatomy_substring=anatomy_filter,
            )
        except RuntimeError as exc:
            if args.verbose:
                print(f"[skip] {subject}: {exc}")
            continue
        if args.skip_no_spike and (len(run.ms_units) == 0):
            if args.verbose:
                print(f"[skip] {run.subject}: no spike units")
            continue

        powerline_hz, powerline_json = _load_powerline_frequency(
            ds_root=ds_root,
            subject=run.subject,
            session=run.session,
        )
        macro_line_noise_hz = powerline_hz if apply_macro_line_noise else None

        out_html = out_dir / f"{run.subject}_{run.session}_signal_pseudo_lfp.html"
        out_psd_html = out_dir / f"{run.subject}_{run.session}_signal_pseudo_lfp_psd.html"
        out_powertime_html = out_dir / f"{run.subject}_{run.session}_signal_pseudo_lfp_power_time.html"
        _build_subject_plot(
            run=run,
            out_html=out_html,
            include_plotlyjs=str(args.plotly_js),
            ms_lfp_overlay=bool(overlay),
            ms_lfp_sigma=float(args.MS_LFP_SIGMA),
            ms_lfp_a=float(args.MS_LFP_A),
            ms_lfp_a0=float(args.MS_LFP_A0),
            ms_lfp_d_default=float(args.MS_LFP_D_DEFAULT),
            ms_lfp_post_smooth_sec=float(args.MS_LFP_POST_SMOOTH_SEC),
            max_points=int(args.interactive_max_points),
            apply_zscore=bool(apply_zscore),
            macro_line_noise_hz=(
                float(macro_line_noise_hz)
                if (macro_line_noise_hz is not None and np.isfinite(macro_line_noise_hz))
                else None
            ),
            macro_line_noise_q=float(args.MACRO_LINE_NOISE_Q),
        )
        _build_subject_psd_plot(
            run=run,
            out_html=out_psd_html,
            include_plotlyjs=str(args.plotly_js),
            ms_lfp_sigma=float(args.MS_LFP_SIGMA),
            ms_lfp_a=float(args.MS_LFP_A),
            ms_lfp_a0=float(args.MS_LFP_A0),
            ms_lfp_d_default=float(args.MS_LFP_D_DEFAULT),
            ms_lfp_post_smooth_sec=float(args.MS_LFP_POST_SMOOTH_SEC),
            max_points=int(args.interactive_max_points),
            apply_zscore=bool(apply_zscore),
            macro_line_noise_hz=(
                float(macro_line_noise_hz)
                if (macro_line_noise_hz is not None and np.isfinite(macro_line_noise_hz))
                else None
            ),
            macro_line_noise_q=float(args.MACRO_LINE_NOISE_Q),
        )
        _build_subject_power_time_plot(
            run=run,
            out_html=out_powertime_html,
            include_plotlyjs=str(args.plotly_js),
            ms_lfp_overlay=bool(overlay),
            ms_lfp_sigma=float(args.MS_LFP_SIGMA),
            ms_lfp_a=float(args.MS_LFP_A),
            ms_lfp_a0=float(args.MS_LFP_A0),
            ms_lfp_d_default=float(args.MS_LFP_D_DEFAULT),
            ms_lfp_post_smooth_sec=float(args.MS_LFP_POST_SMOOTH_SEC),
            max_points=int(args.interactive_max_points),
            apply_zscore=bool(apply_zscore),
            macro_line_noise_hz=(
                float(macro_line_noise_hz)
                if (macro_line_noise_hz is not None and np.isfinite(macro_line_noise_hz))
                else None
            ),
            macro_line_noise_q=float(args.MACRO_LINE_NOISE_Q),
            theta_bands=power_time_theta_bands,
            tf_win_sec=float(args.POWER_TIME_TF_WIN_SEC),
            tf_step_sec=float(args.POWER_TIME_TF_STEP_SEC),
            power_apply_db=bool(power_time_apply_db),
            power_db_eps=float(args.POWER_TIME_DB_EPS),
            power_smooth_win_sec=power_time_smooth_win_sec,
        )

        bids_summary = _load_bids_channel_summary(ds_root=ds_root, subject=run.subject)

        subject_row: dict[str, Any] = {
            "subject": run.subject,
            "session": run.session,
            "nix_file": nix_path.name,
            "trial_count": run.trial_count,
            "fs_hz": run.fs,
            "lfp_channel_count": len(run.channel_labels),
            "lfp_channels": ",".join(run.channel_labels),
            "n_mapped_units": len(run.ms_units),
            "mapped_units": ",".join([u for u, _ in run.ms_units]),
            "powerline_hz": ("" if powerline_hz is None else float(powerline_hz)),
            "powerline_json": powerline_json,
            "macro_line_noise_applied": bool(macro_line_noise_hz is not None),
            "zscore_applied": bool(apply_zscore),
            "plot_html": out_html.name,
            "psd_html": out_psd_html.name,
            "powertime_html": out_powertime_html.name,
            **bids_summary,
        }
        subject_rows.append(subject_row)

        for r in run.unit_inventory_rows:
            unit_rows_all.append(r)

        for r in run.source_inventory_rows:
            source_rows_all.append(
                {
                    "subject": run.subject,
                    "session": run.session,
                    **r,
                }
            )

        if args.verbose:
            print(
                f"[ok] {run.subject} {run.session}: "
                f"trials={run.trial_count}, lfp_ch={len(run.channel_labels)}, "
                f"mapped_units={len(run.ms_units)}, "
                f"line_noise_hz={macro_line_noise_hz}, zscore={bool(apply_zscore)} "
                f"-> {out_html.name}, {out_psd_html.name}, {out_powertime_html.name}"
            )

    if requested and not subject_rows:
        raise RuntimeError(f"No requested subjects found in NIX directory: {sorted(requested)}")

    # summaries
    subject_cols = [
        "subject",
        "session",
        "nix_file",
        "trial_count",
        "fs_hz",
        "lfp_channel_count",
        "lfp_channels",
        "n_mapped_units",
        "mapped_units",
        "powerline_hz",
        "powerline_json",
        "macro_line_noise_applied",
        "zscore_applied",
        "bids_channels_tsv",
        "bids_electrodes_tsv",
        "bids_channel_count",
        "bids_electrode_count",
        "bids_channel_groups",
        "bids_channel_names",
        "plot_html",
        "psd_html",
        "powertime_html",
    ]
    _write_tsv(out_dir / "subject_inventory.tsv", subject_rows, subject_cols)

    unit_cols = [
        "subject",
        "session",
        "nix_file",
        "spike_array",
        "mapped_unit",
        "shank",
        "orig_unit_id",
        "wire_prefix",
        "wire_num",
        "trial",
        "n_spikes",
        "source_uuid",
        "source_name",
        "macro_label",
        "anatomy",
        "soz",
    ]
    _write_tsv(out_dir / "unit_inventory.tsv", unit_rows_all, unit_cols)

    source_cols = [
        "subject",
        "session",
        "source_key",
        "source_uuid",
        "source_name",
        "macro_label",
        "anatomy",
        "soz",
    ]
    _write_tsv(out_dir / "source_inventory.tsv", source_rows_all, source_cols)

    summary_json = {
        "ds_root": str(ds_root),
        "nix_dir": str(nix_dir),
        "n_subject_pages": len(subject_rows),
        "subjects": [r["subject"] for r in subject_rows],
        "outputs": {
            "subject_inventory_tsv": str(out_dir / "subject_inventory.tsv"),
            "unit_inventory_tsv": str(out_dir / "unit_inventory.tsv"),
            "source_inventory_tsv": str(out_dir / "source_inventory.tsv"),
            "index_html": str(out_dir / "index.html"),
        },
        "params": {
            "trial_gap_sec": float(args.trial_gap_sec),
            "MS_LFP_OVERLAY": bool(overlay),
            "MS_LFP_SIGMA": float(args.MS_LFP_SIGMA),
            "MS_LFP_A": float(args.MS_LFP_A),
            "MS_LFP_A0": float(args.MS_LFP_A0),
            "MS_LFP_D_DEFAULT": float(args.MS_LFP_D_DEFAULT),
            "MS_LFP_POST_SMOOTH_SEC": float(args.MS_LFP_POST_SMOOTH_SEC),
            "APPLY_MACRO_LINE_NOISE": bool(apply_macro_line_noise),
            "MACRO_LINE_NOISE_Q": float(args.MACRO_LINE_NOISE_Q),
            "APPLY_ZSCORE": bool(apply_zscore),
            "POWER_TIME_THETA_BANDS": [label for label, _ in power_time_theta_bands],
            "POWER_TIME_TF_WIN_SEC": float(args.POWER_TIME_TF_WIN_SEC),
            "POWER_TIME_TF_STEP_SEC": float(args.POWER_TIME_TF_STEP_SEC),
            "POWER_TIME_APPLY_DB": bool(power_time_apply_db),
            "POWER_TIME_DB_EPS": float(args.POWER_TIME_DB_EPS),
            "POWER_TIME_SMOOTH_WIN_SEC": power_time_smooth_win_sec,
            "interactive_max_points": int(args.interactive_max_points),
            "plotly_js": str(args.plotly_js),
            "anatomy_substring": anatomy_filter,
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_index_html(out_dir / "index.html", subject_rows)

    print(f"-- output dir: {out_dir}")
    print(f"-- subject pages: {len(subject_rows)}")
    for r in subject_rows:
        print(
            f"   {r['subject']} {r['session']}: units={r['n_mapped_units']}, "
            f"lfp={r['lfp_channels']} -> {r['plot_html']} | PSD: {r.get('psd_html', '')} "
            f"| PowerTime: {r.get('powertime_html', '')}"
        )


if __name__ == "__main__":
    main()
