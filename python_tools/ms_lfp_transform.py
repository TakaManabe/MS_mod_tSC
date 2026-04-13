from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np

_UNIT_NAME_RE = re.compile(r"^(?:TT|T)(?P<shank>\d+)_(?P<unit>\d+)$", re.IGNORECASE)


def parse_unit_name(unit_name: str) -> tuple[int, int] | None:
    """
    Parse unit name 'Txx_yy' or 'TTx_y' into (shank, unit) integer IDs.
    Returns None when the format does not match.
    """
    m = _UNIT_NAME_RE.match(str(unit_name).strip())
    if not m:
        return None
    try:
        shank = int(m.group("shank"))
        unit = int(m.group("unit"))
    except Exception:
        return None
    if shank < 0 or unit < 0:
        return None
    return shank, unit


def build_ap_kernel(
    sampling_rate: float,
    sigma_sec: float = 0.0005,
    amplitude: float = 1.0,
    span_sigma: float = 5.0,
) -> np.ndarray:
    """
    Discretize V(t) = -A * (t/sigma^2) * exp(-t^2/(2*sigma^2)).

    The kernel spans [-span_sigma*sigma, +span_sigma*sigma] on the
    sampling grid defined by sampling_rate.
    """
    fs = float(sampling_rate)
    sigma = float(sigma_sec)
    if fs <= 0:
        raise ValueError("sampling_rate must be > 0.")
    if (not np.isfinite(sigma)) or sigma <= 0:
        raise ValueError("sigma_sec must be a positive finite value.")

    span = max(float(span_sigma), 1.0)
    half_width = max(1, int(np.ceil(span * sigma * fs)))
    t = np.arange(-half_width, half_width + 1, dtype=float) / fs
    sigma2 = sigma * sigma
    kernel = -float(amplitude) * (t / sigma2) * np.exp(-(t * t) / (2.0 * sigma2))
    return np.asarray(kernel, dtype=float)


def _normalize_id_key(value: Any) -> str | None:
    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        fv = float(value)
        if not np.isfinite(fv):
            return None
        iv = int(round(fv))
        if abs(fv - iv) > 1e-9:
            return None
        return str(iv)

    s = str(value).strip()
    if not s:
        return None
    if re.fullmatch(r"\d+", s):
        return str(int(s))
    return None


def _canonicalize_distance_map(
    distance_map: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not distance_map or not isinstance(distance_map, Mapping):
        return out

    for shank_raw, unit_map_raw in distance_map.items():
        shank_key = _normalize_id_key(shank_raw)
        if shank_key is None:
            continue
        if not isinstance(unit_map_raw, Mapping):
            continue

        unit_map: dict[str, float] = {}
        for unit_raw, dist_raw in unit_map_raw.items():
            unit_key = _normalize_id_key(unit_raw)
            if unit_key is None:
                continue
            try:
                dist_val = float(dist_raw)
            except Exception:
                dist_val = float("nan")
            unit_map[unit_key] = dist_val

        out[shank_key] = unit_map
    return out


def _gaussian_smooth_1d(
    y: np.ndarray,
    sampling_rate: float,
    sigma_sec: float,
) -> np.ndarray:
    x = np.asarray(y, dtype=float).reshape(-1)
    if x.size < 3:
        return x
    sigma = float(sigma_sec)
    fs = float(sampling_rate)
    if sigma <= 0 or fs <= 0:
        return x
    sigma_samp = sigma * fs
    if sigma_samp < 0.5:
        return x
    half_width = max(1, int(np.ceil(4.0 * sigma_samp)))
    kx = np.arange(-half_width, half_width + 1, dtype=float)
    ker = np.exp(-0.5 * (kx / sigma_samp) ** 2)
    denom = float(np.sum(ker))
    if not np.isfinite(denom) or denom <= 0:
        return x
    ker /= denom
    return np.convolve(x, ker, mode="same")


def synthesize_shank_lfp(
    units: Sequence[tuple[str, np.ndarray]],
    sampling_rate: float,
    time_range_sec: tuple[float, float],
    sigma_sec: float = 0.0005,
    amplitude: float = 1.0,
    gain_a0: float = 1.0,
    distance_map: Mapping[str, Mapping[str, float]] | None = None,
    default_distance: float = 1.0,
    post_smooth_sec: float = 0.012,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Build pseudo-LFP per shank from unit spikes.

    Distance behavior:
      - If distance_map is non-empty, d_ij is read strictly from (shank, unit).
      - If distance_map is empty, default_distance is expanded to all detected
        units in each shank for this session (d=[d0, d0, ...]).
      - post_smooth_sec applies optional Gaussian smoothing to pseudo-LFP.

    Returns:
      rows: list of dict containing
        - shank_int: int
        - shank_label: str
        - time_sec: np.ndarray
        - lfp: np.ndarray
        - spike_t: np.ndarray (for optional raster overlay)
      warnings: list of warning messages (without subject/session prefix)
    """
    fs = float(sampling_rate)
    if fs <= 0:
        raise ValueError("sampling_rate must be > 0.")

    t0 = float(time_range_sec[0])
    t1 = float(time_range_sec[1])
    if (not np.isfinite(t0)) or (not np.isfinite(t1)) or t1 <= t0:
        return [], []

    kernel = build_ap_kernel(
        sampling_rate=fs,
        sigma_sec=sigma_sec,
        amplitude=amplitude,
    )
    radius = int(kernel.size // 2)

    n_samples = int(np.floor((t1 - t0) * fs)) + 1
    if n_samples < 2:
        return [], []

    t_axis = t0 + (np.arange(n_samples, dtype=float) / fs)
    d_lookup = _canonicalize_distance_map(distance_map)
    has_explicit_map = bool(d_lookup)
    d_default = float(default_distance)
    if (not np.isfinite(d_default)) or d_default <= 0.0:
        raise ValueError("default_distance must be a positive finite value.")
    post_smooth = float(post_smooth_sec)
    if (not np.isfinite(post_smooth)) or post_smooth < 0.0:
        raise ValueError("post_smooth_sec must be a finite value >= 0.")

    # If no explicit distance map is provided, auto-expand a scalar default
    # to all units found in each shank for this session.
    if not has_explicit_map:
        for unit_name, _ in units:
            parsed = parse_unit_name(str(unit_name))
            if parsed is None:
                continue
            shank_int, unit_int = parsed
            shank_key = str(shank_int)
            unit_key = str(unit_int)
            d_lookup.setdefault(shank_key, {})[unit_key] = d_default

    # Keep padding so spikes slightly outside [t0, t1] can still contribute
    # through the finite AP kernel support.
    pad_n = n_samples + (2 * radius)
    shank_delta: dict[int, np.ndarray] = {}
    shank_spike_t: dict[int, list[np.ndarray]] = {}
    warnings: list[str] = []

    t_conv_min = t0 - (radius / fs)
    t_conv_max = t1 + (radius / fs)

    for unit_name, spike_idx in units:
        unit_name_s = str(unit_name)
        parsed = parse_unit_name(unit_name_s)
        if parsed is None:
            warnings.append(
                f'"{unit_name_s}" is not in Txx_yy/TTx_y format and was excluded from MS_LFP.'
            )
            continue

        shank_int, unit_int = parsed
        shank_key = str(shank_int)
        unit_key = str(unit_int)

        dist_val = d_lookup.get(shank_key, {}).get(unit_key, None)
        if dist_val is None or (not np.isfinite(dist_val)) or (float(dist_val) <= 0.0):
            warnings.append(
                f'distance d_ij missing/invalid for "{unit_name_s}" '
                f"in shank {shank_key}, unit {unit_key}; excluded from MS_LFP."
            )
            continue

        s = np.asarray(spike_idx, dtype=float).reshape(-1)
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue

        t_spike = (s - 1.0) / fs
        m_conv = (t_spike >= t_conv_min) & (t_spike <= t_conv_max)
        if not np.any(m_conv):
            continue

        local_idx = np.rint((t_spike[m_conv] - t0) * fs).astype(np.int64)
        delta = shank_delta.get(shank_int)
        if delta is None:
            delta = np.zeros(pad_n, dtype=float)
            shank_delta[shank_int] = delta

        idx_pad = local_idx + radius
        m_valid = (idx_pad >= 0) & (idx_pad < delta.size)
        if np.any(m_valid):
            gain = float(gain_a0) / float(dist_val)
            np.add.at(delta, idx_pad[m_valid], gain)

        m_view = (t_spike >= t0) & (t_spike <= t1)
        if np.any(m_view):
            shank_spike_t.setdefault(shank_int, []).append(
                np.asarray(t_spike[m_view], dtype=float)
            )

    out_rows: list[dict[str, Any]] = []
    for shank_int in sorted(shank_delta.keys()):
        delta = shank_delta[shank_int]
        conv = np.convolve(delta, kernel, mode="same")
        y = np.asarray(conv[radius : radius + n_samples], dtype=float)
        y = _gaussian_smooth_1d(y, sampling_rate=fs, sigma_sec=post_smooth)

        spike_blocks = shank_spike_t.get(shank_int, [])
        if spike_blocks:
            spikes = np.sort(np.concatenate(spike_blocks).reshape(-1))
        else:
            spikes = np.array([], dtype=float)

        out_rows.append(
            {
                "shank_int": int(shank_int),
                "shank_label": f"T{int(shank_int):02d}",
                "time_sec": t_axis,
                "lfp": y,
                "spike_t": spikes,
            }
        )

    return out_rows, warnings
