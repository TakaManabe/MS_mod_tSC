#!/usr/bin/env python3
"""
Build an interactive waveform browser from repacked subject/session folders.

Expected input (from python_tools/main_repack.py):
  <root>/<dataset>/<subject>/s<session>_HC_*.mat
  <root>/<dataset>/<subject>/s<session>_MS_spike.mat

Also supports subject-less layout:
  <root>/<dataset>/s<session>_HC_*.mat
  <root>/<dataset>/s<session>_MS_spike.mat

Output:
  - Per-session/subject Plotly HTML pages
  - One matrix-style index HTML per dataset
    (rows=session, columns=subject, iframe viewer)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("h5py is required.") from exc

try:
    from scipy.io import loadmat
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy is required.") from exc

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as exc:  # pragma: no cover
    raise RuntimeError("plotly is required.") from exc


HC_FILE_RE = re.compile(r"^s(?P<session>.+?)_HC_(?P<label>.+)\.mat$", re.IGNORECASE)
MS_FILE_RE = re.compile(r"^s(?P<session>.+?)_MS_spike\.mat$", re.IGNORECASE)
HC_LABEL_CH_RE = re.compile(r"(?:^|_)eeg_ch(?P<ch>\d+)(?:$|_)", re.IGNORECASE)


@dataclass(frozen=True)
class CellEntry:
    dataset: str
    subject: str
    session: str
    hc_path: Optional[Path]
    hc_label: str
    ms_path: Optional[Path]
    page_rel_href: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create interactive waveform HTML pages/index from repacked "
            "session files (rows=session, cols=subject)."
        )
    )
    p.add_argument("--input-root", required=True, help="Repacked root directory.")
    p.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset folder name under input-root. Repeatable. Default: auto-detect.",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Default: input-root.",
    )
    p.add_argument(
        "--output-html",
        default="",
        help=(
            "Output index HTML path (single dataset only). "
            "When omitted, writes interactive_waveform_index_<dataset>.html in output-dir."
        ),
    )
    p.add_argument(
        "--page-dir",
        default="",
        help=(
            "Directory for generated per-cell pages. "
            "Default: <output-dir>/interactive_waveform_pages_<dataset>."
        ),
    )
    p.add_argument(
        "--sampling-rate",
        type=float,
        default=1000.0,
        help="Sampling rate [Hz] for sample index -> time conversion. Default: 1000.",
    )
    p.add_argument(
        "--max-eeg-points",
        type=int,
        default=12000,
        help="Max displayed EEG points per panel. Default: 12000.",
    )
    p.add_argument(
        "--max-spikes-per-unit",
        type=int,
        default=6000,
        help="Max displayed spike events per unit. Default: 6000.",
    )
    p.add_argument(
        "--plotly-js",
        choices=("cdn", "inline"),
        default="cdn",
        help="How to include plotly.js in each page. Default: cdn.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing page/index files.")
    p.add_argument("--dry-run", action="store_true", help="Scan and report without writing files.")
    p.add_argument("--verbose", action="store_true", help="Verbose logs.")
    return p.parse_args()


def is_hidden_name(name: str) -> bool:
    return name.startswith(".") or name.startswith("._")


def list_dirs(path: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(path.iterdir(), key=lambda x: x.name.lower()):
        if p.is_dir() and not is_hidden_name(p.name):
            out.append(p)
    return out


def natural_key(text: str) -> Tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(text))
    key: List[Any] = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return tuple(key)


def sanitize_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-_.")
    return token or "unknown"


def _first_numeric_value(data: Dict[str, Any]) -> Optional[np.ndarray]:
    for key, val in data.items():
        if key.startswith("__"):
            continue
        if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number):
            arr = np.asarray(val).reshape(-1)
            if arr.size > 0:
                return arr
    return None


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


def load_first_numeric_vector(path: Path) -> np.ndarray:
    try:
        mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        first = _first_numeric_value(mat)
        if first is not None:
            return np.asarray(first).reshape(-1)
    except NotImplementedError:
        pass
    except Exception:
        pass

    with h5py.File(path, "r") as f:
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
                return np.asarray(obj).reshape(-1)
    raise ValueError(f"No numeric vector found in MAT file: {path}")


def load_ms_spike_units(path: Path) -> List[Tuple[str, np.ndarray]]:
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    units: List[Tuple[str, np.ndarray]] = []

    ms = mat.get("MS_spike")
    fields = getattr(ms, "_fieldnames", None)
    if ms is not None and fields:
        for field in fields:
            spikes = _normalize_spike_array(np.asarray(getattr(ms, field)))
            if spikes.size > 0:
                units.append((str(field), spikes))
        return units

    # Backward-compat fallback: unit_id + spike_times arrays.
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


def choose_hc_file(files: Sequence[Path]) -> Tuple[Optional[Path], str]:
    if not files:
        return None, ""

    scored: List[Tuple[int, str, Path]] = []
    for p in files:
        label_match = HC_FILE_RE.match(p.name)
        label = label_match.group("label") if label_match else p.stem
        ch_match = HC_LABEL_CH_RE.search(label)
        ch = int(ch_match.group("ch")) if ch_match else 10**9
        scored.append((ch, p.name.lower(), p))

    scored.sort(key=lambda x: (x[0], x[1]))
    chosen = scored[0][2]
    m = HC_FILE_RE.match(chosen.name)
    label = m.group("label") if m else chosen.stem
    return chosen, label


def _linspace_idx(n: int, m: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.int64)
    if m <= 0 or n <= m:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, num=m, dtype=np.int64)


def downsample_eeg(y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).reshape(-1)
    idx = _linspace_idx(y.size, max_points)
    return idx, y[idx]


def downsample_spikes(spikes: np.ndarray, max_points: int) -> np.ndarray:
    spikes = np.asarray(spikes).reshape(-1)
    idx = _linspace_idx(spikes.size, max_points)
    return spikes[idx]


def _extract_session_map(dir_path: Path) -> Dict[str, Dict[str, Any]]:
    sess_map: Dict[str, Dict[str, Any]] = {}
    for f in sorted(dir_path.glob("*.mat"), key=lambda x: x.name.lower()):
        if not f.is_file() or is_hidden_name(f.name):
            continue
        mh = HC_FILE_RE.match(f.name)
        if mh:
            sess = mh.group("session")
            rec = sess_map.setdefault(sess, {"hc": [], "ms": None})
            rec["hc"].append(f)
            continue
        mm = MS_FILE_RE.match(f.name)
        if mm:
            sess = mm.group("session")
            rec = sess_map.setdefault(sess, {"hc": [], "ms": None})
            rec["ms"] = f
    return sess_map


def discover_dataset_dirs(input_root: Path, selected: Sequence[str]) -> List[Path]:
    if selected:
        out: List[Path] = []
        for name in selected:
            ds = input_root / name
            if not ds.is_dir():
                raise FileNotFoundError(f"Dataset directory not found: {ds}")
            out.append(ds)
        return out

    out: List[Path] = []
    for ds in list_dirs(input_root):
        top_map = _extract_session_map(ds)
        if top_map:
            out.append(ds)
            continue
        has_subject_data = any(_extract_session_map(subj) for subj in list_dirs(ds))
        if has_subject_data:
            out.append(ds)
    return out


def collect_cells_for_dataset(dataset_dir: Path) -> List[CellEntry]:
    dataset = dataset_dir.name
    rows: List[CellEntry] = []

    # Subject-less layout: sessions directly under dataset directory.
    top_map = _extract_session_map(dataset_dir)
    if top_map:
        for session, rec in sorted(top_map.items(), key=lambda kv: natural_key(kv[0])):
            hc_path, hc_label = choose_hc_file(rec["hc"])
            ms_path = rec["ms"]
            if hc_path is None and ms_path is None:
                continue
            rows.append(
                CellEntry(
                    dataset=dataset,
                    subject="(no-subject)",
                    session=session,
                    hc_path=hc_path,
                    hc_label=hc_label,
                    ms_path=ms_path,
                    page_rel_href="",
                )
            )

    # Subject layout: dataset/<subject>/...
    for subj_dir in list_dirs(dataset_dir):
        subj_map = _extract_session_map(subj_dir)
        if not subj_map:
            continue
        subject = subj_dir.name
        for session, rec in sorted(subj_map.items(), key=lambda kv: natural_key(kv[0])):
            hc_path, hc_label = choose_hc_file(rec["hc"])
            ms_path = rec["ms"]
            if hc_path is None and ms_path is None:
                continue
            rows.append(
                CellEntry(
                    dataset=dataset,
                    subject=subject,
                    session=session,
                    hc_path=hc_path,
                    hc_label=hc_label,
                    ms_path=ms_path,
                    page_rel_href="",
                )
            )
    return rows


_PLOTLY_RESET_LISTENER = """
<script>
(function () {
  function resetAll() {
    try {
      if (!window.Plotly) return;
      var plots = Array.prototype.slice.call(document.querySelectorAll(".js-plotly-plot"));
      for (var i = 0; i < plots.length; i++) {
        try { window.Plotly.relayout(plots[i], {"xaxis.autorange": true}); } catch (e) {}
        try { window.Plotly.relayout(plots[i], {"yaxis.autorange": true}); } catch (e) {}
      }
    } catch (e) {}
  }
  window.addEventListener("message", function (ev) {
    var d = ev && ev.data ? ev.data : null;
    if (!d || typeof d !== "object") return;
    if (d.type === "reset-plotly-view") resetAll();
  });
})();
</script>
"""


def inject_reset_listener(html_text: str) -> str:
    idx = html_text.rfind("</body>")
    if idx >= 0:
        return html_text[:idx] + _PLOTLY_RESET_LISTENER + html_text[idx:]
    return html_text + _PLOTLY_RESET_LISTENER


def build_cell_figure(
    entry: CellEntry,
    sampling_rate: float,
    max_eeg_points: int,
    max_spikes_per_unit: int,
) -> Optional[go.Figure]:
    hc_vec: Optional[np.ndarray] = None
    if entry.hc_path is not None:
        try:
            hc_vec = np.asarray(load_first_numeric_vector(entry.hc_path), dtype=np.float64).reshape(-1)
        except Exception:
            hc_vec = None

    units: List[Tuple[str, np.ndarray]] = []
    if entry.ms_path is not None:
        try:
            units = load_ms_spike_units(entry.ms_path)
        except Exception:
            units = []

    has_hc = hc_vec is not None and hc_vec.size > 0
    has_ms = len(units) > 0
    if not has_hc and not has_ms:
        return None

    n_rows = (1 if has_hc else 0) + len(units)
    eeg_px = 260
    ms_px = 34
    row_heights_px: List[float] = []
    if has_hc:
        row_heights_px.append(float(eeg_px))
    row_heights_px.extend([float(ms_px)] * len(units))

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.002,
        row_heights=row_heights_px,
    )

    row_idx = 1
    y_label_rows: List[Tuple[int, str]] = []
    x_max_sec = 0.0

    if has_hc:
        assert hc_vec is not None
        eeg_idx, eeg_y = downsample_eeg(hc_vec, max_eeg_points)
        eeg_t = eeg_idx.astype(np.float64) / float(sampling_rate)
        if eeg_t.size > 0:
            x_max_sec = max(x_max_sec, float(eeg_t[-1]))
        fig.add_trace(
            go.Scattergl(
                x=eeg_t,
                y=eeg_y,
                mode="lines",
                line=dict(color="#1f77b4", width=1.0),
                name="HC EEG",
                hovertemplate="t=%{x:.3f}s<br>HC=%{y:.3f}<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )
        y_title = f"HC {entry.hc_label}" if entry.hc_label else "HC"
        y_label_rows.append((row_idx, y_title))
        fig.update_yaxes(
            row=row_idx,
            col=1,
            showgrid=True,
            zeroline=False,
            ticks="outside",
        )
        row_idx += 1

    for unit_name, spikes in units:
        s = downsample_spikes(spikes, max_spikes_per_unit)
        if s.size == 0:
            continue
        t = (s.astype(np.float64) - 1.0) / float(sampling_rate)
        x_max_sec = max(x_max_sec, float(np.max(t)))
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
                name=unit_name,
                hovertemplate=f"{unit_name}<br>t=%{{x:.3f}}s<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(
            row=row_idx,
            col=1,
            range=[-1.0, 1.0],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        )
        y_label_rows.append((row_idx, unit_name))
        row_idx += 1

    total_height = int((eeg_px if has_hc else 120) + len(units) * ms_px + 140)
    title = f"{entry.dataset} | subject={entry.subject} | session={entry.session}"
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        dragmode="pan",
        hovermode="closest",
        margin=dict(l=142, r=20, t=54, b=44),
        height=max(total_height, 320),
        title=dict(text=title, x=0.01, xanchor="left", y=0.98, yanchor="top"),
    )

    # Use horizontal y-label annotations so HC/MS labels stay at the same x-position.
    label_x = -0.085
    ann: List[Dict[str, Any]] = []
    for r, txt in y_label_rows:
        axis_key = "yaxis" if r == 1 else f"yaxis{r}"
        axis_obj = getattr(fig.layout, axis_key, None)
        domain = getattr(axis_obj, "domain", None)
        if not domain or len(domain) != 2:
            continue
        y_mid = 0.5 * (float(domain[0]) + float(domain[1]))
        ann.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": label_x,
                "y": y_mid,
                "text": str(txt),
                "showarrow": False,
                "textangle": 0,
                "xanchor": "right",
                "yanchor": "middle",
                "align": "right",
                "font": {"size": 11, "color": "#2d3748"},
            }
        )
    if ann:
        fig.update_layout(annotations=ann)

    fig.update_xaxes(
        row=n_rows,
        col=1,
        title_text="Time (s)",
        showgrid=True,
        zeroline=False,
        range=[0.0, x_max_sec if x_max_sec > 0 else 1.0],
    )
    return fig


def write_cell_page(
    fig: go.Figure,
    page_path: Path,
    include_plotlyjs: str,
    overwrite: bool,
    dry_run: bool,
) -> str:
    if page_path.exists() and not overwrite:
        return "skip_exists"
    if dry_run:
        return "dry_run"

    page_path.parent.mkdir(parents=True, exist_ok=True)
    include_mode: Any = "cdn" if include_plotlyjs == "cdn" else True
    html_text = fig.to_html(include_plotlyjs=include_mode, full_html=True)
    html_text = inject_reset_listener(html_text)
    page_path.write_text(html_text, encoding="utf-8")
    return "saved"


def build_index_html(entries: Sequence[CellEntry], run_name: str) -> str:
    items = [
        {
            "dataset": e.dataset,
            "subject": e.subject,
            "session": e.session,
            "title": f"{e.dataset} | {e.subject} | {e.session}",
            "href": e.page_rel_href,
        }
        for e in entries
    ]
    items_json = json.dumps(items, ensure_ascii=False)
    run_name_json = json.dumps(run_name, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Interactive Waveform Index</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --line: #d9dde6;
      --txt: #212734;
      --muted: #5e6575;
      --accent: #1f6feb;
      --head-bg: #f8fbff;
      --empty: #f1f3f8;
      --row-col-w: 120px;
      --col-col-w: 132px;
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
      grid-template-columns: clamp(320px, 36vw, 520px) 1fr;
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
      background: var(--empty);
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
      grid-template-columns: auto auto auto 1fr;
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
    .meta {{
      min-width: 0;
      font-size: 13px;
      color: var(--muted);
      padding-left: 8px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
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
        Interactive Waveform Browser
        <span class="side-sub" id="runName"></span>
        <span class="side-sub">Grid: rows=Session, columns=Subject</span>
        <span class="side-sub">Keys: ←/→ subject, ↑/↓ session, R reset zoom</span>
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
        <div class="meta" id="meta"></div>
      </div>
      <div class="viewer">
        <iframe id="frame" title="interactive-plot"></iframe>
      </div>
    </main>
  </div>
  <script>
    const RUN_NAME = {run_name_json};
    const entries = {items_json};
    const runNameEl = document.getElementById("runName");
    const matrixEl = document.getElementById("matrix");
    const metaEl = document.getElementById("meta");
    const frameEl = document.getElementById("frame");
    runNameEl.textContent = "Run: " + RUN_NAME;

    function naturalSort(arr) {{
      return arr.slice().sort((a, b) => a.localeCompare(b, undefined, {{ numeric: true, sensitivity: "base" }}));
    }}

    const subjectList = naturalSort(Array.from(new Set(entries.map(e => e.subject))));

    // Row order: primary sort by subject block, secondary sort by current
    // natural session order within each subject.
    const sessionsBySubject = new Map();
    for (const subj of subjectList) sessionsBySubject.set(subj, []);
    for (const e of entries) {{
      if (!sessionsBySubject.has(e.subject)) sessionsBySubject.set(e.subject, []);
      sessionsBySubject.get(e.subject).push(e.session);
    }}
    const sessionList = [];
    const seenSession = new Set();
    for (const subj of subjectList) {{
      const subjSessions = naturalSort(Array.from(new Set(sessionsBySubject.get(subj) || [])));
      for (const sess of subjSessions) {{
        if (seenSession.has(sess)) continue;
        seenSession.add(sess);
        sessionList.push(sess);
      }}
    }}
    // Safety: append any remaining sessions not covered above.
    const remainingSessions = naturalSort(
      Array.from(new Set(entries.map(e => e.session))).filter(s => !seenSession.has(s))
    );
    for (const sess of remainingSessions) sessionList.push(sess);
    const entryMap = new Map();
    for (const e of entries) {{
      entryMap.set(`${{e.session}}|||${{e.subject}}`, e);
    }}

    let curRow = 0;
    let curCol = 0;

    function getEntry(r, c) {{
      const s = sessionList[r];
      const sub = subjectList[c];
      return entryMap.get(`${{s}}|||${{sub}}`) || null;
    }}

    function findFirstAvailable() {{
      for (let r = 0; r < sessionList.length; r++) {{
        for (let c = 0; c < subjectList.length; c++) {{
          if (getEntry(r, c)) return [r, c];
        }}
      }}
      return [0, 0];
    }}

    function buildMatrix() {{
      matrixEl.innerHTML = "";
      const headTr = document.createElement("tr");
      const corner = document.createElement("th");
      corner.className = "corner";
      corner.textContent = "Session";
      headTr.appendChild(corner);
      for (const subj of subjectList) {{
        const th = document.createElement("th");
        th.className = "colhead";
        th.textContent = subj;
        th.dataset.subject = subj;
        headTr.appendChild(th);
      }}
      matrixEl.appendChild(headTr);

      for (const sess of sessionList) {{
        const tr = document.createElement("tr");
        const rh = document.createElement("th");
        rh.className = "rowhead";
        rh.textContent = sess;
        rh.dataset.session = sess;
        tr.appendChild(rh);

        for (const subj of subjectList) {{
          const td = document.createElement("td");
          const e = entryMap.get(`${{sess}}|||${{subj}}`);
          if (!e) {{
            td.className = "empty";
            tr.appendChild(td);
            continue;
          }}
          td.className = "cell-wrap";
          const btn = document.createElement("button");
          btn.className = "cell";
          btn.textContent = "●";
          btn.title = e.title;
          btn.dataset.session = sess;
          btn.dataset.subject = subj;
          btn.addEventListener("click", () => openByKey(sess, subj));
          td.appendChild(btn);
          tr.appendChild(td);
        }}
        matrixEl.appendChild(tr);
      }}
    }}

    function updateActiveStyles() {{
      const sess = sessionList[curRow];
      const subj = subjectList[curCol];
      for (const el of matrixEl.querySelectorAll(".rowhead")) {{
        el.classList.toggle("active", el.dataset.session === sess);
      }}
      for (const el of matrixEl.querySelectorAll(".colhead")) {{
        el.classList.toggle("active", el.dataset.subject === subj);
      }}
      for (const el of matrixEl.querySelectorAll(".cell")) {{
        el.classList.toggle("active", el.dataset.session === sess && el.dataset.subject === subj);
      }}
    }}

    function openCurrent() {{
      const e = getEntry(curRow, curCol);
      if (!e) return false;
      frameEl.src = e.href;
      metaEl.textContent = `dataset=${{e.dataset}} | subject=${{e.subject}} | session=${{e.session}}`;
      updateActiveStyles();
      return true;
    }}

    function openByKey(session, subject) {{
      const r = sessionList.indexOf(session);
      const c = subjectList.indexOf(subject);
      if (r < 0 || c < 0) return;
      curRow = r;
      curCol = c;
      openCurrent();
    }}

    function move(dRow, dCol) {{
      if (!sessionList.length || !subjectList.length) return;
      let r = curRow;
      let c = curCol;
      const maxSteps = sessionList.length * subjectList.length;
      for (let i = 0; i < maxSteps; i++) {{
        r = (r + dRow + sessionList.length) % sessionList.length;
        c = (c + dCol + subjectList.length) % subjectList.length;
        if (getEntry(r, c)) {{
          curRow = r;
          curCol = c;
          openCurrent();
          return;
        }}
      }}
    }}

    function movePrevFlat() {{
      if (!sessionList.length || !subjectList.length) return;
      let idx = curRow * subjectList.length + curCol;
      const total = sessionList.length * subjectList.length;
      for (let step = 1; step <= total; step++) {{
        const j = (idx - step + total) % total;
        const r = Math.floor(j / subjectList.length);
        const c = j % subjectList.length;
        if (getEntry(r, c)) {{
          curRow = r;
          curCol = c;
          openCurrent();
          return;
        }}
      }}
    }}

    function moveNextFlat() {{
      if (!sessionList.length || !subjectList.length) return;
      let idx = curRow * subjectList.length + curCol;
      const total = sessionList.length * subjectList.length;
      for (let step = 1; step <= total; step++) {{
        const j = (idx + step) % total;
        const r = Math.floor(j / subjectList.length);
        const c = j % subjectList.length;
        if (getEntry(r, c)) {{
          curRow = r;
          curCol = c;
          openCurrent();
          return;
        }}
      }}
    }}

    function resetZoom() {{
      try {{
        frameEl.contentWindow.postMessage({{ type: "reset-plotly-view" }}, "*");
      }} catch (err) {{}}
    }}

    document.getElementById("prevBtn").addEventListener("click", movePrevFlat);
    document.getElementById("nextBtn").addEventListener("click", moveNextFlat);
    document.getElementById("resetBtn").addEventListener("click", resetZoom);
    document.addEventListener("keydown", (ev) => {{
      if (ev.key === "ArrowLeft") {{ ev.preventDefault(); move(0, -1); }}
      else if (ev.key === "ArrowRight") {{ ev.preventDefault(); move(0, 1); }}
      else if (ev.key === "ArrowUp") {{ ev.preventDefault(); move(-1, 0); }}
      else if (ev.key === "ArrowDown") {{ ev.preventDefault(); move(1, 0); }}
      else if (ev.key === "r" || ev.key === "R") {{ ev.preventDefault(); resetZoom(); }}
    }});

    buildMatrix();
    if (entries.length > 0) {{
      const first = findFirstAvailable();
      curRow = first[0];
      curCol = first[1];
      openCurrent();
    }} else {{
      metaEl.textContent = "No pages found.";
    }}
  </script>
</body>
</html>
"""


def build_dataset_outputs(
    dataset_dir: Path,
    output_dir: Path,
    output_html_override: Optional[Path],
    page_dir_override: Optional[Path],
    sampling_rate: float,
    max_eeg_points: int,
    max_spikes_per_unit: int,
    include_plotlyjs: str,
    overwrite: bool,
    dry_run: bool,
    verbose: bool,
) -> Tuple[Path, int, int]:
    dataset = dataset_dir.name
    rows = collect_cells_for_dataset(dataset_dir)
    if not rows:
        raise RuntimeError(f"No s*_HC_*.mat / s*_MS_spike.mat found in dataset: {dataset_dir}")

    if output_html_override is not None:
        output_html = output_html_override
    else:
        output_html = output_dir / f"interactive_waveform_index_{sanitize_token(dataset)}.html"

    if page_dir_override is not None:
        pages_dir = page_dir_override
    else:
        pages_dir = output_dir / f"interactive_waveform_pages_{sanitize_token(dataset)}"

    rows_with_href: List[CellEntry] = []
    n_saved = 0
    n_skipped = 0

    for row in rows:
        token = "__".join(
            [
                sanitize_token(row.dataset),
                sanitize_token(row.subject),
                sanitize_token(row.session),
            ]
        )
        page_path = pages_dir / f"{token}.html"
        rel_href = os.path.relpath(page_path, output_html.parent).replace(os.sep, "/")
        row2 = CellEntry(
            dataset=row.dataset,
            subject=row.subject,
            session=row.session,
            hc_path=row.hc_path,
            hc_label=row.hc_label,
            ms_path=row.ms_path,
            page_rel_href=rel_href,
        )

        fig = build_cell_figure(
            row2,
            sampling_rate=sampling_rate,
            max_eeg_points=max_eeg_points,
            max_spikes_per_unit=max_spikes_per_unit,
        )
        if fig is None:
            n_skipped += 1
            continue

        status = write_cell_page(
            fig=fig,
            page_path=page_path,
            include_plotlyjs=include_plotlyjs,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        if status in ("saved", "dry_run"):
            n_saved += 1
            rows_with_href.append(row2)
        else:
            n_skipped += 1

        if verbose:
            print(f"[{status}] {dataset} | {row.subject} | {row.session} -> {page_path}")

    if not rows_with_href:
        raise RuntimeError(f"No page entries generated for dataset: {dataset}")

    index_html = build_index_html(rows_with_href, run_name=dataset)
    if dry_run:
        print(f"[dry-run] Index would be written: {output_html}")
    else:
        output_html.parent.mkdir(parents=True, exist_ok=True)
        output_html.write_text(index_html, encoding="utf-8")
    return output_html, n_saved, n_skipped


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"input-root does not exist: {input_root}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_root
    output_html_override = Path(args.output_html).expanduser().resolve() if args.output_html else None
    page_dir_override = Path(args.page_dir).expanduser().resolve() if args.page_dir else None

    dataset_dirs = discover_dataset_dirs(input_root, args.dataset)
    if not dataset_dirs:
        raise RuntimeError(f"No dataset folders found under: {input_root}")

    if output_html_override is not None and len(dataset_dirs) != 1:
        raise ValueError("--output-html can be used only with a single dataset selection.")

    summary: List[Tuple[str, Path, int, int]] = []
    for ds_dir in dataset_dirs:
        out_html, n_saved, n_skipped = build_dataset_outputs(
            dataset_dir=ds_dir,
            output_dir=output_dir,
            output_html_override=output_html_override,
            page_dir_override=page_dir_override,
            sampling_rate=float(args.sampling_rate),
            max_eeg_points=int(args.max_eeg_points),
            max_spikes_per_unit=int(args.max_spikes_per_unit),
            include_plotlyjs=args.plotly_js,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
            verbose=bool(args.verbose),
        )
        summary.append((ds_dir.name, out_html, n_saved, n_skipped))

    for ds_name, out_html, n_saved, n_skipped in summary:
        print(
            f"Dataset={ds_name} | pages={n_saved} | skipped={n_skipped} | "
            f"index={out_html}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
