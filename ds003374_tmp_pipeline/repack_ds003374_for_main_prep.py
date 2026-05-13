#!/usr/bin/env python3
"""
Repack ds003374 NIX files into the main_PreP-compatible MAT layout.

Default output:
  /Volumes/T7_Taka/Minnesota/MSHC/Data_VargaV/23798184_repacked/ds003374/
    sub-XX/
      sses-01_ch01_HC_eeg_ch1_mal2-mal1.mat
      sses-01_ch01_MS_spike.mat

The target layout matches python_tools/main_repack.py / main_PreP.py:
  <repack-root>/<dataset>/<subject>/
    s<session>_HC_<label>.mat
    s<session>_MS_spike.mat

By default, each selected macro channel is emitted as a separate pseudo-session
so main_PreP analyzes every HC channel instead of silently selecting one HC file
per session. Use --channel-mode files to emit multiple HC files in the same
session, matching main_repack.py more literally.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import savemat

try:
    from scipy.signal import resample_poly
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy.signal.resample_poly is required for ds003374 repack.") from exc


NIX_FILE_RE = re.compile(r"^Data_Subject_(?P<subject>\d+)_Session_(?P<session>\d+)\.h5$", re.IGNORECASE)
IEEG_TRIAL_RE = re.compile(r"^iEEG_Data_Trial_(?P<trial>\d+)$", re.IGNORECASE)
SPIKE_ARRAY_RE = re.compile(
    r"^Spike_Times_Unit_(?P<unit>\d+)_(?P<wireprefix>[A-Za-z]+)_(?P<wirenum>\d+)_Trial_(?P<trial>\d+)$",
    re.IGNORECASE,
)

DEFAULT_DS_ROOT = "/Users/takamanabe/Documents/Git/ds003374"
DEFAULT_REPACK_ROOT = "/Volumes/T7_Taka/Minnesota/MSHC/Data_VargaV/23798184_repacked"
DEFAULT_DATASET = "ds003374"

MANIFEST_FIELDS = (
    "dataset",
    "subject",
    "session",
    "source_session",
    "channel",
    "kind",
    "source_path",
    "dest_path",
    "status",
    "details",
    "n_units",
)


@dataclass
class SubjectRun:
    subject: str
    session: str
    fs: float
    trial_count: int
    channel_labels: list[str]
    channel_data: dict[str, np.ndarray]
    ms_units: tuple[tuple[str, np.ndarray], ...]
    unit_inventory_rows: list[dict[str, Any]]
    source_inventory_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Repack ds003374 NIX files into MAT files that python_tools/main_PreP.py "
            "can analyze directly."
        )
    )
    p.add_argument("--ds-root", type=str, default=DEFAULT_DS_ROOT, help="Path to the ds003374 BIDS root.")
    p.add_argument(
        "--nix-dir",
        type=str,
        default="",
        help="Path to NIX files. Default: <ds-root>/bidsignore/data_NIX.",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_REPACK_ROOT,
        help=(
            "Repacked root used by main_PreP. Default: "
            "/Volumes/T7_Taka/Minnesota/MSHC/Data_VargaV/23798184_repacked"
        ),
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset folder name created under --output-root. Default: ds003374.",
    )
    p.add_argument(
        "--anatomy-substring",
        type=str,
        default="amyg",
        help="Keep channels/units whose NIX anatomy label contains this substring. Empty disables filtering.",
    )
    p.add_argument(
        "--target-sampling-rate",
        type=float,
        default=1000.0,
        help="Sampling rate written for HC and MS spike indices. Default: 1000 Hz.",
    )
    p.add_argument(
        "--channel-mode",
        choices=("separate", "files"),
        default="separate",
        help=(
            "separate: one pseudo-session per HC channel so main_PreP analyzes all channels. "
            "files: multiple HC files in the original session; main_PreP will choose one."
        ),
    )
    p.add_argument(
        "--subject",
        action="append",
        default=[],
        help="Subject token to include, e.g. sub-01. Repeatable. Default: all NIX files.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    p.add_argument("--dry-run", action="store_true", help="Scan and report without writing files.")
    p.add_argument("--verbose", action="store_true", help="Print per-file progress and warnings.")
    return p.parse_args()


def _decode_text(x: Any) -> str:
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        try:
            return bytes(x).decode("utf-8")
        except Exception:
            return str(x)
    return str(x)


def _natural_key(text: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(text))
    out: list[Any] = []
    for p in parts:
        out.append(int(p) if p.isdigit() else p.lower())
    return tuple(out)


def sanitize_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-_.").lower()
    return token or "unknown"


def matlab_field_name(text: str) -> str:
    field = re.sub(r"[^A-Za-z0-9_]", "_", str(text))
    if not field:
        field = "unit"
    if not re.match(r"^[A-Za-z]", field):
        field = f"u_{field}"
    return field


def _passes_anatomy_filter(anatomy: str, anatomy_substring: str | None) -> bool:
    if anatomy_substring is None:
        return True
    anat = str(anatomy or "").strip().lower()
    if not anat:
        return True
    return str(anatomy_substring).strip().lower() in anat


def _canonical_shank_from_wire(wire_prefix: str, fallback_map: dict[str, int]) -> int:
    up = str(wire_prefix).upper()
    if "AL" in up:
        return 1
    if "AR" in up:
        return 2
    if up not in fallback_map:
        fallback_map[up] = max([2, *fallback_map.values()]) + 1
    return int(fallback_map[up])


def _extract_nix_source_index(base: h5py.Group) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
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


def extract_subject_run(
    nix_path: Path,
    anatomy_substring: str | None,
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

        trial_names = sorted(
            [k for k in data_arrays.keys() if IEEG_TRIAL_RE.match(k)],
            key=_natural_key,
        )
        if not trial_names:
            raise RuntimeError(f"No iEEG trials found in {nix_path.name}")

        labels_ref: list[str] | None = None
        fs_ref: float | None = None
        trial_offsets: dict[int, float] = {}
        trial_t0: dict[int, float] = {}
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
                raise RuntimeError(f"Unexpected iEEG shape in {nix_path.name}::{trial_name}: {arr.shape}")

            dim1 = tg["dimensions"]["1"]["labels"][()]
            labels = [_decode_text(x) for x in np.asarray(dim1).reshape(-1)]
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
                    f"Cannot align data shape {arr.shape} with {len(labels)} labels in {nix_path.name}::{trial_name}"
                )

            n_time = data_ch_by_time.shape[1]
            trial_offsets[trial_no] = running_offset
            trial_t0[trial_no] = t0
            running_offset += float(n_time) / float(fs)

            for ch_idx, ch_label in enumerate(labels):
                channel_segments.setdefault(ch_label, []).append(data_ch_by_time[ch_idx, :].astype(np.float64))

            if labels_ref is None:
                labels_ref = labels
                fs_ref = fs
            else:
                if labels != labels_ref:
                    raise RuntimeError(f"Channel labels changed across trials in {nix_path.name}")
                if abs(fs - float(fs_ref)) > 1e-12:
                    raise RuntimeError(f"Sampling rate changed across trials in {nix_path.name}: {fs_ref} vs {fs}")

        assert labels_ref is not None
        assert fs_ref is not None

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
                    f"No iEEG channels passed anatomy filter '{anatomy_substring}' in {nix_path.name}"
                )

        channel_data = {
            label: np.concatenate(channel_segments[label]).astype(np.float64)
            for label in selected_labels
        }

        spike_names = sorted(
            [k for k in data_arrays.keys() if k.startswith("Spike_Times_Unit_")],
            key=_natural_key,
        )

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

            t_session = float(trial_offsets[trial_no]) + (spike_t - float(trial_t0[trial_no]))
            sample_idx = np.rint(t_session * float(fs_ref)).astype(np.int64) + 1
            sample_idx = sample_idx[sample_idx > 0]
            if sample_idx.size == 0:
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
            merged = np.concatenate(unit_spikes[mapped_unit]).astype(np.float64)
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
            channel_data=channel_data,
            ms_units=tuple(ms_units),
            unit_inventory_rows=unit_rows,
            source_inventory_rows=source_rows,
        )


def resample_signal(y: np.ndarray, source_fs: float, target_fs: float) -> np.ndarray:
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    if yy.size == 0:
        return yy
    if abs(float(source_fs) - float(target_fs)) <= 1e-9:
        return yy
    if source_fs <= 0 or target_fs <= 0:
        raise ValueError(f"Invalid sampling rates: source={source_fs}, target={target_fs}")

    ratio = Fraction(float(target_fs) / float(source_fs)).limit_denominator(100000)
    out = resample_poly(yy, int(ratio.numerator), int(ratio.denominator))
    expected = int(round(yy.size * float(target_fs) / float(source_fs)))
    if out.size > expected:
        out = out[:expected]
    elif out.size < expected:
        out = np.pad(out, (0, expected - out.size), mode="edge")
    return np.asarray(out, dtype=np.float64)


def convert_spike_indices(spikes: np.ndarray, source_fs: float, target_fs: float, max_samples: int) -> np.ndarray:
    x = np.asarray(spikes, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([], dtype=np.int64)
    t_sec = (x - 1.0) / float(source_fs)
    idx = np.rint(t_sec * float(target_fs)).astype(np.int64) + 1
    idx = idx[(idx > 0) & (idx <= int(max_samples))]
    if idx.size == 0:
        return np.array([], dtype=np.int64)
    if np.any(np.diff(idx) < 0):
        idx = np.sort(idx)
    return idx


def save_hc_mat(
    out_path: Path,
    eegx: np.ndarray,
    dataset: str,
    subject: str,
    session: str,
    source_session: str,
    channel_label: str,
    channel_index: int,
    source_fs: float,
    target_fs: float,
    source_file: str,
) -> None:
    savemat(
        out_path,
        {
            "eegx": np.asarray(eegx, dtype=np.float64).reshape(-1, 1),
            "dataset": np.array([dataset], dtype=object),
            "subject": np.array([subject], dtype=object),
            "session": np.array([session], dtype=object),
            "source_session": np.array([source_session], dtype=object),
            "source_file": np.array([source_file], dtype=object),
            "source_channel_label": np.array([channel_label], dtype=object),
            "source_channel_index": np.array([[int(channel_index)]], dtype=np.int32),
            "source_sampling_rate": np.array([[float(source_fs)]], dtype=np.float64),
            "sampling_rate": np.array([[float(target_fs)]], dtype=np.float64),
        },
        do_compression=True,
    )


def save_ms_spike_mat(
    out_path: Path,
    units: tuple[tuple[str, np.ndarray], ...],
    dataset: str,
    subject: str,
    session: str,
    source_session: str,
    source_fs: float,
    target_fs: float,
    max_samples: int,
    source_file: str,
) -> int:
    ms_spike: dict[str, Any] = {}
    field_names: list[str] = []
    unit_ids: list[str] = []
    used: set[str] = set()

    for unit_id, spikes in units:
        converted = convert_spike_indices(spikes, source_fs=source_fs, target_fs=target_fs, max_samples=max_samples)
        if converted.size == 0:
            continue
        base = matlab_field_name(unit_id)
        field = base
        idx = 2
        while field in used:
            field = f"{base}_{idx}"
            idx += 1
        used.add(field)
        field_names.append(field)
        unit_ids.append(str(unit_id))
        ms_spike[field] = converted.reshape(-1, 1)

    if not field_names:
        return 0

    savemat(
        out_path,
        {
            "MS_spike": ms_spike,
            "dataset": np.array([dataset], dtype=object),
            "subject": np.array([subject], dtype=object),
            "session": np.array([session], dtype=object),
            "source_session": np.array([source_session], dtype=object),
            "source_file": np.array([source_file], dtype=object),
            "field_name": np.array(field_names, dtype=object).reshape(-1, 1),
            "unit_id": np.array(unit_ids, dtype=object).reshape(-1, 1),
            "source_sampling_rate": np.array([[float(source_fs)]], dtype=np.float64),
            "sampling_rate": np.array([[float(target_fs)]], dtype=np.float64),
        },
        do_compression=True,
    )
    return len(field_names)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_unlink(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_dir():
            raise IsADirectoryError(f"Refusing to remove directory: {path}")
        path.unlink()


def write_or_skip(path: Path, overwrite: bool, dry_run: bool, writer: Any) -> str:
    if path.exists() or path.is_symlink():
        if not overwrite:
            return "skip_exists"
        if not dry_run:
            safe_unlink(path)
    if dry_run:
        return "dry_run"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer()
    return "saved"


def discover_nix_files(nix_dir: Path, subjects: list[str]) -> list[Path]:
    wanted = {s.strip().lower() for s in subjects if str(s).strip()}
    out: list[Path] = []
    for p in sorted(nix_dir.glob("*.h5"), key=lambda x: _natural_key(x.name)):
        if not p.is_file():
            continue
        m = NIX_FILE_RE.match(p.name)
        if m is None:
            continue
        subject = f"sub-{int(m.group('subject')):02d}"
        if wanted and subject.lower() not in wanted:
            continue
        out.append(p)
    return out


def run(args: argparse.Namespace) -> None:
    ds_root = Path(args.ds_root).expanduser().resolve()
    nix_dir = Path(args.nix_dir).expanduser().resolve() if str(args.nix_dir).strip() else ds_root / "bidsignore" / "data_NIX"
    output_root = Path(args.output_root).expanduser().resolve()
    dataset = sanitize_token(args.dataset)
    dataset_dir = output_root / dataset
    anatomy_substring = str(args.anatomy_substring).strip() or None
    target_fs = float(args.target_sampling_rate)

    if not ds_root.is_dir():
        raise FileNotFoundError(f"ds-root does not exist: {ds_root}")
    if not nix_dir.is_dir():
        raise FileNotFoundError(f"nix-dir does not exist: {nix_dir}")
    if target_fs <= 0 or not np.isfinite(target_fs):
        raise ValueError("--target-sampling-rate must be positive and finite.")

    nix_files = discover_nix_files(nix_dir, args.subject)
    if not nix_files:
        raise RuntimeError(f"No NIX .h5 files found in {nix_dir}")

    if args.verbose or args.dry_run:
        print(f"-- ds-root: {ds_root}")
        print(f"-- nix-dir: {nix_dir}")
        print(f"-- output dataset: {dataset_dir}")
        print(f"-- anatomy filter: {anatomy_substring if anatomy_substring is not None else 'disabled'}")
        print(f"-- target sampling rate: {target_fs:g} Hz")
        print(f"-- channel mode: {args.channel_mode}")
        print(f"-- NIX files: {len(nix_files)}")

    manifest_rows: list[dict[str, Any]] = []
    unit_rows_all: list[dict[str, Any]] = []
    source_rows_all: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    stats = {"nix": 0, "subjects": 0, "sessions": 0, "hc": 0, "ms": 0, "skipped": 0}

    for nix_path in nix_files:
        try:
            run_data = extract_subject_run(nix_path, anatomy_substring=anatomy_substring)
        except Exception as exc:
            stats["skipped"] += 1
            if args.verbose:
                print(f"[skip] {nix_path.name}: {exc}")
            continue

        stats["nix"] += 1
        stats["subjects"] += 1
        out_dir = dataset_dir / run_data.subject
        source_fs = float(run_data.fs)
        source_session = run_data.session
        if abs(source_fs - target_fs) > 1e-9 and args.verbose:
            print(f"-- {run_data.subject}/{source_session}: resampling {source_fs:g} -> {target_fs:g} Hz")

        source_rows_all.extend(
            {**row, "subject": run_data.subject, "session": source_session, "nix_file": nix_path.name}
            for row in run_data.source_inventory_rows
        )
        unit_rows_all.extend(run_data.unit_inventory_rows)

        ms_cache: dict[str, tuple[Path, str, int]] = {}
        for channel_idx, channel_label in enumerate(run_data.channel_labels, start=1):
            session = source_session
            if args.channel_mode == "separate":
                session = f"{source_session}_ch{channel_idx:02d}"

            session_tok = sanitize_token(session)
            label_tok = sanitize_token(f"eeg_ch{channel_idx}_{channel_label}")
            hc_name = f"s{session_tok}_HC_{label_tok}.mat"
            hc_path = out_dir / hc_name

            y_rs = resample_signal(
                run_data.channel_data[channel_label],
                source_fs=source_fs,
                target_fs=target_fs,
            )
            max_samples = int(y_rs.size)

            def _write_hc() -> None:
                save_hc_mat(
                    out_path=hc_path,
                    eegx=y_rs,
                    dataset=dataset,
                    subject=run_data.subject,
                    session=session,
                    source_session=source_session,
                    channel_label=channel_label,
                    channel_index=channel_idx,
                    source_fs=source_fs,
                    target_fs=target_fs,
                    source_file=nix_path.name,
                )

            hc_status = write_or_skip(hc_path, overwrite=bool(args.overwrite), dry_run=bool(args.dry_run), writer=_write_hc)
            if hc_status not in {"skip_exists"}:
                stats["hc"] += 1

            manifest_rows.append(
                {
                    "dataset": dataset,
                    "subject": run_data.subject,
                    "session": session,
                    "source_session": source_session,
                    "channel": channel_label,
                    "kind": "hc",
                    "source_path": str(nix_path),
                    "dest_path": str(hc_path),
                    "status": hc_status,
                    "details": (
                        f"source_fs={source_fs:g};target_fs={target_fs:g};"
                        f"source_samples={run_data.channel_data[channel_label].size};target_samples={max_samples}"
                    ),
                    "n_units": "",
                }
            )

            ms_session = session if args.channel_mode == "separate" else source_session
            if ms_session not in ms_cache:
                ms_name = f"s{sanitize_token(ms_session)}_MS_spike.mat"
                ms_path = out_dir / ms_name

                def _write_ms() -> None:
                    n_saved = save_ms_spike_mat(
                        out_path=ms_path,
                        units=run_data.ms_units,
                        dataset=dataset,
                        subject=run_data.subject,
                        session=ms_session,
                        source_session=source_session,
                        source_fs=source_fs,
                        target_fs=target_fs,
                        max_samples=max_samples,
                        source_file=nix_path.name,
                    )
                    if n_saved <= 0:
                        raise RuntimeError("No spike units remained after sampling-rate conversion.")

                if run_data.ms_units:
                    try:
                        ms_status = write_or_skip(
                            ms_path,
                            overwrite=bool(args.overwrite),
                            dry_run=bool(args.dry_run),
                            writer=_write_ms,
                        )
                        n_units_saved = len(run_data.ms_units)
                    except Exception as exc:
                        ms_status = "ms_error"
                        n_units_saved = 0
                        if args.verbose:
                            print(f"[warn] {run_data.subject}/{ms_session}: failed to save MS spikes: {exc}")
                else:
                    ms_status = "no_units"
                    n_units_saved = 0

                ms_cache[ms_session] = (ms_path, ms_status, n_units_saved)
                if ms_status not in {"skip_exists", "no_units", "ms_error"}:
                    stats["ms"] += 1

                manifest_rows.append(
                    {
                        "dataset": dataset,
                        "subject": run_data.subject,
                        "session": ms_session,
                        "source_session": source_session,
                        "channel": "*" if args.channel_mode == "files" else channel_label,
                        "kind": "ms_units",
                        "source_path": str(nix_path),
                        "dest_path": str(ms_path),
                        "status": ms_status,
                        "details": f"source_fs={source_fs:g};target_fs={target_fs:g}",
                        "n_units": str(n_units_saved),
                    }
                )

            stats["sessions"] += 1
            subject_rows.append(
                {
                    "dataset": dataset,
                    "subject": run_data.subject,
                    "session": session,
                    "source_session": source_session,
                    "channel_label": channel_label,
                    "channel_index": channel_idx,
                    "source_fs": source_fs,
                    "target_fs": target_fs,
                    "source_samples": int(run_data.channel_data[channel_label].size),
                    "target_samples": max_samples,
                    "n_units": len(run_data.ms_units),
                    "trial_count": run_data.trial_count,
                    "nix_file": nix_path.name,
                }
            )

        if args.verbose:
            print(
                f"-- repacked {run_data.subject}/{source_session}: "
                f"channels={len(run_data.channel_labels)}, units={len(run_data.ms_units)}"
            )

    manifest_path = dataset_dir / "repack_manifest.csv"
    summary_path = dataset_dir / "run_summary.json"
    subject_inventory_path = dataset_dir / "subject_inventory.tsv"
    unit_inventory_path = dataset_dir / "unit_inventory.tsv"
    source_inventory_path = dataset_dir / "source_inventory.tsv"

    if args.dry_run:
        print(f"[dry-run] Would write manifest: {manifest_path}")
    else:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(MANIFEST_FIELDS))
            writer.writeheader()
            writer.writerows(manifest_rows)
        write_tsv(subject_inventory_path, subject_rows)
        write_tsv(unit_inventory_path, unit_rows_all)
        write_tsv(source_inventory_path, source_rows_all)
        summary = {
            "ds_root": str(ds_root),
            "nix_dir": str(nix_dir),
            "output_root": str(output_root),
            "dataset": dataset,
            "dataset_dir": str(dataset_dir),
            "anatomy_substring": anatomy_substring,
            "target_sampling_rate": target_fs,
            "channel_mode": args.channel_mode,
            "stats": stats,
            "main_prep_example": (
                f"python python_tools/main_PreP.py --folder {dataset} "
                f"--sampling-rate {target_fs:g} --downsample-rate 500 --modes granger phaselag"
            ),
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Manifest saved: {manifest_path}")
        print(f"Summary saved: {summary_path}")

    print(
        "Summary: "
        f"nix_files={stats['nix']}, "
        f"subjects={stats['subjects']}, "
        f"sessions={stats['sessions']}, "
        f"hc_files={stats['hc']}, "
        f"ms_files={stats['ms']}, "
        f"skipped={stats['skipped']}"
    )
    print(f"main_PreP folder: {dataset}")


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
