#!/usr/bin/env python3
"""
Repackage dataset files into a subject-level hierarchy.

Target layout:
  Subject layout:
    <output_root>/<dataset>/<subject>/
      s<session>_HC_<label>.mat
      s<session>_MS_spike.mat
  Subject-less layout:
    <output_root>/<dataset>/
      s<session>_HC_<label>.mat
      s<session>_MS_spike.mat

The script keeps directory depth at dataset/subject and encodes session/region
in filenames. TT*.mat unit files are merged per session into one MS file.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - import failure is handled at runtime
    raise RuntimeError("numpy is required.") from exc

try:
    import h5py
except Exception as exc:  # pragma: no cover - import failure is handled at runtime
    raise RuntimeError("h5py is required to read v7.3 MAT files.") from exc

try:
    from scipy.io import loadmat, savemat
except Exception as exc:  # pragma: no cover - import failure is handled at runtime
    raise RuntimeError("scipy is required to read/write MAT files.") from exc

TT_RE = re.compile(r"^TT(?P<tetrode>\d+)(?:_(?P<unit>\d+))?\.mat$", re.IGNORECASE)
EEG_RE = re.compile(r"^(?P<prefix>.+)\.eeg\.(?P<ch>\d+)\.mat$", re.IGNORECASE)
REGION_TOKENS: Tuple[str, ...] = ("radiatum", "oriens", "pyramid", "ca1", "ca3", "dg")
OLD_REPACK_SUFFIX = "_repacked_subject_level_v3"
NEW_REPACK_SUFFIX = "_repacked"

MANIFEST_FIELDS: Tuple[str, ...] = (
    "dataset",
    "subject",
    "session",
    "kind",
    "source_path",
    "dest_path",
    "status",
    "details",
    "n_units",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Repackage session folders into dataset/subject hierarchy with "
            "session + region encoded in filenames."
        )
    )
    p.add_argument("--input-root", required=True, help="Original dataset root.")
    p.add_argument(
        "--output-root",
        default="",
        help="Destination root. Default: <input-root>_repacked.",
    )
    p.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset folder name under input-root. Repeatable. Default: auto-detect.",
    )
    p.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="copy",
        help="Deprecated: HC/Position files are always copied.",
    )
    p.add_argument(
        "--ms-format",
        choices=("npz", "mat"),
        default="mat",
        help="Format for aggregated MS unit files.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    p.add_argument(
        "--include-position",
        action="store_true",
        help="Include Position.mat files (default: off).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and print actions without writing files.",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose logs.")
    return p.parse_args()


def is_hidden_name(name: str) -> bool:
    return name.startswith(".") or name.startswith("._")


def sanitize_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-_.").lower()
    return token or "unknown"


def list_dirs(path: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(path.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_dir():
            continue
        if is_hidden_name(p.name):
            continue
        out.append(p)
    return out


def has_relevant_session_data(session_dir: Path) -> bool:
    for f in session_dir.iterdir():
        if not f.is_file():
            continue
        nm = f.name
        if is_hidden_name(nm):
            continue
        low = nm.lower()
        if low == "position.mat":
            return True
        if TT_RE.match(nm):
            return True
        if low.endswith(".mat") and EEG_RE.match(nm):
            return True
        if low.endswith(".mat") and any(tok in low for tok in REGION_TOKENS):
            return True
    return False


def discover_datasets(input_root: Path, selected: Sequence[str]) -> List[Path]:
    if selected:
        out: List[Path] = []
        for name in selected:
            ds = input_root / name
            if not ds.is_dir():
                raise FileNotFoundError(f"Dataset directory not found: {ds}")
            out.append(ds)
        return out

    datasets: List[Path] = []
    for ds in list_dirs(input_root):
        top_dirs = list_dirs(ds)
        if not top_dirs:
            continue
        if any(has_relevant_session_data(d) for d in top_dirs):
            datasets.append(ds)
            continue
        if any(has_relevant_session_data(s) for d in top_dirs for s in list_dirs(d)):
            datasets.append(ds)
    return datasets


def discover_subject_groups(dataset_dir: Path) -> List[Tuple[Optional[str], List[Path]]]:
    """
    Return groups of sessions as (subject_name, session_dirs).

    subject_name:
      - None for subject-less datasets where sessions are directly under dataset.
      - folder name for subject-based datasets.
    """
    groups: List[Tuple[Optional[str], List[Path]]] = []
    top_dirs = list_dirs(dataset_dir)

    direct_sessions = [d for d in top_dirs if has_relevant_session_data(d)]
    if direct_sessions:
        groups.append((None, direct_sessions))

    for subj in top_dirs:
        if has_relevant_session_data(subj):
            continue
        session_dirs = [s for s in list_dirs(subj) if has_relevant_session_data(s)]
        if session_dirs:
            groups.append((subj.name, session_dirs))

    return groups


def is_hc_file(name: str) -> bool:
    low = name.lower()
    if not low.endswith(".mat"):
        return False
    if EEG_RE.match(name):
        return True
    return any(tok in low for tok in REGION_TOKENS)


def hc_source_label(name: str) -> str:
    m = EEG_RE.match(name)
    if m:
        return f"eeg_ch{int(m.group('ch'))}"
    low = name.lower()
    for tok in REGION_TOKENS:
        if tok in low:
            return tok
    return sanitize_token(Path(name).stem)


def pick_hc_file(hc_files: Sequence[Path]) -> Optional[Path]:
    if not hc_files:
        return None
    eeg_candidates: List[Tuple[int, Path]] = []
    for p in hc_files:
        m = EEG_RE.match(p.name)
        if m:
            eeg_candidates.append((int(m.group("ch")), p))
    if eeg_candidates:
        eeg_candidates.sort(key=lambda x: (x[0], x[1].name.lower()))
        return eeg_candidates[0][1]
    return sorted(hc_files, key=lambda p: p.name.lower())[0]


def gather_session_files(session_dir: Path) -> Tuple[List[Path], List[Path], Optional[Path]]:
    hc_files: List[Path] = []
    tt_files: List[Path] = []
    position_file: Optional[Path] = None

    for f in sorted(session_dir.iterdir(), key=lambda x: x.name.lower()):
        if not f.is_file():
            continue
        nm = f.name
        if is_hidden_name(nm):
            continue
        if TT_RE.match(nm):
            tt_files.append(f)
            continue
        if nm.lower() == "position.mat":
            position_file = f
            continue
        if is_hc_file(nm):
            hc_files.append(f)

    return hc_files, tt_files, position_file


def safe_unlink(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir():
        raise IsADirectoryError(f"Refusing to unlink directory: {path}")
    path.unlink()


def materialize_file(
    src: Path,
    dst: Path,
    mode: str,
    overwrite: bool,
    dry_run: bool,
) -> str:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return "skip_exists"
        if not dry_run:
            safe_unlink(dst)

    if dry_run:
        return "dry_run"

    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(str(src.resolve()), str(dst))
        return "linked_symlink"
    if mode == "hardlink":
        try:
            os.link(str(src), str(dst))
            return "linked_hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copied_fallback"
    shutil.copy2(src, dst)
    return "copied"


def resolve_copy_mode(requested_mode: str) -> str:
    if requested_mode != "copy":
        print(
            "[info] --link-mode is deprecated. "
            f"Requested '{requested_mode}', but HC/Position files are always copied."
        )
    return "copy"


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
    if np.any(np.diff(x) < 0):
        x = np.sort(x)
    return x


def load_tt_spikes(path: Path) -> np.ndarray:
    try:
        mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        if "TS" in mat and isinstance(mat["TS"], np.ndarray):
            return _normalize_spike_array(np.asarray(mat["TS"]))
        first = _first_numeric_value(mat)
        if first is not None:
            return _normalize_spike_array(first)
    except NotImplementedError:
        pass
    except Exception:
        # Fall through to h5py attempt for robustness.
        pass

    with h5py.File(path, "r") as f:
        if "TS" in f and isinstance(f["TS"], h5py.Dataset):
            return _normalize_spike_array(np.asarray(f["TS"]))
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
                return _normalize_spike_array(np.asarray(obj))
    raise ValueError(f"No numeric spike vector found in TT file: {path}")


def parse_tt_name(path: Path) -> Tuple[int, int, str]:
    m = TT_RE.match(path.name)
    if not m:
        return -1, -1, sanitize_token(path.stem)
    tetrode = int(m.group("tetrode"))
    unit = int(m.group("unit")) if m.group("unit") is not None else -1
    unit_id = f"TT{tetrode}_{unit}" if unit >= 0 else f"TT{tetrode}"
    return tetrode, unit, unit_id


def to_matlab_field_name(text: str) -> str:
    field = re.sub(r"[^A-Za-z0-9_]", "_", str(text))
    if not field:
        field = "unit"
    if not re.match(r"^[A-Za-z]", field):
        field = f"u_{field}"
    return field


def save_ms_units_npz(
    out_path: Path,
    dataset: str,
    subject: str,
    session: str,
    rows: List[Dict[str, Any]],
) -> None:
    spikes_obj = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        spikes_obj[i] = row["spikes"]
    np.savez_compressed(
        out_path,
        dataset=np.array(dataset),
        subject=np.array(subject),
        session=np.array(session),
        unit_id=np.array([row["unit_id"] for row in rows], dtype=object),
        tetrode=np.array([row["tetrode"] for row in rows], dtype=np.int32),
        cluster=np.array([row["cluster"] for row in rows], dtype=np.int32),
        source_file=np.array([row["source_file"] for row in rows], dtype=object),
        spike_times=spikes_obj,
    )


def save_ms_units_mat(
    out_path: Path,
    dataset: str,
    subject: str,
    session: str,
    rows: List[Dict[str, Any]],
) -> None:
    ms_spike: Dict[str, Any] = {}
    field_names: List[str] = []
    used_fields: set[str] = set()

    for row in rows:
        base_field = to_matlab_field_name(row["unit_id"])
        field = base_field
        idx = 2
        while field in used_fields:
            field = f"{base_field}_{idx}"
            idx += 1
        used_fields.add(field)
        field_names.append(field)
        ms_spike[field] = row["spikes"].reshape(-1, 1)

    savemat(
        out_path,
        {
            "MS_spike": ms_spike,
            "dataset": np.array([dataset], dtype=object),
            "subject": np.array([subject], dtype=object),
            "session": np.array([session], dtype=object),
            "field_name": np.array(field_names, dtype=object).reshape(-1, 1),
            "unit_id": np.array([row["unit_id"] for row in rows], dtype=object).reshape(-1, 1),
            "tetrode": np.array([[row["tetrode"]] for row in rows], dtype=np.int32),
            "cluster": np.array([[row["cluster"]] for row in rows], dtype=np.int32),
            "source_file": np.array([row["source_file"] for row in rows], dtype=object).reshape(-1, 1),
        },
        do_compression=True,
    )


def ensure_unique_path(path: Path, used: set[Path]) -> Path:
    if path not in used:
        used.add(path)
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 2
    while True:
        cand = parent / f"{stem}__idx-{idx:02d}{suffix}"
        if cand not in used:
            used.add(cand)
            return cand
        idx += 1


def resolve_output_root(input_root: Path, output_root_arg: str) -> Tuple[Path, Optional[Path]]:
    if output_root_arg:
        raw = Path(output_root_arg).expanduser()
    else:
        raw = input_root.parent / f"{input_root.name}{NEW_REPACK_SUFFIX}"

    name_low = raw.name.lower()
    if name_low.endswith(OLD_REPACK_SUFFIX):
        stem = raw.name[: -len(OLD_REPACK_SUFFIX)]
        normalized = raw.with_name(f"{stem}{NEW_REPACK_SUFFIX}")
    else:
        normalized = raw
    return normalized.resolve(), raw.resolve()


def run(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).expanduser().resolve()
    output_root, requested_output_root = resolve_output_root(input_root, args.output_root)
    file_mode = resolve_copy_mode(args.link_mode)
    if not input_root.is_dir():
        raise FileNotFoundError(f"input-root does not exist: {input_root}")
    if requested_output_root != output_root:
        print(
            "[info] output-root name normalized: "
            f"{requested_output_root.name} -> {output_root.name}"
        )

    datasets = discover_datasets(input_root, args.dataset)
    if not datasets:
        raise RuntimeError(f"No dataset folders found under {input_root}")

    if args.verbose:
        print("Datasets:")
        for ds in datasets:
            print(f"  - {ds.name}")

    manifest_rows: List[Dict[str, Any]] = []
    stats = {
        "datasets": 0,
        "subjects": 0,
        "sessions": 0,
        "hc": 0,
        "ms": 0,
        "position": 0,
        "subjectless_groups": 0,
    }

    for ds_dir in datasets:
        dataset = ds_dir.name
        stats["datasets"] += 1
        groups = discover_subject_groups(ds_dir)
        if not groups:
            continue

        used_by_outdir: Dict[Path, set[Path]] = {}

        for subject_name, sessions in groups:
            if subject_name is None:
                stats["subjectless_groups"] += 1
                out_dir = output_root / dataset
                subject_for_manifest = ""
            else:
                stats["subjects"] += 1
                out_dir = output_root / dataset / subject_name
                subject_for_manifest = subject_name

            used_out_names = used_by_outdir.setdefault(out_dir, set())

            for sess_dir in sessions:
                session = sess_dir.name
                stats["sessions"] += 1
                hc_files, tt_files, position_file = gather_session_files(sess_dir)
                session_tok = sanitize_token(session)

                hc_path = pick_hc_file(hc_files)
                if hc_path is not None:
                    src_label = sanitize_token(hc_source_label(hc_path.name))
                    out_name = f"s{session_tok}_HC_{src_label}.mat"
                    out_path = ensure_unique_path(out_dir / out_name, used_out_names)
                    status = materialize_file(
                        src=hc_path,
                        dst=out_path,
                        mode=file_mode,
                        overwrite=args.overwrite,
                        dry_run=args.dry_run,
                    )
                    if status != "skip_exists":
                        stats["hc"] += 1
                    manifest_rows.append(
                        {
                            "dataset": dataset,
                            "subject": subject_for_manifest,
                            "session": session,
                            "kind": "hc",
                            "source_path": str(hc_path),
                            "dest_path": str(out_path),
                            "status": status,
                            "details": (
                                f"hc_candidates={len(hc_files)};selected={hc_path.name}"
                                if len(hc_files) > 1
                                else ""
                            ),
                            "n_units": "",
                        }
                    )

                if tt_files:
                    rows: List[Dict[str, Any]] = []
                    load_errors: List[str] = []
                    for tt_path in tt_files:
                        tetrode, cluster, unit_id = parse_tt_name(tt_path)
                        try:
                            spikes = load_tt_spikes(tt_path)
                        except Exception as exc:
                            load_errors.append(f"{tt_path.name}:{exc}")
                            continue
                        rows.append(
                            {
                                "unit_id": unit_id,
                                "tetrode": tetrode,
                                "cluster": cluster,
                                "source_file": tt_path.name,
                                "spikes": spikes,
                            }
                        )

                    ext = "npz" if args.ms_format == "npz" else "mat"
                    out_name = f"s{session_tok}_MS_spike.{ext}"
                    out_path = ensure_unique_path(out_dir / out_name, used_out_names)

                    if rows:
                        if not args.dry_run:
                            out_dir.mkdir(parents=True, exist_ok=True)
                            if out_path.exists() and args.overwrite:
                                safe_unlink(out_path)
                            if out_path.exists() and not args.overwrite:
                                status = "skip_exists"
                            else:
                                if args.ms_format == "npz":
                                    save_ms_units_npz(out_path, dataset, subject_for_manifest, session, rows)
                                else:
                                    save_ms_units_mat(out_path, dataset, subject_for_manifest, session, rows)
                                status = "saved"
                        else:
                            status = "dry_run"
                        if status != "skip_exists":
                            stats["ms"] += 1
                        detail = f"tt_files={len(tt_files)};units_saved={len(rows)}"
                    else:
                        status = "no_units_saved"
                        detail = f"tt_files={len(tt_files)};load_errors={len(load_errors)}"

                    if load_errors and args.verbose:
                        subject_label = subject_for_manifest if subject_for_manifest else "-"
                        print(f"[warn] {dataset}/{subject_label}/{session}: {len(load_errors)} TT load errors")
                        for msg in load_errors[:5]:
                            print(f"       {msg}")

                    manifest_rows.append(
                        {
                            "dataset": dataset,
                            "subject": subject_for_manifest,
                            "session": session,
                            "kind": "ms_units",
                            "source_path": str(sess_dir),
                            "dest_path": str(out_path),
                            "status": status,
                            "details": detail,
                            "n_units": str(len(rows)),
                        }
                    )

                if args.include_position and position_file is not None:
                    out_name = f"s{session_tok}_Position.mat"
                    out_path = ensure_unique_path(out_dir / out_name, used_out_names)
                    status = materialize_file(
                        src=position_file,
                        dst=out_path,
                        mode=file_mode,
                        overwrite=args.overwrite,
                        dry_run=args.dry_run,
                    )
                    if status != "skip_exists":
                        stats["position"] += 1
                    manifest_rows.append(
                        {
                            "dataset": dataset,
                            "subject": subject_for_manifest,
                            "session": session,
                            "kind": "position",
                            "source_path": str(position_file),
                            "dest_path": str(out_path),
                            "status": status,
                            "details": "",
                            "n_units": "",
                        }
                    )

    manifest_path = output_root / "repack_manifest.csv"
    if args.dry_run:
        print(f"[dry-run] Manifest would be written to: {manifest_path}")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(MANIFEST_FIELDS))
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"Manifest saved: {manifest_path}")

    print(
        "Summary: "
        f"datasets={stats['datasets']}, "
        f"subjects={stats['subjects']}, "
        f"sessions={stats['sessions']}, "
        f"hc_files={stats['hc']}, "
        f"ms_files={stats['ms']}, "
        f"subjectless_groups={stats['subjectless_groups']}, "
        f"position_files={stats['position']}"
    )


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
