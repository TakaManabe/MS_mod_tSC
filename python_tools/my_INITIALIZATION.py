import os
from pathlib import Path


REPACK_ROOT = "/Volumes/T7_Taka/Minnesota/MSHC/Data_VargaV/23798184_repacked"


def _validate_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: '{path}'")


def initialization(dataset_folder: str | None = None) -> tuple[str, str]:
    """
    Minimal loader initialization for repacked data.

    - Base root is fixed to REPACK_ROOT.
    - Dataset folder is requested with:
        '-- Enter a dataset folder name:'
      if not explicitly passed.
    """

    root_dir = Path(REPACK_ROOT).expanduser().resolve()
    _validate_dir(root_dir, "Repacked root")

    folder = (dataset_folder or "").strip()
    if not folder:
        folder = input("-- Enter a dataset folder name: ").strip()
    if not folder:
        raise ValueError("Dataset folder name is required.")

    dataset_dir = (root_dir / folder).resolve()
    _validate_dir(dataset_dir, "Dataset")
    print(f"-- Data folder name to be loaded: {dataset_dir}")
    return str(root_dir), str(dataset_dir)
