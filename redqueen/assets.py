"""Extract the LIBERO-plus asset pack wherever its folders sit inside the zip."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

# A file the simulator needs that exists only in the asset pack, not in the git repo.
SENTINEL = "scenes/kitchen250/kitchen_table_FabricCanvas001_COL_VAR1_4K.xml"


def find_assets_prefix(names: list[str]) -> str:
    """The path inside the zip that corresponds to `<libero>/assets/`."""
    for n in names:
        if n.endswith(SENTINEL):
            return n[: -len(SENTINEL)]
    tops = sorted({n.split("/")[0] for n in names})[:20]
    raise RuntimeError(f"asset pack has no {SENTINEL}; top-level entries: {tops}")


def extract_assets(zip_path: str | Path, assets_dir: str | Path) -> int:
    """Copy everything under the zip's assets folder into assets_dir. Returns file count."""
    assets_dir = Path(assets_dir).resolve()
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        prefix = find_assets_prefix(z.namelist())
        for info in z.infolist():
            if info.is_dir() or not info.filename.startswith(prefix):
                continue
            rel = info.filename[len(prefix):]
            if not rel or rel.startswith("__MACOSX"):
                continue
            dest = (assets_dir / rel).resolve()
            if assets_dir not in dest.parents:
                raise RuntimeError(f"unsafe path in zip: {info.filename}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    if not (assets_dir / SENTINEL).exists():
        raise RuntimeError(f"extraction finished but {assets_dir / SENTINEL} is still missing")
    return n
