#!/bin/bash
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

PYTHON=${PYTHON:-python3}

cat <<EOF
Building mrmbc for:
  python: $($PYTHON -c 'import sys; print(sys.executable)')
  version: $($PYTHON -c 'import sys; print(sys.version.split()[0])')
  suffix:  $($PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
EOF

if ! command -v gfortran >/dev/null 2>&1; then
    echo "ERROR: gfortran is not available in PATH." >&2
    echo "Load a compiler module or use an environment that provides gfortran." >&2
    exit 1
fi

$PYTHON - <<'PY'
import importlib.util
import sys

missing = [name for name in ("numpy",) if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"ERROR: missing Python package(s): {', '.join(missing)}")

try:
    import numpy.f2py  # noqa: F401
except Exception as exc:
    raise SystemExit(f"ERROR: numpy.f2py is not available: {exc}") from exc

if sys.version_info >= (3, 12) and importlib.util.find_spec("mesonbuild") is None:
    raise SystemExit(
        "ERROR: Python >=3.12 f2py requires Meson. Install/load meson or use "
        "a Python 3.11 environment such as conda/analysis3-26.02 on NCI."
    )
PY

$PYTHON -m numpy.f2py -c mbc_subroutines.pyf mbc_subroutines.f90

$PYTHON - <<'PY'
from mrmbc import mbc_subroutines as mbc

print("mrmbc import check passed.")
print("mbc_subroutines.day available:", hasattr(mbc, "day"))
PY
