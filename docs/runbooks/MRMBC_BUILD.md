# Building `mrmbc`

`mrmbc` is the compiled Fortran extension used by SDMBCv2 bias correction.
It is built from:

- `mbc_subroutines.f90`
- `mbc_subroutines.pyf`

Compiled Python extensions are tied to the Python ABI. For example:

- `mrmbc.cpython-311-x86_64-linux-gnu.so` works with Python 3.11
- `mrmbc.cpython-312-x86_64-linux-gnu.so` works with Python 3.12

If your Python version changes, rebuild `mrmbc`.

## NCI Current Workflow

The current prebuilt extension in this repository is compatible with Python
3.11. On NCI, use:

```bash
module use /g/data/xp65/public/modules
module load conda/analysis3-26.02
```

Then confirm:

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
python3 - <<'PY'
from mrmbc import mbc_subroutines as mbc
print("mrmbc import ok")
PY
```

## Rebuild

Use the helper script from `src`:

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
./build_mrmbc.sh
```

The script uses the active `python3` by default. To use a specific Python:

```bash
PYTHON=/path/to/python ./build_mrmbc.sh
```

## Python 3.12 Note

NumPy `f2py` on Python 3.12 uses the Meson backend. That means Meson must be
available in the environment. If Meson is missing, either:

- load/install Meson and rebuild, or
- use a Python 3.11 environment compatible with the existing prebuilt module

On NCI, `conda/analysis3-26.03` uses Python 3.12 and currently does not provide
Meson in the tested environment. `conda/analysis3-26.02` uses Python 3.11 and
can import the existing prebuilt `mrmbc` extension.

## Error Meaning

If you see:

```text
ModuleNotFoundError: No module named 'mrmbc'
```

then Python cannot find a compatible compiled extension for the active Python
version. Check:

```bash
python3 - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
ls mrmbc*.so
```

The suffix printed by Python must match one of the available `mrmbc*.so` files.
