#!/usr/bin/env python3
"""
Compute climatological statistics from bias-corrected NetCDF outputs (e.g. bc_corrected*.nc).

NCI Gadi: use the analysis conda module (includes xarray, netCDF4, numpy). Do not rely on
``pip install`` on Gadi; load modules instead::

    module use /g/data/xp65/public/modules
    module load conda/analysis3

Example:
    cd /path/to/sdmbc_v2/src
    python compute_bc_climatology.py \\
      /path/to/bc_corrected_3d_lev_0_...nc \\
      -o /path/to/bc_climatology.nc
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
import glob
import re

import numpy as np
import xarray as xr
import yaml

from stats_functions import build_bc_climatology_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


_GADI_ENV = (
    "On NCI Gadi, load Python via modules (do not pip-install netCDF4): "
    "'module use /g/data/xp65/public/modules' then 'module load conda/analysis3'."
)

def _default_output_path(input_path: str, outdir: str | None) -> str:
    """
    Derive an output filename from the input basename while keeping model info.

    Example:
      bc_corrected_3d_lev_0_6hrLev_ACCESS-ESM1-5_historical_..._1984_2014.nc
    -> bc_climatology_3d_lev_0_6hrLev_ACCESS-ESM1-5_historical_..._1984_2014.nc
    """
    p = Path(input_path)
    name = p.name

    if name.startswith("bc_corrected_"):
        out_name = name.replace("bc_corrected_", "bc_climatology_", 1)
    elif name.startswith("bc_"):
        out_name = name.replace("bc_", "bc_climatology_", 1)
    else:
        out_name = f"{p.stem}_climatology.nc"

    out_parent = Path(outdir) if outdir else p.parent
    return str(out_parent / out_name)


def _default_bias_output_path(stats_output_path: str) -> str:
    p = Path(stats_output_path)
    name = p.name
    if name.startswith("bc_climatology_"):
        out_name = name.replace("bc_climatology_", "bc_climatology_bias_", 1)
    else:
        out_name = f"{p.stem}_bias.nc"
    return str(p.parent / out_name)


def _read_config(path: str) -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)


def _guess_gname_from_bc_input(input_path: str) -> str | None:
    # Example suffix in BC file names:
    # ..._6hrLev_ACCESS-ESM1-5_historical_r6i1p1f1_gn_1984_2014.nc
    stem = Path(input_path).stem
    parts = stem.split("_")
    if len(parts) < 7:
        return None
    # Try to read token after infor (e.g., 6hrLev)
    # ... bc_corrected_3d_lev_0_6hrLev_<gname>_<period>_...
    try:
        idx = parts.index("6hrLev")
        return parts[idx + 1]
    except Exception:
        return None


def _extract_years_from_name(input_path: str) -> tuple[int | None, int | None]:
    stem = Path(input_path).stem
    parts = stem.split("_")
    years = [int(p) for p in parts if p.isdigit() and len(p) == 4]
    if len(years) >= 2:
        return years[-2], years[-1]
    return None, None


def _extract_level_from_name(input_path: str) -> int | None:
    match = re.search(r"(?:^|_)lev_(\d+)(?:_|$)", Path(input_path).stem)
    if match:
        return int(match.group(1))
    return None


def _reference_level_index(config: SimpleNamespace, bc_input_path: str) -> int:
    filename_level = _extract_level_from_name(bc_input_path)
    config_level = int(getattr(config, "slevel", 0))
    if filename_level is not None:
        if filename_level != config_level:
            logger.warning(
                "Using level %s inferred from input filename; config slevel is %s.",
                filename_level,
                config_level,
            )
        return filename_level
    return config_level


def _collect_reference_files(
    config: SimpleNamespace,
    bc_input_path: str,
) -> list[str]:
    obs_path = getattr(config, "obs_path", None)
    if not obs_path:
        raise ValueError("Config is missing 'obs_path'.")

    target_vars = list(getattr(config, "target_variable", []))
    if not target_vars:
        raise ValueError("Config is missing 'target_variable'.")

    gname = getattr(config, "gname", None) or _guess_gname_from_bc_input(bc_input_path)
    if not gname:
        raise ValueError("Cannot determine target model gname for reference file matching.")

    start_year = getattr(config, "startyear_h", None)
    end_year = getattr(config, "endyear_h", None)
    if start_year is None or end_year is None:
        y0, y1 = _extract_years_from_name(bc_input_path)
        start_year = y0
        end_year = y1
    if start_year is None or end_year is None:
        raise ValueError(
            "Cannot determine year range. Provide startyear_h/endyear_h in config."
        )

    files: list[str] = []
    for var in target_vars:
        for y in range(int(start_year), int(end_year) + 1):
            pattern = f"{obs_path}/{var}_*_to_{gname}_{y}*.nc"
            files.extend(glob.glob(pattern))

    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No reference files found under {obs_path} for vars={target_vars}, "
            f"gname={gname}, years={start_year}-{end_year}"
        )
    return files


def _build_encoding(ds: xr.Dataset) -> dict:
    return {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}


def _select_common_bias_variables(
    bc_ds: xr.Dataset,
    ref_ds: xr.Dataset,
    *,
    single_grid_test: bool = False,
    level_index: int | None = None,
) -> tuple[xr.Dataset, xr.Dataset, list[str]]:
    """
    Return the common variable subset to bias against, along with the selected names.

    The one-grid smoke test intentionally writes wind outputs differently from the
    reference file, so bias should only use the overlapping climatology variables.
    """
    common_vars = sorted(set(bc_ds.data_vars) & set(ref_ds.data_vars))
    if not common_vars:
        raise ValueError(
            "Bias inputs do not share any climatology variables. "
            f"Corrected vars={sorted(bc_ds.data_vars)}; "
            f"reference vars={sorted(ref_ds.data_vars)}"
        )

    missing_ref = sorted(set(bc_ds.data_vars) - set(ref_ds.data_vars))
    missing_bc = sorted(set(ref_ds.data_vars) - set(bc_ds.data_vars))
    if missing_ref or missing_bc:
        logger.warning(
            "Bias variables differ; using common subset %s. Missing in reference: %s; missing in corrected: %s",
            common_vars,
            missing_ref or [],
            missing_bc or [],
        )

    bc_sel = bc_ds[common_vars]
    ref_sel = ref_ds[common_vars]

    for dim in ("lev", "lat", "lon"):
        if dim not in bc_sel.dims or dim not in ref_sel.dims:
            continue

        bc_size = int(bc_sel.sizes[dim])
        ref_size = int(ref_sel.sizes[dim])

        if dim == "lev":
            if bc_size != ref_size:
                if bc_size == 1 and ref_size > 1:
                    idx = 0 if level_index is None else int(level_index)
                    ref_sel = ref_sel.isel({dim: [idx]})
                elif ref_size == 1 and bc_size > 1:
                    idx = 0 if level_index is None else int(level_index)
                    bc_sel = bc_sel.isel({dim: [idx]})
                else:
                    raise ValueError(
                        f"Cannot compare climatologies with {dim} sizes "
                        f"corrected={bc_size}, reference={ref_size}."
                    )

            # Corrected outputs often carry lev=0 for the processed level while
            # reference files carry the physical hybrid-level coordinate. The
            # selected level is what matters for bias maps, so normalize labels.
            if bc_sel.sizes.get(dim) == ref_sel.sizes.get(dim):
                idx_coord = np.arange(int(bc_sel.sizes[dim]))
                bc_sel = bc_sel.assign_coords({dim: idx_coord})
                ref_sel = ref_sel.assign_coords({dim: idx_coord})
            continue

        if bc_size != ref_size:
            if bc_size < ref_size:
                targets = np.asarray(bc_sel[dim].values)
                ref_sel = ref_sel.sel({dim: targets}, method="nearest")
            elif ref_size < bc_size:
                targets = np.asarray(ref_sel[dim].values)
                bc_sel = bc_sel.sel({dim: targets}, method="nearest")
            else:
                raise ValueError(
                    f"Cannot compare climatologies with {dim} sizes "
                    f"corrected={bc_size}, reference={ref_size}."
                )

        if bc_sel.sizes.get(dim) == ref_sel.sizes.get(dim):
            # Preserve corrected-output map coordinates for plotting while
            # forcing exact alignment after nearest-neighbour reference subset.
            coord = np.asarray(bc_sel[dim].values)
            ref_sel = ref_sel.assign_coords({dim: coord})

    return bc_sel, ref_sel, common_vars


def _subset_reference_to_corrected_domain(
    ref_ds: xr.Dataset,
    bc_ds: xr.Dataset,
    *,
    level_index: int | None = None,
) -> xr.Dataset:
    """
    Subset the reference time series to the corrected-output spatial domain
    before climatology calculation.

    This is critical for tile tests: opening and aggregating the full reference
    domain is much more expensive than selecting the corrected lev/lat/lon first.
    """
    ds = ref_ds

    if "lev" in ds.dims and "lev" in bc_ds.dims:
        bc_lev_size = int(bc_ds.sizes["lev"])
        ref_lev_size = int(ds.sizes["lev"])
        if bc_lev_size == 1 and ref_lev_size > 1:
            idx = 0 if level_index is None else int(level_index)
            ds = ds.isel(lev=[idx])
        elif ref_lev_size == 1 and bc_lev_size > 1:
            target = np.asarray(ds["lev"].values)
            ds = ds.sel(lev=target, method="nearest")
        elif bc_lev_size == ref_lev_size:
            target = np.asarray(bc_ds["lev"].values)
            ds = ds.sel(lev=target, method="nearest")

    for dim in ("lat", "lon"):
        if dim in ds.dims and dim in bc_ds.dims:
            target = np.asarray(bc_ds[dim].values)
            ds = ds.sel({dim: target}, method="nearest")

    return ds


def main() -> None:
    p = argparse.ArgumentParser(
        description="Climatological mean/std for bias-corrected SDMBC NetCDF time series.",
        epilog=_GADI_ENV,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="Input NetCDF (must contain a time dimension)")
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output NetCDF path (default: derived from input filename)",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory (used only if --output is not provided)",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Optional SDMBC config YAML to locate reference (truth) files for bias",
    )
    p.add_argument(
        "--reference-input",
        default=None,
        help="Optional reference NetCDF path (if provided, bias is computed against this file)",
    )
    p.add_argument(
        "--bias-output",
        default=None,
        help="Optional output path for bias statistics (default: derived from main output name)",
    )
    p.add_argument(
        "--no-bias",
        action="store_true",
        help="Skip bias computation even if --config/--reference-input is provided",
    )
    p.add_argument("--chunks", type=int, default=None, help="Optional time chunk size for dask")
    p.add_argument("--no-monthly", action="store_true", help="Omit monthly climatology")
    p.add_argument(
        "--daily",
        action="store_true",
        help="Include day-of-year climatology (366 levels; more work for long 6-hourly series)",
    )
    p.add_argument("--no-seasonal", action="store_true", help="Omit seasonal climatology")
    p.add_argument("--no-std", action="store_true", help="Omit all-time temporal std")
    p.add_argument(
        "--std-monthly-means",
        action="store_true",
        help="Include interannual std of calendar-month means",
    )
    p.add_argument(
        "--std-seasonal-means",
        action="store_true",
        help="Include interannual std of seasonal means",
    )
    p.add_argument(
        "--std-annual-means",
        action="store_true",
        help="Include interannual std of annual means",
    )
    args = p.parse_args()

    open_kw: dict = {"decode_times": True, "engine": "netcdf4"}
    if args.chunks is not None:
        open_kw["chunks"] = {"time": args.chunks}

    output_path = args.output or _default_output_path(args.input, args.outdir)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bias_output_path = args.bias_output or _default_bias_output_path(output_path)
    Path(bias_output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reading %s", args.input)
    with xr.open_dataset(args.input, **open_kw) as ds:
        out_bc = build_bc_climatology_dataset(
            ds,
            monthly=not args.no_monthly,
            daily=args.daily,
            seasonal=not args.no_seasonal,
            std_all_time=not args.no_std,
            std_monthly_means=args.std_monthly_means,
            std_seasonal_means=args.std_seasonal_means,
            std_annual_means=args.std_annual_means,
        )
        encoding = _build_encoding(out_bc)
        logger.info("Writing %s", output_path)
        out_bc.to_netcdf(output_path, encoding=encoding)

    if args.no_bias:
        logger.info("Skipping bias computation (--no-bias).")
        return

    ref_path = args.reference_input
    if ref_path is None and args.config is not None:
        logger.info("Locating reference files from config: %s", args.config)
        cfg = _read_config(args.config)
        single_grid_test = bool(getattr(cfg, "single_grid_test", False))
        level_index = _reference_level_index(cfg, args.input)
        ref_files = _collect_reference_files(cfg, args.input)
        logger.info("Found %d reference files", len(ref_files))
        ds_ref = xr.open_mfdataset(
            ref_files,
            combine="by_coords",
            decode_times=True,
            chunks=open_kw.get("chunks"),
            engine="netcdf4",
        )
    elif ref_path is not None:
        single_grid_test = False
        level_index = None
        ds_ref = xr.open_dataset(ref_path, **open_kw)
    else:
        logger.info("No --config/--reference-input provided; skipping bias computation.")
        return

    with ds_ref, xr.open_dataset(output_path, **open_kw) as out_bc_saved:
        ds_ref_subset = _subset_reference_to_corrected_domain(
            ds_ref,
            out_bc_saved,
            level_index=level_index,
        )
        out_ref = build_bc_climatology_dataset(
            ds_ref_subset,
            monthly=not args.no_monthly,
            daily=args.daily,
            seasonal=not args.no_seasonal,
            std_all_time=not args.no_std,
            std_monthly_means=args.std_monthly_means,
            std_seasonal_means=args.std_seasonal_means,
            std_annual_means=args.std_annual_means,
        )

        bc_sel, ref_sel, selected_vars = _select_common_bias_variables(
            out_bc_saved,
            out_ref,
            single_grid_test=single_grid_test,
            level_index=level_index,
        )
        bc_aligned, ref_aligned = xr.align(bc_sel, ref_sel, join="exact")
        bias = bc_aligned - ref_aligned
        bias = bias.rename({v: f"{v}_bias" for v in bias.data_vars})
        bias.attrs["title"] = "Bias of climatological statistics (corrected - reference)"
        bias.attrs["bias_definition"] = "bias = corrected - reference"
        bias.attrs["selected_common_variables"] = ", ".join(selected_vars)

        logger.info("Writing %s", bias_output_path)
        bias.to_netcdf(bias_output_path, encoding=_build_encoding(bias))


if __name__ == "__main__":
    main()
