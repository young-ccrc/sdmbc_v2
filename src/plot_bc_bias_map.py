#!/usr/bin/env python3
"""
Plot quick-look bias maps from SDMBCv2 climatology/bias NetCDF files.

Default behavior targets annual-mean bias fields (``*_mean_annual_bias``).
Use --pattern and --facet-dim for monthly or seasonal panels.
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _select_variables(ds: xr.Dataset, patterns: list[str]) -> list[str]:
    if patterns:
        return [name for name in ds.data_vars if any(fnmatch.fnmatch(name, pat) for pat in patterns)]
    return [name for name in ds.data_vars if name.endswith("_mean_annual_bias")]


def _spatial_2d(da: xr.DataArray) -> xr.DataArray:
    spatial_dims = [d for d in da.dims if d in {"lat", "lon"}]
    if set(spatial_dims) != {"lat", "lon"}:
        raise ValueError("Variable does not have lat/lon dimensions.")
    return da.transpose(..., "lat", "lon")


def _symmetric_limits(da: xr.DataArray) -> tuple[float, float]:
    arr = np.asarray(da.values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    vmax = float(np.nanmax(np.abs(finite)))
    if vmax == 0.0:
        vmax = 1.0
    return -vmax, vmax


def _plot_2d(ax, da: xr.DataArray, title: str, cmap: str, vmin: float, vmax: float):
    plot_da = _spatial_2d(da)
    lat = np.asarray(plot_da.lat.values)
    lon = np.asarray(plot_da.lon.values)
    lon2d, lat2d = np.meshgrid(lon, lat)
    mesh = ax.pcolormesh(
        lon2d,
        lat2d,
        np.asarray(plot_da.values),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    return mesh


def _plot_variable(ds: xr.Dataset, varname: str, outdir: Path, cmap: str, facet_dim: str | None):
    da = ds[varname]
    singleton_dims = [
        dim for dim in da.dims if dim not in {"lat", "lon"} and da.sizes.get(dim) == 1
    ]
    if singleton_dims and facet_dim is None:
        da = da.isel({dim: 0 for dim in singleton_dims}, drop=True)
    outdir.mkdir(parents=True, exist_ok=True)

    extra_dims = [d for d in da.dims if d not in {"lat", "lon"}]
    if not extra_dims:
        vmin, vmax = _symmetric_limits(da)
        fig, ax = plt.subplots(figsize=(10, 5))
        mesh = _plot_2d(ax, da, varname, cmap, vmin, vmax)
        fig.colorbar(mesh, ax=ax, orientation="vertical", label=varname)
        fig.tight_layout()
        outfile = outdir / f"{varname}.png"
        fig.savefig(outfile, dpi=200)
        plt.close(fig)
        return outfile

    if facet_dim is None or facet_dim not in da.dims:
        raise ValueError(
            f"{varname} has extra dimensions {extra_dims}; pass --facet-dim {extra_dims[0]} "
            "or use a 2D field."
        )

    facet_values = np.asarray(da[facet_dim].values)
    if facet_values.size > 12:
        raise ValueError(
            f"{varname} facet dimension '{facet_dim}' has {facet_values.size} slices; "
            "use a smaller subset or a 2D field."
        )

    n = facet_values.size
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    vmin, vmax = _symmetric_limits(da)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
    mesh = None
    for idx, value in enumerate(facet_values):
        r, c = divmod(idx, ncols)
        sub = da.sel({facet_dim: value})
        mesh = _plot_2d(
            axes[r][c],
            sub,
            f"{varname} {facet_dim}={value}",
            cmap,
            vmin,
            vmax,
        )
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation="vertical", label=varname)
    fig.tight_layout()
    outfile = outdir / f"{varname}_{facet_dim}.png"
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    return outfile


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SDMBCv2 bias maps from climatology NetCDF.")
    parser.add_argument("input", help="Bias climatology NetCDF file")
    parser.add_argument("-o", "--outdir", default=None, help="Directory for PNG files")
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Glob pattern for variables to plot (repeatable). Default: *_mean_annual_bias",
    )
    parser.add_argument(
        "--facet-dim",
        default=None,
        help="Optional dimension to facet over for non-2D variables (e.g. month or season)",
    )
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir) if args.outdir else input_path.with_suffix("")
    patterns = args.pattern or ["*_mean_annual_bias"]

    with xr.open_dataset(input_path) as ds:
        vars_to_plot = _select_variables(ds, patterns)
        if not vars_to_plot:
            raise ValueError(f"No variables matched patterns: {patterns}")
        for varname in vars_to_plot:
            outfile = _plot_variable(ds, varname, outdir, args.cmap, args.facet_dim)
            print(f"Wrote {outfile}")


if __name__ == "__main__":
    main()
