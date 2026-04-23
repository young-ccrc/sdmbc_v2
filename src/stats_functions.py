"""
Climatological statistics for bias-corrected (or any time-series) gridded datasets.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _data_vars_with_time(ds: xr.Dataset) -> list[str]:
    return [k for k, da in ds.data_vars.items() if "time" in da.dims]


def climatological_mean_annual(ds: xr.Dataset) -> xr.Dataset:
    """Long-term mean at each grid point (reduces time)."""
    return ds[_data_vars_with_time(ds)].mean(dim="time", skipna=True)


def climatological_mean_monthly(ds: xr.Dataset) -> xr.Dataset:
    """Mean for each calendar month (dimension ``month``, size 12)."""
    tvars = _data_vars_with_time(ds)
    return ds[tvars].groupby("time.month").mean("time", skipna=True)


def climatological_mean_daily(ds: xr.Dataset) -> xr.Dataset:
    """Mean for each day-of-year (dimension ``dayofyear``)."""
    tvars = _data_vars_with_time(ds)
    return ds[tvars].groupby("time.dayofyear").mean("time", skipna=True)


def climatological_mean_seasonal(ds: xr.Dataset) -> xr.Dataset:
    """Mean for each meteorological season (dimension ``season``)."""
    tvars = _data_vars_with_time(ds)
    return ds[tvars].groupby("time.season").mean("time", skipna=True)


def climatological_std_all_time(ds: xr.Dataset) -> xr.Dataset:
    """Standard deviation over the full time series at each grid point."""
    return ds[_data_vars_with_time(ds)].std(dim="time", skipna=True)


def climatological_std_monthly_means(ds: xr.Dataset) -> xr.Dataset:
    """
    Interannual variability: for each calendar month, std across years of
    that month's temporal mean (after resampling to monthly means).
    """
    tvars = _data_vars_with_time(ds)
    monthly = ds[tvars].resample(time="ME").mean("time", skipna=True)
    return monthly.groupby("time.month").std("time", skipna=True)


def climatological_std_seasonal_means(ds: xr.Dataset) -> xr.Dataset:
    """Std across years of seasonal-mean time series (Dec–Feb, etc.)."""
    tvars = _data_vars_with_time(ds)
    seasonal = ds[tvars].resample(time="QE-DEC").mean("time", skipna=True)
    return seasonal.groupby("time.season").std("time", skipna=True)


def climatological_std_annual_means(ds: xr.Dataset) -> xr.Dataset:
    """Std across years of annual-mean values."""
    tvars = _data_vars_with_time(ds)
    yearly = ds[tvars].resample(time="YE-DEC").mean("time", skipna=True)
    return yearly.std("time", skipna=True)


def build_bc_climatology_dataset(
    ds: xr.Dataset,
    *,
    monthly: bool = True,
    daily: bool = False,
    seasonal: bool = True,
    std_all_time: bool = True,
    std_monthly_means: bool = False,
    std_seasonal_means: bool = False,
    std_annual_means: bool = False,
) -> xr.Dataset:
    """
    Merge climatological diagnostics into one dataset with prefixed variable names.

    Prefixes: ``*_mean_annual``, ``*_mean_monthly``, ``*_mean_dayofyear``,
    ``*_mean_season``, ``*_std_alltime``, etc.
    """
    pieces: list[xr.Dataset] = []

    mean_ann = climatological_mean_annual(ds)
    pieces.append(mean_ann.rename({v: f"{v}_mean_annual" for v in mean_ann.data_vars}))

    if monthly:
        m = climatological_mean_monthly(ds)
        pieces.append(m.rename({v: f"{v}_mean_monthly" for v in m.data_vars}))
    if daily:
        d = climatological_mean_daily(ds)
        pieces.append(d.rename({v: f"{v}_mean_dayofyear" for v in d.data_vars}))
    if seasonal:
        s = climatological_mean_seasonal(ds)
        pieces.append(s.rename({v: f"{v}_mean_season" for v in s.data_vars}))

    if std_all_time:
        st = climatological_std_all_time(ds)
        pieces.append(st.rename({v: f"{v}_std_alltime" for v in st.data_vars}))
    if std_monthly_means:
        sm = climatological_std_monthly_means(ds)
        pieces.append(sm.rename({v: f"{v}_std_monthly_means" for v in sm.data_vars}))
    if std_seasonal_means:
        ss = climatological_std_seasonal_means(ds)
        pieces.append(ss.rename({v: f"{v}_std_seasonal_means" for v in ss.data_vars}))
    if std_annual_means:
        sy = climatological_std_annual_means(ds)
        pieces.append(sy.rename({v: f"{v}_std_annual_means" for v in sy.data_vars}))

    out = xr.merge(pieces, compat="override")
    out.attrs["title"] = "Climatological statistics (SDMBCv2 stats_functions)"
    out.attrs["source"] = getattr(ds, "attrs", {}).get("history", "")
    return out


# --- Legacy API (kept for older notebooks/scripts) --------------------------------


def cal_mean(ds):
    """
    Legacy: return (daily climatology, monthly, seasonal, annual mean).

    Previously the first return mixed day-of-year and an extra mean; now the first
    element is the proper multi-year mean for each calendar day.
    """
    dsd = climatological_mean_daily(ds)
    dsm = climatological_mean_monthly(ds)
    dss = climatological_mean_seasonal(ds)
    dsy = climatological_mean_annual(ds)
    return dsd, dsm, dss, dsy


def cal_std(ds):
    """Legacy: (all-time std, std of monthly means, std of seasonal means, std of annual means)."""
    dsd = climatological_std_all_time(ds)
    dsm = climatological_std_monthly_means(ds)
    dss = climatological_std_seasonal_means(ds)
    dsy = climatological_std_annual_means(ds)
    return dsd, dsm, dss, dsy


def cal_acor(ds, period, variable):  # period: M, QS-DEC, Y
    if period != "D":
        ds = ds.resample(time=period).mean("time")
    nvar = len(variable)
    cor_matrix = np.zeros([nvar, len(ds["lat"]), len(ds["lon"])])
    for v in range(0, len(variable)):
        var = variable[v]
        for i in range(0, len(ds["lat"])):
            for j in range(0, len(ds["lon"])):
                temp = np.corrcoef(
                    ds[var][1:, i, j], ds[var][0 : (len(ds.time) - 1), i, j]
                )
                cor_matrix[v, i, j] = temp[0, 1]
    return cor_matrix


def cal_ccor(ds, period, v):
    if period != "D":
        ds = ds.resample(time=period).mean("time")
    cor_matrix = np.zeros([3, len(ds["lat"]), len(ds["lon"])])
    for i in range(0, len(ds["lat"])):
        for j in range(0, len(ds["lon"])):
            temp = np.corrcoef(ds[v[0]][:, i, j], ds[v[1]][:, i, j])
            cor_matrix[0, i, j] = temp[0, 1]
            temp = np.corrcoef(ds[v[0]][:, i, j], ds[v[2]][:, i, j])
            cor_matrix[1, i, j] = temp[0, 1]
            temp = np.corrcoef(ds[v[1]][:, i, j], ds[v[2]][:, i, j])
            cor_matrix[2, i, j] = temp[0, 1]
    return cor_matrix
