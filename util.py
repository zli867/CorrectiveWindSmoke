import numpy as np
import pandas as pd
import pyproj
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import colors


def adjust_ratio(current_height, surface_height):
    # For open terrain (grassland) the typical range is 0.01-0.05 m
    z0 = 0.03
    d = 0
    ratio = np.log((surface_height - d)/z0)/np.log((current_height - d)/z0)
    return ratio


def WRFGridInfo(ds):
    crs = pyproj.Proj(proj='lcc',  # projection type: Lambert Conformal Conic
                      lat_1=ds.TRUELAT1, lat_2=ds.TRUELAT2,  # Cone intersects with the sphere
                      lat_0=ds.MOAD_CEN_LAT, lon_0=ds.STAND_LON,  # Center point
                      a=6370000, b=6370000)  # This is it! The Earth is a perfect sphere
    # Grid parameters
    dx, dy = ds.DX, ds.DY
    nx, ny = len(ds.dimensions["west_east"]), len(ds.dimensions["south_north"])
    # Down left corner of the domain
    e, n = crs(ds.CEN_LON, ds.CEN_LAT)
    x0 = -(nx - 1) / 2. * dx + e
    y0 = -(ny - 1) / 2. * dy + n
    # 2d grid
    xx, yy = np.meshgrid(np.arange(nx) * dx + x0, np.arange(ny) * dy + y0)
    Xcenters = xx
    Ycenters = yy
    lat_ctr = ds["XLAT"][0, :, :]
    lon_ctr = ds["XLONG"][0, :, :]
    xcell = dx
    ycell = dy
    # Boundary X, Y
    x_bdry = np.arange(nx + 1) * dx + x0 - dx / 2
    y_bdry = np.arange(ny + 1) * dy + y0 - dy / 2
    Xbounds, Ybounds = np.meshgrid(x_bdry, y_bdry)
    x_max = np.max(Xbounds)
    x_min = np.min(Xbounds)
    y_max = np.max(Ybounds)
    y_min = np.min(Ybounds)
    wrf_time_array = []
    wrf_time = ds["Times"][:]
    for i in range(0, wrf_time.shape[0]):
        current_time_str = ""
        for j in range(0, wrf_time.shape[1]):
            current_time_str = current_time_str + wrf_time[i][j].decode()
        current_time_obj = datetime.strptime(current_time_str, '%Y-%m-%d_%H:%M:%S')
        wrf_time_array.append(current_time_obj)
    res_dict = {"crs": crs, "X": Xcenters, "Y": Ycenters,
                "time": wrf_time_array,
                "Lat": lat_ctr, "Lon": lon_ctr,
                "XCELL": xcell, "YCELL": ycell, "X_bdry": [x_min, x_max], "Y_bdry": [y_min, y_max],}
    return res_dict


def wind_2_uv(wdspd, wddir):
    u = -wdspd * np.sin(np.deg2rad(wddir))
    v = -wdspd * np.cos(np.deg2rad(wddir))
    return u, v


def uv_2_wind(u, v):
    wdspd = np.sqrt(u**2 + v**2)
    wddir = np.mod(180+np.rad2deg(np.arctan2(u, v)), 360)
    return wdspd, wddir


def extract_elev(site_lon, site_lat, dataset):
    elev = dataset["HT"][:].flatten()
    lat = dataset["LAT"][:].flatten()
    lon = dataset["LON"][:].flatten()

    distance = (lon - site_lon) ** 2 + (lat - site_lat) ** 2
    index = np.argmin(distance)
    return elev[index]


def get_model_coord_idx(x, y, model_info):
    # Nearest grid
    distance = (model_info["X"] - x) ** 2 + (model_info["Y"] - y) ** 2
    x_idx, y_idx = np.where(distance == np.min(distance))
    return x_idx, y_idx


def theta_U(u_wind):
    # https://wiki.openwfm.org/wiki/How_to_interpret_WRF_variables
    # It is for MCIP version, WRF version should be a little different
    u_wind_theta = 0.5 * (u_wind[:, :, :-1, :-1] + u_wind[:, :, 1:, 1:])
    return u_wind_theta


def theta_V(v_wind):
    # https://wiki.openwfm.org/wiki/How_to_interpret_WRF_variables
    # It is for MCIP version, WRF version should be a little different
    v_wind_theta = 0.5 * (v_wind[:, :, :-1, :-1] + v_wind[:, :, 1:, 1:])
    return v_wind_theta


def extract_obs(obs_df, start_time, end_time):
    met_df = obs_df[(obs_df["UTC_time"] >= start_time) & (obs_df["UTC_time"] <= end_time)]
    monitors = np.unique(met_df["monitor"].to_numpy())
    monitor_info = {}
    for monitor in monitors:
        monitor_info[monitor] = {}
        current_met = met_df[met_df["monitor"] == monitor]
        if len(current_met) > 0:
            monitor_info[monitor]["lat"] = current_met["lat"].to_numpy()[0]
            monitor_info[monitor]["lon"] = current_met["lon"].to_numpy()[0]
            monitor_info[monitor]["obs_wddir"] = current_met["wddir"].to_numpy()
            monitor_info[monitor]["obs_wdspd"] = current_met["wdspd"].to_numpy()
            monitor_info[monitor]["time"] = pd.to_datetime(current_met["UTC_time"].to_numpy())
    return monitor_info


def findSpatialIndex(fire_x, fire_y, X_ctr, Y_ctr):
    """

    :param fire_x: X of fire location in CMAQ projection
    :param fire_y: Y of fire location in CMAQ projection
    :param X_ctr: CMAQ grid X center
    :param Y_ctr: CMAQ grid Y center
    :return: x_idx, y_idx which are the fire location in CMAQ grid
    """
    dist = np.sqrt((X_ctr - fire_x) ** 2 + (Y_ctr - fire_y) ** 2)
    x_idx, y_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
    return x_idx, y_idx


def windDirectionTransfer(dir1, dir2):
    # Transfer dir1, dir2 to minimum diff
    # Notice that dir1 value will not be changed
    for i in range(0, len(dir1)):
        if np.isnan(dir1[i]) or np.isnan(dir2[i]):
            continue
        else:
            diff = dir1[i] - dir2[i]
            if diff < -180:
                dir2[i] = dir2[i] -360
            elif diff > 180:
                dir2[i] = dir2[i] + 360
    return dir1, dir2


def MetGridInfo(ds):
    crs = pyproj.Proj(proj='lcc',  # projection type: Lambert Conformal Conic
                      lat_1=ds.TRUELAT1, lat_2=ds.TRUELAT2,  # Cone intersects with the sphere
                      lat_0=ds.MOAD_CEN_LAT, lon_0=ds.STAND_LON,  # Center point
                      a=6370000, b=6370000)  # This is it! The Earth is a perfect sphere
    # Grid parameters
    dx, dy = ds.DX, ds.DY
    nx, ny = len(ds.dimensions["west_east"]), len(ds.dimensions["south_north"])
    # Down left corner of the domain
    e, n = crs(ds.CEN_LON, ds.CEN_LAT)
    x0 = -(nx - 1) / 2. * dx + e
    y0 = -(ny - 1) / 2. * dy + n
    # 2d grid
    xx, yy = np.meshgrid(np.arange(nx) * dx + x0, np.arange(ny) * dy + y0)
    Xcenters = xx
    Ycenters = yy
    xcell = dx
    ycell = dy
    lon = ds["XLONG_M"][0, :, :]
    lat = ds["XLAT_M"][0, :, :]
    wrf_time_array = []
    wrf_time = ds["Times"][:]
    for i in range(0, wrf_time.shape[0]):
        current_time_str = ""
        for j in range(0, wrf_time.shape[1]):
            current_time_str = current_time_str + wrf_time[i][j].decode()
        current_time_obj = datetime.strptime(current_time_str, '%Y-%m-%d_%H:%M:%S')
        wrf_time_array.append(current_time_obj)
    res_dict = {"crs": crs, "X": Xcenters, "Y": Ycenters,
                "time": wrf_time_array,
                "Lat": lat, "Lon": lon,
                "XCELL": xcell, "YCELL": ycell}
    return res_dict


def conc_at_obs_WRF(wrf_ds, pollutant_name, obs_lon, obs_lat):
    lat = wrf_ds["XLAT"][0, :, :]
    lon = wrf_ds["XLONG"][0, :, :]
    pollutant = wrf_ds[pollutant_name]
    pollutant_surface = pollutant[:][:, 0, :, :]
    # Nearest grid
    distance = (lat - obs_lat) ** 2 + (lon - obs_lon) ** 2
    x, y = np.where(distance == np.min(distance))
    pollutant_surface_at_obs = pollutant_surface[:, x, y]
    pollutant_surface_at_obs = pollutant_surface_at_obs.flatten()
    return pollutant_surface_at_obs


def wrf_wind_uv(ds, monitor_lon, monitor_lat):
    lat = ds["XLAT"][:]
    lon = ds["XLONG"][:]
    u_10 = ds["U10"][:]
    v_10 = ds["V10"][:]
    lat = lat[0, :, :]
    lon = lon[0, :, :]
    distance = (lat - monitor_lat) ** 2 + (lon - monitor_lon) ** 2
    x, y = np.where(distance == np.min(distance))
    u_10_at_obs = u_10[:, x, y].flatten()
    v_10_at_obs = v_10[:, x, y].flatten()
    return u_10_at_obs, v_10_at_obs


def extract_met_info(wrf_ds, obs_df, monitor_name, select_date):
    model_info = WRFGridInfo(wrf_ds)
    current_df = obs_df[(obs_df["monitor"] == monitor_name) & (obs_df["UTC_time"] >= select_date + timedelta(hours=15))]
    monitor_lon, monitor_lat = current_df["lon"].to_numpy()[0], current_df["lat"].to_numpy()[0]
    daily_df = obs_df[(obs_df["monitor"] == monitor_name) & (obs_df["UTC_time"] >= select_date) & (obs_df["UTC_time"] < select_date + timedelta(days=1))]
    u10, v10 = wrf_wind_uv(wrf_ds, monitor_lon, monitor_lat)
    wrf_time, wrf_u, wrf_v = model_info["time"][::3], u10[::3] * adjust_ratio(10, 2), v10[::3] * adjust_ratio(10, 2)
    obs_wdspd, obs_wddir = daily_df["wdspd"].to_numpy(), daily_df["wddir"].to_numpy()
    obs_u, obs_v = wind_2_uv(obs_wdspd, obs_wddir)
    obs_time = daily_df["UTC_time"].to_numpy()
    return {
        "wrf_time": wrf_time, "wrf_u": wrf_u, "wrf_v": wrf_v,
        "obs_time": obs_time, "obs_u": obs_u, "obs_v": obs_v
    }


def discrete_cmap(invertals, base_color_scheme="Spectral_r", high_value_color='purple'):
    cmap = plt.get_cmap(base_color_scheme, invertals)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    cmap.set_over(color=high_value_color, alpha=1.0)
    return cmap