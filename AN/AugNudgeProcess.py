import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pykrige
import netCDF4 as nc
from util import WRFGridInfo, wind_2_uv, uv_2_wind, extract_elev


def krigingOBS(obsX, obsY, obsConc, predictX, predictY):
    if min(obsConc) == max(obsConc):
        return np.ones(predictX.shape) * max(obsConc)
    else:
        model = pykrige.ok.OrdinaryKriging(obsX, obsY, obsConc, variogram_model='exponential')
        geoSize = predictX.shape
        X = np.reshape(predictX, predictX.size)
        Y = np.reshape(predictY, predictY.size)
        [h, t] = model.execute('points', X, Y)
        h = np.reshape(h, (geoSize[0], geoSize[1]))
        return h


# Generate Augmentation Data
filename = "/Volumes/Expansion/WindUncertainty/Measurements/met/hourly_rounded_FtStwrt.csv"
wrf_filename = "/Volumes/DataStorage/SERDP/data/WRF/wrfout_d03_2022-03-02_00:00:00"
grid_cro_2d = "/Volumes/Expansion/WindUncertainty/static/GRIDCRO2D_FtSt1.nc"

wrf_ds = nc.Dataset(wrf_filename)
wrf_info = WRFGridInfo(wrf_ds)
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
met_obs = pd.read_csv(filename, parse_dates=['UTC_time'], date_parser=dateparse)
start_time = datetime(2022, 3, 1)
end_time = datetime(2022, 3, 6)
cur_time = start_time
data_interval = 1
interpolate_lon, interpolate_lat = wrf_info["Lon"][::data_interval, ::data_interval], wrf_info["Lat"][::data_interval,::data_interval]
# find the elevations
cro_2d_ds = nc.Dataset(grid_cro_2d)
elevations = np.zeros(interpolate_lon.shape)
monitor_name = np.empty(interpolate_lon.shape, dtype=object)
m, n = elevations.shape
for i in range(0, m):
    for j in range(0, n):
        elevations[i, j] = extract_elev(interpolate_lon[i, j], interpolate_lat[i, j], cro_2d_ds)
        monitor_name[i, j] = "monitor_%d_%d" %(i, j)
res = {"UTC_time": [], "monitor": [], "lon": [], "lat": [], "wdspd": [], "wddir": [], "elevation": []}
# Generate the interpolated dataset
while cur_time <= end_time:
    cur_df = met_obs[(met_obs["UTC_time"] >= cur_time) & (met_obs["UTC_time"] < cur_time + timedelta(hours=1))]
    if len(cur_df) > 0:
        wdspd, wddir = cur_df["wdspd"].to_numpy(), cur_df["wddir"].to_numpy()
        lon, lat = cur_df["lon"].to_numpy(), cur_df["lat"].to_numpy()
        u, v = wind_2_uv(wdspd, wddir)
        # interpolate the data
        interpolate_u = krigingOBS(lon, lat, u, interpolate_lon, interpolate_lat)
        interpolate_v = krigingOBS(lon, lat, v, interpolate_lon, interpolate_lat)
        interpolate_wdspd, interpolate_wddir = uv_2_wind(interpolate_u.flatten(), interpolate_v.flatten())
        res["UTC_time"].extend([datetime.strftime(cur_time, "%Y-%m-%d %H:%M:%S")] * m * n)
        res["monitor"].extend(monitor_name.flatten())
        res["lon"].extend(interpolate_lon.flatten())
        res["lat"].extend(interpolate_lat.flatten())
        res["elevation"].extend(elevations.flatten())
        res["wdspd"].extend(interpolate_wdspd)
        res["wddir"].extend(interpolate_wddir)
    cur_time += timedelta(hours=1)

res_df = pd.DataFrame.from_dict(res)
res_df.to_csv("/Volumes/Expansion/WindUncertainty/Interpolation/interpolated_ftstwrt.csv", index=False)




