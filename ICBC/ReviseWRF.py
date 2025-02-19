import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime
from CalculateWindBias import calculate_hourly_bias, revise_u_v
from util import WRFGridInfo, uv_2_wind, theta_U, theta_V
import os

select_date = datetime(2022, 3, 2)
exclude_monitor = "gmet"
wrf_filename = "/Volumes/PubData/WindUncertainty/WRF/results/nudge/wrfout_d03_%s_00:00:00" % select_date.strftime("%Y-%m-%d")
obs_filename = "/Volumes/PubData/WindUncertainty/Measurements/met/hourly_rounded_FtStwrt.csv"
output_filename = "/Volumes/PubData/WindUncertainty/BruteForceWRF/nudge/wrfout_d03_%s_00:00:00" % select_date.strftime("%Y-%m-%d")
wrf_ds = nc.Dataset(wrf_filename)
wrf_info = WRFGridInfo(wrf_ds)
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
met_obs = pd.read_csv(obs_filename, parse_dates=['UTC_time'], date_parser=dateparse)
met_obs = met_obs[met_obs["monitor"] != exclude_monitor]

wind_bias = calculate_hourly_bias(wrf_ds, met_obs)
# rotate u10, v10
u10, v10 = wrf_ds["U10"][:], wrf_ds["V10"][:]
rotated_u10, rotated_v10 = np.zeros(u10.shape), np.zeros(v10.shape)
u, v = wrf_ds["U"][:], wrf_ds["V"][:]
rotated_u, rotated_v = np.zeros(u.shape), np.zeros(v.shape)
for i in range(0, len(wrf_info["time"])):
    cur_time = wrf_info["time"][i]
    cur_hourly_time = datetime(cur_time.year, cur_time.month, cur_time.day, cur_time.hour)
    if cur_hourly_time in wind_bias["time"]:
        t_idx = wind_bias["time"].index(cur_hourly_time)
        cur_deg_bias = wind_bias["degree_bias"][t_idx]
        cur_speed_fct = wind_bias["scaling_factor"][t_idx]
        # u10, v10
        rotated_u10[i, :, :], rotated_v10[i, :, :] = revise_u_v(u10[i, :, :], v10[i, :, :], cur_deg_bias, cur_speed_fct)
        # u, v
        rotated_u[i, :, :, :-1], rotated_v[i, :, :-1, :] = revise_u_v(u[i, :, :, :-1], v[i, :, :-1, :], cur_deg_bias, cur_speed_fct)
        rotated_u[i, :, :, -1], _ = revise_u_v(u[i, :, :, -1], v[i, :, :-1, -1], cur_deg_bias, cur_speed_fct)
        _, rotated_v[i, :, -1, :] = revise_u_v(u[i, :, -1, :-1], v[i, :, -1, :], cur_deg_bias, cur_speed_fct)
    else:
        # u10, v10
        rotated_u10[i, :, :] = u10[i, :, :]
        rotated_v10[i, :, :] = v10[i, :, :]
        # u, v
        rotated_u[i, :, :, :] = u[i, :, :, :]
        rotated_v[i, :, :, :] = v[i, :, :, :]

# Write to wrfout file
# rotate U, V
os.system("cp %s %s" % (wrf_filename, output_filename))
dset = nc.Dataset(output_filename, 'r+')
dset['U10'][:] = rotated_u10
dset['V10'][:] = rotated_v10
dset["U"][:] = rotated_u
dset["V"][:] = rotated_v
dset.close()