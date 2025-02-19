import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime
from CalculateWindBias import calculate_metgrid_hourly_bias, revise_u_v
from util import MetGridInfo, uv_2_wind, theta_U, theta_V
import os


exclude_monitor = "gmet"
metgrid_dir = "/Volumes/Expansion/WindUncertainty/WRF/metgrid_result/0305"
output_dir = "/Volumes/Expansion/WindUncertainty/BruteForceWRF/metgrid/0305"

for filename in os.listdir(metgrid_dir):
    metgrid_filename = os.path.join(metgrid_dir, filename)
    output_filename = os.path.join(output_dir, filename)
    obs_filename = "/Volumes/Expansion/WindUncertainty/Measurements/met/hourly_rounded_FtStwrt.csv"
    metgrid_ds = nc.Dataset(metgrid_filename)
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    met_obs = pd.read_csv(obs_filename, parse_dates=['UTC_time'], date_parser=dateparse)
    met_obs = met_obs[met_obs["monitor"] != exclude_monitor]
    metgrid_info = MetGridInfo(metgrid_ds)
    wind_bias = calculate_metgrid_hourly_bias(metgrid_ds, met_obs)

    # rotate u, v
    u, v = metgrid_ds["UU"][:], metgrid_ds["VV"][:]
    rotated_u, rotated_v = np.zeros(u.shape), np.zeros(v.shape)
    for i in range(0, len(metgrid_info["time"])):
        cur_time = metgrid_info["time"][i]
        print(cur_time)
        cur_hourly_time = datetime(cur_time.year, cur_time.month, cur_time.day, cur_time.hour)
        if cur_hourly_time in wind_bias["time"]:
            t_idx = wind_bias["time"].index(cur_hourly_time)
            cur_deg_bias = wind_bias["degree_bias"][t_idx]
            cur_speed_fct = wind_bias["scaling_factor"][t_idx]
            # u, v
            rotated_u[i, :, :, :-1], rotated_v[i, :, :-1, :] = revise_u_v(u[i, :, :, :-1], v[i, :, :-1, :], cur_deg_bias, cur_speed_fct)
            rotated_u[i, :, :, -1], _ = revise_u_v(u[i, :, :, -1], v[i, :, :-1, -1], cur_deg_bias, cur_speed_fct)
            _, rotated_v[i, :, -1, :] = revise_u_v(u[i, :, -1, :-1], v[i, :, -1, :], cur_deg_bias, cur_speed_fct)
        else:
            # u, v
            rotated_u[i, :, :, :] = u[i, :, :, :]
            rotated_v[i, :, :, :] = v[i, :, :, :]

    # Write to wrfout file
    # rotate U, V
    os.system("cp %s %s" % (metgrid_filename, output_filename))
    dset = nc.Dataset(output_filename, 'r+')
    dset["UU"][:] = rotated_u
    dset["VV"][:] = rotated_v
    dset.close()