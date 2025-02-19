import netCDF4 as nc
import pandas as pd
from util import WRFGridInfo, adjust_ratio, get_model_coord_idx, uv_2_wind, wind_2_uv, MetGridInfo
from datetime import datetime
import numpy as np


def calculate_hourly_bias(wrf_ds, obs_df):
    # degree bias: model + bias = obs => bias = obs - model
    # scaling factor: model * factor = obs => factor = obs / model
    # get all obs in the WRF domain and time period
    wrf_info = WRFGridInfo(wrf_ds)
    # calculate bias hour by hour: time, u_bias, v_bias
    select_df = obs_df[(obs_df["lon"] >= np.min(wrf_info["Lon"])) & (obs_df["lon"] <= np.max(wrf_info["Lon"])) &
                       (obs_df["lat"] >= np.min(wrf_info["Lat"])) & (obs_df["lat"] <= np.max(wrf_info["Lat"])) &
                       (obs_df["UTC_time"] >= np.min(wrf_info["time"])) & (obs_df["UTC_time"] <= np.max(wrf_info["time"]))]
    u_model, v_model = wrf_ds["U10"], wrf_ds["V10"]
    res = {"time": [], "degree_bias": [], "scaling_factor": []}
    for cur_time_idx in range(0, len(wrf_info["time"])):
        cur_time = wrf_info["time"][cur_time_idx]
        cur_obs = select_df[select_df["UTC_time"] == cur_time]
        if len(cur_obs) == 0:
            continue
        else:
            domain_obs_u, domain_obs_v, domain_model_u, domain_model_v = [], [], [], []
            for _, row in cur_obs.iterrows():
                cur_lon, cur_lat = row["lon"], row["lat"]
                monitor_x, monitor_y = wrf_info["crs"](cur_lon, cur_lat)
                cur_wdspd, cur_wddir = row["wdspd"], row["wddir"]
                cur_obs_u, cur_obs_v = wind_2_uv(cur_wdspd, cur_wddir)
                x_idx, y_idx = get_model_coord_idx(monitor_x, monitor_y, wrf_info)
                cur_model_u, cur_model_v = np.squeeze(u_model[cur_time_idx, x_idx, y_idx]), np.squeeze(v_model[cur_time_idx, x_idx, y_idx])
                domain_obs_u.append(cur_obs_u)
                domain_obs_v.append(cur_obs_v)
                domain_model_u.append(cur_model_u)
                domain_model_v.append(cur_model_v)
                if row["monitor"] in ["gpem", "USFS 1079"]:
                    print(row["monitor"])
                    print(row["UTC_time"])
                    # print("obs speed: %.2f" % (np.sqrt(cur_obs_u ** 2 + cur_obs_v ** 2)))
                    # print("wrf speed: %.2f" % (np.sqrt(cur_model_u ** 2 + cur_model_v ** 2)))
                    print("obs / wrf speed ratio %.2f" % (np.sqrt(cur_obs_u ** 2 + cur_obs_v ** 2) / np.sqrt(cur_model_u ** 2 + cur_model_v ** 2)))

            domain_obs_u, domain_obs_v, domain_model_u, domain_model_v = np.mean(domain_obs_u), np.mean(domain_obs_v), np.mean(domain_model_u), np.mean(domain_model_v)
            domain_obs_wdspd, domain_obs_dir = uv_2_wind(domain_obs_u, domain_obs_v)
            domain_model_wdspd, domain_model_dir = uv_2_wind(domain_model_u, domain_model_v)
            cur_factor = domain_obs_wdspd / (domain_model_wdspd * adjust_ratio(10, 2))
            cur_degree_bias = domain_obs_dir - domain_model_dir
            res["time"].append(cur_time)
            res["degree_bias"].append(cur_degree_bias)
            res["scaling_factor"].append(cur_factor)
    return res


def revise_u_v(u, v, deg_bias, scale_factor):
    c_u, c_v = rotate_u_v(u, v, deg_bias)
    c_u, c_v = scale_u_v(c_u, c_v, scale_factor)
    return c_u, c_v


def rotate_u_v(u, v, deg_bias):
    # TODO: Check the sign
    rad_bias = np.deg2rad(-deg_bias)
    c_u = u * np.cos(rad_bias) - v * np.sin(rad_bias)
    c_v = u * np.sin(rad_bias) + v * np.cos(rad_bias)
    return c_u, c_v


def scale_u_v(u, v, scale_factor):
    return u * scale_factor, v * scale_factor


def calculate_metgrid_hourly_bias(metgrid, obs_df):
    # Firstly, calculate the domain averaged u and domain averaged v
    # get the dir and spd for the domain averaged u and domain averaged v
    # calculate the bias and scaling factor
    # degree bias: model + bias = obs => bias = obs - model
    # scaling factor: model * factor = obs => factor = obs / model
    # get all obs in the WRF domain and time period, notice that scale 10m to 2m for WRF or scale 2m to 10m for obs
    # do not affect the results
    metgrid_info = MetGridInfo(metgrid)
    # calculate bias hour by hour: time, u_bias, v_bias
    select_df = obs_df[(obs_df["lon"] >= np.min(metgrid_info["Lon"])) & (obs_df["lon"] <= np.max(metgrid_info["Lon"])) &
                       (obs_df["lat"] >= np.min(metgrid_info["Lat"])) & (obs_df["lat"] <= np.max(metgrid_info["Lat"])) &
                       (obs_df["UTC_time"] >= np.min(metgrid_info["time"])) & (obs_df["UTC_time"] <= np.max(metgrid_info["time"]))]
    u_model, v_model = metgrid["UU"][:, 0, :, :], metgrid["VV"][:, 0, :, :]
    res = {"time": [], "degree_bias": [], "scaling_factor": []}
    for cur_time_idx in range(0, len(metgrid_info["time"])):
        cur_time = metgrid_info["time"][cur_time_idx]
        cur_obs = select_df[select_df["UTC_time"] == cur_time]
        if len(cur_obs) == 0:
            continue
        else:
            domain_obs_u, domain_obs_v, domain_model_u, domain_model_v = [], [], [], []
            for _, row in cur_obs.iterrows():
                cur_lon, cur_lat = row["lon"], row["lat"]
                monitor_x, monitor_y = metgrid_info["crs"](cur_lon, cur_lat)
                cur_wdspd, cur_wddir = row["wdspd"], row["wddir"]
                cur_obs_u, cur_obs_v = wind_2_uv(cur_wdspd, cur_wddir)
                x_idx, y_idx = get_model_coord_idx(monitor_x, monitor_y, metgrid_info)
                cur_model_u, cur_model_v = np.squeeze(u_model[cur_time_idx, x_idx, y_idx]), np.squeeze(v_model[cur_time_idx, x_idx, y_idx])
                domain_obs_u.append(cur_obs_u)
                domain_obs_v.append(cur_obs_v)
                domain_model_u.append(cur_model_u)
                domain_model_v.append(cur_model_v)
            domain_obs_u, domain_obs_v, domain_model_u, domain_model_v = np.mean(domain_obs_u), np.mean(domain_obs_v), np.mean(domain_model_u), np.mean(domain_model_v)
            domain_obs_wdspd, domain_obs_dir = uv_2_wind(domain_obs_u, domain_obs_v)
            domain_model_wdspd, domain_model_dir = uv_2_wind(domain_model_u, domain_model_v)
            cur_factor = domain_obs_wdspd / (domain_model_wdspd * adjust_ratio(10, 2))
            cur_degree_bias = domain_obs_dir - domain_model_dir
            res["time"].append(cur_time)
            res["degree_bias"].append(cur_degree_bias)
            res["scaling_factor"].append(cur_factor)
    return res