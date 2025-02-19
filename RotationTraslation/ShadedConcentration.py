import pandas as pd
from datetime import datetime, timedelta
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import json
from shapely.geometry import shape
from util import WRFGridInfo, adjust_ratio, conc_at_obs_WRF, wind_2_uv, wrf_wind_uv, uv_2_wind
from shapely.ops import unary_union
from uncertainty_util import STR_relocate_idx, relocate_region_idx
import pickle

uncertainty_res = {}
select_dates = [datetime(2022, 3, 2), datetime(2022, 3, 3), datetime(2022, 3, 5)]
monitor_met_map = {
    datetime(2022, 3, 2): {
        "Trailer_FS": "gpem",
        "USFS 1078": "gpem",
        "USFS 1079": "USFS 1079"
    },
    datetime(2022, 3, 3): {
        "Trailer_FS": "USFS 1079",
        "USFS 1078": "USFS 1079",
        "USFS 1079": "USFS 1079"
    },
    datetime(2022, 3, 5): {
        "Trailer_FS": "USFS 1079",
        "USFS 1078": "USFS 1079",
        "USFS 1079": "USFS 1079"
    }
}

for select_date in select_dates:
    uncertainty_res[select_date] = {}
    conc_filename = "/Volumes/PubData/WindUncertainty/Measurements/conc/combined_PM25_conc.csv"
    # wrf_sfire_filename = "/Volumes/Expansion/WindUncertainty/SFIRE_Results/nudge/%s/wrfout_d01_%s_00:00:00" % (select_date.strftime("%m%d"), select_date.strftime("%Y-%m-%d"))
    wrf_sfire_filename = "/Volumes/PubData/WindUncertainty/SFIRE/brute_force/%s/wrfout_d01_%s_00:00:00" % (select_date.strftime("%m%d"), select_date.strftime("%Y-%m-%d"))
    wind_obs = "/Volumes/PubData/WindUncertainty/Measurements/met/hourly_rounded_FtStwrt.csv"

    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    conc_df = pd.read_csv(conc_filename, parse_dates=['UTC_time'], date_parser=dateparse)
    met_df = pd.read_csv(wind_obs, parse_dates=['UTC_time'], date_parser=dateparse)

    select_conc_df = conc_df[(conc_df["UTC_time"] >= select_date) & (conc_df["UTC_time"] < select_date + timedelta(days=1))]
    select_conc_df = select_conc_df.reset_index(drop=True)
    select_wind_df = met_df[(met_df["UTC_time"] >= select_date) & (met_df["UTC_time"] < select_date + timedelta(days=1))]
    select_wind_df = select_wind_df.reset_index(drop=True)

    wind_monitor_locations = {}
    for monitor_name in list(set(select_wind_df["monitor"].to_numpy())):
        current_df = select_wind_df[(select_wind_df["monitor"] == monitor_name) & (select_wind_df["UTC_time"] >= select_date + timedelta(hours=15))]
        wind_monitor_locations[monitor_name] = (current_df["lon"].to_numpy()[0], current_df["lat"].to_numpy()[0])

    conc_monitor_locations = {}
    for monitor_name in list(set(select_conc_df["monitor"].to_numpy())):
        current_df = select_conc_df[(select_conc_df["monitor"] == monitor_name) & (select_conc_df["UTC_time"] >= select_date + timedelta(hours=15))]
        conc_monitor_locations[monitor_name] = (current_df["lon"].to_numpy()[0], current_df["lat"].to_numpy()[0])

    wrf_sfire_ds = nc.Dataset(wrf_sfire_filename)

    # fire polygon
    fire_file = "/Volumes/Shield/FireFrameworkCF/Stewart/obs_data/fire/FtStwrt_BurnInfo.json"

    with open(fire_file) as json_file:
        fire_events = json.load(json_file)

    select_fire_polygons = []
    for fire_event in fire_events["fires"]:
        fire_date = datetime.strptime(fire_event["date"], "%Y-%m-%d")
        if fire_date == select_date and fire_event["type"] == "rx":
            select_fire_polygons.append(shape(fire_event["perimeter"]))
    fire_polygon = unary_union(select_fire_polygons)
    polygon_ctr = fire_polygon.centroid
    unit_coord_lon, unit_coord_lat = polygon_ctr.x, polygon_ctr.y

    model_info = WRFGridInfo(wrf_sfire_ds)
    wind_info = {}

    # get bias
    for monitor_name in wind_monitor_locations.keys():
        current_lon = wind_monitor_locations[monitor_name][0]
        current_lat = wind_monitor_locations[monitor_name][1]
        if current_lon < np.min(model_info["Lon"]) or current_lon > np.max(model_info["Lon"]) \
                or current_lat < np.min(model_info["Lat"]) or current_lat > np.max(model_info["Lat"]):
            continue
        # meteorological condition
        current_wind_df = select_wind_df[select_wind_df["monitor"] == monitor_name]
        met_obs_time = current_wind_df["UTC_time"].to_numpy()
        met_obs_wdspd, met_obs_wddir = current_wind_df["wdspd"].to_numpy(), current_wind_df["wddir"].to_numpy()
        wrf_u_10, wrf_v_10 = wrf_wind_uv(wrf_sfire_ds, current_lon, current_lat)
        obs_u, obs_v = wind_2_uv(met_obs_wdspd, met_obs_wddir)
        wrf_u_2, wrf_v_2 = wrf_u_10 * adjust_ratio(10, 2), wrf_v_10 * adjust_ratio(10, 2)
        wrf_spd, wrf_dir = uv_2_wind(wrf_u_2, wrf_v_2)
        wind_info[monitor_name] = {
            "time": [], "spd_bias": [], "dir_bias": [], "obs_u": [], "obs_v": [], "wrf_u": [], "wrf_v": []
        }
        # common time met_obs_time
        for i in range(0, len(met_obs_time)):
            current_hour = pd.to_datetime(met_obs_time[i])
            current_model_idx = model_info["time"].index(current_hour)
            spd_bias = met_obs_wdspd[i] - wrf_spd[current_model_idx]
            dir_bias = met_obs_wddir[i] - wrf_dir[current_model_idx]
            cur_time = model_info["time"][current_model_idx]
            wind_info[monitor_name]["time"].append(cur_time)
            wind_info[monitor_name]["spd_bias"].append(spd_bias)
            wind_info[monitor_name]["dir_bias"].append(dir_bias)
            wind_info[monitor_name]["obs_u"].append(obs_u[i])
            wind_info[monitor_name]["obs_v"].append(obs_v[i])
            wind_info[monitor_name]["wrf_u"].append(wrf_u_2[current_model_idx])
            wind_info[monitor_name]["wrf_v"].append(wrf_v_2[current_model_idx])

    # correct the concentration
    for monitor_name in conc_monitor_locations.keys():
        fig, ax = plt.subplots(figsize=(8, 6))
        current_lon = conc_monitor_locations[monitor_name][0]
        current_lat = conc_monitor_locations[monitor_name][1]
        wrf_time = model_info["time"]
        wrf_conc = conc_at_obs_WRF(wrf_sfire_ds, "tr17_2", current_lon, current_lat)
        current_df = select_conc_df[select_conc_df["monitor"] == monitor_name]
        current_obs_time = current_df["UTC_time"]
        current_obs_conc = current_df["PM25"]

        # plot figure
        # conc
        ax.plot(wrf_time, wrf_conc, label="WRF_SFIRE")
        ax.plot(current_obs_time, current_obs_conc, label=monitor_name, marker='o')
        adjust_conc, max_uncertainty_conc, min_uncertainty_conc = [], [], []
        # uncertainty
        cur_wind_bias = wind_info[monitor_met_map[select_date][monitor_name]]
        for cur_idx in range(0, len(cur_wind_bias["time"])):
            cur_spd_bias, cur_dir_bias = cur_wind_bias["spd_bias"][cur_idx], cur_wind_bias["dir_bias"][cur_idx]
            if cur_idx == 0:
                # This will not affect Rx since Rx does not happen in UTC 0 in U.S. SE case
                prev_spd_bias, prev_dir_bias = 0, 0
            else:
                prev_spd_bias, prev_dir_bias = cur_wind_bias["spd_bias"][cur_idx - 1], cur_wind_bias["dir_bias"][cur_idx - 1]
            cur_time_idx = model_info["time"].index(cur_wind_bias["time"][cur_idx])
            search_mask = relocate_region_idx([prev_spd_bias, cur_spd_bias], [prev_dir_bias, cur_dir_bias], current_lon, current_lat, [fire_polygon], model_info)
            x_idx, y_idx = STR_relocate_idx(cur_spd_bias, cur_dir_bias, current_lon, current_lat, unit_coord_lon, unit_coord_lat, model_info)
            current_model_conc = wrf_sfire_ds["tr17_2"][cur_time_idx, 0, :, :]
            current_adjust_conc_region = current_model_conc[search_mask].flatten()
            current_adjust_conc = current_model_conc[x_idx, y_idx][0]
            adjust_conc.append(current_adjust_conc)
            max_uncertainty_conc.append(np.max(current_adjust_conc_region))
            min_uncertainty_conc.append(np.min(current_adjust_conc_region))
        # plot uncertainty
        ax.plot(cur_wind_bias["time"], adjust_conc, '--k')
        plt.fill_between(cur_wind_bias["time"], min_uncertainty_conc, max_uncertainty_conc, alpha=0.5, color='gray')
        plt.legend()
        plt.xticks(rotation=45, ha='right')
        datefmt = DateFormatter("%m/%d %H")
        ax.xaxis.set_major_formatter(datefmt)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
        # plt.show()

        uncertainty_res[select_date][monitor_name] = {
            "time": cur_wind_bias["time"],
            "lower": min_uncertainty_conc,
            "upper": max_uncertainty_conc,
            "mean": adjust_conc
        }

# # save data for future use
# with open("/Volumes/Shield/WindUncertaintyImpacts/data/conc_eval/rotation_translation.pickle", 'wb') as handle:
#     pickle.dump(uncertainty_res, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save data for future use
with open("/Volumes/Shield/WindUncertaintyImpacts/data/icbc_conc_eval/rotation_translation.pickle", 'wb') as handle:
    pickle.dump(uncertainty_res, handle, protocol=pickle.HIGHEST_PROTOCOL)