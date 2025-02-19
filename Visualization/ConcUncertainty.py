import pandas as pd
from datetime import datetime, timedelta
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from util import conc_at_obs_WRF, WRFGridInfo, extract_met_info
import json
import pickle

monitor_names = ["Trailer_FS", "USFS 1079", "USFS 1078"]
colors = ["#F27970", "#54B345", "#05B9E2"]
select_dates = [datetime(2022, 3, 2), datetime(2022, 3, 3), datetime(2022, 3, 5)]

conc_filename = "/Volumes/PubData/WindUncertainty/Measurements/conc/combined_PM25_conc.csv"
fire_file = "/Volumes/Shield/FireFrameworkCF/Stewart/obs_data/fire/FtStwrt_BurnInfo.json"
uncertainty_file = "/Volumes/Shield/WindUncertaintyImpacts/TrajectoryNew/res/equal_time_nudge.pickle"
# uncertainty_file = "/Volumes/Shield/WindUncertaintyImpacts/TrajectoryNew/res/equal_dist_nudge.pickle"
# uncertainty_file = "/Volumes/Shield/WindUncertaintyImpacts/data/conc_eval/rotation_translation.pickle"
with open(uncertainty_file, "rb") as f:
    uncertainty_data = pickle.load(f)

# method_name = "Equal Distance Back/forward Trajectory"
method_name = "Equal Time Back/forward Trajectory"
# method_name = "Rotation and Translation"
fire_time = {}

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

wind_obs = "/Volumes/PubData/WindUncertainty/Measurements/met/hourly_rounded_FtStwrt.csv"
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
met_df = pd.read_csv(wind_obs, parse_dates=['UTC_time'], date_parser=dateparse)

# fire time
with open(fire_file) as json_file:
    fire_events = json.load(json_file)
for fire_event in fire_events["fires"]:
    fire_date = datetime.strptime(fire_event["date"], "%Y-%m-%d")
    fire_start_time = datetime.strptime(fire_event["start_UTC"], "%Y-%m-%d %H:%M:%S")
    fire_start_time = datetime(fire_start_time.year, fire_start_time.month, fire_start_time.day, fire_start_time.hour)
    if fire_date not in fire_time.keys():
        fire_time[fire_date] = fire_start_time
    else:
        fire_time[fire_date] = min(fire_time[fire_date], fire_start_time)

datefmt = DateFormatter("%m/%d %H")
fig, axs = plt.subplots(3, 3, figsize=(9, 6))
ax_ravel = axs.ravel()
ax_idx = 0
for monitor_name in monitor_names:
    for select_date in select_dates:
        wrf_sfire_filename = "/Volumes/PubData/WindUncertainty/SFIRE_Results/nudge/%s/wrfout_d01_%s_00:00:00" % (select_date.strftime("%m%d"), select_date.strftime("%Y-%m-%d"))
        # wrf_sfire_filename = "/Volumes/Expansion/WindUncertainty/SFIRE/brute_force/%s/wrfout_d01_%s_00:00:00" % (select_date.strftime("%m%d"), select_date.strftime("%Y-%m-%d"))
        wrf_sfire_ds = nc.Dataset(wrf_sfire_filename)
        wrf_info = WRFGridInfo(wrf_sfire_ds)
        dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        df = pd.read_csv(conc_filename, parse_dates=['UTC_time'], date_parser=dateparse)
        cur_fire_time = fire_time[select_date]
        select_df = df[(df["UTC_time"] >= cur_fire_time) & (df["UTC_time"] < select_date + timedelta(days=1)) & (df["monitor"] == monitor_name)]
        select_df = select_df.reset_index(drop=True)

        current_lon, current_lat = select_df["lon"].to_numpy()[0], select_df["lat"].to_numpy()[0]

        wrf_time = wrf_info["time"]
        wrf_start_index = wrf_time.index(cur_fire_time)
        wrf_conc = conc_at_obs_WRF(wrf_sfire_ds, "tr17_2", current_lon, current_lat)
        current_obs_time = select_df["UTC_time"]
        current_obs_conc = select_df["PM25"]

        plot_row, plot_col = ax_idx // 3, ax_idx % 3
        # plot figure
        markers = ["x", "*", None]
        ax = ax_ravel[ax_idx]
        # WRF-SFIRE
        ax.plot(wrf_time[wrf_start_index:], wrf_conc[wrf_start_index:], label="NB", color=colors[plot_row])
        ax.plot(current_obs_time, current_obs_conc, label="obs", marker='o', linestyle="--", color=colors[plot_row])
        ax.plot(wrf_time[wrf_start_index:], 35 * np.ones(len(wrf_time[wrf_start_index:])), linestyle="--", color="orange")
        ax.text(wrf_time[wrf_start_index:][0], 35, '35', color='orange', verticalalignment='bottom')
        ax.plot(uncertainty_data[select_date][monitor_name]["time"],
                uncertainty_data[select_date][monitor_name]["mean"], 'k--')
        ax.fill_between(uncertainty_data[select_date][monitor_name]["time"],
                         uncertainty_data[select_date][monitor_name]["lower"],
                         uncertainty_data[select_date][monitor_name]["upper"], alpha=0.3, color='gray')

        if plot_col == 1:
            ax.set_title(monitor_names[plot_row])
        if plot_col == 0 and plot_row == 1:
            ax.set_ylabel("$PM_{2.5}$ $\mu g/m^3$", fontsize=16)
        ax.set_xlim([min(wrf_time[wrf_start_index:]), max(wrf_time[wrf_start_index:])])
        if plot_row == 2:
            xlabels = ax.get_xticklabels()
            ax.set_xticklabels(xlabels, rotation=45, ha='right')
            ax.xaxis.set_major_formatter(datefmt)
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
        else:
            ax.xaxis.set_major_formatter(datefmt)
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
            ax.xaxis.set_ticklabels([])
        # plot met
        cur_met_monitor = monitor_met_map[select_date][monitor_name]
        ymin, ymax = ax.get_ylim()
        met_info = extract_met_info(wrf_sfire_ds, met_df, cur_met_monitor, select_date)
        y_value = ymax * np.ones(met_info["obs_u"].shape) * 0.95
        # obs
        Q = ax.quiver(met_info["obs_time"], y_value, met_info["obs_u"], met_info["obs_v"], scale=20, scale_units='height', angles='uv', zorder=100)
        y_value = ymax * np.ones(met_info["wrf_u"].shape) * 0.95
        ax.quiver(met_info["wrf_time"], y_value, met_info["wrf_u"], met_info["wrf_v"], scale=20, scale_units='height', angles='uv', alpha=0.3, zorder=100)
        if plot_col == 2 and plot_row == 2:
            ax.legend(loc="upper left", frameon=False)
            qk = ax.quiverkey(Q, 0.9, 0.2, 2, r'$2 \frac{m}{s}$', labelpos='E', coordinates='figure')
        ax.set_ylim([ymin, ymax * 1.1])
        ax_idx += 1
plt.suptitle(method_name, fontsize=16)
plt.show()