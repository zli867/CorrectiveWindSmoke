import pandas as pd
from datetime import datetime, timedelta
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from util import WRFGridInfo
import json
from shapely.geometry import shape
from metrics import RMSE
from matplotlib import cm
from shapely.ops import unary_union
from util import discrete_cmap
from uncertainty_util import get_model_coord_idx
from mpl_toolkits.axes_grid1 import make_axes_locatable


def STR_relocate_idx(wind_spd_bias, wind_dir_bias, monitor_lon, monitor_lat, unit_lon, unit_lat, model_info):
    """

    :param wind_spd_bias: observed_value - modeled_value
    :param wind_dir_bias: observed_value - modeled_value
    :param monitor_lon: longitude location of monitor
    :param monitor_lat: latitude location of monitor
    :param unit_lon: longitude location of burned area
    :param unit_lat: latitude location of burned area
    :param model_info: model information dictionary which includes lat, lon, X, Y, projection information, etc.
    :return: x_idx and y_idx related to relocated monitor location -> (lat[x_idx, y_idx], lon[x_idx, y_idx])
    """
    # time_step = (model_info["time"][1] - model_info["time"][0]).seconds
    # TODO: I update it here since we use hourly wind bias
    time_step = 3600
    unit_x, unit_y = model_info["crs"](unit_lon, unit_lat)
    monitor_x, monitor_y = model_info["crs"](monitor_lon, monitor_lat)
    distance = np.sqrt((unit_x - monitor_x) ** 2 + (unit_y - monitor_y) ** 2)
    delta_distance = time_step * wind_spd_bias
    search_distance = distance - delta_distance
    degree = np.degrees(np.arctan2((monitor_y - unit_y), (monitor_x - unit_x)))
    search_degree = degree + wind_dir_bias
    # delta_x = relocate monitor x - unit x
    delta_x = search_distance * np.cos(search_degree * np.pi / 180)
    delta_y = search_distance * np.sin(search_degree * np.pi / 180)
    relocate_monitor_x = delta_x + unit_x
    relocate_monitor_y = delta_y + unit_y
    if model_info["X_bdry"][0] <=relocate_monitor_x <= model_info["X_bdry"][1] and model_info["Y_bdry"][0] <=relocate_monitor_y <= model_info["Y_bdry"][1]:
        x_idx, y_idx = get_model_coord_idx(relocate_monitor_x, relocate_monitor_y, model_info)
    else:
        x_idx, y_idx = -1, -1
    return x_idx, y_idx


cmap = discrete_cmap(20)
select_date = datetime(2022, 3, 5)
specie_name = "tr17_2"
conc_filename = "/Volumes/PubData/WindUncertainty/Measurements/conc/combined_PM25_conc.csv"
wrf_sfire_filename = "/Volumes/PubData/WindUncertainty/SFIRE_Results/nudge/%s/wrfout_d01_%s_00:00:00" % (select_date.strftime("%m%d"), select_date.strftime("%Y-%m-%d"))
# fire polygon
fire_file = "/Volumes/Shield/FireFrameworkCF/Stewart/obs_data/fire/FtStwrt_BurnInfo.json"

dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv(conc_filename, parse_dates=['UTC_time'], date_parser=dateparse)

select_df = df[(df["UTC_time"] >= select_date) & (df["UTC_time"] < select_date + timedelta(days=1))]
select_df = select_df.reset_index(drop=True)

monitor_names = list(set(select_df["monitor"].to_numpy()))

monitor_locations = {}
for monitor_name in monitor_names:
    current_df = select_df[(select_df["monitor"] == monitor_name) & (select_df["UTC_time"] >= select_date + timedelta(hours=15))]
    monitor_locations[monitor_name] = (current_df["lon"].to_numpy()[0], current_df["lat"].to_numpy()[0])

wrf_sfire_ds = nc.Dataset(wrf_sfire_filename)

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
wrf_met_time = model_info["time"]
model_conc = wrf_sfire_ds[specie_name][:]


rotated_degrees = np.arange(-70, 70, 1)
transformed_speeds = np.arange(0, 1.5 + 0.1, 0.1)
degrees, speeds = np.meshgrid(rotated_degrees, transformed_speeds)
speed_num, degree_num = degrees.shape
total_rmse = np.zeros(degrees.shape)

# monitor info
monitor_info = {}
for monitor_name in monitor_locations.keys():
    monitor_info[monitor_name] = {}
    current_lon = monitor_locations[monitor_name][0]
    current_lat = monitor_locations[monitor_name][1]
    common_time = []
    obs_conc = []
    valid_idx = []
    current_df = select_df[select_df["monitor"] == monitor_name]
    idx = 0
    for current_time_idx in range(0, len(model_info["time"])):
        current_time = model_info["time"][current_time_idx]
        if current_time.minute != 0:
            continue
        tmp_df = current_df[
            (current_df["UTC_time"] >= current_time) & (current_df["UTC_time"] < current_time + timedelta(hours=1))]
        if len(tmp_df) > 0:
            common_time.append(current_time)
            current_obs_conc = np.mean(tmp_df["PM25"])
            obs_conc.append(current_obs_conc)
            valid_idx.append(current_time_idx)
    monitor_info[monitor_name]["conc"] = obs_conc
    monitor_info[monitor_name]["valid_idx"] = valid_idx

for i in range(0, speed_num):
    for j in range(0, degree_num):
        current_degree_bias = degrees[i, j]
        current_speed_bias = speeds[i, j]
        total_obs_conc = np.array([])
        total_model_conc = np.array([])
        for monitor_name in monitor_info.keys():
            current_lon = monitor_locations[monitor_name][0]
            current_lat = monitor_locations[monitor_name][1]
            valid_idx = monitor_info[monitor_name]["valid_idx"]
            obs_conc = monitor_info[monitor_name]["conc"]
            current_model_conc = model_conc[valid_idx, 0, :, :]
            current_x_idx, current_y_idx = STR_relocate_idx(current_speed_bias, current_degree_bias, current_lon, current_lat, unit_coord_lon, unit_coord_lat, model_info)
            if current_x_idx != -1:
                total_model_conc = np.concatenate((total_model_conc, current_model_conc[:, current_x_idx, current_y_idx].flatten()))
                total_obs_conc = np.concatenate((total_obs_conc, np.array(obs_conc)))
        total_rmse[i, j] = RMSE(total_obs_conc, total_model_conc)

fig, ax = plt.subplots(figsize=(8, 6))
c = ax.pcolor(degrees, speeds, total_rmse, cmap=cmap, vmin=0, vmax=80)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(c, cax=cax, orientation='vertical', extend='max')
ax.set_title("Concentration Root Mean Squared Error (%s)" % select_date.strftime("%Y-%m-%d"), fontsize=16)
ax.set_xlabel("Wind Direction Bias (degree)", fontsize=16)
ax.set_ylabel("Wind Speed Bias (m/s)", fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.show()
