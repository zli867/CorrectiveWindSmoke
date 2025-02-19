import pyproj
import numpy as np
from datetime import datetime, timedelta
from shapely import geometry
import pandas as pd
from shapely.geometry import shape
import shapely.geometry as geom
from util import uv_2_wind, wind_2_uv
from shapely.ops import unary_union
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as geom
from util import WRFGridInfo, adjust_ratio
from datetime import datetime, timedelta
from util import uv_2_wind, wind_2_uv


def get_model_coord_idx(x, y, model_info):
    # Nearest grid
    distance = (model_info["X"] - x) ** 2 + (model_info["Y"] - y) ** 2
    x_idx, y_idx = np.where(distance == np.min(distance))
    return x_idx, y_idx


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
    x_idx, y_idx = get_model_coord_idx(relocate_monitor_x, relocate_monitor_y, model_info)
    return x_idx, y_idx


def relocate_region_idx(wind_spd_bias, wind_dir_bias, monitor_lon, monitor_lat, fire_polygons, model_info):
    """

    :param wind_spd_bias: observed_value - modeled_value
    :param wind_dir_bias: observed_value - modeled_value
    :param monitor_lon: longitude location of monitor
    :param monitor_lat: latitude location of monitor
    :param fire_polygon: fire polygon
    :param model_info: model information dictionary which includes lat, lon, X, Y, projection information, etc.
    :return: x_idx and y_idx related to relocated monitor location -> (lat[x_idx, y_idx], lon[x_idx, y_idx])
    """
    mask_res = np.full(model_info["X"].shape, False)
    fire_polygon = unary_union(fire_polygons)
    fire_ctr_lon, fire_ctr_lat = fire_polygon.centroid.xy
    fire_ctr_lon, fire_ctr_lat = fire_ctr_lon[0], fire_ctr_lat[0]
    for current_wind_spd_bias in np.linspace(min(wind_spd_bias), max(wind_spd_bias), num=10):
        for current_wind_dir_bias in np.linspace(min(wind_dir_bias), max(wind_dir_bias), num=10):
            x_idx, y_idx = STR_relocate_idx(current_wind_spd_bias, current_wind_dir_bias, monitor_lon,
                                        monitor_lat, fire_ctr_lon, fire_ctr_lat, model_info)
            mask_res[x_idx, y_idx] = True
    return mask_res



def hourly_conc_data(conc_df, select_date):
    """

    :param conc_df: dataframe of conc file
    :param select_date: datetime object (hour = 0, minute = 0, seconds = 0)
    :return: a hourly dict {"monitor_name": {"loc": [], "conc": [], "time": []}}
    """
    res = {}
    select_df = conc_df[(conc_df["UTC_time"] >= select_date) & (conc_df["UTC_time"] < select_date + timedelta(days=1))]
    select_df = select_df[select_df["PM25"] >= 0]
    select_df = select_df.reset_index(drop=True)
    # monitor location
    monitor_names = list(set(select_df["monitor"].to_numpy()))
    for monitor_name in monitor_names:
        # filter day time (~ 9am)
        current_df = select_df[
            (select_df["monitor"] == monitor_name) & (select_df["UTC_time"] > select_date + timedelta(hours=15))]
        if len(current_df) > 0:
            res[monitor_name] = {}
            res[monitor_name]["loc"] = [current_df["lon"].to_numpy()[0], current_df["lat"].to_numpy()[0]]
    # hourly conc and time array
    for monitor_name in res.keys():
        monitor_cur_df = select_df[select_df["monitor"] == monitor_name]
        cur_conc = []
        cur_time = []
        start_utc_time = select_date
        end_utc_time = select_date + timedelta(days=1)
        while start_utc_time < end_utc_time:
            cur_df = monitor_cur_df[(monitor_cur_df["UTC_time"] >= start_utc_time) & (
                        monitor_cur_df["UTC_time"] < start_utc_time + timedelta(hours=1))]
            if len(cur_df) > 0:
                cur_conc.append(np.mean(cur_df["PM25"]))
                cur_time.append(start_utc_time)
            start_utc_time += timedelta(hours=1)
        res[monitor_name]["conc"] = np.array(cur_conc)
        res[monitor_name]["time"] = cur_time
        res[monitor_name]["raw_conc"] = monitor_cur_df["PM25"].to_numpy()
        res[monitor_name]["raw_time"] = monitor_cur_df["UTC_time"].to_numpy()
    return res


def hourly_met_data(met_df, select_date):
    """

    :param met_df: dataframe of met file
    :param select_date: datetime object
    :return: an hourly {"monitor_name": {"loc": [], "wdspd": [], "wddir": [], "time": []}}
    """
    res = {}
    select_df = met_df[(met_df["UTC_time"] >= select_date) & (met_df["UTC_time"] < select_date + timedelta(days=1))]
    select_df = select_df.reset_index(drop=True)
    # monitor location
    monitor_names = list(set(select_df["monitor"].to_numpy()))
    for monitor_name in monitor_names:
        # filter day time (~ 9am)
        current_df = select_df[
            (select_df["monitor"] == monitor_name) & (select_df["UTC_time"] > select_date + timedelta(hours=15))]
        if len(current_df) > 0:
            res[monitor_name] = {}
            res[monitor_name]["loc"] = [current_df["lon"].to_numpy()[0], current_df["lat"].to_numpy()[0]]
    # hourly wddir, wdspd and time array
    for monitor_name in res.keys():
        monitor_cur_df = select_df[select_df["monitor"] == monitor_name]
        cur_wdspd = []
        cur_wddir = []
        cur_time = []
        start_utc_time = select_date
        end_utc_time = select_date + timedelta(days=1)
        while start_utc_time < end_utc_time:
            cur_df = monitor_cur_df[(monitor_cur_df["UTC_time"] >= start_utc_time) & (
                        monitor_cur_df["UTC_time"] < start_utc_time + timedelta(hours=1))]
            if len(cur_df) > 0:
                if len(cur_df) == 1:
                    cur_wdspd.append(cur_df["wdspd"].values[0])
                    cur_wddir.append(cur_df["wddir"].values[0])
                else:
                    u, v = wind_2_uv(cur_df["wdspd"].to_numpy(), cur_df["wddir"].to_numpy())
                    mean_u, mean_v = np.mean(u), np.mean(v)
                    mean_spd, mean_dir = uv_2_wind(mean_u, mean_v)
                    cur_wdspd.append(mean_spd)
                    cur_wddir.append(mean_dir)
                cur_time.append(start_utc_time)
            start_utc_time += timedelta(hours=1)
        res[monitor_name]["wdspd"] = np.array(cur_wdspd)
        res[monitor_name]["wddir"] = np.array(cur_wddir)
        res[monitor_name]["time"] = cur_time
    return res


def calculate_transport_dist(traj_dict):
    x = traj_dict["x"].copy()
    y = traj_dict["y"].copy()
    x.append(traj_dict['dist'][0])
    y.append(traj_dict['dist'][1])
    dist = 0
    for i in range(1, len(x)):
        dist += np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2)
    return dist


def get_model_coord_idx(x, y, model_info):
    # Nearest grid
    distance = (model_info["X"] - x) ** 2 + (model_info["Y"] - y) ** 2
    x_idx, y_idx = np.where(distance == np.min(distance))
    return x_idx, y_idx


def line_equation(sx, sy, ex, ey):
    A = ey - sy
    B = sx - ex
    C = ex * sy - sx * ey
    return A, B, C


def point_to_line_distance(sx, sy, ex, ey, x0, y0):
    A, B, C = line_equation(sx, sy, ex, ey)
    d = np.abs(A * x0 + B * y0 + C) / np.sqrt(A * A + B * B)
    x_d = (B * B * x0 - A * B * y0 - A * C) / (A * A + B * B)
    y_d = (- A * B * x0 + A * A * y0 - B * C) / (A * A + B * B)
    # whether the (x_d, y_d) is in the line
    product = (x_d - sx) * (x_d - ex) + (y_d - sy) * (y_d - ey)
    if product < 0:
        # in the line
        return d, x_d, y_d
    else:
        # dist1 = np.sqrt((x_d - sx) ** 2 + (y_d - sy) ** 2)
        # dist2 = np.sqrt((x_d - ex) ** 2 + (y_d - ey) ** 2)
        dist1 = np.sqrt((x0 - sx) ** 2 + (y0 - sy) ** 2)
        dist2 = np.sqrt((x0 - ex) ** 2 + (y0 - ey) ** 2)
        if dist2 > dist1:
            return dist1, sx, sy
        else:
            return dist2, ex, ey


def get_polygons_ctr(polygons):
    multi_poly = geom.MultiPolygon(polygons)
    multi_poly_centroid = multi_poly.centroid
    unit_coord_lon, unit_coord_lat = multi_poly_centroid.x, multi_poly_centroid.y
    return unit_coord_lon, unit_coord_lat
# DEBUG for distance
# sx, sy = 0.5, 0.5
# ex, ey = 1, 1
# # sx, sy = -1, 0
# # ex, ey = 1, 0
# point_x, point_y = -1, -1
# plt.plot([sx, ex], [sy, ey])
# plt.scatter([point_x], [point_y])
# d, x_d, y_d = point_to_line_distance(sx, sy, ex, ey, point_x, point_y)
# plt.scatter([x_d], [y_d], marker='x')
# plt.title("d: " + str(d))
# plt.show()


def hourly_met_data(met_df, select_date):
    """

    :param met_df: dataframe of met file
    :param select_date: datetime object
    :return: an hourly {"monitor_name": {"loc": [], "wdspd": [], "wddir": [], "time": []}}
    """
    res = {}
    select_df = met_df[(met_df["UTC_time"] >= select_date) & (met_df["UTC_time"] < select_date + timedelta(days=1))]
    select_df = select_df.reset_index(drop=True)
    # monitor location
    monitor_names = list(set(select_df["monitor"].to_numpy()))
    for monitor_name in monitor_names:
        # filter day time (~ 9am)
        current_df = select_df[
            (select_df["monitor"] == monitor_name) & (select_df["UTC_time"] > select_date + timedelta(hours=15))]
        if len(current_df) > 0:
            res[monitor_name] = {}
            res[monitor_name]["loc"] = [current_df["lon"].to_numpy()[0], current_df["lat"].to_numpy()[0]]
    # hourly wddir, wdspd and time array
    for monitor_name in res.keys():
        monitor_cur_df = select_df[select_df["monitor"] == monitor_name]
        cur_wdspd = []
        cur_wddir = []
        cur_time = []
        start_utc_time = select_date
        end_utc_time = select_date + timedelta(days=1)
        while start_utc_time < end_utc_time:
            cur_df = monitor_cur_df[(monitor_cur_df["UTC_time"] >= start_utc_time) & (
                        monitor_cur_df["UTC_time"] < start_utc_time + timedelta(hours=1))]
            if len(cur_df) > 0:
                if len(cur_df) == 1:
                    cur_wdspd.append(cur_df["wdspd"].values[0])
                    cur_wddir.append(cur_df["wddir"].values[0])
                else:
                    u, v = wind_2_uv(cur_df["wdspd"].to_numpy(), cur_df["wddir"].to_numpy())
                    mean_u, mean_v = np.mean(u), np.mean(v)
                    mean_spd, mean_dir = uv_2_wind(mean_u, mean_v)
                    cur_wdspd.append(mean_spd)
                    cur_wddir.append(mean_dir)
                cur_time.append(start_utc_time)
            start_utc_time += timedelta(hours=1)
        res[monitor_name]["wdspd"] = np.array(cur_wdspd)
        res[monitor_name]["wddir"] = np.array(cur_wddir)
        res[monitor_name]["time"] = cur_time
    return res


def hourly_u_v_sigma(met_data_dict, select_time):
    # calculate std(u) and std(v)
    wdspd, wddir = [], []
    for monitor_name in met_data_dict.keys():
        time_array = met_data_dict[monitor_name]["time"]
        if select_time in time_array:
            time_idx = time_array.index(select_time)
            wdspd.append(met_data_dict[monitor_name]["wdspd"][time_idx])
            wddir.append(met_data_dict[monitor_name]["wddir"][time_idx])
    u_bias, v_bias = wind_2_uv(np.array(wdspd), np.array(wddir))
    sigma_u = np.std(u_bias)
    sigma_v = np.std(v_bias)
    return sigma_u, sigma_v


def calculate_uv_sigma(met_data_dict, select_date):
    time_array, u_sigma, v_sigma = [select_date + timedelta(hours=t) for t in range(0, 24)], [], []
    for select_time in time_array:
        cur_u_sigma, cur_v_sigma = hourly_u_v_sigma(met_data_dict, select_time)
        u_sigma.append(cur_u_sigma)
        v_sigma.append(cur_v_sigma)
    return {"time": time_array, "u_sigma": u_sigma, "v_sigma": v_sigma}


def agent_based_backward_loc(monitor_x, monitor_y, unit_x, unit_y, sampling_time, cur_u_series, cur_v_series):
    traj = {"start_time": [], "start_points": [], "end_time": [], "end_points": []}
    backward_res = {"time": [], "x": [], "y": [], "dist": []}
    x_traj, y_traj = [monitor_x], [monitor_y]
    distances = []
    transport_distance_array = [0]
    traj_idx = 0
    for cur_time_idx in range(len(sampling_time) - 2, -1, -1):
        start_point = (x_traj[traj_idx], y_traj[traj_idx])
        traj["start_points"].append(start_point)
        traj["start_time"].append(sampling_time[cur_time_idx + 1])
        cur_u, cur_v = cur_u_series[cur_time_idx] + 0.00001, cur_v_series[cur_time_idx] + 0.00001
        # # TODO: check this, original I use cur_time_idx
        # cur_u, cur_v = cur_u_series[cur_time_idx + 1] + 0.00001, cur_v_series[cur_time_idx + 1] + 0.00001
        end_point = start_point[0] - 3600 * cur_u, start_point[1] - 3600 * cur_v
        traj["end_time"].append(sampling_time[cur_time_idx])
        traj["end_points"].append(end_point)
        x_traj.append(end_point[0])
        y_traj.append(end_point[1])
        d, x_d, y_d = point_to_line_distance(start_point[0], start_point[1], end_point[0], end_point[1], unit_x, unit_y)
        # (x_d, y_d) to start_point
        cur_transport_dist = np.sqrt((start_point[0] - x_d) ** 2 + (start_point[1] - y_d) ** 2)
        transport_distance = transport_distance_array[-1] + cur_transport_dist
        ratio = np.sqrt((start_point[0] - x_d) ** 2 + (start_point[1] - y_d) ** 2) / np.sqrt((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2)
        intercept_time = sampling_time[cur_time_idx + 1] + (sampling_time[cur_time_idx] - sampling_time[cur_time_idx + 1]) * ratio
        distances.append((d, x_d, y_d, intercept_time, transport_distance, len(traj["start_time"])))
        traj_idx += 1

    # select the minimum distance one, the intersection point is (x_d, y_d)
    distances.sort()
    nearest_point = distances[0]
    d, source_x, source_y, d_time, trans_dist, f_idx = nearest_point[0], nearest_point[1], nearest_point[2], nearest_point[3], nearest_point[4], nearest_point[5]
    # forward time duration
    forward_duration = traj["start_time"][0] - d_time
    # get backward_res
    for i in range(0, f_idx):
        backward_res["time"].append(traj["start_time"][i])
        backward_res["x"].append(traj["start_points"][i][0])
        backward_res["y"].append(traj["start_points"][i][1])
    backward_res["dist"] = (source_x, source_y)
    trans_dist = calculate_transport_dist(backward_res)
    return d_time, forward_duration, source_x, source_y, trans_dist


def floor_round_time(datetime_obj):
    number_20 = timedelta(minutes=20 * (datetime_obj.minute // 20))
    return datetime(datetime_obj.year, datetime_obj.month, datetime_obj.day, datetime_obj.hour) + number_20


def neareast_round_time(datetime_obj):
    number_20 = timedelta(minutes=20 * np.round(datetime_obj.minute / 20))
    return datetime(datetime_obj.year, datetime_obj.month, datetime_obj.day, datetime_obj.hour) + number_20


def ceil_round_time(datetime_obj):
    if datetime_obj == floor_round_time(datetime_obj):
        return datetime_obj
    else:
        number_20 = timedelta(minutes=20 * ((datetime_obj.minute // 20) + 1))
        return datetime(datetime_obj.year, datetime_obj.month, datetime_obj.day, datetime_obj.hour) + number_20


def agent_based_relocate_idx(wind_obs, sigma_obs, model_ds, monitor_x, monitor_y, unit_x, unit_y, start_time, end_time, model_info):
    np.random.seed(0)
    sampling_size = 1000
    # backward traj
    current_u_obs, current_v_obs = wind_2_uv(wind_obs["wdspd"], wind_obs["wddir"])
    sampling_time, sampling_u, sampling_v = [], [], []
    sampling_x_idx, sampling_y_idx = [], []
    for i in range(0, len(wind_obs["time"])):
        if start_time - timedelta(hours=1) <= wind_obs["time"][i] <= end_time:
            mean = [current_u_obs[i], current_v_obs[i]]
            sigma_idx = sigma_obs["time"].index(wind_obs["time"][i])
            cov = [[sigma_obs["u_sigma"][sigma_idx] ** 2, 0], [0, sigma_obs["v_sigma"][sigma_idx] ** 2]]
            cur_samping_u, cur_sampling_v = np.random.multivariate_normal(mean, cov, sampling_size).T
            sampling_u.append(cur_samping_u)
            sampling_v.append(cur_sampling_v)
            sampling_time.append(wind_obs["time"][i])
    sampling_u, sampling_v = np.array(sampling_u), np.array(sampling_v)
    for i in range(0, sampling_size):
        cur_u_series, cur_v_series = sampling_u[:, i], sampling_v[:, i]
        d_time, forward_duration, forward_x, forward_y, _ = agent_based_backward_loc(monitor_x, monitor_y, unit_x, unit_y, sampling_time, cur_u_series, cur_v_series)
        # forward the none integer part, wind use previous time step wind
        round_d_time = floor_round_time(d_time)
        x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
        cur_idx = model_info["time"].index(round_d_time)
        cur_model_u, cur_model_v = model_ds["U10"][:][cur_idx, x_idx, y_idx][0], model_ds["V10"][:][cur_idx, x_idx, y_idx][0]
        transport_seconds = (ceil_round_time(d_time) - d_time).seconds
        forward_x += cur_model_u * transport_seconds
        forward_y += cur_model_v * transport_seconds
        # then, treat the integer part
        steps = int((forward_duration - timedelta(seconds=transport_seconds)).seconds / (20 * 60))
        cur_forward_time = d_time + timedelta(seconds=transport_seconds)
        cur_idx = model_info["time"].index(neareast_round_time(cur_forward_time))
        for j in range(cur_idx, cur_idx + steps):
            x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
            cur_model_u, cur_model_v = model_ds["U10"][:][j, x_idx, y_idx][0], model_ds["V10"][:][j, x_idx, y_idx][0]
            forward_x += cur_model_u * 1200
            forward_y += cur_model_v * 1200
        forward_x_idx, forward_y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
        sampling_x_idx.append(forward_x_idx)
        sampling_y_idx.append(forward_y_idx)
    return sampling_x_idx, sampling_y_idx


def agent_based_relocate_idx_mean(wind_obs, model_ds, monitor_x, monitor_y, unit_x, unit_y, start_time, end_time, model_info):
    # backward traj
    current_u_obs, current_v_obs = wind_2_uv(wind_obs["wdspd"], wind_obs["wddir"])
    sampling_time, sampling_u, sampling_v = [], [], []
    for i in range(0, len(wind_obs["time"])):
        if start_time - timedelta(hours=1) <= wind_obs["time"][i] <= end_time:
            cur_samping_u, cur_sampling_v = current_u_obs[i], current_v_obs[i]
            sampling_u.append(cur_samping_u)
            sampling_v.append(cur_sampling_v)
            sampling_time.append(wind_obs["time"][i])
    sampling_u, sampling_v = np.array(sampling_u), np.array(sampling_v)

    cur_u_series, cur_v_series = sampling_u[:], sampling_v[:]
    d_time, forward_duration, forward_x, forward_y, _ = agent_based_backward_loc(monitor_x, monitor_y, unit_x, unit_y, sampling_time, cur_u_series, cur_v_series)
    # forward the none integer part, wind use previous time step wind
    round_d_time = floor_round_time(d_time)
    x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
    cur_idx = model_info["time"].index(round_d_time)
    cur_model_u, cur_model_v = model_ds["U10"][:][cur_idx, x_idx, y_idx][0], model_ds["V10"][:][cur_idx, x_idx, y_idx][0]
    transport_seconds = (ceil_round_time(d_time) - d_time).seconds
    forward_x += cur_model_u * transport_seconds
    forward_y += cur_model_v * transport_seconds
    # then, treat the integer part
    steps = int((forward_duration - timedelta(seconds=transport_seconds)).seconds / (20 * 60))
    cur_forward_time = d_time + timedelta(seconds=transport_seconds)
    cur_idx = model_info["time"].index(neareast_round_time(cur_forward_time))
    for j in range(cur_idx, cur_idx + steps):
        x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
        cur_model_u, cur_model_v = model_ds["U10"][:][j, x_idx, y_idx][0], model_ds["V10"][:][j, x_idx, y_idx][0]
        forward_x += cur_model_u * 1200
        forward_y += cur_model_v * 1200
    forward_x_idx, forward_y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
    return forward_x_idx, forward_y_idx


def equal_time_trajectory(wrf_sfire_ds, met_df, fire_obj, monitor_info):
    model_info = WRFGridInfo(wrf_sfire_ds)
    fire_date = datetime(fire_obj["start_time"].year, fire_obj["start_time"].month, fire_obj["start_time"].day)
    obs_met_data = hourly_met_data(met_df, fire_date)
    obs_sigma = calculate_uv_sigma(obs_met_data, fire_date)
    unit_x, unit_y = model_info["crs"](fire_obj["coord"][0], fire_obj["coord"][1])
    uncertainty_res = {}
    for conc_monitor_name in monitor_info.keys():
        uncertainty_res[conc_monitor_name] = {}
        lower_quantile, upper_quantile, mean_conc = [], [], []
        assigned_met_monitor = monitor_info[conc_monitor_name]["met"]
        conc_monitor_lon, conc_monitor_lat = monitor_info[conc_monitor_name]["coord"][0], monitor_info[conc_monitor_name]["coord"][1]
        conc_monitor_x, conc_monitor_y = model_info["crs"](conc_monitor_lon, conc_monitor_lat)
        # get current met
        current_wind_obs = obs_met_data[assigned_met_monitor]
        for i in range(0, len(current_wind_obs["time"])):
            current_time_idx = model_info["time"].index(current_wind_obs["time"][i])
            current_model_conc = np.squeeze(wrf_sfire_ds["tr17_2"][current_time_idx, 0, :, :])

            if current_wind_obs["time"][i] < fire_obj["start_time"]:
                x_idx, y_idx = get_model_coord_idx(conc_monitor_x, conc_monitor_y, model_info)
                conc_val = current_model_conc[x_idx, y_idx][0]
                lower_quantile.append(conc_val)
                upper_quantile.append(conc_val)
                mean_conc.append(conc_val)
            else:
                traj_start_time, traj_end_time = fire_obj["start_time"], current_wind_obs["time"][i]
                x_idx, y_idx = agent_based_relocate_idx(current_wind_obs, obs_sigma, wrf_sfire_ds, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time, model_info)
                # calculate correct concentration based x_idx, y_idx
                sampling_conc = np.array([current_model_conc[x_idx[idx], y_idx[idx]] for idx in range(0, len(x_idx))])
                lower_quantile.append(np.quantile(sampling_conc, 0.16))
                upper_quantile.append(np.quantile(sampling_conc, 0.84))
                # calculate the mean concentration
                # agent_based_relocate_idx_mean(wind_obs, model_ds, monitor_x, monitor_y, unit_x, unit_y, start_time, end_time, model_info):
                x_idx, y_idx = agent_based_relocate_idx_mean(current_wind_obs, wrf_sfire_ds, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time, model_info)
                mean_conc.append(current_model_conc[x_idx, y_idx][0])

        uncertainty_res[conc_monitor_name]["time"] = current_wind_obs["time"]
        uncertainty_res[conc_monitor_name]["lower"] = lower_quantile
        uncertainty_res[conc_monitor_name]["upper"] = upper_quantile
        uncertainty_res[conc_monitor_name]["mean"] = mean_conc
    return uncertainty_res


def dist_based_relocate_idx(wind_obs, sigma_obs, model_ds, monitor_x, monitor_y, unit_x, unit_y, start_time, end_time, model_info):
    np.random.seed(0)
    sampling_size = 1000
    # backward traj
    current_u_obs, current_v_obs = wind_2_uv(wind_obs["wdspd"], wind_obs["wddir"])
    sampling_time, sampling_u, sampling_v = [], [], []
    for i in range(0, len(wind_obs["time"])):
        if start_time - timedelta(hours=1) <= wind_obs["time"][i] <= end_time:
            mean = [current_u_obs[i], current_v_obs[i]]
            sigma_idx = sigma_obs["time"].index(wind_obs["time"][i])
            cov = [[sigma_obs["u_sigma"][sigma_idx] ** 2, 0], [0, sigma_obs["v_sigma"][sigma_idx] ** 2]]
            cur_samping_u, cur_sampling_v = np.random.multivariate_normal(mean, cov, sampling_size).T
            sampling_u.append(cur_samping_u)
            sampling_v.append(cur_sampling_v)
            sampling_time.append(wind_obs["time"][i])
    sampling_u, sampling_v = np.array(sampling_u), np.array(sampling_v)
    sampling_x_idx, sampling_y_idx, sampling_t_idx = [], [], []
    for i in range(0, sampling_size):
        cur_u_series, cur_v_series = sampling_u[:, i], sampling_v[:, i]
        d_time, forward_duration, forward_x, forward_y, target_dist = agent_based_backward_loc(monitor_x, monitor_y,
                                                                                               unit_x, unit_y,
                                                                                               sampling_time,
                                                                                               cur_u_series,
                                                                                               cur_v_series)
        round_d_time = floor_round_time(d_time)
        x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
        cur_idx = model_info["time"].index(round_d_time)
        cur_model_u, cur_model_v = model_ds["U10"][:][cur_idx, x_idx, y_idx][0], model_ds["V10"][:][cur_idx, x_idx, y_idx][0]
        transport_seconds = (ceil_round_time(d_time) - d_time).seconds
        delta_x, delta_y = cur_model_u * transport_seconds, cur_model_v * transport_seconds
        final_t_idx, final_forward_x, final_forward_y = cur_idx, forward_x, forward_y
        if target_dist < np.sqrt(delta_x ** 2 + delta_y ** 2):
            delta_t = target_dist / np.sqrt(cur_model_u ** 2 + cur_model_v ** 2)
            forward_x += delta_t * cur_model_u
            forward_y += delta_t * cur_model_v
            final_t_idx, final_forward_x, final_forward_y = cur_idx, forward_x, forward_y
            target_dist = 0
            d_time += timedelta(seconds=delta_t)
        else:
            forward_x += cur_model_u * transport_seconds
            forward_y += cur_model_v * transport_seconds
            target_dist = target_dist - np.sqrt((cur_model_u * transport_seconds) ** 2 + (cur_model_v * transport_seconds) ** 2)
            d_time = d_time + timedelta(seconds=transport_seconds)
            cur_idx = model_info["time"].index(neareast_round_time(d_time))
            for j in range(cur_idx, len(model_info["time"]) - 1):
                x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
                cur_model_u, cur_model_v = model_ds["U10"][:][j, x_idx, y_idx][0], model_ds["V10"][:][j, x_idx, y_idx][0]
                cur_delta_x, cur_delta_y = cur_model_u * 1200, cur_model_v * 1200
                if target_dist <= np.sqrt(cur_delta_x ** 2 + cur_delta_y ** 2):
                    delta_t = target_dist / np.sqrt(cur_model_u ** 2 + cur_model_v ** 2)
                    forward_x += delta_t * cur_model_u
                    forward_y += delta_t * cur_model_v
                    d_time += timedelta(seconds=delta_t)
                    final_forward_x, final_forward_y = forward_x, forward_y
                    target_dist = 0
                    break
                else:
                    forward_x += cur_delta_x
                    forward_y += cur_delta_y
                    final_forward_x, final_forward_y = forward_x, forward_y
                    target_dist = target_dist - np.sqrt(cur_delta_x ** 2 + cur_delta_y ** 2)
                    d_time += timedelta(seconds=20 * 60)
        final_t_idx = model_info["time"].index(neareast_round_time(d_time))
        forward_x_idx, forward_y_idx = get_model_coord_idx(final_forward_x, final_forward_y, model_info)
        sampling_x_idx.append(forward_x_idx)
        sampling_y_idx.append(forward_y_idx)
        sampling_t_idx.append(final_t_idx)
    # should I do concentration interpolation?
    return sampling_t_idx, sampling_x_idx, sampling_y_idx


def dist_based_relocate_idx_mean(wind_obs, model_ds, monitor_x, monitor_y, unit_x, unit_y, start_time, end_time, model_info):
    # backward traj
    current_u_obs, current_v_obs = wind_2_uv(wind_obs["wdspd"], wind_obs["wddir"])
    sampling_time, sampling_u, sampling_v = [], [], []
    for i in range(0, len(wind_obs["time"])):
        if start_time - timedelta(hours=1) <= wind_obs["time"][i] <= end_time:
            cur_samping_u, cur_sampling_v = current_u_obs[i], current_v_obs[i]
            sampling_u.append(cur_samping_u)
            sampling_v.append(cur_sampling_v)
            sampling_time.append(wind_obs["time"][i])
    sampling_u, sampling_v = np.array(sampling_u), np.array(sampling_v)

    cur_u_series, cur_v_series = sampling_u[:], sampling_v[:]
    d_time, forward_duration, forward_x, forward_y, target_dist = agent_based_backward_loc(monitor_x, monitor_y, unit_x, unit_y, sampling_time, cur_u_series, cur_v_series)
    round_d_time = floor_round_time(d_time)
    x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
    cur_idx = model_info["time"].index(round_d_time)
    cur_model_u, cur_model_v = model_ds["U10"][:][cur_idx, x_idx, y_idx][0], model_ds["V10"][:][cur_idx, x_idx, y_idx][0]
    transport_seconds = (ceil_round_time(d_time) - d_time).seconds
    delta_x, delta_y = cur_model_u * transport_seconds, cur_model_v * transport_seconds
    final_t_idx, final_forward_x, final_forward_y = cur_idx, forward_x, forward_y
    if target_dist < np.sqrt(delta_x ** 2 + delta_y ** 2):
        delta_t = target_dist / np.sqrt(cur_model_u ** 2 + cur_model_v ** 2)
        forward_x += delta_t * cur_model_u
        forward_y += delta_t * cur_model_v
        final_t_idx, final_forward_x, final_forward_y = cur_idx, forward_x, forward_y
        target_dist = 0
        d_time += timedelta(seconds=delta_t)
    else:
        forward_x += cur_model_u * transport_seconds
        forward_y += cur_model_v * transport_seconds
        target_dist = target_dist - np.sqrt((cur_model_u * transport_seconds) ** 2 + (cur_model_v * transport_seconds) ** 2)
        d_time = d_time + timedelta(seconds=transport_seconds)
        cur_idx = model_info["time"].index(neareast_round_time(d_time))
        for j in range(cur_idx, len(model_info["time"]) - 1):
            x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
            cur_model_u, cur_model_v = model_ds["U10"][:][j, x_idx, y_idx][0], model_ds["V10"][:][j, x_idx, y_idx][0]
            cur_delta_x, cur_delta_y = cur_model_u * 1200, cur_model_v * 1200
            if target_dist <= np.sqrt(cur_delta_x ** 2 + cur_delta_y ** 2):
                delta_t = target_dist / np.sqrt(cur_model_u ** 2 + cur_model_v ** 2)
                forward_x += delta_t * cur_model_u
                forward_y += delta_t * cur_model_v
                d_time += timedelta(seconds=delta_t)
                final_forward_x, final_forward_y = forward_x, forward_y
                target_dist = 0
                break
            else:
                forward_x += cur_delta_x
                forward_y += cur_delta_y
                final_forward_x, final_forward_y = forward_x, forward_y
                target_dist = target_dist - np.sqrt(cur_delta_x ** 2 + cur_delta_y ** 2)
                d_time += timedelta(seconds=20 * 60)
    final_t_idx = model_info["time"].index(neareast_round_time(d_time))
    forward_x_idx, forward_y_idx = get_model_coord_idx(final_forward_x, final_forward_y, model_info)
    return final_t_idx, forward_x_idx, forward_y_idx


def equal_dist_trajectory(wrf_sfire_ds, met_df, fire_obj, monitor_info):
    model_info = WRFGridInfo(wrf_sfire_ds)
    fire_date = datetime(fire_obj["start_time"].year, fire_obj["start_time"].month, fire_obj["start_time"].day)
    obs_met_data = hourly_met_data(met_df, fire_date)
    obs_sigma = calculate_uv_sigma(obs_met_data, fire_date)
    unit_x, unit_y = model_info["crs"](fire_obj["coord"][0], fire_obj["coord"][1])
    uncertainty_res = {}
    for conc_monitor_name in monitor_info.keys():
        uncertainty_res[conc_monitor_name] = {}
        lower_quantile, upper_quantile, mean_conc = [], [], []
        assigned_met_monitor = monitor_info[conc_monitor_name]["met"]
        conc_monitor_lon, conc_monitor_lat = monitor_info[conc_monitor_name]["coord"][0], monitor_info[conc_monitor_name]["coord"][1]
        conc_monitor_x, conc_monitor_y = model_info["crs"](conc_monitor_lon, conc_monitor_lat)
        # get current met
        current_wind_obs = obs_met_data[assigned_met_monitor]
        surface_model_conc = np.squeeze(wrf_sfire_ds["tr17_2"][:, 0, :, :])
        for i in range(0, len(current_wind_obs["time"])):
            current_time_idx = model_info["time"].index(current_wind_obs["time"][i])

            if current_wind_obs["time"][i] < fire_obj["start_time"]:
                x_idx, y_idx = get_model_coord_idx(conc_monitor_x, conc_monitor_y, model_info)
                conc_val = surface_model_conc[i, x_idx, y_idx][0]
                lower_quantile.append(conc_val)
                upper_quantile.append(conc_val)
                mean_conc.append(conc_val)
            else:
                traj_start_time, traj_end_time = fire_obj["start_time"], current_wind_obs["time"][i]
                t_idx, x_idx, y_idx = dist_based_relocate_idx(current_wind_obs, obs_sigma, wrf_sfire_ds, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time, model_info)
                # calculate correct concentration based x_idx, y_idx
                sampling_conc = np.array([surface_model_conc[t_idx[idx], x_idx[idx], y_idx[idx]] for idx in range(0, len(x_idx))])
                lower_quantile.append(np.quantile(sampling_conc, 0.16))
                upper_quantile.append(np.quantile(sampling_conc, 0.84))
                # calculate the mean concentration
                t_idx, x_idx, y_idx = dist_based_relocate_idx_mean(current_wind_obs, wrf_sfire_ds, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time, model_info)
                mean_conc.append(surface_model_conc[t_idx, x_idx, y_idx][0])

        uncertainty_res[conc_monitor_name]["time"] = current_wind_obs["time"]
        uncertainty_res[conc_monitor_name]["lower"] = lower_quantile
        uncertainty_res[conc_monitor_name]["upper"] = upper_quantile
        uncertainty_res[conc_monitor_name]["mean"] = mean_conc
    return uncertainty_res







