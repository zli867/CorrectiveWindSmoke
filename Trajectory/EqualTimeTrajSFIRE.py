import pandas as pd
from datetime import datetime
import netCDF4 as nc
import numpy as np
from uncertainty_util import get_polygons_ctr, equal_time_trajectory
import json
from shapely.geometry import shape
import pickle


sfire_uncertainty_res = {}
select_dates = [datetime(2022, 3, 2), datetime(2022, 3, 3), datetime(2022, 3, 5)]

monitor_met_map = {
    datetime(2022, 3, 2): {
        "Trailer_FS": {"coord": [-81.64287409999999, 32.0663665], "met": "gpem"},
        "USFS 1078": {"coord": [-81.67125, 32.01727], "met": "gpem"},
        "USFS 1079": {"coord": [-81.5189, 32.071220000000004], "met": "USFS 1079"},
    },
    datetime(2022, 3, 3): {
        "Trailer_FS": {"coord": [-81.64287409999999, 32.0663665], "met": "USFS 1079"},
        "USFS 1078": {"coord": [-81.6712, 32.01726], "met": "USFS 1079"},
        "USFS 1079": {"coord": [-81.67526, 32.04538], "met": "USFS 1079"}
    },
    datetime(2022, 3, 5): {
        "Trailer_FS": {"coord": [-81.88420579999998, 32.054632500000004], "met": "USFS 1079"},
        "USFS 1078": {"coord": [-81.89891, 32.03498], "met": "USFS 1079"},
        "USFS 1079": {"coord": [-81.86196, 32.06538], "met": "USFS 1079"}
    }
}

for select_date in select_dates:
    print(select_date)
    conc_filename = "/Volumes/Expansion/WindUncertainty/Measurements/conc/combined_PM25_conc.csv"
    wrf_sfire_filename = "/Volumes/Expansion/WindUncertainty/SFIRE/brute_force/%s/wrfout_d01_%s_00:00:00" % (select_date.strftime("%m%d"), select_date.strftime("%Y-%m-%d"))
    # wrf_sfire_filename = "/Volumes/Expansion/WindUncertainty/SFIRE_Results/nudge/%s/wrfout_d01_%s_00:00:00" % (select_date.strftime("%m%d"), select_date.strftime("%Y-%m-%d"))
    wind_obs = "/Volumes/Expansion/WindUncertainty/Measurements/met/hourly_rounded_FtStwrt.csv"
    cluster_file = "/Volumes/Shield/WindUncertaintyImpacts/data/BurnInfo/BurnCluster.json"
    fire_file = "/Volumes/Shield/WindUncertaintyImpacts/data/BurnInfo/Select_BurnInfo.json"

    with open(cluster_file) as json_file:
        cluster_data = json.load(json_file)
    main_cluster = cluster_data["BurnCluster"][datetime.strftime(select_date, "%Y-%m-%d")]["Main"]

    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    met_df = pd.read_csv(wind_obs, parse_dates=['UTC_time'], date_parser=dateparse)
    wrf_sfire_ds = nc.Dataset(wrf_sfire_filename)

    # fire polygon and fire info
    with open(fire_file) as json_file:
        fire_events = json.load(json_file)

    rx_polygons, fire_start_time = [], []
    for fire_event in fire_events["fires"]:
        fire_date = datetime.strptime(fire_event["date"], "%Y-%m-%d")
        if fire_date == select_date and fire_event["id"] in main_cluster and fire_event["type"] == "rx":
            fire_start_time.append(datetime.strptime(fire_event["start_UTC"], "%Y-%m-%d %H:%M:%S"))
            rx_polygons.append(shape(fire_event["perimeter"]))
    fire_start_time = min(fire_start_time)
    fire_start_hour = datetime(fire_start_time.year, fire_start_time.month, fire_start_time.day, fire_start_time.hour)
    unit_coord_lon, unit_coord_lat = get_polygons_ctr(rx_polygons)
    monitor_mappings = monitor_met_map[select_date]
    fire_obj = {"coord": [unit_coord_lon, unit_coord_lat], "start_time": fire_start_hour}

    uncertainty_res = equal_time_trajectory(wrf_sfire_ds, met_df, fire_obj, monitor_mappings)
    sfire_uncertainty_res[select_date] = uncertainty_res

with open("/Volumes/Shield/WindUncertaintyImpacts/TrajectoryNew/res/equal_time_icbc_test.pickle", 'wb') as handle:
    pickle.dump(sfire_uncertainty_res, handle, protocol=pickle.HIGHEST_PROTOCOL)