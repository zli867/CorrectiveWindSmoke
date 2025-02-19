import DataReader
import little_r as lr
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from util import adjust_ratio

met_filename = "/Volumes/Expansion/WindUncertainty/Interpolation/interpolated_ftstwrt.csv"
file_time_interval = timedelta(hours=3)
start_time = datetime(2022, 3, 1, 0)
end_time = datetime(2022, 3, 6, 0)
current_time = start_time
output_path = "/Users/zongrunli/Desktop/obs_little_r/"

FM = 'FM-35 TEMP'
bogus = False
data_list = DataReader.convert_standard_MET_to_dict(met_filename)

while current_time <= end_time:
    print(current_time)
    # SURFACE_OBS: 2021031900
    output_name = "SELF_OBS:" + datetime.strftime(current_time, "%Y%m%d%H")
    output_file = output_path + output_name
    with open(output_file, "w") as file:
        for file_index in range(0, len(data_list)):
            select_data = data_list[file_index]
            select_file_time = np.array(select_data["Time"])
            select_index = (select_file_time >= current_time) & (select_file_time < current_time + file_time_interval)
            select_data_time = select_file_time[select_index]
            select_wind_spd = select_data["wdspd"][select_index]
            # speed adjustment
            select_wind_spd = select_wind_spd * adjust_ratio(2, 10)

            select_wind_dir = select_data["wddir"][select_index]
            site_elev = select_data["elev"][select_index]
            site_lon = select_data["lon"][select_index]
            site_lat = select_data["lat"][select_index]
            # Check whether the monitor is moved or not
            if len(site_lat) == 0:
                continue

            if (np.max(site_lat) - np.min(site_lat)) > 0.01 and (np.max(site_lon) - np.min(site_lon)) > 0.01:
                continue
            else:
                site_elev = np.mean(site_elev)
                site_lon = np.mean(site_lon)
                site_lat = np.mean(site_lat)
            site_id = select_data["ID"]
            site_name = select_data["Name"]
            for i in range(0, len(select_data_time)):
                if (not np.isnan(select_wind_spd[i])) and (not np.isnan(select_wind_dir[i])):
                    date = datetime.strftime(pd.to_datetime(select_data_time[i]), "%Y%m%d%H%M%S")
                    header_str = lr.header_record(site_lat, site_lon, site_id, site_name, FM, site_elev, bogus, date)
                    file.write(header_str + "\n")
                    # for data
                    pres = -888888.00000  # Pa
                    h = site_elev  # m
                    t = -888888.00000  # K
                    td = -888888.00000  # K
                    wspd = select_wind_spd[i]  # m/s
                    wdir = select_wind_dir[i]  # deg
                    u = -888888.00000  # m/s
                    v = -888888.00000  # m/s
                    rh = -888888.00000  # %
                    tk = -888888.00000  # m

                    data_str = lr.data_record(pres, h, t, td, wspd, wdir, u, v, rh, tk)
                    file.write(data_str + "\n")
                    ending_str = lr.ending_record()
                    file.write(ending_str + "\n")

                    # for tail
                    num_fields = 3

                    tail_str = lr.tail_record(num_fields)
                    file.write(tail_str + "\n")
        file.close()
    current_time = current_time + file_time_interval