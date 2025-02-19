import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import netCDF4 as nc
import matplotlib.pyplot as plt


def extract_elev(site_lon, site_lat, GRIDCRO2D):
    dataset = nc.Dataset(GRIDCRO2D)
    elev = dataset["HT"][:].flatten()
    lat = dataset["LAT"][:].flatten()
    lon = dataset["LON"][:].flatten()

    distance = (lon - site_lon) ** 2 + (lat - site_lat) ** 2
    index = np.argmin(distance)
    return elev[index]


# Time is UTC time?
def convert_ebam_to_dict(ID, Name, ebam_files, GRIDCRO2D):
    """

    :param ebam_files:
    :return: observation dict: {"Time": datetime object array,
                                "lat": float,
                                "lon": float,
                                "ID": string,
                                "Name": string,
                                "elev": float,
                                "pressure": numpy array,
                                "temperature": numpy array,
                                "dew point": numpy array,
                                "wdspd": numpy array,
                                "wddir": numpy array,
                                "u": numpy array,
                                "v": numpy array,
                                "rh": numpy array,
                                "thickness": numpy array}
    """
    res_dict = {"Time": None,
                "lat": None,
                "lon": None,
                "ID": ID,
                "Name": Name,
                "elev": None,
                "pressure": None,
                "temperature": None,
                "dew point": None,
                "wdspd": None,
                "wddir": None,
                "u": None,
                "v": None,
                "rh": None,
                "thickness": None}
    dataframe = pd.read_csv(ebam_files)
    dataframe = dataframe[["Latitude", "Longitude", "Date_Time_GMT", "W.S", "W.D_declination"]]
    dataframe = dataframe.dropna()

    res_dict["lat"] = dataframe["Latitude"].to_numpy()
    res_dict["lon"] = dataframe["Longitude"].to_numpy()
    time_str = dataframe["Date_Time_GMT"].to_numpy()
    time_array = []
    elev = []
    for i in range(0, len(res_dict["lat"])):
        elev.append(extract_elev(res_dict["lon"][i], res_dict["lat"][i], GRIDCRO2D))
        time_array.append(datetime.strptime(time_str[i], "%Y-%m-%d %H:%M:%S"))
    res_dict["Time"] = time_array
    res_dict["elev"] = np.array(elev)
    res_dict["wdspd"] = dataframe["W.S"].to_numpy()
    res_dict["wddir"] = dataframe["W.D_declination"].to_numpy()
    return res_dict


def convert_RAWS_to_dict(ID, Name, raws_files, lon, lat, GRIDCRO2D):
    """

    :param ebam_files:
    :return: observation dict
    """
    res_dict = {"Time": None,
                "lat": None,
                "lon": None,
                "ID": ID,
                "Name": Name,
                "elev": None,
                "pressure": None,
                "temperature": None,
                "dew point": None,
                "wdspd": None,
                "wddir": None,
                "u": None,
                "v": None,
                "rh": None,
                "thickness": None}
    # dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    dataframe = pd.read_csv(raws_files)
    dataframe = dataframe[["Time_LST", "Wind V Dir Deg", "Ave m/s"]]
    dataframe = dataframe.dropna()
    elev = extract_elev(lon, lat, GRIDCRO2D)
    time_array = []
    time_data = dataframe["Time_LST"].to_numpy()
    time_zone = -5
    for i in range(0, len(time_data)):
        # revert local standard time to UTC
        current_time = datetime.strptime(time_data[i], '%Y-%m-%d %H:%M:%S')
        time_array.append(current_time - timedelta(hours=time_zone))
    res_dict["Time"] = time_array
    res_dict["lat"] = np.array([lat] * len(time_array))
    res_dict["lon"] = np.array([lon] * len(time_array))
    res_dict["elev"] = np.array([elev] * len(time_array))
    res_dict["wdspd"] = dataframe["Ave m/s"].to_numpy()
    res_dict["wddir"] = dataframe["Wind V Dir Deg"].to_numpy()
    return res_dict


def convert_standard_MET_to_dict(filename):
    """

    :param ebam_files:
    :return: observation dict
    """
    res_list = []
    df = pd.read_csv(filename)
    monitor_set = list(set(df["monitor"]))
    for current_monitor in monitor_set:
        select_df = df[df["monitor"] == current_monitor]
        select_df = select_df.reset_index(drop=True)
        current_monitor = current_monitor.replace(" ", "_")
        res_dict = {"Time": None,
                    "lat": None,
                    "lon": None,
                    "ID": current_monitor,
                    "Name": current_monitor,
                    "elev": None,
                    "pressure": None,
                    "temperature": None,
                    "dew point": None,
                    "wdspd": None,
                    "wddir": None,
                    "u": None,
                    "v": None,
                    "rh": None,
                    "thickness": None}

        time_array = []
        for i in range(0, len(select_df["UTC_time"])):
            current_time = datetime.strptime(select_df["UTC_time"][i], '%Y-%m-%d %H:%M:%S')
            time_array.append(current_time)
        res_dict["Time"] = time_array
        res_dict["lat"] = select_df["lat"].to_numpy()
        res_dict["lon"] = select_df["lon"].to_numpy()
        res_dict["elev"] = select_df["elevation"].to_numpy()
        res_dict["wdspd"] = select_df["wdspd"].to_numpy()
        res_dict["wddir"] = select_df["wddir"].to_numpy()
        res_list.append(res_dict)
    return res_list