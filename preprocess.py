import json
import cv2
import numpy as np
from utils import *
import os
import pickle as pkl

def collate_data(input_folder, output_path, start_id, end_id):
    session_data = list()
    for k in range(start_id, end_id+1, 1):
        data_path = os.path.join(input_folder, f"{k}.json")
        with open(data_path, "r") as f:
            data = json.load(f)["_items"]
        for i in range(25):
            connect_time = data[i]["connectionTime"]
            done_time = data[i]["doneChargingTime"]
            pilot_signal = data[i]["pilotSignal"]
            current_signal = data[i]["chargingCurrent"]
            warning = f"remove json{k}-item{i}!"
            if connect_time is None or done_time is None or pilot_signal is None or current_signal is None:
                print(warning)
                continue
            connect_timestamp = gmt_to_timestamp(connect_time)
            done_timestamp = gmt_to_timestamp(done_time)
            pilot_time = pilot_signal["timestamps"]  # arrival time is the first element of pilot time
            pilot_timestamp = gmt_to_timestamp(pilot_time)
            pilot_value = pilot_signal["pilot"]
            pilot_timestamp, pilot_value = interpolate_signal(pilot_timestamp, pilot_value)  # resolution: 1s/p
            current_time = current_signal["timestamps"]
            current_timestamp = gmt_to_timestamp(current_time)
            current_value = current_signal["current"]
            current_timestamp, current_value = interpolate_signal(current_timestamp, current_value)  # resolution: 1s/p
            if connect_timestamp < pilot_timestamp[0]:
                connect_timestamp = pilot_timestamp[0]
            if done_timestamp > pilot_timestamp[-1]:
                done_timestamp = pilot_timestamp[-1]
            pilot_connect_index = pilot_timestamp.index(connect_timestamp)
            pilot_done_index = pilot_timestamp.index(done_timestamp)
            pilot_data = pilot_value[pilot_connect_index:pilot_done_index+1]
            current_connect_index = current_timestamp.index(connect_timestamp)
            current_done_index = current_timestamp.index(done_timestamp)
            current_data = current_value[current_connect_index:current_done_index+1]  # the last current is 0
            session_timestamp = pilot_timestamp[pilot_connect_index:pilot_done_index+1]
            user_input = data[i]["userInputs"]
            if user_input is not None:
                user_id = user_input[0]["userID"]
                min_requested = user_input[0]["minutesAvailable"]
                energy_requested = user_input[0]["kWhRequested"]
                energy_delivered = data[i]["kWhDelivered"]
                session_id = data[i]["sessionID"]
                charging_session = {"sessionID": session_id, "energyD": energy_delivered,
                                    "userID": user_id, "duration": min_requested, "energyR": energy_requested,
                                    "timestamp": session_timestamp, "pilot": pilot_data, "current": current_data}
                session_data.append(charging_session)
                print(f"get json{k}-item{i}-user{user_id}!")
    output_collation = {"sessions": session_data}
    with open(f"{output_path}.pkl", "wb") as f:
        pkl.dump(output_collation, f)

def get_driver_data(input_path, output_path, down_scale):
    with open(input_path, "rb") as f:
        raw_data = pkl.load(f)
    session_data = raw_data["sessions"]
    for s in session_data:
        session_id, energy_delivered = s["sessionID"], s["energyD"]
        user_id, min_requested, energy_requested = s["userID"], s["duration"], s["energyR"]
        timestamp, pilot_data, current_data = s["timestamp"], s["pilot"], s["current"]
        # down sample to 1min/p
        pilot_data = down_sample(pilot_data, scale=down_scale)
        current_data = down_sample(current_data, scale=down_scale)
        # output session data
        file_path = f"{output_path}/user{user_id}-{session_id.split(' ')[0]}"
        output_session = {"pilot": pilot_data, "current": current_data, "kwhRequested": energy_requested, "minAvailable": min_requested, "kwhDelivered": energy_delivered}
        pkl_path = file_path + ".pkl"
        with open(pkl_path, "wb") as f:
            pkl.dump(output_session, f)
        img_title = f"minutesAvailable:{min_requested}, kwhRequested:{energy_requested}, kwhDelivered:{energy_delivered}"
        img_path = file_path + ".png"
        plot_session(pilot_data, current_data, img_title, img_path)
        print(f"{file_path} done!")

def get_station_data(input_folder, site, start_id, end_id):  # 5min resolution
    station_data = dict()
    for k in range(start_id, end_id+1, 1):
        json_path = os.path.join(input_folder, f"{k}.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path, "r") as f:
            raw_data = json.load(f)["_items"]
        for i in range(len(raw_data)):
            connect_time = raw_data[i]["connectionTime"]
            done_time = raw_data[i]["doneChargingTime"]
            if done_time is None:
                done_time = raw_data[i]["disconnectTime"]
            energy_delivered = raw_data[i]["kWhDelivered"]
            st = time.strptime(connect_time, gmt_format)
            et = time.strptime(done_time, gmt_format)
            syear, smon, sday, shour, smin = st.tm_year, st.tm_mon, st.tm_mday, st.tm_hour, st.tm_min
            eyear, emon, eday, ehour, emin = et.tm_year, et.tm_mon, et.tm_mday, et.tm_hour, et.tm_min
            skey, ekey = f"{syear}-{smon}-{sday}", f"{eyear}-{emon}-{eday}"
            if skey not in station_data:
                station_data[skey] = [np.zeros(288), 0]  # [daily charging load profile, number of EVs]
            if ekey not in station_data:
                station_data[ekey] = [np.zeros(288), 0]
            if sday == eday:  # within the same day
                duration = (ehour-shour)*12 + (emin-smin)//5
                if duration == 0:
                    continue
                unit_power = energy_delivered*12/duration
                start_index = shour*12 + smin//5
                end_index = ehour*12 + emin//5
                station_data[skey][0][start_index:end_index+1] += unit_power
            else:  # span two days
                duration0 = (23-shour)*12 + (60-smin)//5
                duration1 = ehour*12 + emin//5
                duration = duration0 + duration1
                if duration == 0:
                    continue
                unit_power = energy_delivered * 12 / duration
                start_index0 = shour*12 + smin//5
                station_data[skey][0][start_index0:] += unit_power
                end_index1 = ehour*12 + emin//5
                station_data[ekey][0][0:end_index1+1] += unit_power
            station_data[skey][1] += 1
        print(f"{k}.json get done!")
    with open(f"ACN-data/{site}/station_data.pkl", "wb") as f:
        pkl.dump(station_data, f)

def output_station_data(input_path, output_folder, delay):
    with open(input_path, "rb") as f:
        station_data = pkl.load(f)
    keys = list(station_data.keys())
    for i in range(len(keys)-1):
        # timezone delay
        date0, date1 = keys[i], keys[i+1]
        s0, s1 = station_data[date0][0][delay:], station_data[date1][0][0:delay]  # caltech-8 hours, jpl-7 hours
        ev_num = station_data[date0][1]
        station_power = np.concatenate((s0, s1))
        station_dict = {"power": station_power, "number": ev_num}
        output_path = f"{output_folder}/{date0}"
        with open(f"{output_path}.pkl", "wb") as f:
            pkl.dump(station_dict, f)
        plt.plot(station_power)
        plt.xlim((0, 24))
        positions = list(range(0, 288+1, 12))
        ticks = list(range(0, 24+1, 1))
        plt.xticks(positions, ticks)
        plt.xlabel("Daily time [hour]")
        plt.ylabel("Charging power [kW]")
        plt.title(f"{ev_num} EVs")
        plt.savefig(f"{output_path}.png")
        plt.clf()
        print(f"{date0} output done!")

if __name__ == "__main__":
    # 2019.8.1—2019.8.31
    # collate_data("ACN-data/jpl/meta", "ACN-data/jpl/2019.8.1_31", start_id=595, end_id=656)
    # get_driver_data("ACN-data/jpl/2019.8.1_31.pkl", "ACN-data/jpl/driver", down_scale=60)

    # workplace
    # get_station_data("D:\\Datasets\\ACN-Data\\jpl", site="jpl", start_id=50, end_id=1346)
    # output_station_data("ACN-data/jpl/station_data.pkl", "ACN-data/jpl/station", delay=84)

    # residence
    # get_station_data("D:\\Datasets\\ACN-Data\\caltech", site="caltech", start_id=1, end_id=1257)
    # output_station_data("ACN-data/caltech/station_data.pkl", "ACN-data/caltech/station", delay=96)

    # get_station_data("D:\\Datasets\\ACN-Data\\office001", site="office001", start_id=1, end_id=68)
    # output_station_data("ACN-data/office001/station_data.pkl", "ACN-data/office001/station")
    pass