import numpy as np
import pandas as pd
import datetime
import os
import argparse
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.spatial.transform import Rotation as R


def construct_physical_property(data_path, imu_timestamps, speed_timestamps, localizatoin_timestamps, all_timestamps, debug=False):
    speed_df = pd.DataFrame(
        columns=["time", "speed"])
    imu_df = pd.DataFrame(
        columns=["time", "o_x", "o_y", "o_z", "o_w", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z", "r_x", "r_y", "r_z"])
    for timestamp in speed_timestamps:
        df = pd.read_csv(os.path.join(
            data_path, "other_data", f"{timestamp}_raw_speed.csv"), header=None)
        time = datetime.datetime.fromtimestamp(
            float(timestamp.replace("_", ".")))
        speed_df.loc[len(speed_df)] = [time, df[0][0]]
    speed_df = speed_df.sort_values(by=['time'], ignore_index=True)
    if debug:
        print(speed_df)
    for timestamp in imu_timestamps:
        df = pd.read_csv(os.path.join(
            data_path, "other_data", f"{timestamp}_raw_imu.csv"), header=None)
        time = datetime.datetime.fromtimestamp(
            float(timestamp.replace("_", ".")))
        r = R.from_quat([df[0][0], df[1][0], df[2][0], df[3][0]])
        r = r.as_euler("xyz")
        imu_df.loc[len(imu_df)] = [time, df[0][0], df[1][0], df[2][0], df[3][0],
                                   df[0][1], df[1][1], df[2][1], df[0][2], df[1][2], df[2][2], r[0], r[1], r[2]]
    imu_df = imu_df.sort_values(by=['time'], ignore_index=True)
    if debug:
        print(imu_df)
    if debug:
        f = plt.figure()
        for index,  target_column in enumerate(["x", "y", "z"]):
            plt.plot(imu_df[f"r_{target_column}"],
                     label=f"rotation-{target_column}")
            plt.plot(imu_df[f"w_{target_column}"],
                     label=f"angular_velocity-{target_column}")
            if target_column == "z":
                continue
                # omit gravity
            plt.plot(imu_df[f"a_{target_column}"],
                     label=f"linear_acceleration-{target_column}")
        plt.legend()
        plt.savefig(os.path.join(
            data_path, f"physics-analysis-imu.png"))
    groundtruth_df = pd.DataFrame(columns=["time", "pos_x", "pos_y"])
    for timestamp in localizatoin_timestamps:
        try:
            df = pd.read_csv(os.path.join(data_path, "dataset",
                                          timestamp, "gound_turth_pose.csv"), header=None)
            time = pd.Timestamp(
                float(timestamp.replace("_", ".")), unit="s")
            groundtruth_df.loc[len(groundtruth_df)] = [
                time, df[0][0], df[1][0]]
        except FileNotFoundError:
            pass
    groundtruth_df = groundtruth_df.sort_values(by=['time'], ignore_index=True)

    if debug:
        print(groundtruth_df)
    if debug:
        f = plt.figure()
        ax = groundtruth_df.plot.scatter(
            x="pos_x", y="pos_y", c="time", colormap='viridis')
        plt.savefig(os.path.join(
            data_path, f"physics-analysis-ground_truth.png"))

    result_df = pd.DataFrame(columns=["time", "pos_x", "pos_y"])

    for timestamp in all_timestamps:
        time = datetime.datetime.fromtimestamp(
            float(timestamp.replace("_", ".")))
        result_df.loc[len(result_df)] = [time, 0, 0]

    result_df = result_df.sort_values(by=['time'], ignore_index=True)
    if debug:
        print(result_df)
    delta_t = result_df.loc[len(result_df)-1][0].to_numpy() - \
        result_df.loc[0][0].to_numpy()
    delta_t = delta_t / np.timedelta64(1, 'ns')
    kalman_filter_step_size = 8000
    t = np.linspace(0, delta_t, kalman_filter_step_size)
    dt = t[1]-t[0]
    vx = np.zeros(kalman_filter_step_size)
    vy = np.zeros(kalman_filter_step_size)
    x = np.zeros(kalman_filter_step_size)
    y = np.zeros(kalman_filter_step_size)
    ax = np.zeros(kalman_filter_step_size)
    ay = np.zeros(kalman_filter_step_size)
    speed_ptr = 0
    imu_ptr = 0
    for kalman_index in range(kalman_filter_step_size):
        pass


if __name__ == '__main__':
    mpl.rcParams['figure.dpi'] = 300
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    speed_timestamps = []
    imu_timestamps = []
    localization_timestamps = []
    all_timestamps = []

    for file_entry in list(os.scandir(os.path.join(args.data_path, "other_data"))):
        filename = file_entry.name
        if filename.endswith("raw_imu.csv"):
            imu_timestamps.append(re.findall(r"\d+_\d+", filename)[0])
        if filename.endswith("raw_speed.csv"):
            speed_timestamps.append(re.findall(r"\d+_\d+", filename)[0])
    with open(os.path.join(args.data_path, "localization_timestamp.txt")) as reader:
        for line in reader.readlines():
            localization_timestamps.append(line.replace("\n", ""))
    with open(os.path.join(args.data_path, "all_timestamp.txt")) as reader:
        for line in reader.readlines():
            all_timestamps.append(line.replace("\n", ""))
    construct_physical_property(
        args.data_path, imu_timestamps, speed_timestamps,  localization_timestamps, all_timestamps, True)
