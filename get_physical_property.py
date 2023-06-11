import numpy as np
import pandas as pd
import datetime
import os
import argparse
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.spatial.transform import Rotation as RR

from filterpy.kalman import KalmanFilter


def linear_interpolation(ts, ts_a, ts_b, ya, yb):
    pt = pd.Timestamp(2023, 1, 1)
    x = (ts - pt).to_numpy()/np.timedelta64(1, 'ns')
    xa = (ts_a - pt).to_numpy()/np.timedelta64(1, 'ns')
    xb = (ts_b - pt).to_numpy()/np.timedelta64(1, 'ns')
    return np.interp(x, [xa, xb], [ya, yb])


def construct_physical_property(data_path, imu_timestamps, speed_timestamps, localizatoin_timestamps, all_timestamps, kalman_filter_step_size,  debug=False):
    speed_df = pd.DataFrame(
        columns=["time", "speed"])
    imu_df = pd.DataFrame(
        columns=["time", "o_x", "o_y", "o_z", "o_w", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z", "r_x", "r_y", "r_z"])
    for timestamp in speed_timestamps:
        df = pd.read_csv(os.path.join(
            data_path, "other_data", f"{timestamp}_raw_speed.csv"), header=None)
        time = pd.Timestamp(
            float(timestamp.replace("_", ".")), unit="s")
        speed_df.loc[len(speed_df)] = [time, df[0][0]]
    speed_df = speed_df.sort_values(by=['time'], ignore_index=True)
    if debug:
        print(speed_df)
    for timestamp in imu_timestamps:
        df = pd.read_csv(os.path.join(
            data_path, "other_data", f"{timestamp}_raw_imu.csv"), header=None)
        time = pd.Timestamp(
            float(timestamp.replace("_", ".")), unit="s")
        r = RR.from_quat([df[0][0], df[1][0], df[2][0], df[3][0]])
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
    pos_df = pd.DataFrame(columns=["time", "pos_x", "pos_y"])
    for timestamp in localizatoin_timestamps:
        try:
            df = pd.read_csv(os.path.join(data_path, "dataset",
                                          timestamp, "gound_turth_pose.csv"), header=None)
            time = pd.Timestamp(
                float(timestamp.replace("_", ".")), unit="s")
            pos_df.loc[len(pos_df)] = [
                time, df[0][0], df[1][0]]
        except FileNotFoundError:
            pass
    pos_df = pos_df.sort_values(by=['time'], ignore_index=True)

    if debug:
        print(pos_df)
    if debug:
        f = plt.figure()
        ax = pos_df.plot.scatter(
            x="pos_x", y="pos_y", c="time", colormap='viridis')
        plt.savefig(os.path.join(
            data_path, f"physics-analysis-ground_truth.png"))

    result_df = pd.DataFrame(columns=["time", "pos_x", "pos_y"])
    for timestamp in all_timestamps:
        time = pd.Timestamp(
            float(timestamp.replace("_", ".")), unit="s")
        result_df.loc[len(result_df)] = [time, 0, 0]

    result_df = result_df.sort_values(by=['time'], ignore_index=True)
    if debug:
        print(result_df)
    start_time = result_df.loc[0][0]
    end_time = result_df.loc[len(result_df)-1][0]
    kalman_data_df = pd.DataFrame(
        columns=["time", "x", "y", "vx", "vy", "ax", "ay"])
    t = pd.date_range(start=start_time,
                      end=end_time,
                      periods=kalman_filter_step_size)
    kalman_data_df["time"] = t
    pos_index = 0
    speed_index = 0
    imu_index = 0
    for index in range(kalman_filter_step_size):
        current_time = t[index]
        while pos_df["time"][pos_index] < current_time and pos_index < len(pos_df)-1:
            pos_index += 1
        while speed_df["time"][speed_index] < current_time and speed_index < len(speed_df)-1:
            speed_index += 1
        while imu_df["time"][imu_index] < current_time and imu_index < len(imu_df)-1:
            imu_index += 1
        if pos_index == 0:
            x = pos_df["pos_x"][pos_index]
            y = pos_df["pos_y"][pos_index]
        else:
            x = linear_interpolation(current_time, pos_df["time"][pos_index-1], pos_df["time"][pos_index],
                                     pos_df["pos_x"][pos_index-1], pos_df["pos_x"][pos_index])
            y = linear_interpolation(current_time, pos_df["time"][pos_index-1], pos_df["time"][pos_index],
                                     pos_df["pos_y"][pos_index-1], pos_df["pos_y"][pos_index])
        if imu_index == 0:
            ax = imu_df["a_x"][imu_index]
            ay = imu_df["a_y"][imu_index]
        else:
            ax = linear_interpolation(current_time, imu_df["time"][imu_index-1], imu_df["time"][imu_index],
                                      imu_df["a_x"][imu_index-1], imu_df["a_x"][imu_index])
            ay = linear_interpolation(current_time, imu_df["time"][imu_index-1], imu_df["time"][imu_index],
                                      imu_df["a_y"][imu_index-1], imu_df["a_y"][imu_index])
        if speed_index == 0 or imu_index == 0:
            vx = np.cos(
                imu_df["r_z"][imu_index])*speed_df["speed"][speed_index]
            vy = np.sin(
                imu_df["r_z"][imu_index])*speed_df["speed"][speed_index]
        else:
            r = linear_interpolation(current_time, imu_df["time"][imu_index-1], imu_df["time"][imu_index],
                                     imu_df["r_z"][imu_index-1], imu_df["r_z"][imu_index])
            s = linear_interpolation(current_time, speed_df["time"][speed_index-1], speed_df["time"][speed_index],
                                     speed_df["speed"][speed_index-1], speed_df["speed"][speed_index])
            vx = np.cos(r)*s
            vy = np.sin(r)*s
        kalman_data_df.loc[index] = [
            kalman_data_df["time"][index], x, y, vx, vy, ax, ay]
    if debug:
        print(kalman_data_df)
        f = plt.figure()
        ax = kalman_data_df.plot.scatter(
            x="x", y="y", c="time", colormap='cool')
        prev = None
        if kalman_filter_step_size < 200:
            for row in kalman_data_df[::-1].itertuples():
                if prev:
                    ax.annotate('',
                                xy=(prev.x, prev.y),
                                xytext=(row.x, row.y),
                                arrowprops=dict(arrowstyle="->"),
                                color='k',
                                )
                prev = row
        plt.savefig(os.path.join(
            data_path, f"physics-analysis-kalman_filter_measure.png"))
    kalman_result_df = pd.DataFrame(
        columns=["time", "x", "y", "vx", "vy", "ax", "ay"])
    kalman_result_df["time"] = t
    # source: https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CA-2.ipynb?create=1
    x = np.matrix(kalman_data_df.loc[0][1:]).T
    P = np.diag([1.0, 1.0, 3.0, 3.0, 3.0, 3.0])
    dt = (t[1]-t[0]).total_seconds()
    A = np.matrix([[1, 0, dt, 0, 0.5*(dt**2), 0],
                   [0, 1, 0, dt, 0, 0.5*(dt**2)],
                   [0, 0, 1,  0, dt, 0],
                   [0, 0, 0,  1, 0, dt],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1]
                   ])
    H = np.diag([1, 1, 1, 1, 1, 1])
    R = np.diag([0.05**2, 0.05**2, 0.3**2, 0.3**2, 0.3**2, 0.3**2])
    sj = 0.1
    Q = np.matrix([[(dt**6)/36, 0, (dt**5)/12, 0, (dt**4)/6, 0],
                   [0, (dt**6)/36, 0, (dt**5)/12, 0, (dt**4)/6],
                   [(dt**5)/12, 0, (dt**4)/4, 0, (dt**3)/2, 0],
                   [0, (dt**5)/12, 0, (dt**4)/4, 0, (dt**3)/2],
                   [(dt**4)/6, 0, (dt**3)/2, 0, (dt**2), 0],
                   [0, (dt**4)/6, 0, (dt**3)/2, 0, (dt**2)]]) * sj**2
    I = np.eye(6)
    for filter_step in range(kalman_filter_step_size):
        x = A*x
        P = A*P*A.T + Q
        S = H*P*H.T + R
        K = (P*H.T) * np.linalg.pinv(S)
        Z = np.array(kalman_data_df.loc[filter_step][1:]).reshape(
            H.shape[0], 1)
        y = Z - (H*x)
        x = x + (K*y)
        P = (I - (K*H))*P
        kalman_result_df.loc[filter_step] = [
            kalman_result_df["time"][filter_step], x[0, 0], x[1, 0], x[2, 0], x[3, 0], x[4, 0], x[5, 0]]
    if debug:
        print(kalman_result_df)
        f = plt.figure()
        ax = kalman_result_df.plot.scatter(
            x="x", y="y", c="time", alpha=0.5,
            colormap="cool")
        prev = None
        if kalman_filter_step_size < 200:
            for row in kalman_result_df[::-1].itertuples():
                if prev:
                    ax.annotate('',
                                xy=(prev.x, prev.y),
                                xytext=(row.x, row.y),
                                arrowprops=dict(arrowstyle="->"),
                                color='k',
                                )
                prev = row
        plt.savefig(os.path.join(
            data_path, f"physics-analysis-kalman_predict.png"))
    if debug:
        f = plt.figure
        ax_kalman = kalman_result_df.plot.scatter(
            x="x", y="y", c="blue", alpha=0.5,
        )
        ax_real = pos_df.plot.scatter(
            x="pos_x", y="pos_y", c="red", s=1, alpha=0.1)
        plt.savefig(os.path.join(
            data_path, f"physics-analysis-validate.png"))
    return kalman_result_df


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
        args.data_path, imu_timestamps, speed_timestamps,  localization_timestamps, all_timestamps, 100,  True)
