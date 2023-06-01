import numpy as np
import pandas as pd
import datetime
import os
import argparse
from tqdm import tqdm
import re
import matplotlib.pyplot as plt


def construct_physical_property(data_path, imu_timestamps, speed_timestamps):
    speed_df = pd.DataFrame(
        columns=["time", "speed"])
    imu_df = pd.DataFrame(
        columns=["time", "o_x", "o_y", "o_z", "o_w", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z"])
    for i,  timestamp in enumerate(speed_timestamps):
        df = pd.read_csv(os.path.join(
            data_path, "other_data", f"{timestamp}_raw_speed.csv"), header=None)
        time = datetime.datetime.fromtimestamp(
            float(timestamp.replace("_", ".")))
        speed_df.loc[len(speed_df)] = [time, df[0][0]]
    print(speed_df)
    for i, timestamp in enumerate(imu_timestamps):
        df = pd.read_csv(os.path.join(
            data_path, "other_data", f"{timestamp}_raw_imu.csv"), header=None)
        time = datetime.datetime.fromtimestamp(
            float(timestamp.replace("_", ".")))
        imu_df.loc[len(imu_df)] = [time, df[0][0], df[1][0], df[2][0], df[3][0],
                                   df[0][1], df[1][1], df[2][1], df[0][2], df[1][2], df[2][2]]
    for section in range(len(imu_df)//100):
        f = plt.figure(section)
        for index,  target_column in enumerate(["x", "y", "z", "w"]):
            plt.plot(imu_df[f"o_{target_column}"][section*100:(section+1)*100].ewm(span=10).mean(),
                     label=f"orientation-{target_column}")
            if target_column == "w":
                continue
            plt.plot(imu_df[f"w_{target_column}"][section*100:(section+1)*100].ewm(span=10).mean(),
                     label=f"angular_velocity-{target_column}")
            plt.plot(imu_df[f"a_{target_column}"][section*100:(section+1)*100].ewm(span=10).mean(),
                     label=f"linear_acceleration-{target_column}")
        plt.legend()
        plt.savefig(os.path.join(
            data_path, f"physics-analysis-{section}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    speed_timestamps = []
    imu_timestamps = []

    for file_entry in list(os.scandir(os.path.join(args.data_path, "other_data"))):
        filename = file_entry.name
        if filename.endswith("raw_imu.csv"):
            imu_timestamps.append(re.findall(r"\d+_\d+", filename)[0])
        if filename.endswith("raw_speed.csv"):
            speed_timestamps.append(re.findall(r"\d+_\d+", filename)[0])
    construct_physical_property(
        args.data_path, imu_timestamps, speed_timestamps)
