import numpy as np
import pandas as pd
import cv2
import os
import argparse
from tqdm import tqdm
import re


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    speed_timestamps = []
    imu_timestamps = []

    for file_entry in list(os.scandir(os.path.join(args.data_path, "other_data"))):
        filename = file_entry.name
        if filename.endswith("raw_imu.csv"):
            imu_timestamps.append(re.findall(r"\d+_\d+", filename))
        if filename.endswith("raw_speed.csv"):
            speed_timestamps.append(re.findall(r"\d+_\d+", filename))

    for timestamp in tqdm(imu_timestamps):
        pass
