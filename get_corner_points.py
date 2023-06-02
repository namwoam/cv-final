import numpy as np
import pandas as pd
import cv2
import os
import argparse
from tqdm import tqdm

ZERBRACROSS = 0
STOPLINE = 1
ARROW = 2
JUNCTIONBOX = 3
OTHER = 4
CLASS_ID_TO_TYPE = ['ZERBRACROSS', 'STOPLINE', 'ARROW', 'JUNCTIONBOX', 'OTHER']


def get_corner_points(data_path, timestamp, confidence = 0.2, output_path = None):
    # ==================================================================================================
    # data_path (string): path to the folder containing the sequence. ex. public/seq1
    # timestamp (string): timestamp of the frame to be processed. ex. 1681710717_532211005
    # confidence (float): the prediction boxes with confidence lower than this value will be ignored
    # output_path (string): path to the folder to save the output. the output will not be saved if this is None
    #
    # return: a list of corner points (x, y)
    # ==================================================================================================

    img_path = os.path.join(data_path, 'dataset', timestamp, 'raw_image.jpg')
    pred_path = os.path.join(data_path, 'dataset', timestamp, 'detect_road_marker.csv')
    output_path = os.path.join(output_path, timestamp) if output_path else None

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    pred = pd.read_csv(pred_path, names = ['x1', 'y1', 'x2', 'y2', 'class', 'confidence'])
    pred = pred[pred['confidence'] >= confidence]

    pred[['x1', 'x2']] = np.clip(pred[['x1', 'x2']].astype(int), 0, w - 1)
    pred[['y1', 'y2']] = np.clip(pred[['y1', 'y2']].astype(int), 0, h - 1)

    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
    output_img = img.copy()

    corner_points = []

    for row in pred.itertuples():
        index, x1, y1, x2, y2, class_id, confidence = row
        index += 1
        class_type = CLASS_ID_TO_TYPE[class_id]

        crop_img = img[y1:y2, x1:x2]
        crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        mean = np.mean(crop_gray)
        std = np.std(crop_gray)
        crop_gray = cv2.threshold(crop_gray, mean + std / 2, 255, cv2.THRESH_BINARY)[1]

        if output_path:
            cv2.imwrite(os.path.join(output_path, f'threshold_{index}_{class_type}.jpg'), crop_gray)

        contours = cv2.findContours(crop_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

        if output_path:
            contour_img = np.zeros((crop_gray.shape[0], crop_gray.shape[1], 3), np.uint8)
            cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)
            cv2.imwrite(os.path.join(output_path, f'contours_{index}_{class_type}.jpg'), contour_img)
            approx_img = np.zeros((crop_gray.shape[0], crop_gray.shape[1], 3), np.uint8)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 5, True)
            if output_path:
                cv2.drawContours(approx_img, [approx], -1, (255, 255, 255), 1)

            for corner in approx:
                corner_points.append((corner[0][0] + x1, corner[0][1] + y1))
                if output_path:
                    cv2.circle(output_img, (corner[0][0] + x1, corner[0][1] + y1), 2, (0, 0, 255), -1)

        if output_path:
            cv2.imwrite(os.path.join(output_path, f'approx_{index}_{class_type}.jpg'), approx_img)

    if output_path:
        cv2.imwrite(os.path.join(output_path, 'output.jpg'), output_img)

    return corner_points


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, required = True, help = 'path to the folder containing the sequence')
    args = parser.parse_args()

    with open(os.path.join(args.data_path, 'all_timestamp.txt'), 'r') as f:
        timestamps = f.readlines()

    timestamps = [timestamp.strip() for timestamp in timestamps]

    for timestamp in tqdm(timestamps):
        get_corner_points(args.data_path, timestamp, output_path = 'outputs')




