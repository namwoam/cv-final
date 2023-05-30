import numpy as np
import pandas as pd
import cv2
import os
import argparse
from tqdm import tqdm

def get_corner_points(data_path, timestamp, confidence = 0.2):
    img_path = os.path.join(data_path, 'dataset', timestamp, 'raw_image.jpg')
    pred_path = os.path.join(data_path, 'dataset', timestamp, 'detect_road_marker.csv')

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w, _ = img.shape

    pred = pd.read_csv(pred_path, names=['x1', 'y1', 'x2', 'y2', 'class', 'confidence'])
    pred = pred[pred['confidence'] > confidence]

    boxes = pred[['x1', 'y1', 'x2', 'y2']].values.astype(np.int32)
    # Some values may be out of bound
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h)

    corner_points = []

    if not os.path.exists(os.path.join('output', timestamp)):
        os.makedirs(os.path.join('output', timestamp))

    for id, box in enumerate(boxes):
        crop_img = img[box[1]:box[3], box[0]:box[2]]
        crop_gray = gray[box[1]:box[3], box[0]:box[2]].copy() # There could be overlapping, so copy is needed in order not to change the original gray image

        mean = np.mean(crop_gray)
        std = np.std(crop_gray)
        crop_gray = cv2.threshold(crop_gray, mean + std / 2, 255, cv2.THRESH_BINARY)[1]

        cv2.imwrite(os.path.join('output', timestamp, f'AfterThreshold_{id}.jpg'), crop_gray)

        contours, _ = cv2.findContours(crop_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # save contours
        black_img = np.zeros((crop_img.shape[0], crop_img.shape[1], 3), np.uint8)
        cv2.drawContours(black_img, contours, -1, (255, 255, 255), 1)
        cv2.imwrite(os.path.join('output', timestamp, f'contours_{id}.jpg'), black_img)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 5, True)
            for corner in approx:
                cv2.circle(crop_img, tuple(corner[0]), 3, (0, 0, 255), -1)
                corner_points.append(corner[0] + box[:2])

    cv2.imwrite(os.path.join('output', timestamp, 'points.jpg'), img)
    cv2.imwrite(os.path.join('output', timestamp + '_points.jpg'), img)
    return corner_points

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    with open(os.path.join(args.data_path, 'all_timestamp.txt'), 'r') as f:
        timestamps = f.readlines()

    timestamps = [timestamp.strip() for timestamp in timestamps]

    if not os.path.exists('output'):
        os.makedirs('output')

    for timestamp in tqdm(timestamps):
        get_corner_points(args.data_path, timestamp)


