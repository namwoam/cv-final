import numpy as np
import pandas as pd
import cv2
import os
import argparse
from tqdm import tqdm

ZEBRACROSS = 0
STOPLINE = 1
ARROW = 2
JUNCTIONBOX = 3
OTHER = 4
CLASS_ID_TO_TYPE = ['ZEBRACROSS', 'STOPLINE', 'ARROW', 'JUNCTIONBOX', 'OTHER']
POINT_COLOR = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
SHORTEST_DISTANCE_TO_BORDER = 2
SHORTEST_DISTNACE_TO_POINT = 4
MIN_AREA_RATIO = 1/200
EPSILON = 4

class CornerPointDetector:
    def __init__(self, data_path, camera_info_path, output_path = None, confidence = 0.2):
        # ==================================================================================================
        # data_path (string): path to the folder containing the sequence. ex. public/seq1
        # camera_info_path (string): path to the camera info directory. ex. public/camera_info/lucid_cameras_x00
        # output_path (string): path to the folder to save the output. the output will not be saved if this is None
        # confidence (float): the prediction boxes with confidence lower than this value will be ignored
        # ==================================================================================================

        self.data_path = data_path
        self.confidence = confidence
        self.output_path = output_path

        self.camera_mask = {}
        self.camera_mask['f'] = cv2.imread(os.path.join(camera_info_path, 'gige_100_f_hdr_mask.png'), cv2.IMREAD_GRAYSCALE)
        self.camera_mask['b'] = cv2.imread(os.path.join(camera_info_path, 'gige_100_b_hdr_mask.png'), cv2.IMREAD_GRAYSCALE)
        self.camera_mask['fl'] = cv2.imread(os.path.join(camera_info_path, 'gige_100_fl_hdr_mask.png'), cv2.IMREAD_GRAYSCALE)
        self.camera_mask['fr'] = cv2.imread(os.path.join(camera_info_path, 'gige_100_fr_hdr_mask.png'), cv2.IMREAD_GRAYSCALE)

        self.camera_mask['f'] = cv2.threshold(self.camera_mask['f'], 127, 255, cv2.THRESH_BINARY)[1]
        self.camera_mask['b'] = cv2.threshold(self.camera_mask['b'], 127, 255, cv2.THRESH_BINARY)[1]
        self.camera_mask['fl'] = cv2.threshold(self.camera_mask['fl'], 127, 255, cv2.THRESH_BINARY)[1]
        self.camera_mask['fr'] = cv2.threshold(self.camera_mask['fr'], 127, 255, cv2.THRESH_BINARY)[1]

        kernel = np.ones((9, 9), np.uint8)
        self.camera_mask['f'] = cv2.dilate(self.camera_mask['f'], kernel, iterations = 1)
        self.camera_mask['b'] = cv2.dilate(self.camera_mask['b'], kernel, iterations = 1)
        self.camera_mask['fl'] = cv2.dilate(self.camera_mask['fl'], kernel, iterations = 1)
        self.camera_mask['fr'] = cv2.dilate(self.camera_mask['fr'], kernel, iterations = 1)

    def get_corner_points(self, timestamp):
        # ==================================================================================================
        # timestamp (string): timestamp of the frame to be processed. ex. 1681710717_532211005
        #
        # return: a list of corner points (x, y)
        # ==================================================================================================

        img_path = os.path.join(self.data_path, 'dataset', timestamp, 'raw_image.jpg')
        pred_path = os.path.join(self.data_path, 'dataset', timestamp, 'detect_road_marker.csv')
        output_path = os.path.join(self.output_path, timestamp) if self.output_path else None

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        pred = pd.read_csv(pred_path, names = ['x1', 'y1', 'x2', 'y2', 'class', 'confidence'])
        pred = pred[pred['confidence'] >= self.confidence]

        pred[['x1', 'x2']] = np.clip(pred[['x1', 'x2']].astype(int), 0, w - 1)
        pred[['y1', 'y2']] = np.clip(pred[['y1', 'y2']].astype(int), 0, h - 1)

        with open(os.path.join(self.data_path, 'dataset', timestamp, 'camera.csv'), 'r') as f:
            camera_info = f.read()
        camera_info = camera_info.strip().split('_')[-2]
        camera_mask = self.camera_mask[camera_info]

        if output_path and not os.path.exists(output_path):
            os.makedirs(output_path)

        if output_path:
            output_img = img.copy()
            box_img = img.copy()

        point_mask = np.zeros((h, w), dtype = np.uint8)
        corner_points = []

        for row in pred.itertuples():
            index, x1, y1, x2, y2, class_id, confidence = row
            index += 1
            class_type = CLASS_ID_TO_TYPE[class_id].lower()

            # Save box image
            if output_path:
                cv2.rectangle(box_img, (x1, y1), (x2, y2), POINT_COLOR[class_id], 2)
                cv2.putText(box_img, f'{index}_{class_type}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, POINT_COLOR[class_id], 2)

            crop_img = img[y1:y2, x1:x2]
            crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

            ###################################################
            #Modify grayscale
            ####################################################
            # B,G,R = crop_img[:,:,0],crop_img[:,:,1],crop_img[:,:,2]
            # a = np.array([3,-1,-1]).astype(np.float32)
            # a /= np.sum(a)
            # crop_gray = a[0]*B +a[1]*G+a[2]*R
            # crop_gray = crop_gray.astype(np.uint8)
            #############################################

            mean = np.mean(crop_gray)
            std = np.std(crop_gray)
            crop_gray = cv2.threshold(crop_gray, mean + std * 2 / 3 , 255, cv2.THRESH_BINARY)[1]

            # Save threshold image
            if output_path:
                cv2.imwrite(os.path.join(output_path, f'{index}_{class_type}_threshold.jpg'), crop_gray)

            # Erode and Dilate
            h_bar,w_bar = crop_gray.shape
            white = np.sum(crop_gray) //255
            total = h_bar*w_bar
            if class_id == 0 and white < total/2.5 and total > 250*130:
                kernal = np.ones((6,6),np.uint8)
                crop_gray = cv2.dilate(crop_gray,kernal,iterations=1)
                crop_gray = cv2.erode(crop_gray,kernal,iterations=1)
                kernal2 = np.ones((2,2),np.uint8)
                crop_gray = cv2.erode(crop_gray,kernal2,iterations=1)
                crop_gray = cv2.dilate(crop_gray,kernal2,iterations=1)

            # Save erode and dilate image
            if output_path:
                cv2.imwrite(os.path.join(output_path, f'{index}_{class_type}_erode_dilate.jpg'), crop_gray)

            contours = cv2.findContours(crop_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

            # Save contour image
            if output_path:
                contour_img = np.zeros((crop_gray.shape[0], crop_gray.shape[1], 3), np.uint8)
                cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)
                cv2.imwrite(os.path.join(output_path, f'{index}_{class_type}_contours.jpg'), contour_img)
                approx_img = np.zeros((crop_gray.shape[0], crop_gray.shape[1], 3), np.uint8)

            for contour in contours:
                if cv2.contourArea(contour) < MIN_AREA_RATIO * crop_gray.shape[0] * crop_gray.shape[1]:
                    continue

                approx = cv2.approxPolyDP(contour, EPSILON, True)
                if output_path:
                    cv2.drawContours(approx_img, [approx], -1, (255, 255, 255), 1)

                for corner in approx:
                    actual_x, actual_y = corner[0][0] + x1, corner[0][1] + y1
                    if corner[0][0] < SHORTEST_DISTANCE_TO_BORDER or corner[0][0] > crop_img.shape[1] - SHORTEST_DISTANCE_TO_BORDER or corner[0][1] < SHORTEST_DISTANCE_TO_BORDER or corner[0][1] > crop_img.shape[0] - SHORTEST_DISTANCE_TO_BORDER or \
                       camera_mask[actual_y, actual_x] > 0 or \
                       point_mask[actual_y, actual_x] > 0:
                        continue
                    corner_points.append((corner[0][0] + x1, corner[0][1] + y1))
                    point_mask[actual_y - SHORTEST_DISTNACE_TO_POINT:actual_y + SHORTEST_DISTNACE_TO_POINT, actual_x - SHORTEST_DISTNACE_TO_POINT:actual_x + SHORTEST_DISTNACE_TO_POINT] = 255
                    if output_path:
                        cv2.circle(output_img, (corner[0][0] + x1, corner[0][1] + y1), 2, POINT_COLOR[class_id], -1)

            # Save approx image
            if output_path:
                cv2.imwrite(os.path.join(output_path, f'{index}_{class_type}_approx.jpg'), approx_img)


        if output_path:
            # Save image with corner points
            cv2.imwrite(os.path.join(output_path, 'output.jpg'), output_img)
            # Save box image
            cv2.imwrite(os.path.join(output_path, 'box.jpg'), box_img)

        return corner_points


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, required = True, help = 'path to the folder containing the sequence')
    parser.add_argument('--camera_info_path', type = str, required = True, help = 'path to the folder containing the camera info')
    args = parser.parse_args()

    detector = CornerPointDetector(args.data_path, args.camera_info_path, 'outputs')

    with open(os.path.join(args.data_path, 'all_timestamp.txt'), 'r') as f:
        timestamps = f.readlines()

    timestamps = [timestamp.strip() for timestamp in timestamps]

    for timestamp in tqdm(timestamps):
        detector.get_corner_points(timestamp)




