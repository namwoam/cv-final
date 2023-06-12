import numpy as np
import pandas as pd
import cv2
import os
import argparse
from tqdm import tqdm
import yaml
import csv



def get_camerainfo_matrix(matirx_name='camera_matrix'):
    camera_path = os.path.join(args.data_path, 'dataset',timestamp, 'camera.csv')
    camera_path = pd.read_csv(camera_path, header=None)[0][0]
    camera = camera_path.split('_')[-2]
    camera_path += '_camera_info.yaml'
    
    with open("./public/camera_info"+ camera_path, 'r') as f:
        data = yaml.load(f.read(), yaml.FullLoader)
        camerainfo_matrix = np.array(data[matirx_name]['data']).reshape(data[matirx_name]['rows'], data[matirx_name]['cols'])
        camerainfo_matrix = np.hstack((camerainfo_matrix, [[0], [0], [0]]))
    return camera, camerainfo_matrix

def pinHoleModel_trans(camera_matrix,Trans, P1):
    # 相機內參和相機外參 矩陣相乘
    temp = np.dot(camera_matrix, Trans)
    Pp = np.linalg.pinv(temp)
    
    # （u, v, 1) 
    p1 = np.array(P1, np.float32)
    X = np.dot(Pp, p1)
    # 與Zc相除 得到世界座標系的某一個點
    X1 = np.array(X[:3], np.float32)/X[3]
    return X1
   
def get_corner_points(data_path, timestamp, confidence=0.2):
    img_path = os.path.join(data_path, 'dataset', timestamp, 'raw_image.jpg')
    pred_path = os.path.join(data_path, 'dataset',
                             timestamp, 'detect_road_marker.csv')

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w, _ = img.shape

    pred = pd.read_csv(pred_path, names=[
                       'x1', 'y1', 'x2', 'y2', 'class', 'confidence'])
    pred = pred[pred['confidence'] > confidence]

    boxes = pred[['x1', 'y1', 'x2', 'y2']].values.astype(np.int32)
    # Some values may be out of bound
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h)

    corner_points = []

    if not os.path.exists(os.path.join(data_path, 'output', timestamp)):
        os.makedirs(os.path.join(data_path, 'output', timestamp))

    for id, box in enumerate(boxes):
        crop_img = img[box[1]:box[3], box[0]:box[2]]
        crop_gray = gray[box[1]:box[3], box[0]:box[2]].copy() # There could be overlapping, so copy is needed in order not to change the original gray image

        mean = np.mean(crop_gray)
        std = np.std(crop_gray)
        crop_gray = cv2.threshold(
            crop_gray, mean + std /2, 255, cv2.THRESH_BINARY)[1]
        kernal = np.ones((6,6),np.uint8)

        crop_gray = cv2.dilate(crop_gray,kernal,iterations=1)
        crop_gray = cv2.erode(crop_gray,kernal,iterations=1)
        kernal2 = np.ones((2,2),np.uint8)
        crop_gray = cv2.erode(crop_gray,kernal2,iterations=1)
        crop_gray = cv2.dilate(crop_gray,kernal2,iterations=1)
        # crop_gray = cv2.GaussianBlur(crop_gray,(5,5),5)

        #cv2.imwrite(os.path.join(data_path, 'output', timestamp,f'AfterThreshold_{id}_improved.jpg'), crop_gray)
        # h,w = crop_gray.shape
        contours, _ = cv2.findContours(
            crop_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # threshold = h/20 * w/20
        # for i in range(len(contours)):
        #     area = cv2.contourArea(contours[i])
        #     if area < threshold:
        #         cv2.drawContours(img,[contours[i]],-1,0,thickness=-1)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 5, True)
            for corner in approx:
                corner_points.append(corner[0] + box[:2])
    return corner_points

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    with open(os.path.join(args.data_path, 'localization_timestamp.txt'), 'r') as f:
        timestamps = f.readlines()

    timestamps = [timestamp.strip() for timestamp in timestamps]

    if not os.path.exists(os.path.join(args.data_path, 'output')):
        os.makedirs(os.path.join(args.data_path, 'output'))
    
    

    camera_transform_dict = {'fl' : np.matrix( [[-0.19917452,  0.92702531,  0.31773045, -0.564697  ],
                                                [-0.96289232, -0.24538819,  0.11235196,  0.0402756 ],
                                                [ 0.1821203,  -0.28356257,  0.94150066, -0.028059  ],
                                                [ 0.,          0.,          0.,          1.        ]]),
                             'f' :  np.matrix( [[ 0.08330063,  0.03058046,  0.99605514,  0.        ],
                                                [-0.99632541,  0.02253289,  0.08263144,  0.        ],
                                                [-0.01991709, -0.99927829,  0.03234509,  0.        ],
                                                [ 0.,          0.,          0.,          1.        ]]),
                             'fr' : np.matrix( [[-0.125135,    0.00752639, -0.99210955, -1.05741083],
                                                [-0.86734111, -0.48635875,  0.10570865,  0.38842579],
                                                [-0.48172621,  0.87372511,  0.06738862, -1.42484287],
                                                [ 0.,          0.,          0.,          1.        ]]),
                             'b' : np.matrix(  [[-0.125135,    0.00752639, -0.99210955, -1.05741083],
                                                [-0.86734111, -0.48635875,  0.10570865,  0.38842579],
                                                [-0.48172621,  0.87372511,  0.06738862, -1.42484287],
                                                [ 0.,          0.,          0.,          1.        ]]),                               
                            }
    record_Loss_Time = []
    np.seterr(divide = 'ignore') 
    for timestamp in tqdm(timestamps):
        P1 = get_corner_points(args.data_path, timestamp)
        if P1:
            P1 = np.array(P1)
            P1 = np.concatenate((P1, np.ones((len(P1), 1))), axis=1)
            P1 = np.transpose(P1)
            camera, camera_matrix = get_camerainfo_matrix()
            A = pinHoleModel_trans(camera_matrix,camera_transform_dict[camera],P1)
            A = np.transpose(A)

            my_df = pd.DataFrame(A)
            
            filrname = os.path.join(args.data_path, 'dataset', timestamp,'output_XYZ_final.csv')
            my_df.to_csv(filrname, index=False, header=False)
        else:
            record_Loss_Time.append(timestamp)
    my_df = pd.DataFrame(record_Loss_Time)  
    my_df.to_csv('./public/solution//seq1/lossTime.txt', index=False, header=False, sep='\t')
            
        
        
        
    
        
                
        

        




