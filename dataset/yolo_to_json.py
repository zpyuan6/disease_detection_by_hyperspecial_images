import os
import cv2
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import base64

if __name__ == "__main__":

    names = ["health", "anarsiaLineatella","grapholitaMolesta","dead"]

    for file in os.listdir('F:\\Hyperspecial\\pear_processed\\yolo_object_detection\\ann\\images'):
        if 'png' in file:
            dic = {}
            dic['version'] = '5.2.0.post4'
            dic['flags'] = {}
            dic['shapes'] = []
            img = cv2.imread('F:\\Hyperspecial\\pear_processed\\yolo_object_detection\\ann\\images\\{}'.format(file))
            imageHeight,imageWidth,_ = img.shape
            with open('F:\\Hyperspecial\\pear_processed\\yolo_object_detection\\ann\\labels\\{}.txt'.format(file.split('.')[0])) as f:
                datas = f.readlines()
                for data in datas:
                    shape = {}
                    shape['label'] = names[int(data[0])]
                    shape['line_color'] = None
                    shape['fill_color'] = None
                    data = data.strip().split(' ')
                    x = float(data[1]) * imageWidth
                    y = float(data[2]) * imageHeight
                    w = float(data[3]) * imageWidth
                    h = float(data[4]) * imageHeight
                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x1 + w
                    y2 = y1 + h
                    shape['points'] = [[x1,y1],[x2,y2]]
                    shape['shape_type'] = 'polygon'
                    shape['flags'] = {}
                    dic['shapes'].append(shape)
            dic['imagePath'] = file
            dic['imageData'] = base64.b64encode(open('F:\\Hyperspecial\\pear_processed\\yolo_object_detection\\ann\\images\\{}'.format(file),"rb").read()).decode('utf-8')
            dic['imageHeight'] = imageHeight
            dic['imageWidth'] = imageWidth
            fw = open('F:\\Hyperspecial\\pear_processed\\yolo_object_detection\\ann\\images\\{}.json'.format(file.split('.')[0]),'w')
            json.dump(dic,fw)
            fw.close()