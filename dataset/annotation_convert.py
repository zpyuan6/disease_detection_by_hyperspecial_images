import os
import cv2
import sys
import PIL
import copy
import json
import yaml
import base64
import numpy as np
import skimage.io as io
from glob import glob
import matplotlib.pyplot as plt
# try:
#     from labelme import __version__ as labelme_version
# except:
#     labelme_version = '4.2.9'

sys.path.append('..')
currentCV_version = cv2.__version__

def read_name_file(name_path):
    names = []
    with open(name_path, "r") as name_file:
        for name in name_file:
            names.append(name.replace("\n", "").strip())
    return names

def convert_coor(size, xy):
    dw = size[0]
    dh = size[1]
    x, y = xy
    return x / dw, y / dh
def convert(file, txt_name=None):

    if txt_name is None:
        txt_name = file.rstrip(".json") + ".txt"
    """ Open input text files """
    txt_path = file
    names = ['anarsiaLineatella','dead','grapholitaMolesta','health']

    print("Input:" + txt_path)
    txt_file = open(txt_path, "r")

    """ Open output text files """
    txt_outpath = txt_name
    print("Output:" + txt_outpath)
    txt_outfile = open(txt_outpath, "w")

    """ Convert the data to YOLO format """
    js = json.loads(txt_file.read())

    for item in js["shapes"]:
        label = item["label"]
        for i, name in enumerate(names):
            if label == name:
                cls = str(i)

        height = js["imageHeight"]
        width = js["imageWidth"]
        point = item["points"]

        for idx, pt in enumerate(point):
            if idx == 0:
                txt_outfile.write(cls)

            x, y = pt
            bb = convert_coor((width, height), [x, y])
            txt_outfile.write(" " + " ".join([str(a) for a in bb]))
        txt_outfile.write("\n")


def rm(filepath):
    p = open(filepath, 'r+')
    lines = p.readlines()
    d = ""
    for line in lines:
        c = line.replace('"group_id": "null",', '"group_id": null,')
        d += c
    p.seek(0)
    p.truncate()
    p.write(d)
    p.close()


def imgEncode(img_or_path):
    if isinstance(img_or_path, np.ndarray):
        """
        copy from labelme image.py    
        """
        img_pil = PIL.Image.fromarray(img_or_path)
        f = io.BytesIO()
        img_pil.save(f, format='PNG')
        img_bin = f.getvalue()
        if hasattr(base64, 'encodebytes'):
            img_b64 = base64.encodebytes(img_bin)
        else:
            img_b64 = base64.encodestring(img_bin)
        return img_b64
    else:
        if isinstance(img_or_path, str):
            i = open(img_or_path, 'rb')
        elif isinstance(img_or_path, io.BufferedReader):
            i = img_or_path
        else:
            raise TypeError('Input type error!')
        base64_data = base64.b64encode(i.read())
        return base64_data.decode()


def rs(st: str):
    s = st.replace('\n', '').strip()
    return s


def readYmal(filepath, labeledImg=None):
    if os.path.exists(filepath):
        if filepath.endswith('.yaml'):
            f = open(filepath)
            y = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
            # print(y)
            tmp = y['label_names']
            # print(tmp["tag1"])
            objs = zip(tmp.keys(), tmp.values())
            return sorted(objs)
        elif filepath.endswith('.txt'):
            f = open(filepath, 'r', encoding='utf-8')
            classList = f.readlines()
            f.close()
            l3 = [rs(i) for i in classList]
            l = list(range(1, len(classList)+1))
            objs = zip(l3, l)
            return sorted(objs)
    elif labeledImg is not None and filepath == "":
        """
        should make sure your label is correct!!!
        """
        labeledImg = np.array(labeledImg, dtype=np.uint8)

        labeledImg[labeledImg > 0] = 255
        labeledImg[labeledImg != 255] = 0
        # print(labeledImg)
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(
            labeledImg)

        labels = np.max(labels) + 1
        labels = [x for x in range(1, labels)]

        classes = []
        for i in range(0, len(labels)):
            classes.append("class{}".format(i))

        return zip(classes, labels)
    else:
        raise FileExistsError('file not found')


def get_approx(img, contour, length_p=0.005):
    """获取逼近多边形
    :param img: 处理图片
    :param contour: 连通域
    :param length_p: 逼近长度百分比
    """
    img_adp = img.copy()
    # 逼近长度计算
    epsilon = length_p * cv2.arcLength(contour, True)
    # 获取逼近多边形
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx


def getBinary(img_or_path, minConnectedArea=1):
    if isinstance(img_or_path, str):
        i = cv2.imread(img_or_path)
    elif isinstance(img_or_path, np.ndarray):
        i = img_or_path
    else:
        raise TypeError('Input type error')

    if len(i.shape) == 3:
        img_gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

    else:
        img_gray = i

    ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    _, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=4)
    # labels：图像上每一像素的标记，用数字1、2、3…表示（不同的数字表示不同的连通域）
    # stats：每一个标记的统计信息，是一个5列的矩阵，每一行对应每个连通区域的外接矩形的x、y、width、height和面积，示例如下： 0 0 720 720 291805
    # centroids：连通域的中心点
    # print(stats.shape)  (19,5)
    # 删除区域小的图片
    for index in range(1, stats.shape[0]):
        if stats[index][4] < minConnectedArea or stats[index][4] < 0.0001 * (
                stats[index][2] * stats[index][3]):
            labels[labels == index] = 0

    labels[labels != 0] = 1

    img_bin = np.array(img_bin * labels).astype(np.uint8)
    return i, img_bin


def getMultiRegion(img, img_bin):
    """
    for multiple objs in same class
    """
    if float(currentCV_version[0:3]) < 3.5:
        img_bin, contours, hierarchy = cv2.findContours(
            img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    if len(contours) >= 1:
        for i in range(0, len(contours)):
            if i:
                # print(len(contours[i]))
                region = get_approx(img, contours[i], 0.0001)
                # print(region)
                if region.shape[0] > 3:
                    regions.append(region)

        return regions
    else:
        return []


def process(oriImg):
    img, img_bin = getBinary(oriImg)
    return getMultiRegion(img, img_bin)


def getMultiShapes(oriImgPath, labelPath, savePath='', labelYamlPath='', flag=False):
    """
    oriImgPath : for change img to base64  \n
    labelPath : after fcn/unet or other machine learning objects outlining , the generated label img
                or labelme labeled imgs(after json files converted to mask files)  \n
    savePath : json file save path  \n
    labelYamlPath : after json files converted to mask files. if doesn't have this file,should have a labeled img.
                    but the classes should change by yourself(labelme 4.2.9 has a bug,when change the label there will be an error.
                    )   \n
    """
    if isinstance(labelPath, str):
        if os.path.exists(labelPath):
            # label_img = io.imread(labelPath)
            label_img = np.load(labelPath, allow_pickle=True)
        else:
            raise FileNotFoundError('mask/labeled image not found')
    else:
        label_img = labelPath

    # plt.imshow(label_img)
    # plt.show()

    # print(np.max(label_img))

    if np.max(label_img) > 127:
        # print('too many classes! \n maybe binary?')
        label_img[label_img > 127] = 255
        label_img[label_img != 255] = 0
        label_img = label_img / 255

    labelShape = label_img.shape

    # labels = readYmal(labelYamlPath, label_img)
    labels = [0,2,3,4]
    labels_names = ['anarsiaLineatella','dead','grapholitaMolesta','health']
    # print(list(labels))
    shapes = []
    obj = dict()
    obj['version'] = "5.2.0.post4"
    obj['flags'] = {}
    for index, la in enumerate(labels):
        
        img = copy.deepcopy(label_img)  # img = label_img.copy()
        img = img.astype(np.uint8)
        img[img == la] = 255
        img[img != 255] = 0

        region = process(img.astype(np.uint8))

        print(len(region))

        if isinstance(region, np.ndarray):

            points = []
            for i in range(0, region.shape[0]):
                print(len(region[i][0]))
                points.append(region[i][0].tolist())
            shape = dict()
            shape['label'] = labels_names[index]
            shape['points'] = points
            shape['group_id'] = 'null'
            shape['shape_type'] = 'polygon'
            shape['flags'] = {}
            shapes.append(shape)

        elif isinstance(region, list):
            # print(index, len(region))
            for subregion in region:
                points = []
                for i in range(0, subregion.shape[0]):
                    points.append(subregion[i][0].tolist())
                shape = dict()
                shape['label'] = labels_names[index]
                shape['points'] = points
                shape['group_id'] = 'null'
                shape['shape_type'] = 'polygon'
                shape['flags'] = {}
                shapes.append(shape)

    # print(len(shapes))
    obj['shapes'] = shapes
    # print(shapes)
    (_, imgname) = os.path.split(oriImgPath)
    obj['imagePath'] = imgname
    # print(obj['imagePath'])
    obj['imageData'] = str(imgEncode(oriImgPath))

    obj['imageHeight'] = labelShape[0]
    obj['imageWidth'] = labelShape[1]

    j = json.dumps(obj, sort_keys=True, indent=4)
    # print(j)

    if not flag:
        saveJsonPath = savePath + os.sep + obj['imagePath'][:-4] + '.json'
        # print(saveJsonPath)
        with open(saveJsonPath, 'w') as f:
            f.write(j)

        rm(saveJsonPath)
    else:
        return j


def convert_from_mask_to_json():
    IMAGE_DIR = "F:\Hyperspecial\pear_processed\segmentation_data\images"
    ANNOTATION_DIR = "F:\Hyperspecial\pear_processed\segmentation_data\label"
    
    #适当修改类别
    CATEGORIES = [
        {
            'id': 0,
            'name': 'anarsiaLineatella',
        },
        {
            'id': 1,
            'name': 'background',
        },
        {
            'id': 2,
            'name': 'dead',
        },
        {
            'id': 3,
            'name': 'grapholitaMolesta',
        },
        {
            'id': 4,
            'name': 'health',
        },
    ]

    yaml_file = '' 
    save_json = 'F:\\Hyperspecial\\pear_processed\\segmentation_data\\seg'

    mask_images_list = glob(os.path.join(ANNOTATION_DIR, "*.npy"))
    init_images_list = glob(os.path.join(IMAGE_DIR, "*.png"))

    if not os.path.exists(save_json):
        os.mkdir(save_json)

    for mask_image, init_image in zip(mask_images_list, init_images_list):
        print(mask_image)
        getMultiShapes(init_image, mask_image, save_json, yaml_file)
        # break

def convert_from_json_to_txt():
    items = []
    for root, folders, files in os.walk("F:\Hyperspecial\pear_processed\segmentation_data\images"):
        for file in files:
            if file.split(".")[-1] == "json":
                # items.append(os.path.join(root,file))
                convert(os.path.join(root,file))

    

if __name__ == "__main__":
    # convert_from_mask_to_json()

    convert_from_json_to_txt()
            