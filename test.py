import torch
import cv2
import numpy as np
from  libtiff import TIFF
from skimage.io  import imread, imsave

if __name__ == "__main__":
    # loss_function = torch.nn.CrossEntropyLoss()

    # input_tensor = torch.rand((2,4))
    # label_tensor = torch.rand((2,4))

    # print(input_tensor.shape, label_tensor.shape)

    # loss = loss_function(input_tensor,label_tensor)
    # print(loss)
    i = np.load("F:\\Hyperspecial\\pear_processed\\segmentation_data\\label\\27_07_21_4_4.npy")
    print(i.shape)
    print(i)

    # path = "F:\\Hyperspecial\\pear_processed\\yolo_object_detection\\images\\train\\14_09_21_4_2.tif"
    # img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # print(img.shape)
    cv2.imshow('image1', i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # tif = TIFF.open(path)
    # image = imread(path)
    # print(image)

    # cv2.imshow('image2',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # imsave(path,i)

    

    
