from skimage import io
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    image_path = "F:\\Hyperspecial\\pear\\15_07_22\\Aerial_UAV_Photos\\Orthomosaic.rgb.tif"
    tif = io.imread(image_path)
    print(tif.shape)

    print(tif)
    # if len(tif.shape)>2:
    #     for i in range(tif.shape[2]):
    #         plt.subplot(1,tif.shape[2],i+1)
    #         plt.imshow(tif[:,:,i])

    #     plt.show()
    # else:
    # plt.imshow(tif, cmap="gray")
    # [x1,x2],[y1,y2]
    # for i in range(29):
    for i in range(30):
        plt.plot([0,2828],[-100+i*91,248+i*91], color = "red")
    # plt.plot([0,2828],[-10,338], color = "red")
    # plt.plot([0,2828],[80,428], color = "red")

    for i in range(38):
        plt.plot([-15+i*74,160+i*74],[0,2806],color="blue")

    plt.imshow(tif)
    plt.show()