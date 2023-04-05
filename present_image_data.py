from skimage import io
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    image_path = "F:\\Hyperspecial\\pear\\04_11_21\\Aerial_UAV_Photos\\nir.rgb.tif"
    tif = io.imread(image_path)
    print(tif.shape)

    # if len(tif.shape)>2:
    #     for i in range(tif.shape[2]):
    #         plt.subplot(1,tif.shape[2],i+1)
    #         plt.imshow(tif[:,:,i])

    #     plt.show()
    # else:
    # plt.imshow(tif, cmap="gray")
    plt.imshow(tif)
    plt.show()