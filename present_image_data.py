from skimage import io
import matplotlib.pyplot as plt

if __name__ == "__main__":


    tif = io.imread("F:\\Hyperspecial\\Aerial_UAV_Photos\\green.rgb.tif")
    print(tif.shape)
    print(type(tif))

    # print(tif[:,:,1])

    plt.subplot(1,2,1)
    plt.imshow(tif[:,:,1],cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(tif[:,:,2],cmap="gray")

    plt.show()