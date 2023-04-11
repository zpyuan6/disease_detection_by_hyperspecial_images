from skimage import io
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    date = "15_07_22"
    # 14_09_21

    green_path = f"F:\\Hyperspecial\\pear\\{date}\\Aerial_UAV_Photos\\green.data.tif"
    NDVI_path = f"F:\\Hyperspecial\\pear\\{date}\\Aerial_UAV_Photos\\NDVI.data.tif"
    nir_path = f"F:\\Hyperspecial\\pear\\{date}\\Aerial_UAV_Photos\\nir.data.tif"
    red_path = f"F:\\Hyperspecial\\pear\\{date}\\Aerial_UAV_Photos\\red.data.tif"
    rededge_path = f"F:\\Hyperspecial\\pear\\{date}\\Aerial_UAV_Photos\\rededge.data.tif"

    mosaic_path = f"F:\\Hyperspecial\\pear\\{date}\\Aerial_UAV_Photos\\Orthomosaic.data.tif"

    green_tif = io.imread(green_path)
    NDVI_tif = io.imread(NDVI_path)
    nir_tif = io.imread(nir_path)
    red_tif = io.imread(red_path)
    rededge_tif = io.imread(rededge_path)

    mosaic_tif = io.imread(mosaic_path)

    print("green_tif mean",np.mean(green_tif), np.max(green_tif),np.min(green_tif))
    print("NDVI_tif mean",np.mean(NDVI_tif), np.max(NDVI_tif),np.min(NDVI_tif))
    print("nir_tif mean",np.mean(nir_tif), np.max(nir_tif),np.min(nir_tif))
    print("red_tif mean",np.mean(red_tif), np.max(red_tif),np.min(red_tif))
    print("rededge_tif mean",np.mean(rededge_tif), np.max(rededge_tif),np.min(rededge_tif))

    for i in range(mosaic_tif.shape[-1]):
        print(mosaic_tif[:,:,i].shape, np.mean(mosaic_tif[:,:,i]), np.max(mosaic_tif[:,:,i]),np.min(mosaic_tif[:,:,i]))

    # mosaic_tif[:,:,0] == green_tif
    # mosaic_tif[:,:,1] == red_tif
    # mosaic_tif[:,:,2] == rededge_tif
    # mosaic_tif[:,:,3] == nir_tif

    print(np.argmax(green_tif),np.argmax(mosaic_tif[:,:,0]))
    print(np.argmax(red_tif),np.argmax(mosaic_tif[:,:,1]))
    print(np.argmax(rededge_tif),np.argmax(mosaic_tif[:,:,2]))
    print(np.argmax(nir_tif),np.argmax(mosaic_tif[:,:,3]))

    plt.imshow(mosaic_tif[:,:,4])
    plt.show()




