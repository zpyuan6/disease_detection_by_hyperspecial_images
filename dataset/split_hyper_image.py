import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import os

SAVE_PATH = "F:\Hyperspecial\pear_processed"

def generate_mosiac_image(date, with_NDVI=False):
    # NDVI=（NIR-RED）/（NIR+RED）

    green_path = f"F:\\Hyperspecial\\pear\\{date}\\Aerial_UAV_Photos\\green.data.tif"
    NDVI_path = f"F:\\Hyperspecial\\pear\\{date}\\Aerial_UAV_Photos\\NDVI.data.tif"
    nir_path = f"F:\\Hyperspecial\\pear\\{date}\\Aerial_UAV_Photos\\nir.data.tif"
    red_path = f"F:\\Hyperspecial\\pear\\{date}\\Aerial_UAV_Photos\\red.data.tif"
    rededge_path = f"F:\\Hyperspecial\\pear\\{date}\\Aerial_UAV_Photos\\rededge.data.tif"

    green_tif = io.imread(green_path)
    NDVI_tif = io.imread(NDVI_path)
    nir_tif = io.imread(nir_path)
    red_tif = io.imread(red_path)
    rededge_tif = io.imread(rededge_path)

    # mosaic_tif[:,:,0] == green_tif
    # mosaic_tif[:,:,1] == red_tif
    # mosaic_tif[:,:,2] == rededge_tif
    # mosaic_tif[:,:,3] == nir_tif
    # mosaic_tif[:,:,4] == NDVI_path
    mosaic_numpy = np.array([green_tif,red_tif,rededge_tif,nir_tif])
    if with_NDVI:
        mosaic_numpy = np.array([green_tif,red_tif,rededge_tif,nir_tif, NDVI_tif])
    
    save_numpy_folder = os.path.join(SAVE_PATH, date)
    if not os.path.exists(save_numpy_folder):
        os.makedirs(save_numpy_folder)

    np.save(os.path.join(save_numpy_folder, f"mosaic_with_NDVI_{with_NDVI}.npy"), mosaic_numpy)


# def split_images():
    

if __name__ == "__main__":
    date_list = ["04_11_21","14_09_21","14_09_22","15_07_22","25_05_22","27_07_21"]

    for date in date_list:
        generate_mosiac_image(date,True)
