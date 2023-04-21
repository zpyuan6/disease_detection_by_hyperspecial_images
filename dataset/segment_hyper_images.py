from skimage import io
from skimage.segmentation import felzenszwalb,slic,quickshift,watershed, mark_boundaries, slic_superpixels
from skimage.filters import sobel
from skimage.color import rgb2gray
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms
import torch
import time
from sklearn.cluster import KMeans
import joblib

import tqdm

def segmentation_image(tif_img) -> np.ndarray:
    # segments_fz = felzenszwalb(tif_img, scale=200, sigma=0.8, min_size=50)
    # print("Finished felzenszwalb")
    # plt.figure(figsize=(10,10))
    # plt.imshow(mark_boundaries(tif_img,segments_fz))
    # plt.show()

    segments_slic = slic(tif_img, n_segments=2500, compactness=20, sigma=1, start_label=1)
    print(type(segments_slic),segments_slic)
    print("Finished slic")
    # plt.imshow(mark_boundaries(tif_img,segments_slic))
    # plt.show()

    # segments_quick = quickshift(tif_img, kernel_size=5, max_dist=10, ratio=1.0)
    # print("Finished quickshift")
    # plt.imshow(mark_boundaries(tif_img,segments_quick))
    # plt.show()

    # gradient = sobel(rgb2gray(tif_img))
    # segments_watershed = watershed(gradient, markers=250, compactness=0.001)
    # print("Finished watershed")
    # plt.imshow(mark_boundaries(tif_img,segments_watershed))
    # plt.show()

    return segments_slic

def segmentation_and_save(tif_img, save_path):

    segment_map = segmentation_image(tif_img)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # segments = []
    # superpixels = []
    i = 0
    with tqdm.tqdm(total=len(np.unique(segment_map))) as tbar:
        for s in np.unique(segment_map):
            mask = (segment_map == s).astype(float)
            mask_expanded = np.expand_dims(mask, -1) if len(mask.shape)!= len(tif_img.shape) else mask
            patch = (mask_expanded * tif_img + (1 - mask_expanded) * float(0))
            patch_image = Image.fromarray((patch).astype(np.uint8))
            # print("patch",patch.shape, np.mean(patch), np.min(patch), np.max(patch))
            # segments.append(patch)

            ones = np.where(mask == 1)
            h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
            super_pixel = patch[h1:h2, w1:w2]
            # print("patch",super_pixel.shape, np.mean(super_pixel), np.min(super_pixel), np.max(super_pixel))
            superpixel_image = Image.fromarray((patch[h1:h2, w1:w2]).astype(np.uint8))
            # superpixels.append(super_pixel)
            
            # if 50 < np.sum(patch[h1:h2, w1:w2]) < 200:
            # if 50 < np.mean(patch[h1:h2, w1:w2]) < 240 and np.sum(mask)>300:
                # plt.subplot(1,2,1)
                # plt.imshow(patch)
                # plt.subplot(1,2,2)
                # plt.imshow(superpixel_image)

                # plt.show()

            patch_image.save(os.path.join(save_path,f"{i}_patch.jpg"))
            superpixel_image.save(os.path.join(save_path,f"{i}_superpixel.jpg"))

            i+=1

            del patch_image
            del superpixel_image
            del super_pixel
            del patch
            del mask

            tbar.update(1)

    print(i)

    

    # for i,segmentation in enumerate(masks):
    #     s

    # for i,mask in enumerate(masks):
    #     superpixel = Image.fromarray()


def cluster_segments(segment_folder:str):
    print("Start load images")

    num_clusters = 3

    # cluster_torch = None
    # if os.path.exists(torch_path):
    #     cluster_torch = torch.load(torch_path)
    # else:
    image_size = (64,64)
    cluster_torch = None
    trans = transforms.ToTensor()
    file_id_list = []
    for root, folders, files in os.walk(segment_folder):
        for file in files:
            if str(file).find("superpixel") != -1:
                file_id_list.append(file.split('_')[0])
                superpixel = Image.open(os.path.join(root,file))
                superpixel = superpixel.resize(image_size).convert('L')
                superpixel_torch = torch.flatten(trans(superpixel)).unsqueeze(0)
                if cluster_torch == None:
                    cluster_torch = superpixel_torch 
                else:
                    cluster_torch = torch.cat([cluster_torch,superpixel_torch], dim=0)

                print(file,cluster_torch.shape)

    num_segs = cluster_torch.shape[0]
    torch.save(cluster_torch,os.path.join("\\".join(segment_folder.split("\\")[0:-1]),f"torch_{num_segs}_superpixels.pt"))

    now = time.time()
    print("Start cluster")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(cluster_torch)
    print(f"End cluster, spend {time.time()-now} s")
    joblib.dump(kmeans, os.path.join("\\".join(segment_folder.split("\\")[0:-1]),f"cluster_{num_clusters}_classes.joblib"))

    result = kmeans.predict(cluster_torch)

    del cluster_torch

    print(file_id_list)
    print(result)

    return file_id_list, result


def present_cluster(save_path, file_id_list, cluster_result):
    COLOR_MAP = [[0,0,0],[128,0,0],[0,128,0],[0,0,128],[128,128,0],[128,0,128]]
    GRAY_MAP = [0.0,51.0,102.0,153.0,204.0,255.0]

    print(len(file_id_list), len(cluster_result))

    max_index = np.max(cluster_result)+1
    print(max_index)

    mosaic_images = None
    with tqdm.tqdm(total=len(file_id_list)) as tbar:
        for i, file_id in enumerate(file_id_list):
            patch_mask = transforms.ToTensor()(Image.open(os.path.join(save_path, f"{file_id}_patch.jpg")).convert('L')).squeeze()

            if mosaic_images == None:
                mosaic_images = (patch_mask != 0) * GRAY_MAP[cluster_result[i]+1]
            else:
                mosaic_images = mosaic_images + ((patch_mask != 0) * GRAY_MAP[cluster_result[i]+1])
            
            tbar.update(1)

    print(mosaic_images, mosaic_images.shape)

    mask_images = transforms.ToPILImage()(mosaic_images)

    plt.imshow(mask_images)
    plt.show()

    

if __name__ == "__main__":

    data = "15_07_22"
    # image_path = "F:\\Hyperspecial\\pear\\14_09_21\\Aerial_UAV_Photos\\Orthomosaic.rgb.tif"
    image_path = f"F:\\Hyperspecial\\pear\\{data}\\Aerial_UAV_Photos\\Orthomosaic.rgb.tif"
    save_path = f"F:\\Hyperspecial\\pear\\{data}\\segment"
    tif = io.imread(image_path)[:,:,0:3]
    print(tif.shape)

    # plt.imshow(tif)
    # plt.show()

    # segmentation_and_save(tif,save_path)
    file_id_list, cluster_result = cluster_segments(save_path)
    present_cluster(save_path, file_id_list, cluster_result)

