import os
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

def get_pixel_value_from_file_id(classify_result:list, file_id):
    GRAY_MAP = [0.0,51.0,102.0,153.0,204.0,255.0]

    for i, id_list in enumerate(classify_result):
        if file_id in id_list:
            return GRAY_MAP[i+1]

    raise Exception(f'Can not find classify result for file_id {file_id}')

def merge_segments_and_present_segement_map(classify_result:list, patch_path):

    mosaic_images = None
    original_images = None
    for root, folders, files in os.walk(patch_path):
        with tqdm.tqdm(total=len(files)) as tbar:
            for file in files:
                if str(file).find("mask") != -1:
                    patch_mask = transforms.ToTensor()(Image.open(os.path.join(root, file))).squeeze()
                    
                    file_id = file.split('_')[0]

                    pixel_value = get_pixel_value_from_file_id(classify_result,file_id)

                    if mosaic_images == None:
                        mosaic_images = (patch_mask != 0) * pixel_value
                        original_images = patch_mask
                    else:
                        mosaic_images = mosaic_images + ((patch_mask != 0) * pixel_value)
                        original_images = original_images + patch_mask
                
                tbar.update(1)

    print(mosaic_images, mosaic_images.shape)

    mask_images = transforms.ToPILImage()(mosaic_images)
    original_images = transforms.ToPILImage()(original_images)

    mask_images.save(os.path.join("\\".join(patch_path.split("\\")[0:-1]), 'segment_map.jpg'))

    plt.subplot(1,2,1)
    plt.imshow(mask_images)
    plt.subplot(1,2,2)
    plt.imshow(original_images)
    plt.show()

if __name__ == "__main__":
    patch_path = "F:\\Hyperspecial\\pear\\15_07_22\\segment"

    classified_id_path = "F:\\Hyperspecial\\pear_processed\\classifier_training_data"

    background_path = os.path.join(classified_id_path, "background")
    shadow_path = os.path.join(classified_id_path, "shadow")
    tree_path = os.path.join(classified_id_path, "tree")

    background_ids = []
    shadow_ids = []
    tree_ids = []
    
    for root, folders, files in os.walk(background_path):
        for file in files:
            background_ids.append(file.split('_')[0])
    print("background_ids: ",len(background_ids),background_ids)

    for root, folders, files in os.walk(shadow_path):
        for file in files:
            shadow_ids.append(file.split('_')[0])
    print("shadow_ids: ",len(shadow_ids),shadow_ids)

    for root, folders, files in os.walk(tree_path):
        for file in files:
            tree_ids.append(file.split('_')[0])
    print("tree_ids: ", len(tree_ids),tree_ids)

    merge_segments_and_present_segement_map([background_ids,shadow_ids,tree_ids],patch_path)



