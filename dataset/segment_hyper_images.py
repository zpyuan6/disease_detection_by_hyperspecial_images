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
import yaml
import random
import shutil

import tqdm
from libtiff import TIFF
from skimage.io  import imread, imsave

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

            if not os.path.exists(os.path.join(save_path,f"{i}_mask.jpg")):
                mask_image = Image.fromarray(np.repeat((mask_expanded*255),3,axis=-1).astype(np.uint8))
                mask_image.save(os.path.join(save_path,f"{i}_mask.jpg"))
            

            patch = (mask_expanded * tif_img + (1 - mask_expanded) * float(0))

            if not os.path.exists(os.path.join(save_path,f"{i}_patch.jpg")):
                patch_image = Image.fromarray((patch).astype(np.uint8))
                patch_image.save(os.path.join(save_path,f"{i}_patch.jpg"))
            # print("patch",patch.shape, np.mean(patch), np.min(patch), np.max(patch))
            # segments.append(patch)

            if not os.path.exists(os.path.join(save_path,f"{i}_superpixel.jpg")):
                ones = np.where(mask == 1)
                h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
                super_pixel = patch[h1:h2, w1:w2]
                # print("patch",super_pixel.shape, np.mean(super_pixel), np.min(super_pixel), np.max(super_pixel))
                superpixel_image = Image.fromarray((patch[h1:h2, w1:w2]).astype(np.uint8))
                superpixel_image.save(os.path.join(save_path,f"{i}_superpixel.jpg"))
            
            # if 50 < np.sum(patch[h1:h2, w1:w2]) < 200:
            # if 50 < np.mean(patch[h1:h2, w1:w2]) < 240 and np.sum(mask)>300:
                # plt.subplot(1,2,1)
                # plt.imshow(patch)
                # plt.subplot(1,2,2)
                # plt.imshow(superpixel_image)

                # plt.show()

            i+=1

            # del patch_image
            # del superpixel_image
            # del super_pixel
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
    COLOR_MAP = [[131,60,11],[112,173,71],[0,112,192],[255,192,0],[0,0,0]]
    # GRAY_MAP = [0.0,51.0,102.0,153.0,204.0,255.0]
    GRAY_MAP = [0.0,1.0,2.0,3.0,4.0,5.0]

    max_index = np.max(cluster_result)+1
    print(max_index)

    mosaic_images = None
    with tqdm.tqdm(total=len(file_id_list)) as tbar:
        for i, file_id in enumerate(file_id_list):
            patch_mask = transforms.ToTensor()(Image.open(os.path.join(save_path, f"{file_id}_patch.jpg")).convert('L')).squeeze()

            if mosaic_images == None:
                # mosaic_images = Image.new("RGB",(patch_mask.shape[1],patch_mask.shape[0]))
                mosaic_images = (patch_mask != 0) * GRAY_MAP[cluster_result[i]]
            else:
                mosaic_images = mosaic_images + ((patch_mask != 0) * GRAY_MAP[cluster_result[i]+1])
            
            tbar.update(1)

    print(mosaic_images, mosaic_images.shape)

    # mask_images = transforms.ToPILImage()(mosaic_images)
    plt.imshow(mosaic_images)
    plt.show()

    mosaic_plt_image = Image.new("RGB", (mosaic_images.shape[1],mosaic_images.shape[0]))

    with tqdm.tqdm(total=mosaic_images.shape[1]) as tbar:
        for i in range(0, mosaic_images.shape[1]):
            for j in range(0, mosaic_images.shape[0]):
                color = COLOR_MAP[1]
                if mosaic_images[j][i] == 3 or mosaic_images[j][i] == 4:
                    color = COLOR_MAP[0]
                mosaic_plt_image.putpixel([i,j],(color[0],color[1],color[2]))
            tbar.update(1)

    mosaic_plt_image = mosaic_plt_image.crop((1000, 2000, 1512, 2512))

    plt.imshow(mosaic_plt_image)
    plt.savefig("show_cluster_result.jpg")
    plt.show()

    return mosaic_plt_image


def segmentation_and_annotation(tif_img, tree_centres_path, tree_annotation:dict, save_path=None):
    """
    This function is generated the segment id list for healthy, anarsiaLineatella, grapholitaMolesta, and dead trees.
    Arg:
        tif: the Orthomosaic.rgb.tif image
        tree_centre_path: A path of numpy, which stores the annotated tree centre
        tree_annotation: A list of annotated unhealthy tree in the following form 
            {
                "anarsiaLineatella":[[index_row,index_col],[],[],...],
                "grapholitaMolesta":[[index_row,index_col],[],[],...],
                "dead":[[index_row,index_col],[],[],...],
                ...
            }
    Return:
        healthy tree: a list of segment id for health tree
        anarsiaLineatella tree: a list of segment id for AnarsiaLineatella(Yellow)
        grapholitaMolesta tree: a list of segment id for GrapholitaMolesta(Blue)
        dead tree: a list of segment id for DeadTree(Black)
    """

    # First, segment image
    segment_map = segmentation_image(tif_img)

    tree_centres = np.load(tree_centres_path,allow_pickle=True)

    healthy_tree, anarsiaLineatella_tree, grapholitaMolesta_tree, dead_tree = [],[],[],[]

    for row_index_of_tree_centre in range(len(tree_centres)):
        for col_index_of_tree_centre in range(len(tree_centres[row_index_of_tree_centre])):
            tree_centre = tree_centres[row_index_of_tree_centre][col_index_of_tree_centre]

            segment_id = segment_map[int(tree_centre[1])][int(tree_centre[0])]

            is_added = False
            for tree_type in tree_annotation:
                a_type_tree_list = tree_annotation[tree_type]   
                if [row_index_of_tree_centre, col_index_of_tree_centre] in a_type_tree_list:
                    if tree_type == "anarsiaLineatella":
                        anarsiaLineatella_tree.append(segment_id)
                    elif tree_type == "grapholitaMolesta":
                        grapholitaMolesta_tree.append(segment_id)
                    elif tree_type == "dead":
                        dead_tree.append(segment_id)
                    
                    is_added = True
                    break

            if not is_added:
                healthy_tree.append(segment_id)

    print("healthy_tree",healthy_tree)
    print("anarsiaLineatella_tree",anarsiaLineatella_tree)
    print("grapholitaMolesta_tree",grapholitaMolesta_tree)
    print("dead_tree",dead_tree)

    if not save_path == None:
        np.save(os.path.join(save_path, "segmentation.npy"), segment_map)
        np.save(os.path.join(save_path, "healthy_tree_segmentation_id.npy"), np.array(healthy_tree))
        np.save(os.path.join(save_path, "anarsiaLineatella_tree_segmentation_id.npy"), np.array(anarsiaLineatella_tree))
        np.save(os.path.join(save_path, "grapholitaMolesta_tree_segmentation_id.npy"), np.array(grapholitaMolesta_tree))
        np.save(os.path.join(save_path, "dead_tree_segmentation_id.npy"), np.array(dead_tree))

    return healthy_tree, anarsiaLineatella_tree, grapholitaMolesta_tree, dead_tree

def generate_segment_annotation(save_path, shadow_folder=None):
    # load data
    config_file = open("class_list.yaml")
    class_list = yaml.load(config_file, Loader=yaml.FullLoader)['class_name']
    print(class_list)

    segment_map = np.load(os.path.join(save_path, "segmentation.npy"), allow_pickle=True)
    print("segmentation map", segment_map.shape)
    healthy_tree_list = np.load(os.path.join(save_path, "healthy_tree_segmentation_id.npy"), allow_pickle=True)
    anarsiaLineatella_tree_list = np.load(os.path.join(save_path, "anarsiaLineatella_tree_segmentation_id.npy"), allow_pickle=True)
    grapholitaMolesta_tree_list = np.load(os.path.join(save_path, "grapholitaMolesta_tree_segmentation_id.npy"), allow_pickle=True)
    dead_tree_list = np.load(os.path.join(save_path, "dead_tree_segmentation_id.npy"), allow_pickle=True)
    print(f"Num of health tree {len(healthy_tree_list)}")
    print(f"Num of anarsiaLineatella tree {len(anarsiaLineatella_tree_list)}")
    print(f"Num of grapholitaMolesta tree {len(grapholitaMolesta_tree_list)}")
    print(f"Num of dead tree {len(dead_tree_list)}")


    # generate shadow segment id list
    shadow_ids = []
    if shadow_folder!=None:
        for root, folders, files in os.walk(shadow_folder):
            for file in files:
                segment_id = file.split("_")[0]
                if (not segment_id in healthy_tree_list.data) and (not segment_id in anarsiaLineatella_tree_list.data) and (not segment_id in grapholitaMolesta_tree_list.data) and (not segment_id in dead_tree_list.data):
                    shadow_ids.append(int(file.split("_")[0]))

    shadow_ids = np.array(shadow_ids)
    print(f"Num of shadow tree {len(shadow_ids)}")

    # Generate segmentation map
    segmentation_annotation = torch.full(segment_map.shape,1)

    for segment_id in anarsiaLineatella_tree_list:
        segmentation_annotation[segment_map==segment_id] = 0
    print("Finish anarsiaLineatella_tree_list",type(anarsiaLineatella_tree_list))
    for segment_id in dead_tree_list:
        segmentation_annotation[segment_map==segment_id] = 2
    print("Finish dead_tree_list",type(dead_tree_list))
    for segment_id in grapholitaMolesta_tree_list:
        segmentation_annotation[segment_map==segment_id] = 3
    print("Finish grapholitaMolesta_tree_list",type(grapholitaMolesta_tree_list))
    for segment_id in healthy_tree_list:
        segmentation_annotation[segment_map==segment_id] = 4
    print("Finish healthy_tree_list",type(healthy_tree_list))

    # for segment_id in shadow_ids:
    #     segmentation_annotation[segment_map==segment_id] = 5
    # print(type(shadow_ids),type(shadow_ids[0]))

    segment_img = transforms.ToPILImage()(segmentation_annotation.numpy().astype(np.uint8))

    plt.imshow(segment_img)
    plt.show()

    segment_img.save(os.path.join(save_path, "segment_annotation.jpg"))
    np.save(os.path.join(save_path, "segment_annotation.npy"),segmentation_annotation.numpy())

def present_segmentation(tif,save_path):
    # annotation = np.load(os.path.join(save_path, "segment_annotation.npy"),allow_pickle=True) 

    annotation = np.load("F:\\Hyperspecial\\pear_processed\\segmentation_data\\label\\14_09_22_4_4.npy",allow_pickle=True) 
    
    # plt.subplot(1,3,1)
    # plt.imshow(tif)
    # plt.subplot(1,3,2)
    # plt.imshow(annotation)
    # plt.subplot(1,3,3)
    # plt.imshow(tif)
    # plt.imshow(annotation,alpha=0.2)

    COLOR_MAP = [[131,60,11],[112,173,71],[0,112,192],[255,192,0],[0,0,0]]

    # plt.imshow(annotation)
    mosaic_plt_image = Image.new("RGB", (annotation.shape[1],annotation.shape[0]))
    with tqdm.tqdm(total=annotation.shape[1]) as tbar:
        for i in range(0, annotation.shape[1]):
            for j in range(0, annotation.shape[0]):
                color = COLOR_MAP[0]

                if annotation[j][i] == 3:
                    color = COLOR_MAP[2]
                elif annotation[j][i] == 4:
                    color = COLOR_MAP[1]
                elif annotation[j][i] == 0:
                    color = COLOR_MAP[3]

                mosaic_plt_image.putpixel([i,j],(color[0],color[1],color[2]))
            tbar.update(1)

    plt.imshow(mosaic_plt_image)
    plt.show()


def cut_segment_model_input_and_output(save_path,date):
    input_size = 512
    step_length = 512
    target_image_path = "F:\\Hyperspecial\\pear_processed\\segmentation_data_unoverlap\\input"
    target_label_path = "F:\\Hyperspecial\\pear_processed\\segmentation_data_unoverlap\\label"

    if not os.path.exists(target_image_path):
        os.makedirs(target_image_path)
        os.makedirs(target_label_path)


    source_image = np.load(os.path.join(save_path,"mosaic_with_NDVI_True.npy"),allow_pickle=True)
    print("source_image",source_image.shape)
    source_label = np.load(os.path.join(save_path,"segment_annotation.npy"),allow_pickle=True)
    print("source_label",source_label.shape)

    num_x = int((source_label.shape[1] - input_size)/step_length)+1
    num_y = int((source_label.shape[0] - input_size)/step_length)+1

    x_index = 0
    y_index = 0
    while y_index <= num_y:
        while x_index <= num_x:
            end_x = 512 + (step_length * x_index)
            end_y = 512 + (step_length * y_index)
            if x_index == num_x:
                end_x = source_label.shape[1]
            if y_index == num_y:
                end_y = source_label.shape[0]

            input_np = source_image[:, end_y-512:end_y,end_x-512:end_x]
            input_np[input_np == -1e4] = 0
            label_np = source_label[end_y-512:end_y,end_x-512:end_x]

            np.save(os.path.join(target_image_path,f"{date}_{y_index}_{x_index}.npy"),input_np)
            np.save(os.path.join(target_label_path,f"{date}_{y_index}_{x_index}.npy"),label_np)
            x_index += 1

        x_index = 0
        y_index += 1


def cut_image_for_object_detection_model(save_path,date):
    input_size = 512
    step_length = 206

    target_image_path = "F:\\Hyperspecial\\pear_processed\\yolo_object_detection\\images"
    target_label_path = "F:\\Hyperspecial\\pear_processed\\yolo_object_detection\\labels"

    if not os.path.exists(target_image_path):
        os.makedirs(target_image_path)
        os.makedirs(target_label_path)

    # source_image = np.load(os.path.join(save_path,"mosaic_with_NDVI_False.npy"),allow_pickle=True)
    source_image = io.imread(os.path.join(save_path,"Orthomosaic.rgb.tif"))
    print(source_image.shape)

    # source_label = open(os.path.join(save_path,"annotation.txt")).read().split("\n")
    
    num_x = int((source_image.shape[1] - input_size)/step_length)+1
    num_y = int((source_image.shape[0] - input_size)/step_length)+1

    print("num_x,num_y",num_x,num_y)

    x_index = 0
    y_index = 0
    while y_index <= num_y:
        while x_index <= num_x:
            end_x = 512 + (step_length * x_index)
            end_y = 512 + (step_length * y_index)
            if x_index == num_x:
                end_x = source_image.shape[1]
            if y_index == num_y:
                end_y = source_image.shape[0]

            input_np = source_image[end_y-512:end_y,end_x-512:end_x,:]
            input_np[input_np == -1e4] = 0

            # plt.imshow(input_np)
            # plt.show()

            # np.save(os.path.join(target_image_path,f"{date}_{y_index}_{x_index}.npy"),input_np)
            io.imsave(os.path.join(target_image_path,f"{date}_{y_index}_{x_index}.png"),input_np)

            # start_x = end_x - input_size
            # start_y = end_y - input_size

            # with open(os.path.join(target_label_path, f"{date}_{y_index}_{x_index}.txt"),'w') as annotation_file:
            #     for source_label_line in source_label:
            #         if len(source_label_line)>1:
            #             label_arr = [float(i) for i in source_label_line.split()] 
            #             xc,yc,w,h = label_arr[1],label_arr[2],label_arr[3],label_arr[4]

            #             x_min = (xc-(w/2)) * source_image.shape[2]
            #             y_min = (yc-(h/2)) * source_image.shape[1]
            #             x_max = (xc+(w/2)) * source_image.shape[2]
            #             y_max = (yc+(h/2)) * source_image.shape[1]

            #             print(x_max > start_x and x_min < end_x and y_max > start_y and y_min < end_y, x_max,start_x,x_min, end_x, y_max, start_y, y_min, end_y)

            #             if x_max > start_x and x_min < end_x and y_max > start_y and y_min < end_y:
            #                 print(x_max,start_x,x_min, end_x, y_max, start_y, y_min, end_y)
            #                 x_start = (max(start_x, x_min) - start_x)/input_size
            #                 x_end = (min(end_x, x_max) - start_x)/input_size
            #                 y_start = (max(start_y, y_min) - start_y)/input_size
            #                 y_end = (min(end_y, y_max) - start_y)/input_size

            #                 annotation_file.write(f"{int(label_arr[0])} {(x_start+x_end)/2} {(y_start+y_end)/2} {(x_end-x_start)} {(y_end-y_start)}\n")

            x_index += 1

        x_index = 0
        y_index += 1
    

def generate_img_for_paper(tif_img):
    image_start_x = 1000
    image_start_y = 2000
    image_size = 512
    plt.axis("off")
    plt.imshow(tif_img[image_start_y:image_start_y+image_size,image_start_x:image_start_x+image_size,:])
    plt.savefig("show_source_image.jpg")
    
    # mask = segmentation_image(tif_img)
    # unsupervised_segment_img = mark_boundaries(tif_img,mask)
    # plt.imshow(unsupervised_segment_img[image_start_y:image_start_y+image_size,image_start_x:image_start_x+image_size,:])
    # plt.savefig("show_unsupervised_segment_image.jpg")
    

def split_train_val_for_object_detection():
    img_path = "F:\Hyperspecial\pear_processed\yolo_object_detection\images"
    label_path = "F:\Hyperspecial\pear_processed\yolo_object_detection\labels"

    img_id_list = []
    for root, folders, files in os.walk(img_path):
        img_id_list.extend(files)
        break
    
    print(len(img_id_list))

    random.shuffle(img_id_list)

    for i in range(len(img_id_list)):

        if i < len(img_id_list)*0.9:  
            shutil.move(os.path.join(img_path, img_id_list[i]),os.path.join(img_path,"train", img_id_list[i]))
            annotation_file = img_id_list[i].split(".")[0]+".txt"
            shutil.move(os.path.join(label_path, annotation_file),os.path.join(label_path,"train", annotation_file))
        else:
            shutil.move(os.path.join(img_path, img_id_list[i]),os.path.join(img_path,"val", img_id_list[i]))
            annotation_file = img_id_list[i].split(".")[0]+".txt"
            shutil.move(os.path.join(label_path, annotation_file),os.path.join(label_path,"val", annotation_file))
        

def convert_np_to_tif():
    path = "F:\Hyperspecial\pear_processed\yolo_object_detection_times_255\images"

    for root, folders, files in os.walk(path):
        for file in files:
            n = np.load(os.path.join(root, file)) * 255.0
            print(n.shape)
            # n = n.transpose(1,2,0)
            # img = transforms.ToPILImage()(n)
            # img.save(os.path.join(root, file.split(".")[0]+".tiff"))
            # tif = TIFF.open(os.path.join(root, file.split(".")[0]+".tif"), mode='w')
            # tif.write_image(n)
            # imsave(os.path.join(root, file.split(".")[0]+".tif"), n)
            np.save(os.path.join(root, file), n)


if __name__ == "__main__":
    # date = "15_07_22"
    # date = "14_09_21"
    # date = "27_07_21"
    # image_path = "F:\\Hyperspecial\\pear\\14_09_21\\Aerial_UAV_Photos\\Orthomosaic.rgb.tif"
    # image_path = f"F:\\Hyperspecial\\pear\\{date}\\Aerial_UAV_Photos\\Orthomosaic.rgb.tif"
    # save_path = f"F:\\Hyperspecial\\pear\\{date}\\segment"
    # tif = io.imread(image_path)[:,:,0:3]
    # print(tif.shape)

    # plt.imshow(tif)
    # plt.show()
    # generate_img_for_paper(tif)

    # segmentation_and_save(tif,save_path)
    # file_id_list, cluster_result = cluster_segments(save_path)
    # present_cluster(save_path, file_id_list, cluster_result)

    # tree_annotation = {
    #     "anarsiaLineatella":[[0,18],[0,24],[1,10],[1,18],[1,28],[2,17],[2,19],[3,3],[3,9],[3,19],[4,22],[5,24],[6,13],[6,16],[6,24],[8,28],[10,6],[18,23]],
    #     "grapholitaMolesta":[[3,16],[4,11],[4,13],[4,19],[5,1],[5,3],[5,14],[5,22],[6,6],[7,0],[14,1],[14,16],[14,28],[14,24],[15,13],[16,14],[23,28]],
    #     "dead":[[2,5],[5,2],[12,16],[13,34],[18,34],[19.34],[22,0]]
    #     }
    # tree_annotation = {
    #     "anarsiaLineatella":[[0,18],[0,24],[1,10],[1,18],[1,28],[2,17],[2,19],[3,3],[3,9],[3,19],[4,22],[5,24],[6,13],[6,16],[6,24],[8,28],[10,6],[18,23]],
    #     "grapholitaMolesta":[[3,15],[4,13],[4,19],[5,1],[5,3],[5,14],[5,22],[6,6],[7,0],[14,1],[14,16],[15,13],[16,14],[23,28]],
    #     "dead":[]
    #     }
    # tree_annotation = {
    #     "anarsiaLineatella":[],
    #     "grapholitaMolesta":[],
    #     "dead":[]
    #     }

    # save_path = f"F:\Hyperspecial\pear_processed\{date}"

    # segmentation_and_annotation(tif, f"tree_centre_{date}.npy", tree_annotation, save_path)
    # shadow_folder = "F:\Hyperspecial\pear_processed\classifier_training_data\Shadow"
    # generate_segment_annotation(save_path, shadow_folder)
    # generate_segment_annotation(save_path)

    # present_segmentation(tif, save_path)

    # cut_segment_model_input_and_output(save_path,date)

    dates = ["27_07_21","14_09_21","25_05_22","15_07_22","14_09_22"]

    # for date in dates:
    #     save_path = f"F:\Hyperspecial\pear_processed\{date}"
    #     cut_segment_model_input_and_output(save_path,date)

    # for date in dates:
    #     # save_path = f"F:\Hyperspecial\pear_processed\{date}"
    #     # save_path = f"F:\\Hyperspecial\\pear\\{date}\\Aerial_UAV_Photos"
    #     save_path = f"F:\\Hyperspecial\\pear_processed\\{date}"
    #     # cut_image_for_object_detection_model(save_path,date)

    #     generate_segment_annotation(save_path)

    # split_train_val_for_object_detection()
    # convert_np_to_tif()

    present_segmentation("","")
