
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

def plot_point(tif_img,date):
    global index
    index = 0
    global tree_centre
    tree_centre = [[]]
    if os.path.exists(f"tree_centre_{date}.npy"):
        tree_centre = np.load(f"tree_centre_{date}.npy", allow_pickle=True)

    fig = plt.figure()

    if os.path.exists(f"tree_centre_{date}.npy"):
        for i in range(tree_centre.shape[0]):

            arr = tree_centre[i]

            x_list = []
            y_list = []
            for point in arr:
                x_list.append(point[0])
                y_list.append(point[1])
                
            plt.plot(x_list,y_list,'o')

    def on_press(event):
        global index
        global tree_centre
        if event.button==1: #鼠标左键点击
            tree_centre[index].append([event.xdata, event.ydata])
            print("add position:" , event.xdata, event.ydata, len(tree_centre[index]))
            plt.plot([event.xdata],[event.ydata], 'o')
            fig.canvas.draw_idle()

        if event.button==3: #鼠标右键点击
            print("Row:",len(tree_centre[index]))
            index += 1
            tree_centre.append([])
        if event.button==2:
            tree_centre[index].pop()
            print("delete last point", len(tree_centre[index]))

    
    for i in range(30):
        plt.plot([0,2828],[-100+i*91,248+i*91], color = "red")

    for i in range(38):
        plt.plot([-15+i*74,160+i*74],[0,2806], color="blue", alpha=0.5)

    plt.imshow(tif_img)
    fig.canvas.mpl_connect('button_press_event', on_press)

    plt.show()

    a = np.array(tree_centre)
    np.save(f"tree_centre_{date}.npy",a)

def check_tree_centre(tif_img,date):
    annotation_yellow = [[0,18],[0,24],[1,10],[1,18],[1,28],[2,17],[2,19],[3,3],[3,9],[3,19],[4,22],[5,24],[6,13],[6,16],[6,24],[8,28],[10,6],[18,23]]
    annotation_blue = [[3,16],[4,13],[4,19],[5,1],[5,3],[5,14],[5,22],[6,6],[7,0],[14,1],[14,16],[15,13],[15,14],[23,28]]

    tree_centre = np.load(f"tree_centre_{date}.npy",allow_pickle=True)
    print("Row number: ",tree_centre.shape[0])

    # tree_centre = np.delete(tree_centre, [21])

    yellow_list_x = []
    yellow_list_y = []
    blue_list_x = []
    blue_list_y = []

    for index in range(tree_centre.shape[0]):

        arr = tree_centre[index]

        print(type(arr))

        # if index == 20:
        #     del arr[-14]
        #     arr.extend(tree_centre[21])


        print("Col num: ", len(arr), index)
        tree_centre[index] = arr

        x_list = []
        y_list = []
        for j, point in enumerate(arr):

            if [index,j] in annotation_yellow:
                yellow_list_x.append(point[0])
                yellow_list_y.append(point[1])
            elif [index,j] in annotation_blue:
                blue_list_x.append(point[0])
                blue_list_y.append(point[1])
            else:
                x_list.append(point[0])
                y_list.append(point[1])

            # if index == 20:
            #     print(point[0], point[1])
            
        plt.plot(x_list,y_list,'o', color = 'Green')
    
    plt.plot(yellow_list_x,yellow_list_y, 'o', color = 'yellow')
    plt.plot(blue_list_x,blue_list_y, 'o', color = 'blue')

    plt.imshow(tif_img)
    plt.show()

    # np.save(f"tree_centre_{date}.npy",tree_centre)

def present_with_rectangle(tif_img,date):
    annotation_yellow = [[0,18],[0,24],[1,10],[1,18],[1,28],[2,17],[2,19],[3,3],[3,9],[3,19],[4,22],[5,24],[6,13],[6,16],[6,24],[8,28],[10,6],[18,23]]
    annotation_blue = [[3,16],[4,13],[4,19],[5,1],[5,3],[5,14],[5,22],[6,6],[7,0],[14,1],[14,16],[15,13],[15,14],[23,28]]

    fig, ax = plt.subplots()
    ax.imshow(tif_img)
    
    tree_centre = np.load(f"tree_centre_{date}.npy",allow_pickle=True)

    for index in range(tree_centre.shape[0]):

        arr = tree_centre[index]

        for i in range(len(arr)):
            xc,yc = arr[i][0],arr[i][1]

            color = 'green'
            if [index,i] in annotation_yellow:
                color = 'yellow'

            if [index,i] in annotation_blue:
                color = 'blue'

            rect = mpatches.Rectangle((xc-25,yc-25),50,50,fill=False,edgecolor = color, linewidth=2)

            ax.add_patch(rect)
    
    plt.show()

def present_plot_in_segment_map(date):
    annotation_yellow = [[0,18],[0,24],[1,10],[1,18],[1,28],[2,17],[2,19],[3,3],[3,9],[3,19],[4,22],[5,24],[6,13],[6,16],[6,24],[8,28],[10,6],[18,23]]
    annotation_blue = [[3,16],[4,13],[4,19],[5,1],[5,3],[5,14],[5,22],[6,6],[7,0],[14,1],[14,16],[15,13],[15,14],[23,28]]
    annotation_black = [[1,1]]

    tree_centre = np.load(f"tree_centre_{date}.npy",allow_pickle=True)

    mosaic = f"F:\\Hyperspecial\\pear\\{date}\\Aerial_UAV_Photos\\Orthomosaic.rgb.tif"
    # segment_mask = io.imread(f"F:\\Hyperspecial\\pear\\{date}\\segment_map.jpg")
    segment_mask = io.imread(f"show_cluster_result.jpg")
    mosaic_img = io.imread(mosaic)

    fig, ax = plt.subplots()
    ax.imshow(mosaic_img)

    # ax.imshow(segment_mask, alpha=0.4)
    

    for index in range(tree_centre.shape[0]):

        arr = tree_centre[index]

        for i in range(len(arr)):
            xc,yc = arr[i][0],arr[i][1]

            color = 'green'
            if [index,i] in annotation_yellow:
                color = 'yellow'

            if [index,i] in annotation_blue:
                color = 'blue'

            if [index,i] in annotation_black:
                color = 'black'

            rect = mpatches.Rectangle((xc-25,yc-25),50,50, fill=False, edgecolor = color, linewidth=4)

            ax.add_patch(rect)
    
    plt.show()

    # ax.set_axis_off()
    # fig.savefig('myfig.png',dpi=300)

def generate_object_detection_annotation_for_one_data(tif_img, date, save_path):
    annotation_yellow = [[0,18],[0,24],[1,10],[1,18],[1,28],[2,17],[2,19],[3,3],[3,9],[3,19],[4,22],[5,24],[6,13],[6,16],[6,24],[8,28],[10,6],[18,23]]
    annotation_blue = [[3,16],[4,13],[4,19],[5,1],[5,3],[5,14],[5,22],[6,6],[7,0],[14,1],[14,16],[15,13],[16,14],[23,28]]
    annotation_black = []
    # 14_07_21
    # annotation_yellow = [[0,18],[0,24],[1,10],[1,18],[1,28],[2,17],[2,19],[3,3],[3,9],[3,19],[4,22],[5,24],[6,13],[6,16],[6,24],[8,28],[10,6],[18,23]]
    # annotation_blue = [[3,16],[4,11],[4,13],[4,19],[5,1],[5,3],[5,14],[5,22],[6,6],[7,0],[14,1],[14,16],[14,28],[14,34],[15,13],[16,14],[23,28]]
    # annotation_black = [[12,16],[13,35],[18,34],[22,0]]

    tree_centre = np.load(f"tree_centre_{date}.npy",allow_pickle=True)

    img_shape = tif_img.shape

    print("img_shape", img_shape)
    
    with open(save_path, "w") as annotation_file:
        for index in range(tree_centre.shape[0]):

            arr = tree_centre[index]

            for i in range(len(arr)):
                xc,yc = arr[i][0],arr[i][1]

                class_type = 0 # healthy tree
                if [index,i] in annotation_yellow:
                    class_type = 1 # yellow disease tree 

                if [index,i] in annotation_blue:
                    class_type = 2 # blue disease tree 

                if [index,i] in annotation_black:
                    class_type = 3 # dead tree

                # rect = mpatches.Rectangle((xc-25,yc-25),50,50,fill=False,edgecolor = color, linewidth=2)

                annotation_file.write(f"{class_type} {xc/img_shape[1]} {yc/img_shape[0]} {50/img_shape[1]} {50/img_shape[0]}\n")

if __name__ == "__main__":

    # image_path = "F:\\Hyperspecial\\pear\\14_09_21\\Aerial_UAV_Photos\\Orthomosaic.rgb.tif"
    DATE = "15_07_22"
    # DATE = "14_09_22"
    # DATE = "15_07_22"
    # DATE = "27_07_21"
    # DATE = "14_09_21"

    image_path = f"F:\\Hyperspecial\\pear\\{DATE}\\Aerial_UAV_Photos\\Orthomosaic.rgb.tif"
    tif = io.imread(image_path)[:,:,0:3]
    print(tif.shape)

    # plot_point(tif,DATE)
    # check_tree_centre(tif,DATE)
    # present_with_rectangle(tif,DATE)

    # present_plot_in_segment_map(DATE)

    generate_object_detection_annotation_for_one_data(tif, DATE, f"F:\\Hyperspecial\\pear_processed\\{DATE}\\annotation.txt")