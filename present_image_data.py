from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


def plot_point(tif_img):
    global index
    index = 0
    global tree_centre
    tree_centre = [[]]
    fig = plt.figure()

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
    np.save("tree_centre.npy",a)

def check_tree_centre(tif_img):
    tree_centre = np.load("tree_centre.npy",allow_pickle=True)
    print("Row number: ",tree_centre.shape[0])

    for index in range(tree_centre.shape[0]):

        arr = tree_centre[index]

        tree_centre[index] = arr
        print("Col num: ", len(arr))
        x_list = []
        y_list = []
        for point in arr:
            x_list.append(point[0])
            y_list.append(point[1])

            print(point[0], point[1])
            
        plt.plot(x_list,y_list,'o')

    plt.imshow(tif_img)
    plt.show()

    # np.save("tree_centre.npy",tree_centre)
    

def present_with_rectangle(tif_img):

    fig, ax = plt.subplots()
    ax.imshow(tif_img)
    
    tree_centre = np.load("tree_centre.npy",allow_pickle=True)

    for index in range(tree_centre.shape[0]):

        arr = tree_centre[index]

        for i in range(len(arr)):
            xc,yc = arr[i][0],arr[i][1]
            rect = mpatches.Rectangle((xc-25,yc-25),50,50,fill=False,edgecolor = 'red', linewidth=2)

            ax.add_patch(rect)

    
    plt.show()

def save_to_jpg(tif_img,save_path):
    im = Image.fromarray(tif_img)
    im.show()
    im.save(save_path)

if __name__ == "__main__":

    # image_path = "F:\\Hyperspecial\\pear\\14_09_21\\Aerial_UAV_Photos\\Orthomosaic.rgb.tif"
    # image_path = "F:\\Hyperspecial\\pear\\15_07_22\\Aerial_UAV_Photos\\Orthomosaic.rgb.tif"
    # "F:\Hyperspecial\pear\27_07_21\Aerial_UAV_Photos"
    image_path = "F:\\Hyperspecial\\cherry\\13-07-2022\\Aerial_UAV_photos\\rededge.rgb.tif"
    tif = io.imread(image_path)[:,:,0:3]
    print(tif.shape)

    # plt.imshow(tif)
    # plt.show()

    # plot_point(tif)
    # check_tree_centre(tif)
    # segmentation_image(tif)
    # present_with_rectangle(tif)
    save_to_jpg(tif, "F:\\Hyperspecial\\cherry\\13-07-2022\\Aerial_UAV_photos\\rededge.jpg")
