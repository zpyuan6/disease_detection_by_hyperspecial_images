from skimage import io
from skimage.segmentation import felzenszwalb,slic,quickshift,watershed, mark_boundaries, slic_superpixels
from skimage.filters import sobel
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def segmentation_image(tif_img):
    # segments_fz = felzenszwalb(tif_img, scale=200, sigma=0.8, min_size=50)
    # print("Finished felzenszwalb")
    # plt.figure(figsize=(10,10))
    # plt.imshow(mark_boundaries(tif_img,segments_fz))
    # plt.show()

    segments_slic = slic(tif_img, n_segments=2500, compactness=20, sigma=1, start_label=1)
    print("Finished slic")
    plt.imshow(mark_boundaries(tif_img,segments_slic))
    plt.show()

    # segments_quick = quickshift(tif_img, kernel_size=5, max_dist=10, ratio=1.0)
    # print("Finished quickshift")
    # plt.imshow(mark_boundaries(tif_img,segments_quick))
    # plt.show()

    # gradient = sobel(rgb2gray(tif_img))
    # segments_watershed = watershed(gradient, markers=250, compactness=0.001)
    # print("Finished watershed")
    # plt.imshow(mark_boundaries(tif_img,segments_watershed))
    # plt.show()

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


if __name__ == "__main__":

    # image_path = "F:\\Hyperspecial\\pear\\14_09_21\\Aerial_UAV_Photos\\Orthomosaic.rgb.tif"
    image_path = "F:\\Hyperspecial\\pear\\15_07_22\\Aerial_UAV_Photos\\Orthomosaic.rgb.tif"
    tif = io.imread(image_path)[:,:,0:3]
    print(tif.shape)

    # plt.imshow(tif)
    # plt.show()

    plot_point(tif)
    check_tree_centre(tif)
    # segmentation_image(tif)
    # present_with_rectangle(tif)
