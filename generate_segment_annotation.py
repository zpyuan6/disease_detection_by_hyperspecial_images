from skimage import io
from dataset.segment_hyper_images import segmentation_image
import torchvision

def generate_annotation_semi_automatically(dates):
    raise Exception("Waiting for implement")

def generate_annotation_automatically(dates):
    for date in dates:
        mosaic_image_path = f"F:\\Hyperspecial\\pear\\{date}\\Aerial_UAV_Photos\\Orthomosaic.rgb.tif"
        tif_img = io.imread(mosaic_image_path)

        segment_map = segmentation_image(tif_img)


if __name__ == "__main__":
    dates = ["14_09_21","14_09_22","15_07_22","25_05_22","27_07_21"]

    classify_model = torchvision.models.alexnet(pretrained=False)

    generate_annotation_automatically(dates)