import torch
import torch.utils.data as data_utils
import torchvision
from torch.utils.tensorboard import SummaryWriter
import random
import tqdm

if __name__ == "__main__":
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True,num_classes=3)