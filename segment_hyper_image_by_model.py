import torch
import numpy as np
import torch.utils.data as data_utils
import torchvision
from dataset.SegmentationDataset import SegmentationDataset
import matplotlib.pyplot as plt

def load_parameters(model:torch.nn.Module, parameter_path:str) -> torch.nn.Module:
    model_dict = model.state_dict()

    pretrained_dict = torch.load(parameter_path)
    pretrained_dict = {k: pretrained_dict[k] for k, v in model_dict.items() if np.shape(pretrained_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"Load parameter {len(model_dict)}/{len(pretrained_dict)}")

    return model

def model_prediction(input_tensor, model:torch.nn.Module, device):
    model.eval()

    model = model.to(device=device)
    input_tensor = input_tensor.to(device=device)
    model_output = model(input_tensor)['out']
    pred = torch.argmax(model_output, 1).squeeze()

    return pred

def split_image(input_numpy, input_size):
    input_list = []

    y_num, x_num = int(input_numpy.shape[1]/input_size)+1, int(input_numpy.shape[2]/input_size)+1
    print(f"For segment whole image, we need split the image in {y_num} rows, and {x_num} cols")

    for y_index in range(y_num):
        for x_index in range(y_num):
            y_end = input_numpy.shape[1] if (y_index+1) * 512 > input_numpy.shape[1] else (y_index+1) * 512
            x_end = input_numpy.shape[2] if (x_index+1) * 512 > input_numpy.shape[2] else (x_index+1) * 512

            input_split = input_numpy[:, y_end-input_size:y_end, x_end-input_size:x_end]
            input_list.append(input_split)

    return input_list, y_num, x_num

def merge_preds(pred_list, y_num, x_num, label_shape, input_size, device):
    pred_mask = torch.zeros(label_shape).to(device=device)

    for y_index in range(y_num):
        row_mask = torch.cat(pred_list[y_index*x_num: (y_index+1)*x_num-1], 1)
        last_start_pixel = label_shape[1] - (x_num-1)*input_size
        row_mask = torch.cat((row_mask, pred_list[(y_index+1)*x_num-1][:,input_size-last_start_pixel:input_size]), 1)

        print(row_mask.shape, pred_mask.shape)

        if (y_index+1)*512 > label_shape[0]:
            pred_mask[label_shape[0]-512:label_shape[0]] = row_mask
        else:
            pred_mask[y_index*512:(y_index+1)*512] = pred_mask[y_index*512:(y_index+1)*512] +row_mask
            
    
    return pred_mask




if __name__ == "__main__":
    parameter_path = "log\\segment_best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 512
    # input_data_path = "F:\\Hyperspecial\\pear_processed\\segmentation_data\\input\\14_09_21_1_0.npy"
    # output_label_path = "F:\\Hyperspecial\\pear_processed\\segmentation_data\\label\\14_09_21_1_0.npy"
    # input_data_path = "F:\\Hyperspecial\\pear_processed\\14_09_21\\mosaic_with_NDVI_True.npy"
    # output_label_path = "F:\\Hyperspecial\\pear_processed\\14_09_21\\segment_annotation.npy"
    input_data_path = "F:\\Hyperspecial\\pear_processed\\25_05_22\\mosaic_with_NDVI_True.npy"
    output_label_path = "F:\\Hyperspecial\\pear_processed\\14_09_21\\segment_annotation.npy"

    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
    model.backbone.conv1 = torch.nn.Conv2d(5,64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.aux_classifier[4] = torch.nn.Conv2d(256, 5, kernel_size=(1, 1), stride=(1, 1))
    model.classifier[4] = torch.nn.Conv2d(256, 5, kernel_size=(1, 1), stride=(1, 1))

    model = load_parameters(model,parameter_path)

    input_data = np.load(input_data_path, allow_pickle=True)
    output_label = np.load(output_label_path, allow_pickle=True)

    if input_data.shape[-1] == input_size:
        print(f"Input shape {input_data.shape}")
        input_tensor = torch.from_numpy(input_data).unsqueeze(0)
        pred = model_prediction(input_tensor, model, device)
    else:
        print(f"Input shape {input_data.shape}")
        input_data[input_data == -1e4] = 0
        input_list, y_num, x_num = split_image(input_data, input_size)
        pred_list = []
        for input_tensor in input_list:
            input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)
            pred = model_prediction(input_tensor, model, device)
            pred_list.append(pred)

        pred = merge_preds(pred_list, y_num, x_num, output_label.shape, input_size, device)


    print(input_data.shape)
    print(output_label.shape)
    print(pred.shape)
    plt.subplot(1,3,1)
    plt.imshow(input_data.squeeze()[0,:,:])
    plt.title("Visualised multispectral images")
    plt.subplot(1,3,2)
    plt.imshow(output_label)
    plt.title("Annotation")
    plt.subplot(1,3,3)
    plt.imshow(pred.cpu().detach().numpy())
    plt.title("Prediction (Segmentation)")

    plt.show()


