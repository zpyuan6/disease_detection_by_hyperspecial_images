import os
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt

class SegmentationDataset(data.Dataset):
    def __init__(self, dataset_path) -> None:
        input_path = os.path.join(dataset_path,"input")
        output_path = os.path.join(dataset_path,"label")

        for root,folders, files in os.walk(input_path):
            self.input_file_list = [os.path.join(root,file) for file in files]

        for root,folders, files in os.walk(output_path):
            self.segment_file_list = [os.path.join(root,file) for file in files]

        print(f"Read {len(self.input_file_list)} valid input samples, and {len(self.segment_file_list)} annotation")

    def __getitem__(self, index):
        input_image = torch.from_numpy(np.load(self.input_file_list[index],allow_pickle=True))

        label = torch.from_numpy(np.load(self.segment_file_list[index],allow_pickle=True))

        return input_image, label

    def get_file_path(self, index):
        return self.input_file_list[index]

    def __len__(self):
        return len(self.input_file_list)

if __name__ == "__main__":

    dataset = SegmentationDataset("F:\\Hyperspecial\\pear_processed\\segmentation_data")

    for i in range(len(dataset)):
        input_sample,label = dataset.__getitem__(i)
        if label.shape != torch.Size([512, 512]):
            print(input_sample.shape, label.shape)
        else:
            plt.subplot(1,4,1)
            plt.title(dataset.get_file_path(i))
            plt.imshow(input_sample[0,:,:])
            plt.subplot(1,4,2)
            plt.imshow(input_sample[1,:,:])
            plt.subplot(1,4,3)
            plt.imshow(input_sample[2,:,:])
            plt.subplot(1,4,4)
            plt.imshow(label)
            plt.show()

    print(torch.Size([512,512]).numel())