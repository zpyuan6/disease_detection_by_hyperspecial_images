import torch.utils.data as data

class SegmentationDataset(data.Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index) -> T_co:
        return super().__getitem__(index)

    def __len__(self):
        return len(self.images)