import torch
import torch.utils.data as data_utils
import torchvision
import random

def load_dataset(data_folder):
    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize([64,64]),
        torchvision.transforms.ToTensor()
    ])

    dataset = torchvision.datasets.ImageFolder(data_folder, transform=trans)

    num_of_samples = len(dataset)

    index_training = random.sample(range(1,num_of_samples), int(0.9*num_of_samples))

    train_dataset = data_utils.Subset(dataset, index_training)

    val_dataset = data_utils.Subset(dataset, list(set(range(1,num_of_samples)).difference(set(index_training))))

    return train_dataset, val_dataset

def generate_dataloader(train_dataset, val_dataset, batch_size):
    train_dataloader = data_utils.DataLoader(train_dataset, )
    val_dataloader = data_utils.DataLoader(val_dataset)

def train_model(model, device, batch_size, epoch_num, epoch, train_dataset):

def val_model(model, device, batch_size, epoch_num, epoch, train_dataset):

if __name__ == "__main__":
    data_folder = "F:\\Hyperspecial\\pear_processed\\classifier_training_data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_num = 50
    batch_size = 128

    train_dataset, val_dataset = load_dataset(data_folder)

    

    model = torchvision.models.vgg11(pretrained=True)

    for i in range(epoch_num):
        train_model(model, device, batch_size, epoch_num, epoch, train_dataset)
        val_model(model, device, batch_size, epoch_num, epoch, val_dataset)



    