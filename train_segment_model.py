import torch
import torch.utils.data as data_utils
import torchvision
from torch.utils.tensorboard import SummaryWriter
import random
import tqdm
import numpy as np
from dataset.SegmentationDataset import SegmentationDataset

def load_data(data_path,batch_size):
    dataset = SegmentationDataset(data_path)

    num_of_samples = len(dataset)

    index_training = random.sample(range(1,num_of_samples), int(0.9*num_of_samples))

    train_dataset = data_utils.Subset(dataset, index_training)

    val_dataset = data_utils.Subset(dataset, list(set(range(1,num_of_samples)).difference(set(index_training))))

    train_dataloader = data_utils.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=6)
    val_dataloader = data_utils.DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=6)

    return train_dataloader, val_dataloader

def train_model(model:torch.nn.Module, loss_function, optimizer, device, epoch_num, epoch, train_datasetloader:data_utils.DataLoader):
    model.train()
    model.to(device=device)

    sum_loss = 0
    step_num = len(train_datasetloader)

    with tqdm.tqdm(total= step_num) as tbar:
        for data, target in train_datasetloader:
            data, target = data.to(device), target.to(device)
            if data.shape[0] == 1:
                continue
            output = model(data)["out"]
            print(output.shape, target.shape)
            loss = loss_function(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print_loss = loss.data.item()
            sum_loss += print_loss
            
            tbar.set_description('Training Epoch: {}/{} Loss: {:.6f}'.format(epoch, epoch_num, loss.item()))
            tbar.update(1)
    
    ave_loss = sum_loss / step_num
    return ave_loss


def val_model(model:torch.nn.Module, device, loss_function, val_datasetloader):
    model.eval()
    test_loss = 0
    correct = 0
    total_num = 0
    with torch.no_grad():
        with tqdm.tqdm(total = len(val_datasetloader)) as pbar:
            for data, target in val_datasetloader:
                data, target = data.to(device), target.to(device)
                output = model(data)["out"]
                loss = loss_function(output, target)
                pred = torch.argmax(output, 1)
                correct += torch.sum(pred == target)
                total_num += pred.shape.numel()
                print_loss = loss.data.item()
                test_loss += print_loss
                pbar.update(1)

        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(val_datasetloader)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, total_num, 100 * acc))
    
    return avgloss, correct, acc

def load_parameters(model:torch.nn.Module, parameter_path:str) -> torch.nn.Module:
    model_dict = model.state_dict()

    pretrained_dict = torch.load(parameter_path)
    pretrained_dict = {k: v for k, v in model_dict.items() if np.shape(pretrained_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"Load parameter {len(model_dict)}/{len(pretrained_dict)}")

    return model

if __name__ == "__main__":
    data_path = "F:\Hyperspecial\pear_processed\segmentation_data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_num = 50
    batch_size = 8
    learn_rate = 0.0001
    pretrained_model_path = "log\\segment_best.pth"

    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    print(model)
    model.backbone.conv1 = torch.nn.Conv2d(5,64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.aux_classifier[4] = torch.nn.Conv2d(256, 5, kernel_size=(1, 1), stride=(1, 1))
    model.classifier[4] = torch.nn.Conv2d(256, 5, kernel_size=(1, 1), stride=(1, 1))

    if pretrained_model_path != None:
        model = load_parameters(model, pretrained_model_path)

    model.to(device)

    train_dataloader, val_dataloader = load_data(data_path, batch_size)

    loss_function = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(),learn_rate)
    writer = SummaryWriter(log_dir="log")

    total_num = len(train_dataloader.dataset)
    step_num = len(train_dataloader)
    print("training dataset total images: {}, step number: {}".format(total_num, step_num))

    total_num = len(val_dataloader.dataset)
    step_num = len(val_dataloader)
    print("val dataset total images: {}, step number: {}".format(total_num, step_num))

    for i in range(epoch_num):
        best_acc = 0

        train_avgloss = train_model(model, loss_function, optimizer, device, epoch_num, i, train_dataloader)
        val_avgloss, correct, acc = val_model(model, device, loss_function, val_dataloader)

        writer.add_scalar(tag="training_loss", scalar_value=train_avgloss, global_step=i)
        writer.add_scalar(tag="val_avgloss", scalar_value=val_avgloss, global_step=i)
        writer.add_scalar(tag="val_acc", scalar_value=acc, global_step=i)
        
        if acc>best_acc:
            torch.save(model.state_dict(), 'log/segment_best.pth')
            best_acc = acc

        if (i+1) % 5 == 0:
            torch.save(model.state_dict(), 'log/segment_{}.pth'.format(str(i)))