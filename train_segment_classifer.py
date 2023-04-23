import torch
import torch.utils.data as data_utils
import torchvision
from torch.utils.tensorboard import SummaryWriter
import random
import tqdm

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
            output = model(data)
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
    total_num = len(val_datasetloader.dataset)
    # print(total_num, len(test_loader))
    with torch.no_grad():
        with tqdm.tqdm(total = len(val_datasetloader)) as pbar:
            for data, target in val_datasetloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_function(output, target)
                _, pred = torch.max(output.data, 1)
                correct += torch.sum(pred == target)
                print_loss = loss.data.item()
                test_loss += print_loss
                pbar.update(1)

        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(val_datasetloader)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, len(val_datasetloader.dataset), 100 * acc))
    
    return avgloss, correct, acc


if __name__ == "__main__":
    data_folder = "F:\\Hyperspecial\\pear_processed\\classifier_training_data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_num = 50
    batch_size = 128
    learn_rate = 0.001

    train_dataset, val_dataset = load_dataset(data_folder)

    train_dataloader, val_dataloader = generate_dataloader(train_dataset, val_dataset, batch_size)

    model = torchvision.models.alexnet(pretrained=True)
    print(model)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, out_features=3)

    model.to(device=device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),learn_rate)

    writer = SummaryWriter(log_dir="log")

    total_num = len(train_dataloader.dataset)
    step_num = len(train_dataloader)
    print("training dataset total images: {}, step number: {}".format(total_num, step_num))

    total_num = len(val_dataloader.dataset)
    step_num = len(val_dataloader)
    print("val dataset total images: {}, step number: {}".format(total_num, step_num))

    for i in range(epoch_num):
        train_avgloss = train_model(model, loss, optimizer, device, epoch_num, i, train_dataloader)
        val_avgloss, correct, acc = val_model(model, device, loss, val_dataloader)

        writer.add_scalar(tag="training_loss", scalar_value=train_avgloss, global_step=i)
        writer.add_scalar(tag="val_avgloss", scalar_value=val_avgloss, global_step=i)
        writer.add_scalar(tag="val_acc", scalar_value=acc, global_step=i)
        
        if (i+1) % 5 == 0:
            torch.save(model.state_dict(), 'log/alex_{}.pth'.format(str(i)))



    