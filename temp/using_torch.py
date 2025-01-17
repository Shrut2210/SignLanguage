import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

data_path = os.path.join("../Data")
train_data_path = os.path.join(data_path, "asl_alphabet_train/asl_alphabet_train")
test_data_path = os.path.join(data_path, "asl_alphabet_test/asl_alphabet_test")

# calculate mean and standard deviation
# data_transforms = transforms.Compose([transforms.Resize((224,224)) ,transforms.ToTensor()])

# train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=data_transforms)

# train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

# def get_mean_std():
#     mean = 0.0
#     std = 0.0
#     nb_samples = 0
    
#     for data, _ in train_loader:
#         batch_samples = data.size(0)
#         data = data.view(batch_samples, data.size(1), -1)
#         mean += data.mean(2).sum(0)
#         std += data.std(2).sum(0)
#         nb_samples += batch_samples
        
#     mean /= nb_samples
#     std /= nb_samples
    
#     return mean, std

# mean, std = get_mean_std()
# print(mean, std)

mean = [0.5180, 0.4967, 0.5127]
std = [0.2037, 0.2336, 0.2420]

train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

train_dataset = torchvision.datasets.ImageFolder(root = train_data_path, transform = train_transforms)
test_dataset = torchvision.datasets.ImageFolder(root = test_data_path, transform = test_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

def set_device():
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    return torch.device(dev)

def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs) :
    device = set_device()
    best_acc = 0
    
    for epoch in range(n_epochs) :
        print(f"epoch = {epoch + 1}")
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        i = 0 
        
        for data in train_loader :
            print(i)
            i += 1
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()
            
        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.00 * running_correct / total
        
        print(f"Train --- Running correct - {running_correct}, Total - {total}, Epochs - {epoch_acc}, Epoch Loss - {epoch_loss}")
        
        test_dataset_acc = eval_model_on_test(model, test_loader)
        
        if (test_dataset_acc > best_acc) :
            best_acc = test_dataset_acc
            save_checkpoint(model, epoch, optimizer, best_acc)
            
    print("Done!!")

    return model

def save_checkpoint(model, epoch, optimizer, best_acc) :
    state = {
        'epoch' : epoch + 1,
        'model' : model.state_dict(),
        'best accuracy' : best_acc,
        'optimizer' : optimizer.state_dict(),
    }
    
    torch.save(state, 'model_best_checkpoint.pth.tar')
    
def eval_model_on_test(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()
    
    with torch.no_grad() :
        for data in test_loader :
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            predicted_correctly_on_epoch += (predicted == labels).sum().item()
    
    epoch_acc = 100.00 * predicted_correctly_on_epoch / total

    print(f" Test --- Correct - {predicted_correctly_on_epoch} Total - {total} Epoch - {epoch_acc}")
    
    return epoch_acc

resnet18_model = models.resnet18(pretrained=False)
num_ftrs = resnet18_model.fc.in_features
number_of_class = 26
resnet18_model.fc = nn.Linear(num_ftrs, number_of_class)
device = set_device()
resnet_18_model = resnet18_model.to(device)
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)

train_nn(resnet18_model, train_loader, test_loader, loss_fn, optimizer, 50)

checkpoint = torch.load('model_best_checkpoint.pth.tar')

resnet18_model = models.resnet18()
num_ftrs = resnet18_model.fc.in_features
number_of_class = 26
resnet18_model.fc = nn.Linear(num_ftrs, number_of_class)
resnet18_model.load_state_dict(checkpoint['model'])

torch.save(resnet18_model.state_dict(), 'best_model.pth')