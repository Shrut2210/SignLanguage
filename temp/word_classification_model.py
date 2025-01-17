import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from PIL import Image

data_path = os.path.join("../Data")
train_data_path = os.path.join(data_path, "Words/downloaded_images")

# print(train_data_path)

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

mean = [0.7421, 0.7358, 0.7252]
std = [0.3584, 0.3613, 0.3622]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

train_dataset = torchvision.datasets.ImageFolder(root = train_data_path, transform = train_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

def set_device() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def train_nn(model, train_loader, criterion, optimizer, n_epochs) :
    device = set_device()
    model = model.to(device)
    
    for epoch in range(n_epochs) :
        print(f"epoch = {epoch + 1}")
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        i = 0
        
        for data, labels in train_loader :
            i += 1
            print(i)
            images, labels = data.to(device), labels.to(device)
            total += labels.size(0)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * running_correct / total
        
        print(f"Train --- Running correct - {running_correct}, Total - {total}, Epochs - {epoch_acc}, Epoch Loss - {epoch_loss}")
        
    return model

model = models.resnet18(pretrained=False)
criterion = nn.CrossEntropyLoss()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.003)

model = train_nn(model, train_loader, criterion, optimizer, 20)

torch.save(model.state_dict(), 'sign_language_word_model.pth')