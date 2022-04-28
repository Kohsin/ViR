from __future__ import print_function

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import csv

from reservoir.parallel_reservoir import Parallel_Reservoir

print(f"Torch: {torch.__version__}")

# Training settings
batch_size = 100
epochs = 100
lr = 3e-4
gamma = 0.7
seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

#Image Augumentation

data_type = "cifar10" # cifar100

if data_type == "cifar10":
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
else:
    mean = 0.5
    std = 0.5

train_transforms = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
)

# load data (cifar10 or cifar100)
if data_type == "cifar10":
    train_list = datasets.CIFAR10('./datasets', train=True, transform=train_transforms, download=True)
    test_list = datasets.CIFAR10('./datasets', train=False, transform=test_transforms, download=True)
    classes = 10
else:
    train_list = datasets.CIFAR100('./datasets', train=True, transform=train_transforms, download=True)
    test_list = datasets.CIFAR100('./datasets', train=False, transform=test_transforms, download=True)
    classes = 100



train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)

#print(f"{data_type} train data num: {train_list.data.shape}")
#print(f"{data_type} test data num: {test_list.data.shape}")

# image setting
dim = 128
image_size = 32
patch_size = 4
mlp_dim = 512
channels = 3

# reservoir setting
depth = 3
input_scaling = 1
spectral_radius = 0.9
leaky = 1
sparsity = 0.05
reservoir_units = 1000

# cycle reservoir setting
cycle_weight = 0.05
jump_weight = 0.5
jump_size = 137
connection_weight = 0.08

print(f"{reservoir_units} reservoir units with {depth} layers")
print(f"spectral_radius: {spectral_radius}, sparsity: {sparsity}")
print(f"cycle reservoir parameters - cycle_weight: {cycle_weight}, jump_weight: {jump_weight} jump_size: {jump_size} connection_weight: {connection_weight}")
print("learning rate:", lr)

# reservoir model
model = Parallel_Reservoir(
    dim=dim,
    image_size=image_size,
    patch_size=patch_size,
    num_classes=classes,
    depth=depth,
    mlp_dim=mlp_dim,
    channels=channels,
    device=device,
    reservoir_units=reservoir_units,
    input_scaling=input_scaling,
    spectral_radius=spectral_radius,
    leaky=leaky,
    sparsity=sparsity,
    cycle_weight=cycle_weight,
    jump_weight=jump_weight,
    jump_size=jump_size,
    connection_weight=connection_weight,
).to(device)

#Training
# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

best_acc = 0

current_time = time.strftime("%Y-%b-%d_%H;%M;%S", time.localtime())
# save model
# saved_model_path = "./model/" + current_time + "-reservoir_VIT_cifar10_cnn_reservoir" + "/"
# if not os.path.exists(saved_model_path):
#     os.makedirs(saved_model_path)

print("time:", current_time)
headers = ['epoch','loss','acc','test_loss','acc']
with open('test.csv','w')as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    
# training
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print("len", len(train_loader))
        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_test_accuracy = 0
        epoch_test_loss = 0
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            test_output = model(data)
            test_loss = criterion(test_output, label)

            acc = (test_output.argmax(dim=1) == label).float().mean()
            epoch_test_accuracy += acc / len(test_loader)
            epoch_test_loss += test_loss / len(test_loader)
    with open('test.csv','w')as f:
        f_csv.writerows(rows[epoch+1])
    print(
            f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - test_loss : {epoch_test_loss:.4f} - test_acc: {epoch_test_accuracy:.4f}\n"
    )

    if epoch_test_accuracy > best_acc:
        print(f"Best development accuracy improved from {best_acc} to {epoch_test_accuracy}\n")
        best_acc = epoch_test_accuracy
        # torch.save(model.state_dict(), saved_model_path + 'reservoir_VIT_best.dat')

print(f"Best development accuracy: {best_acc}")
