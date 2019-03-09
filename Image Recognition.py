# Importing Packages
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode





# Loading the data with the use of torchvision and torch.utils.data packages
# Normalizing Data
data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=data_transforms)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=False, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=data_transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=4)
# Defining Classes in DataSet
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Creating Dataloader for train and test 
dataloaders={}
dataloaders["train"] = trainloader
dataloaders["test"] = testloader


# Length of dataset
dataset_sizes={}
dataset_sizes["train"]=len(trainset)
dataset_sizes["test"]=len(testset)

# Verifying the size
dataiter = iter(trainloader)
images, labels = dataiter.next()

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))

# Visualizing Images in Trainset

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images[:4]))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")










#Convnet as a feature extractor
#Freezing all the layers other than the final layer
#Freezing the gradient parameter so that the gradient is not computed backwards
#Testing Data Accuracy : 62.59%
#
#Choices made:
#Increased the batch size to 32 and parellel processing channels to 4(to optimize the computation time)
#Computation Time reduced by 30% (from ~100 min to ~60 min)
#Difference in training and testing accuracies reduced toa round 1-2% from 10-15%
#Changed the Learning Rate to 0.01 (from 0.001) and Momentum to 0.5 (from 0.9) that helped to converge faster


# Building Model

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode if it is a training set
            else:
                model.eval()   # Set model to evaluate mode if otherwise

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in  dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



# Using RESNET-18 Pretrained model to extract features

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 10)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.01, momentum=0.5)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


# Training Model
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)









# Finetuning the Convnet
# Load a pretrained model and reset final fully connected layer.
# Cross-entropy loss captures error on the target class. It discards any notion of errors that you might consider "false positive" and does not care how predicted
# probabilities are distributed other than predicted probability of the true class.
# Testing Dataset Accuracy : 87.08%


# using pretrained RESNET-18 Model for Fine-Tuning
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features

# Adding Final Fully Connected Layer
model_ft.fc = nn.Linear(num_ftrs, 10)

model_ft = model_ft.to(device)

# Defining Loss 
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.5)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Training Model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)







# Visualizing Top 5 Correct and Top 5 Incorrect Predictions

def Top5_Correct(model, num_images=5,out=1):
    images_so_far = 0
    fig = plt.figure()

    for b in range(1000): 
        data = next(iter(testloader))
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        b += 1
        for a in range(31):
            if labels[a] == out:
                if preds[a] == out:
                    print('predicted: {}'.format(classes[preds[a]]))
                    imshow(torchvision.utils.make_grid(inputs.cpu().data[a]))
                    images_so_far += 1

                if images_so_far == num_images:
                        return  

def Top5_Incorrect(model, num_images=25,out=1):
    images_so_far = 0
    fig = plt.figure()

    for b in range(1000): 
        data = next(iter(testloader))
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        b += 1
        for a in range(31):
            if labels[a] == out:
                if preds[a] != out:
                    print('predicted: {}'.format(classes[preds[a]]))
                    imshow(torchvision.utils.make_grid(inputs.cpu().data[a]))
                    images_so_far += 1

                if images_so_far == num_images:
                        return 



# For Feature Extraction Model

for i in range(10):
    print('Top correct predictions for class: {}'.format(classes[i]))
    Top5_Correct(model_conv,num_images=5,out=i)
    print('Top Incorrect predictions for class: {}'.format(classes[i]))
    Top5_Incorrect(model_conv,num_images=5,out=i)

# For Fine-Tuning Model

for i in range(10):
    print('Top correct predictions for class: {}'.format(classes[i]))
    Top5_Correct(model_ft,num_images=5,out=i)
    print('Top Incorrect predictions for class: {}'.format(classes[i]))
    Top5_Incorrect(model_ft,num_images=5,out=i)