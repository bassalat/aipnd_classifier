# Imports
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import imageio
import matplotlib.pyplot as plt
import os, random
import json
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import argparse
from time import time, sleep



def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='/home/workspace/aipnd-project/flowers/', 
                        help='path to folder of images')
    parser.add_argument('--arch', type=str, default='vgg16',choices=['vgg16', 'vgg13'], 
                        help='chosen model')
    parser.add_argument('--learning_rate', type=float, default='0.0001', help='learning rate')
    parser.add_argument('--epochs', type=int, default='3', help='epochs')
    parser.add_argument('--hidden_layer1', type=int, default='4096', help='hidden layer 1')
    parser.add_argument('--hidden_layer2', type=int, default='1024', help='hidden layer 2')
    parser.add_argument('--gpu', type=str, default='yes', help='train with gpu')
    
    return parser.parse_args()
    
    

def main():
    start_time = time()
    
    in_arg = get_input_args()
    
    training_fun()

    end_time = time()
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    
    
def training_fun():
    in_arg = get_input_args()
    
    train_dir = in_arg.dir + 'train'
    valid_dir = in_arg.dir + 'valid'
    test_dir = in_arg.dir + 'test'
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomRotation(45),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])


    validation_transforms = test_transforms
    
    train_data = torchvision.datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform = test_transforms)
    validation_data = torchvision.datasets.ImageFolder(valid_dir, transform = validation_transforms)


    train_data_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=4,
                                              shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=4)
    validation_data_loader = torch.utils.data.DataLoader(validation_data,
                                              batch_size=4)
    
    
    
    if in_arg.arch == 'vgg16':
        vgg16 = models.vgg16(pretrained=True)
    else: vgg16 = models.vgg13(pretrained=True)

    for param in vgg16.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, in_arg.hidden_layer1)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(in_arg.hidden_layer1, in_arg.hidden_layer2)),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(p=0.5)),
                          ('fc3', nn.Linear(in_arg.hidden_layer2, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    vgg16.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(vgg16.classifier.parameters(), lr=in_arg.learning_rate)
    epochs = in_arg.epochs
    print_every = 40
    steps = 0

    
   #Train a model with a pre-trained network
    print("training started")
    vgg16.train()

    # change to cuda
    if torch.cuda.is_available() and in_arg.gpu == 'yes':
        #vgg16.to('cuda')
        vgg16.cuda()
    else:
        vgg16.cpu()
        

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_data_loader):
            steps += 1
        
            if torch.cuda.is_available() and in_arg.gpu == 'yes':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                inputs, labels = Variable(inputs), Variable(labels)
        
            optimizer.zero_grad()
        
            # Forward and backward passes
            outputs = vgg16.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                    "Loss: {:.4f}".format(running_loss/print_every))
            
            running_loss = 0
                
     # accuracy on the validation set:
    print("validation set")
    vgg16.eval()
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(validation_data_loader):
            steps += 1
        
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        #optimizer.zero_grad()
        
        # Forward pass and loss
            outputs = vgg16.forward(inputs)
            loss = criterion(outputs, labels)
            #loss.backward()
            #optimizer.step()
        
            running_loss += loss.item()
        
            ps = torch.exp(outputs).data
            equality = (labels.data == ps.max(1)[1])
            accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()
        
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                    "Loss: {:.4f}....".format(running_loss/print_every),
                    "accuracy: {:.4f}".format(accuracy))
            
                running_loss = 0
                
                
                
   #Do validation on the test set
    print("Validation on the test set")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
            #images, labels = images.to(device), labels.to(device) 
        
            outputs = vgg16(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test images: %d %%' % (100 * correct / total))
    
    # save checkpoint
    vgg16.class_to_idx = train_data.class_to_idx

    checkpoint = {'input_size': 25088,
              'output_size': 102,
              'arch': 'vgg16',
              'learning_rate': 0.0001,
              'batch_size': 4,
              'classifier' : classifier,
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'state_dict': vgg16.state_dict(),
              'class_to_idx': vgg16.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')
    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()