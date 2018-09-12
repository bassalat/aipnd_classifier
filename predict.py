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
    parser.add_argument('--dir', type=str, default='/home/workspace/aipnd-project/flowers/test/13/image_05761.jpg', 
                        help='test image')
    parser.add_argument('--topk', type=int, default='5', 
                        help='select top k classes')
    parser.add_argument('--gpu', type=str, default='yes', help='train with gpu')
    parser.add_argument('--json', type=str, default='/home/workspace/aipnd-project/cat_to_name.json', 
                        help='json file for category names')
    return parser.parse_args()

in_arg = get_input_args()

# Load checkpoint from train.py
def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    learning_rate = checkpoint['learning_rate']
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model

model_checkpoint = load_checkpoint('checkpoint.pth')
#print(model_checkpoint)


# Process image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    width = image.size[0]
    length = image.size[1]
    
    if length > width:
        #image = image.resize((256, length))
        image.thumbnail((256, length), Image.ANTIALIAS)
    else:
        #image = image.resize((width, 256))
        image.thumbnail((width, 256), Image.ANTIALIAS)

    image = image.crop((256/2 - 224/2,
                       256/2 - 224/2,
                       256/2 + 224/2,
                       256/2 + 224/2))
    
    image = (np.array(image))/255
    
    mean = np.array([0.485, 0.456, 0.406])
    st_dev = np.array([0.229, 0.224, 0.225])
    
    image = (image - mean) / st_dev
    image = image.transpose((2, 0, 1))
    
    
    return image


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    if torch.cuda.is_available() and in_arg.gpu == 'yes':
        model = model.cuda()
    else:
        model = model.cpu()
    image = Image.open(image_path)
    np_array = process_image(image)
    tensor = torch.from_numpy(np_array)
    
    if torch.cuda.is_available() and in_arg.gpu == 'yes':
        inputs = Variable(tensor.float().cuda())
    else: 
        inputs = Variable(tensor.float().cpu())
        
    inputs = inputs.unsqueeze(0)
    output = model.forward(inputs)
    
    ps = torch.exp(output).data.topk(topk)
    probabilities = ps[0].cpu()
    classes = ps[1].cpu()
    class_to_idx_nw = {model.class_to_idx[k]: k for k in model.class_to_idx}
    tag_classes = list()
    
    for label in classes.numpy()[0]:
        tag_classes.append(class_to_idx_nw[label])
        
    return probabilities.numpy()[0], tag_classes



with open(in_arg.json, 'r') as f:
    cat_to_name = json.load(f)

image_path = in_arg.dir
prob, classes = predict(image_path, model_checkpoint, topk= in_arg.topk)
print(prob)
print(classes)
print(np.array([cat_to_name[x] for x in classes]))
#%matplotlib inline
#plt.imshow(in_arg.dir, 'r')



