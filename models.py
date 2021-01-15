## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # second conv layer: 32 inputs, 64 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output tensor will have dimensions: (64, 106, 106)
        # after another pool layer this becomes (64, 53, 53);
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        # dropout with p=0.5
        self.conv2_drop = nn.Dropout(p=0.5)
        
        # third conv layer: 64 inputs, 96 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (53-5)/1 +1 = 49
        # the output tensor will have dimensions: (96, 49, 49)
        # after another pool layer this becomes (96, 24, 24); 24.5 is rounded down
        self.conv3 = nn.Conv2d(64, 96, 5)
        
        # dropout with p=0.4
        self.conv3_drop = nn.Dropout(p=0.4)
        
        # fourth conv layer: 96 inputs, 64 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (24-3)/1 +1 = 22
        # the output tensor will have dimensions: (128, 22, 22)
        # after another pool layer this becomes (128, 11, 11); 
        self.conv4 = nn.Conv2d(96, 128, 3)
        
        # 128 outputs * the 11*11 filtered/pooled map size
        self.fc1 = nn.Linear(128*11*11, 816)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # second fully connected layer
        self.fc2 = nn.Linear(816, 272)
        
        # dropout with p=0.3
        self.fc2_drop = nn.Dropout(p=0.3)
        
        # finally, create 136 output channels
        self.fc3 = nn.Linear(272, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv2_drop(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.conv3_drop(x)
        x = self.pool(F.relu(self.conv4(x)))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
                        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
