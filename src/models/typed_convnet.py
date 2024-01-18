import torch
from typing import Callable, Union, List, Optional, Tuple

from torch import nn

class ConvNet2D(nn.Module):
    """ Convolutional neural network class with 2 convolutional layers and 1 fully connected layer.
    With batch normalization and maxpooling.
    
    Args:
        in_features: integer, number of input features
        out_features: integer, number of output features
    
    """
    def __init__(self, 
                 img_shape: Tuple[int, int],
                in_channels: int,
                conv_features_layer1: int, 
                conv_features_layer2: int, 
                kernel_size_layer1: int, 
                kernel_size_layer2: int,
                maxpool_dim: int):
        super().__init__()
        self.img_shape = img_shape #this should bea variable also and not a constant
        self.conv1 = nn.Conv2d(in_channels, conv_features_layer1, kernel_size_layer1)  # [B, 1, 160, 106] --> [B, 64, 156, 102]
        self.conv2 = nn.Conv2d(conv_features_layer1, conv_features_layer2, kernel_size_layer2)  # [B, 64, 156, 102] --> [B, 64, 154, 100]
        self.maxpool2 = nn.MaxPool2d(maxpool_dim, padding=0)  # [B, 64, 154, 100] --> [B, 64, 77, 50]
        dim_reduction = (kernel_size_layer1 - 1) + (kernel_size_layer2 - 1)
        reduced_img_shape = [self.img_shape[0] - dim_reduction, self.img_shape[1] - dim_reduction]
        reduced_img_shape = [reduced_img_shape[0] // maxpool_dim, reduced_img_shape[1] // maxpool_dim]
        self.fc3 = nn.Linear(conv_features_layer2 * reduced_img_shape[0] * reduced_img_shape[1], 1)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(1) don't use activation for regression problem
        self.batchnorm1 = torch.nn.BatchNorm2d(conv_features_layer1)
        self.batchnorm2 = torch.nn.BatchNorm2d(conv_features_layer2)
        self.batchnorm3 = torch.nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor):
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        x #.resize_(x.shape[0], 1, 28, 28)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = torch.tensor(x, dtype = torch.float32)
        x = torch.flatten(x, start_dim=1)
        x = self.fc3(x)
        x = self.batchnorm3(x)
        #x = self.softmax(x)
        return x

