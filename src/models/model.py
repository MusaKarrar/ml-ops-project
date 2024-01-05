import torch
from torch import nn

class ConvNet2D(nn.Module):
    """ Convolutional neural network class with 2 convolutional layers and 1 fully connected layer.
    With batch normalization and maxpooling.
    
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self):
        super().__init__()
        # torch.set_default_dtype(torch.float32)
        self.conv1 = nn.Conv2d(1, 64, 5)  # [B, 1, 28, 28] --> [B, 64, 24, 24]
        self.conv2 = nn.Conv2d(64, 64, 3)  # [B, 64, 26, 26] --> [B, 64, 22, 22]
        self.maxpool2 = nn.MaxPool2d(2, padding=0)  # [B, 64, 24, 24] --> [B, 64, 11, 11]

        self.fc3 = nn.Linear(64 * 11 * 11, 10)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)
        self.batchnorm1 = torch.nn.BatchNorm2d(64)
        self.batchnorm2 = torch.nn.BatchNorm2d(64)
        self.batchnorm3 = torch.nn.BatchNorm1d(10)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        x.resize_(x.shape[0], 1, 28, 28)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc3(x)
        x = self.batchnorm3(x)
        x = self.softmax(x)
        return x
