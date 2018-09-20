"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    super(ConvNet, self).__init__()

    # conv1 block
    conv1 = [
      nn.Conv2d(n_channels, 64, kernel_size=(3, 3), stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU()
    ]

    maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

    # conv2 block
    conv2 = [
      nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU()
    ]

    maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

    # conv3 block
    conv3 = [
      nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
      nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU()
    ]

    maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

    # conv4 block
    conv4 = [
      nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1),
      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU()
    ]

    maxpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

    # conv5 block
    conv5 = [
      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU()
    ]

    maxpool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
    avgpool = nn.AvgPool2d(kernel_size=(1, 1), stride=1, padding=0)

    linear = nn.Linear(512, n_classes)

    softmax = nn.Softmax()

    feature_extraction_layers = [
      *conv1,
      maxpool1,
      *conv2,
      maxpool2,
      *conv3,
      maxpool3,
      *conv4,
      maxpool4,
      *conv5,
      maxpool5,
      avgpool,
    ]

    self.feature_extraction = nn.Sequential(
      *feature_extraction_layers
    )

    classification_layers = [
      linear
    ]

    self.classification = nn.Sequential(
      *classification_layers
    )
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    features = self.feature_extraction(x)
    features = features.reshape(-1, 512)
    out = self.classification(features)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
