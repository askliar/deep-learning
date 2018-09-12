"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 
from functools import reduce

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # Check there is at least one node in the input layer
    if n_inputs < 1:
        raise ValueError(
            "Number of units in the input layer is incorrect. There should be at least one unit.")

    # Check there is at least one node in each of the hidden layers.
    # Using `any` instead of all to speed up the check by using short circuit evaluation.
    if len(n_hidden) > 0 and any(n_layer < 0 for n_layer in n_hidden):
        raise ValueError(
            "Number of units in one of the hidden layer is incorrect. There should be at least one unit.")

    # Check there is at least one node in the output layer
    if n_classes < 1:
        raise ValueError(
            "Number of units in the output layer is incorrect. There should be at least one unit.")

    # Create list with sizes of all the layers.
    sizes = [n_inputs] + n_hidden + [n_classes]

    self.layers = []
    # Go over all the layers, excluding the last one
    for idx in range(len(sizes) - 1):
        input_size, output_size = sizes[idx], sizes[idx + 1]
        self.layers.append(LinearModule(input_size, output_size))

        # avoid adding ReLU activation in the very end, instead add softmax
        if idx < len(sizes) - 2:
            self.layers.append(ReLUModule())
        else:
            self.layers.append(SoftMaxModule())
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
    out = reduce(lambda res, f: f.forward(res), self.layers, x)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    reduce(lambda res, f: f.backward(res), self.layers[::-1], dout)
    ########################
    # END OF YOUR CODE    #
    #######################

    return
