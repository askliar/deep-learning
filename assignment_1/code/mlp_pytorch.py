"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

<<<<<<< HEAD
    def __init__(self, n_inputs, n_hidden, n_classes, dropouts=None):
=======
    def __init__(self, n_inputs, n_hidden, n_classes, dropouts = None):
>>>>>>> finish assignment 1
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
        super(MLP, self).__init__()

        # Check there is at least one node in the input layer
        if n_inputs < 1:
<<<<<<< HEAD
            raise ValueError(
                "Number of units in the input layer is incorrect. There should be at least one unit.")
=======
            raise ValueError("Number of units in the input layer is incorrect. There should be at least one unit.")
>>>>>>> finish assignment 1

        # Check there is at least one node in each of the hidden layers.
        # Using `any` instead of all to speed up the check by using short circuit evaluation.
        if len(n_hidden) > 0 and any(n_layer < 0 for n_layer in n_hidden):
            raise ValueError(
                "Number of units in one of the hidden layer is incorrect. There should be at least one unit.")

        # Check there is at least one node in the output layer
        if n_classes < 1:
<<<<<<< HEAD
            raise ValueError(
                "Number of units in the output layer is incorrect. There should be at least one unit.")
=======
            raise ValueError("Number of units in the output layer is incorrect. There should be at least one unit.")
>>>>>>> finish assignment 1

        # Create list with sizes of all the layers.
        sizes = [n_inputs] + n_hidden + [n_classes]

        # Check dropout parameter
        # if dropouts is not None:
<<<<<<< HEAD
        #     # Check if number of dropouts is the same as number of layers in the MLP
=======
        #     # Check if number of dropouts is the same as number of layers in the MLP 
>>>>>>> finish assignment 1
        #     if isinstance(dropouts, list):
        #         if len(dropouts) > len(sizes) - 2:
        #             raise ValueError("Length of dropouts list is too large. It should be equal to the number "
        #                             "of layers in your MLP (excluding output layer).")
        #         elif len(dropouts) < len(sizes) - 2:
        #             raise ValueError("Length of dropouts list is too small. It should be equal to the number "
        #                             "of layers in your MLP (excluding output layer).")
        #     else:
        #         raise ValueError("Length of dropouts list is too small. It should be equal to the number "
        #                         "of layers in your MLP (excluding output layer).")

        layers = []
        # Go over all the layers, excluding the last one
        for idx in range(len(sizes) - 1):
            input_size, output_size = sizes[idx], sizes[idx + 1]
            layers.append(nn.Linear(input_size, output_size))

            # avoid adding ReLU activation in the very end, instead add softmax
            if idx < len(sizes) - 2:
                layers.append(nn.ReLU())
                # add dropout layer
                # if dropouts is not None:
                #     dropout_rate = dropouts[idx]
                #     if dropout_rate > 0:
                #         layers.append(nn.Dropout(dropout_rate))

        # define sequential model
        self.mlp = nn.Sequential(*layers)

        ########################
        # END OF YOUR CODE    #
        #######################

    # return string representation for debugging purposes
    def __str__(self):
        return str(self.mlp)

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
<<<<<<< HEAD

        # Check if x is of type Tensor - if not, convert it to Tensor type
        if not isinstance(x, torch.Tensor):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            elif isinstance(x, list):
                x = torch.FloatTensor(x)
            else:
                x = torch.FloatTensor([x])

        # propagate x through sequential
        out = self.mlp(x)

        ########################
        # END OF YOUR CODE    #
        #######################

=======

        # Check if x is of type Tensor - if not, convert it to Tensor type
        if not isinstance(x, torch.Tensor):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            elif isinstance(x, list):
                x = torch.FloatTensor(x)
            else:
                x = torch.FloatTensor([x])

        # propagate x through sequential
        out = self.mlp(x)
>>>>>>> finish assignment 1
        return out
