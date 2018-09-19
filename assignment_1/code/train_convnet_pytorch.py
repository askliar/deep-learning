"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
import torch.nn as nn

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  raise NotImplementedError
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  # Set pytorch seeds as well
  random.seed(42)
  torch.manual_seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  cifar10 = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir, one_hot=True, validation_size=0)
  n_classes = 10
  n_channels = 3

  vgg = nn.DataParallel(ConvNet(n_channels, n_classes).to(device))

  optimizer = torch.optim.Adam(vgg.parameters(), FLAGS.learning_rate)
  loss_criterion = torch.nn.CrossEntropyLoss()

  # set mode to train
  vgg.train()

  arrays for storing accuracies, losses and steps in which evaluations were made
  train_steps = []
  train_losses = []
  train_accuracies = []
  test_steps = []
  test_losses = []
  test_accuracies = []

  for step in range(FLAGS.max_steps + 1):
    
    images, labels = cifar10['train'].next_batch(FLAGS.batch_size)
    labels = torch.from_numpy(labels).long().to(device)
    _, labels_indices = labels.max(1)
    input_data = torch.from_numpy(images).to(device)

    optimizer.zero_grad()

    outputs = vgg(input_data)
    loss = loss_criterion(outputs, labels_indices)
    train_accuracy = accuracy(outputs, labels)

    # save train accuracies, losses and steps in which evaluations were made into corresponding arrays
    # train_steps.append(step)
    # train_losses.append(loss.item())
    # train_accuracies.append(train_accuracy.item())

    loss.backward()
    optimizer.step()
 
    if (step % FLAGS.eval_freq) == 0:
      
      # set mode to evaluation
      vgg.eval()

      test_loss = 0.0
      test_accuracy = 0.0

      # number of batches to go through the whole test dataset once
      num_batches = cifar10['test'].num_examples // FLAGS.batch_size

      # evaluate using batches, otherwise can run out of memory trying to process all images on GPU
      for i in range(num_batches):
        test_images, test_labels = cifar10['train'].next_batch(FLAGS.batch_size)
        test_labels = torch.from_numpy(test_labels).long().to(device)
        _, test_labels_indices = test_labels.max(1)
        test_input_data = torch.from_numpy(test_images).to(device)

        test_outputs = vgg(test_input_data)
        test_loss += loss_criterion(test_outputs, test_labels_indices).item()
        test_accuracy += accuracy(test_outputs, test_labels).item()

      # average accuracy and loss over batches
      test_accuracy /= num_batches
      test_loss /= num_batches

      # save test accuracies, losses and steps in which evaluations were made into corresponding arrays
      # test_accuracies.append(test_accuracy)
      # test_losses.append(test_loss)
      # test_steps.append(step)
      
      print(f"Test loss is: {test_loss}")
      print(f"Test accuracy is: {test_accuracy}")

      # set mode back to train
      vgg.train()

  # save losses and accuracies to a file
  # with open(f'output_convnet.txt', 'w') as f:
  #       f.write(f'Test steps: \n{test_steps}')
  #       f.write(f'Test losses: \n{test_losses}')
  #       f.write(f'Test accuracies: \n{test_accuracies}')

  #       f.write(f'Train steps: \n{train_steps}')
  #       f.write(f'Train losses: \n{train_losses}')
  #       f.write(f'Train accuracies: \n{train_accuracies}')

  print('Finished Training')
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
