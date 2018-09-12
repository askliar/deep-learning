"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

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
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  cifar10 = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir, one_hot=True, validation_size=0)
  n_classes = 10
  n_inputs = 3 * 32 * 32

  mlp = MLP(n_inputs, dnn_hidden_units, n_classes).to(device)

  optimizer = torch.optim.SGD(mlp.parameters(), FLAGS.learning_rate)
  loss_criterion = torch.nn.CrossEntropyLoss()
  mlp.train()

  _, (loss_axis, accuracy_axis) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

  train_steps = []
  train_losses = []
  train_accuracies = []

  test_steps = []
  test_losses = []
  test_accuracies = []

  for step in range(FLAGS.max_steps):
      images, labels = cifar10['train'].next_batch(FLAGS.batch_size)
      labels = torch.from_numpy(labels).long().to(device)
      _, labels_indices = labels.max(1)
      input_data = torch.from_numpy(images).reshape((FLAGS.batch_size, -1)).to(device)

      optimizer.zero_grad()

      outputs = mlp(input_data)
      loss = loss_criterion(outputs, labels_indices)
      train_accuracy = accuracy(outputs, labels)

      train_steps.append(step)
      train_losses.append(loss.data)
      train_accuracies.append(train_accuracy)

      loss.backward()
      optimizer.step()

      if (step % FLAGS.eval_freq) == 0:
          mlp.eval()

          test_images = cifar10['test'].images
          test_labels = cifar10['test'].labels
          test_labels = torch.from_numpy(test_labels).long().to(device)

          _, test_labels_indices = test_labels.max(1)
          test_input_data = torch.from_numpy(test_images).reshape((test_images.shape[0], -1)).to(device)

          test_outputs = mlp(test_input_data)

          test_loss = loss_criterion(test_outputs, test_labels_indices).data
          test_accuracy = accuracy(test_outputs, test_labels)

          test_accuracies.append(test_accuracy)
          test_losses.append(test_loss)
          test_steps.append(step)


          # num_batches = cifar10['test'].num_examples//FLAGS.batch_size
          #
          # for i in range(num_batches):
          #     test_images, test_labels = cifar10['test'].next_batch(FLAGS.batch_size)
          #     test_labels = torch.from_numpy(test_labels).long().to(device)
          #     _, test_labels_indices = test_labels.max(1)
          #     test_input_data = torch.from_numpy(test_images).reshape((FLAGS.batch_size, -1)).to(device)
          #
          #     test_outputs = mlp(test_input_data)
          #     test_loss += loss_criterion(test_outputs, test_labels_indices).data
          #     test_accuracy += accuracy(test_outputs, test_labels)
          #
          # test_accuracy /= num_batches
          # test_accuracies.append(test_accuracy)
          # test_loss /= num_batches
          # test_losses.append(test_loss)
          # test_steps.append(step)

          print(f"Test loss at {step} is: {test_loss}")
          print(f"Test accuracy at {step} is: {test_accuracy}")

          mlp.train()

      # if (step % 3) == 0:
      #     loss_axis.cla()
      #     loss_axis.plot(train_steps, train_losses, label="train loss")
      #     loss_axis.plot(test_steps, test_losses, label="test loss")
      #     loss_axis.legend()
      #     loss_axis.set_title('Train and Test Losses')
      #     loss_axis.set_ylabel('Loss')
      #     loss_axis.set_xlabel('Step')
      #
      #     accuracy_axis.cla()
      #     accuracy_axis.plot(train_steps, train_accuracies, label="train accuracy")
      #     accuracy_axis.plot(test_steps, test_accuracies, label="test accuracy")
      #     accuracy_axis.set_title('Train and Test Accuracies')
      #     accuracy_axis.legend()
      #     accuracy_axis.set_ylabel('Accuracy')
      #     accuracy_axis.set_xlabel('Step')
      #
      #     plt.draw()
      #     plt.ion()
      #     plt.show()
      #
      #     plt.pause(0.00001)

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
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
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