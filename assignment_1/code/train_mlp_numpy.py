"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt 

# Commented for running on display-less systems like surfsara
# import matplotlib.pyplot as plt

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
  predictions_indices = predictions.argmax(1)
  targets_indices = targets.argmax(1)

  accuracy = (predictions_indices == targets_indices).sum() / predictions_indices.shape[0]
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

  mlp = MLP(n_inputs, dnn_hidden_units, n_classes)

  loss_criterion = CrossEntropyModule()

  # Commented for running on display-less systems like surfsara
  # _, (loss_axis, accuracy_axis) = plt.subplots(
  #     nrows=1, ncols=2, figsize=(10, 4)
  # )

  # arrays for storing accuracies, losses and steps in which evaluations were made
  # train_steps = []
  # train_losses = []
  # train_accuracies = []
  # test_steps = []
  # test_losses = []
  # test_accuracies = []

  for step in range(FLAGS.max_steps + 1):
      images, labels = cifar10['train'].next_batch(FLAGS.batch_size)
      input_data = images.reshape((FLAGS.batch_size, -1))

      outputs = mlp.forward(input_data)
      loss = loss_criterion.forward(outputs, labels)
      train_accuracy = accuracy(outputs, labels)

      # save train accuracies, losses and steps in which evaluations were made into corresponding arrays
      # train_steps.append(step)
      # train_losses.append(loss)
      # train_accuracies.append(train_accuracy)

      mlp.backward(loss_criterion.backward(outputs, labels))
      
      # update layer parameters using SGD
      for layer in mlp.layers:
        if hasattr(layer, 'params'):
          weight_grad = layer.grads['weight']
          bias_grad = layer.grads['bias']

          layer.params['weight'] -= FLAGS.learning_rate * weight_grad
          layer.params['bias'] -= FLAGS.learning_rate * bias_grad

      if (step % FLAGS.eval_freq) == 0:
          test_loss = 0.0
          test_accuracy = 0.0

          # number of batches to go through the whole test dataset once
          num_batches = cifar10['test'].num_examples//FLAGS.batch_size
          
          # evaluate using batches
          for i in range(num_batches):
              test_images, test_labels = cifar10['test'].next_batch(FLAGS.batch_size)
              test_input_data = test_images.reshape((test_images.shape[0], -1))
              test_outputs = mlp.forward(test_input_data)

              test_loss += loss_criterion.forward(test_outputs, test_labels)
              test_accuracy += accuracy(test_outputs, test_labels)
          
          test_accuracy /= num_batches
          test_loss /= num_batches

          # save test accuracies, losses and steps in which evaluations were made into corresponding arrays
          # test_accuracies.append(test_accuracy)
          # test_losses.append(test_loss)
          # test_steps.append(step)

          print(f"Test loss at {step} is: {test_loss}")
          print(f"Test accuracy at {step} is: {test_accuracy}")

      # Commented for running on display-less systems like surfsara
      # If uncommented - will dynamically plot loss and accuracy curves
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
      
      # save losses and accuracies to files
      # np.savetxt('test_steps.txt', np.array(test_steps))
      # np.savetxt('test_losses.txt', np.array(test_losses))
      # np.savetxt('test_accuracies.txt', np.array(test_accuracies))

      # np.savetxt('train_steps.txt', np.array(train_steps))
      # np.savetxt('train_losses.txt', np.array(train_losses))
      # np.savetxt('train_accuracies.txt', np.array(train_accuracies))
      
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
