"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}
    
    self.params['weight'] = np.random.normal(loc=0, scale=0.0001, size=(out_features, in_features))
    self.params['bias'] = np.zeros(shape=out_features)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = x @ self.params['weight'].T + self.params['bias'][None, :]
    self.x = x

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = dout @ self.params['weight']
    self.grads['weight'] = dout.T @ self.x
    self.grads['bias'] = np.sum(dout, axis=0)
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = np.maximum(x, 0)

    self.x = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    dx = dout * (self.x > 0)

    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    b = x.max(axis=1, keepdims=True)
    x_exp = np.exp(x - b)
    out = x_exp/x_exp.sum(axis=1, keepdims=True)

    self.x = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    batch_size = self.x.shape[0]
    n_classes = self.x.shape[1]

    diag = np.vsplit(self.x, batch_size) * np.eye(n_classes)
    temp = diag - self.x[:, :, None] * self.x[:, None, :]

    dx = (dout[:, None, :] @ temp).reshape(batch_size, n_classes)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
<<<<<<< HEAD
    out = np.mean(np.sum(-y * np.log(np.clip(x, a_min=1e-25, a_max=None)), axis=1))
=======
    out = np.mean(np.sum(-y * np.log(x), axis=1))
>>>>>>> start numpy implementation;
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    batch_size = x.shape[0]
<<<<<<< HEAD
    dx = (-y * 1/np.clip(x, a_min=1e-25, a_max=None)) / batch_size
=======
    dx = (-y * 1/x) / batch_size
>>>>>>> start numpy implementation;
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
