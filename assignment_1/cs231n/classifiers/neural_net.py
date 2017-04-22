from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange
import copy 

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.std = std
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    first_layer_out = np.einsum("dh,nd->hn",self.params['W1'],X) + self.params["b1"].reshape(-1,1)
    first_layer_out = np.maximum(np.zeros(first_layer_out.shape), first_layer_out)
    second_layer_out = np.einsum("hc,hn->nc",self.params['W2'], first_layer_out) + self.params["b2"].T
    scores = second_layer_out
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    num_train = X.shape[0] 
    #scores is n*c, get max score for each class per example and subract it out 
    scores -= np.max(scores,axis=1,keepdims=True)
    #extract the correct score out 
    correct_class_score = scores[np.arange(num_train), y] 
    #exponentiate the scores 
    exp_scores = np.exp(scores)
    #sum the exp scores along each example 
    sum_exp_scores = np.sum(exp_scores, axis=1)
    loss = -np.sum(correct_class_score) + np.sum(np.log(sum_exp_scores))

    loss /= num_train
    W1 = np.vstack((self.params['W1'],self.params['b1']))
    W2 = np.vstack((self.params['W2'],self.params['b2']))
    loss += reg * np.sum(W1*W1)+\
	    reg * np.sum(W2*W2)


    #dW += reg*W

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    #
   # get derivative of loss function wrt the function params(aka soft-max )    
    probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True) #N*C
    dscores = probs
    dscores[np.arange(num_train), y] -= 1
    dscores = dscores / num_train
    grads["W2"] = np.dot(first_layer_out, dscores) + 2*reg*self.params["W2"]
    #sum the gradient along each class(axis=0)
    grads["b2"] = np.sum(dscores, axis=0)


    #first_layer_out
    #dhidden should of size h*n
    #loss = softmax(W2(hidden))
    #dhidden =  softmax_output*W2
    #softmax_output = dscores
    #W2 = weights in 2nd hidden node
    dhidden = np.dot(self.params["W2"], dscores.T)
    #only send back that which are not 0 aka those that were sent forward
    mask = first_layer_out!=0
    dhidden = mask*dhidden
    #loss = softmax(W2(hidden(W1(x)+b)))
    #dl/db = sofmax*W2*hiddenmask= dhidden
    #sum along each class
    grads["b1"] = np.sum(dhidden, axis=1)#, keepdims=True)
    # dl/dw1 = dhidden*X
    grads["W1"] = np.dot(X.T, dhidden.T) + 2*reg*self.params["W1"]
    #dFbydb2 = self.params['b1']
   
#  norm_exp_score = np.divide(exp_scores,sum_exp_scores.reshape(-1,1))
 # norm_exp_score[np.arange(num_train), y] -= 1
 # dW = np.einsum("nc,nd->dc",norm_exp_score,X)
 # dW /= num_train
    #W2: Second layer weights; has shape (H, C)
    #grads["W2"] = np.einsum("->hc"dLbydF*dFbydW2)
    #grads["b2"]=0

    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False,optim='sgd'):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    if optim=="adam":
      mv_dict={}
      for param_name in self.params:
        mass = self.std * np.abs(np.random.randn(*self.params[param_name].shape))
        vel = self.std * np.abs(np.random.randn(*self.params[param_name].shape))
        mv_dict[param_name]  = [mass,vel]
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None
      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      inds = np.random.choice(range(num_train),batch_size)
      X_batch = np.array([X[i] for i in inds])
      y_batch = np.array([y[i] for i in inds])
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      for param_name in grads:
	if optim=='sgd':
	  self.params[param_name] += -learning_rate * grads[param_name]
        elif optim=='adam':
	  beta1 = 0.9
	  beta2 = 0.999
          eps = 1e-8
          dx = grads[param_name]
          m = mv_dict[param_name][0]
          v = mv_dict[param_name][1]
          m = beta1*m + (1-beta1)*dx
	  v = beta2*v + (1-beta2)*(dx**2)
          self.params[param_name] += - learning_rate * m / (np.sqrt(v) + eps)
	  if(np.any(np.isnan(learning_rate * m / (np.sqrt(v) + eps)))):
	    print(m,v)
          mv_dict[param_name][0] = m
	  mv_dict[param_name][1] = v
	else:
	  raise ValueError("Sorry optim has to be either sgd or adam")
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
	
        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    loss = self.loss(X)
    y_pred = np.argmax(loss, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


