import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    correct_class_score = scores[y[i]]
    sum_exp_correct_score = np.exp(correct_class_score)
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores)
    loss += -correct_class_score + np.log(sum_exp_scores)
    
    #W/dW is D*C and X is N*D. we are removing X[i] from the weight of the 
    #correct class 
    # for all the classes, we add the
    for c in range(num_classes):
	dW[:,c] += X[i]*exp_scores[c]/sum_exp_scores 
	if c==y[i]:
	  dW[:,c] -= X[i]
#   Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W


  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #W/dW is D*C and X is N*D. 
  scores = np.einsum("dc,nd->nc",W,X)
  
  #subtract row max from each row
  scores -= np.max(scores, axis=1, keepdims=True)
  #calculate the exp of the score for the correct class 
  correct_class_score = scores[np.arange(num_train),y]
  sum_exp_correct_score = np.exp(correct_class_score)
  #calculate all the exponentials for all n-examples* c-classes 
  exp_scores = np.exp(scores)
  #sum the exp along each example 
  sum_exp_scores = np.sum(exp_scores,axis=1)
  #add everything together 
  loss = -np.sum(correct_class_score) + np.sum(np.log(sum_exp_scores))
  
  loss /= num_train
  loss +=  0.5* reg * np.sum(W * W)
  
  #dW is D*C and X is N*D.
  norm_exp_score = np.divide(exp_scores,sum_exp_scores.reshape(-1,1))
  norm_exp_score[np.arange(num_train), y] -= 1
  dW = np.einsum("nc,nd->dc",norm_exp_score,X)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

