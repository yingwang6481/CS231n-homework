import numpy as np
from random import shuffle

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
  num_train=X.shape[0]
  num_class=W.shape[1]
  for i in range(num_train):
    scores=np.dot(X[i],W)
    shift_scores=scores-np.max(scores)
    exp_scores=np.exp(shift_scores)
    exp_sum=np.sum(exp_scores)
    Li=-shift_scores[y[i]]+np.log(exp_sum)

    loss +=Li


    for j in range(num_class):
      if j==y[i]:
        ratio = exp_scores[y[i]]/exp_sum
        dW[:,j] += -X[i].T + ratio * X[i].T
      else:
        ratio = exp_scores[j]/exp_sum
        dW[:,j] += ratio * X[i].T
  dW /= num_train

  dW += reg * W

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  shift_socres = scores - np.max(scores)
  sy=shift_socres[np.arange(num_train),y]
  sy = np.reshape(sy,(num_train,1))
  sigma=np.sum(np.exp(shift_socres),axis=1,keepdims=True)
  loss=np.sum(-sy+np.log(sigma))
  loss /=num_train
  loss += 0.5 * reg * np.sum(W*W)

  temp = np.exp(shift_socres)/sigma
  temp[np.arange(num_train),y] -= 1
  dW=np.dot(X.T,temp)/num_train + reg * W



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

