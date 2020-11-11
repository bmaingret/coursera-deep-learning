# Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

- [1. Practical aspects of Deep Learning](#1-practical-aspects-of-deep-learning)
  - [1.1. A. Setting up your ML application](#11-a-setting-up-your-ml-application)
    - [1.1.1. Train/Dev/Test sets](#111-traindevtest-sets)
    - [1.1.2. Bias/Variance](#112-biasvariance)
    - [1.1.3. 3.Basic recipe for ML](#113-3basic-recipe-for-ml)
  - [1.2. B. Regularizing your NN](#12-b-regularizing-your-nn)
    - [1.2.1. Logistic regression regularization](#121-logistic-regression-regularization)
    - [1.2.2. Neural Network *L2* regularization](#122-neural-network-l2-regularization)
    - [1.2.3. Dropout regularization](#123-dropout-regularization)
    - [1.2.4. Other regularization methods](#124-other-regularization-methods)
  - [1.3. C. Setting up your optimization problem](#13-c-setting-up-your-optimization-problem)
    - [1.3.1. Normalizing inputs](#131-normalizing-inputs)
    - [1.3.2. Vanishing/Exploding gradients](#132-vanishingexploding-gradients)
    - [1.3.3. Weight init in DNN](#133-weight-init-in-dnn)
    - [1.3.4. Numerical approximation of gradients](#134-numerical-approximation-of-gradients)
    - [1.3.5. Gradient checking](#135-gradient-checking)
- [2. Week 2 - Optimization algorithms](#2-week-2---optimization-algorithms)
  - [2.1. A. Mini-batch gradient descent](#21-a-mini-batch-gradient-descent)
  - [2.2. B. Exponentially weighted averages](#22-b-exponentially-weighted-averages)
  - [2.3. C.Gradient descent with momentum](#23-cgradient-descent-with-momentum)
  - [2.4. D.RMSprop (Root Mean Square Propa)](#24-drmsprop-root-mean-square-propa)
  - [2.5. E. Adam optimization algorithm (ADAptive Moment estimation)](#25-e-adam-optimization-algorithm-adaptive-moment-estimation)
  - [2.6. E. Local optima in NN](#26-e-local-optima-in-nn)
  - [2.7. F. Learning rate decay](#27-f-learning-rate-decay)
- [3. Week 3 - Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization](#3-week-3---improving-deep-neural-networks-hyperparameter-tuning-regularization-and-optimization)
  - [3.1. A. Hyperparameter tuning](#31-a-hyperparameter-tuning)
    - [3.1.1. Tuning process](#311-tuning-process)
    - [3.1.2. Scale for hyperparameters](#312-scale-for-hyperparameters)
  - [3.2. B. Batch normalization](#32-b-batch-normalization)
  - [3.3. C. Multi-class classification / Softmax regression](#33-c-multi-class-classification--softmax-regression)

## 1. Practical aspects of Deep Learning

### 1.1. A. Setting up your ML application

#### 1.1.1. Train/Dev/Test sets

**Ratios**

Usually: 70/30 or 60/20/20

But with DL usually data set are larger. For instance with 1M examples, could be 98/1/1, or even 99.5/0.4/0.1


**Mismatched train/test distribution**

Make sure examples come from the same distribution. For instance issues can occur training on images taken from website versus photos from smartphone camera.

**No test sets**

Can be OK to have only a dev set

#### 1.1.2. Bias/Variance

Compare train and dev set errors, taking into account the optimal (Bayes) error.

#### 1.1.3. 3.Basic recipe for ML

**High Bias**:

* Bigger Networks
* Train longer
* (different NN architecture)

**High variance**:

* More data
* Regularization
* (different NN architecture)

In DL era, bias/variance trade-off occurs less as long as you can impact on both (for instance get more data and train longer)



### 1.2. B. Regularizing your NN

#### 1.2.1. Logistic regression regularization

Add a term depending on your parameters to the cost function J.

L2 norm -> L2 regularization

L1 norm -> L1 regularization: ends up producing compressed model with W being sparser than with L2 reg. Only helps a little bit in practice.

Usually only regularize W and not b.

It add an additional parameters to your model: lambda

#### 1.2.2. Neural Network *L2* regularization

Similar to LR, but we use the Frobenius norm (sum of squared terms of the matrix) as a regularization for the cost function.

It leads to *weight decay*: W = W - alpha * lambda/m * W.

#### 1.2.3. Dropout regularization

For each training example, selecting a subset of neurons to train.

**Implementation (Inverted dropout)**

Generate a mask matrix with appropriate randomness and *keep-probability*, and mask the A[l]

Scale up by dividing by the *keep-probability* (--> inverted dropout) since we have removed several activations, so that Z[l+1] = W[l+1]*A[l]+b[l+1] remains of the same order (and don't end up with scale issue especially with the test set).

You can have different keep-prob for different layers (especially between larger and smaller layers)

**At test time**

No drop-out. It would simply add noise/randomness

**Intuition**

Spread the weight, and as such shrink the weight similarly to L2, since we cant rely on any specific features

**Cost function**

J cost function is no longer well defined and might not got down for each iteration. For dev testing we can turn off dropout and check that the cost function is acting as normal.

#### 1.2.4. Other regularization methods

**Data augmentation**

Images:

* Flipping (as long as it still makes sense, for instance only horizontally)
* Rotation
* Distortion (for instance for OCR)

**Early stopping**

Stop the network when you get the best minimized error on your dev set.

Basically you select smaller W.

Downside: trying to solve both optimization problem of cost function and minimize overfitting with a same tool.

Upside: with only one run of cost optimization we can then select several W and test it directly versus chossing several lambda for L2 reg and rerunning all the optimization.

### 1.3. C. Setting up your optimization problem

#### 1.3.1. Normalizing inputs

Really useful when you have different scales on your input features, but usually not any harm in doing it.

#### 1.3.2. Vanishing/Exploding gradients

If we simplify a very deep network to using activation=identity, b=0, you end up with W**(L-1) and thus with either initial weight >1 or <1 we end up with exponentially large or small activations. Same issue with the gradient making learning very difficult.

#### 1.3.3. Weight init in DNN

To limit the vanishing/exploding gradient we can carefully init the weights.

Basically we want small w_i the larger the input feature space dim is for the neuron. We can set variance(w_i) = 1/ dim(input space).

If using a ReLU activation function, var(w_i[l])=2/n[l-1]. (He Initialization)

tanh : var(w_i[l]) = 1/n[l-1] (Xavier initialization)

Init W[l] = np.random.randn(shape) * np.sqrt(var(w_i[l]))


#### 1.3.4. Numerical approximation of gradients

Just use defitinion of derivative: f'(x) = lim(eps->0) ( f(x+eps) - f(x-eps) )/ (2*eps)

#### 1.3.5. Gradient checking

Can be used for debugging gradient calculation.

Concatenate all W[l] and b[l] into one matrix Theta. Do the same with dW[l] and db[l] into dTheta.

For each i, compute dTheta(i)_approx and comare to dTheta(i): ||dTheta_approx - dTheta||_2 / (||dTheta_approx||_2 + ||dTheta||_2), and check that the error is of the same order of epsilon used to compute dTheta_approx.

Else try to identify if error is larger for any specific component of dTheta.

Some notes:

* Be sure to include the regularization if used.
* Does not work with dropout, since J is not well-defined.
* Run gradient checking also after some training (algo might work well with W close to zero but not so much after).


## 2. Week 2 - Optimization algorithms

### 2.1. A. Mini-batch gradient descent

Vectorization allows fo compute on all examples at once (at least mathematically). However you update your weights only once you have processed all examples. This is called 
batch gradient descent*.

We can split training examples in smaller batches (mini-batches): X{t} (shape nx, mini-batch size), Y{t}, with t being the mini-batch. This is called *mini-batch gradient descent*. 

N.B. Shuffle training examples before.

As a result, the cost may not decrease at every iteration of mini-batch

**Choosing mini-batch size**

N.B. If small training set, just use Batch GD

* size=m: batch gradient descent
* size=1: stochastic gradient descent
* size in between: 2**(6-10) trying to have mini-batch fits in memory

### 2.2. B. Exponentially weighted averages

Definition: v_t = beta*v_t-1 + (&-beta)*x_t, would average roughly over 1/(1-beta) examples

**Bias correction**

At start, v_0 = 0, and v_1 = 0*beta + (1-beta)*x_1 -> small and wrong.

Correct it calculating: v_t / (1 - beta**t)

### 2.3. C.Gradient descent with momentum

Usually works better than batch GD.

Update W and b with weighted average of dW and dB on iteration t of mini-batch:

* V_dW = beta*V_dW + (1-beta)*dW
* V_db = beta*V_db + (1-beta)*db
  
* W = W-alpha*V_dW
* b = b-alpha*V_db

In practice we don't use bias correction for GD with momentum.

### 2.4. D.RMSprop (Root Mean Square Propa)

Calculate the exp. weighted average of the element-wise squared dW and dB, and then update:

* S_dW = beta*S_dW + (1-beta)*dW^2
* S_db = beta*S_db + (1-beta)*db^2

* W = W-alpha*dW/sqrt(S_dW)
* b = b-alpha*db/sqrt(S_db)

Allows for smoother updates, since we penalize large gradient, and to use a bit larger learnin rate.

### 2.5. E. Adam optimization algorithm (ADAptive Moment estimation)

Combine both GD with momentum and RMSprop.

* V_dW = beta_1*V_dW + (1-beta_1)*dW
* V_db = beta_1*V_db + (1-beta_1)*db
  
* S_dW = beta_2*S_dW + (1-beta_2)*dW^2
* S_db = beta_2*S_db + (1-beta_2)*db^2

Usually uses the bias corrected version of exp. weighted avg:


* V_dW = V_dW / (2-beta_1**t)
* V_db = V_db / (2-beta_1**t)
  
* S_dW = S_dW / (2-beta_2**t)
* S_db = S_db / (2-beta_2**t)
  
* W = W-alpha*V_dW/sqrt(S_dW+espilon)
* b = b-alpha*V_db/sqrt(S_db+epsilon)

epsilon being a small constant to prevent from dividing by zero

**Hyperparameters choice**

* alpha: needs to be tune
* beta_1: 0.9
* beta_2: 0.999
* epsilon: 10^-8

### 2.6. E. Local optima in NN

Local optima are not much a problem in neural network, because usually the optimization problem is a high dimensional problem, so the probability of having all dimension being convex is very low.

We find more saddle point than local optima.


However **plateaus** can slow down learning.

### 2.7. F. Learning rate decay

**1 epoch**: 1 pass through data (all mini-batches)

Learning rate decay: alpha = 1/(1 + decay_rate * epoch_num)
Exponential decay: alpha = 0.95**epoch_num * alpha_0
Others: alpha = k / sqrt(epoch_num) * alpha_0, or k / sqrt(t (mini-batch)) * alpha_0
Discrete staircase
Manual decay: just set it manually

## 3. Week 3 - Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

### 3.1. A. Hyperparameter tuning

#### 3.1.1. Tuning process

First:
* learning rate

* beta for momentum GD
* #hidden units
* #mini-batch size

Third:
* #layers

**Don't use a grid but try random values**: allows to try more values for more important hyperparameters.

**Coarse to fine**: Progressively focus on smaller hyper-parameter value space

#### 3.1.2. Scale for hyperparameters

Pay attention to the log of the hyperparameters.

For instance for learning rate, it makes more sense to search over a log scale. 
It can also be used when searching beta between 0.9 and 0.999 by searching over the log space of 1-beta.

### 3.2. B. Batch normalization

In deeper network we might want to normalize activation outputs so ease learning of the weights of the innner layers.

Normalization can be done on Z or on A. Here we look at normalizing Z.

<img src="https://render.githubusercontent.com/render/math?math=z_{norm}^{(i)}=\frac{z^{(i)}-\mu}{\sqrt{\sigma^{2}%2B\epsilon}}">

To be able to chose what you Z distributions are (for instance depending on the activation function), we can use:

<img src="https://render.githubusercontent.com/render/math?math=\tilde{z}^{(i)}=\gamma*z_{norm}^{(i)}%2B\beta">

This add two additional parameters per layer, <img src="https://render.githubusercontent.com/render/math?math=\gamma^{[l]}"> and <img src="https://render.githubusercontent.com/render/math?math=\beta^{[l]}">, that needs to pe upgraded during backprop (works with GD, GD w/ momentum, Adam).

When using batch norm, the b parameters get cancelled out, and get sort of replaced by beta.

**Covariate shift**

If the function to learn stay the sam, but that the data may no be representative of the whole feature space, and ends up shifting inside, the learn parameters might not do as well.

This can also applies inside of the NN, where when we update the weight or the previous layer, it may shift the distribution of the input for the following layer. Batch normalization reduces this shift, and sort of reducing the coupling between layers.

**Regularization effect (as a remark)**

Since mean/variance are computed only on mini-batches, it adds some noise to the values and acts a bit like some (small) regularization (depending on the batch size).

**Batch normalization at test time**

We may have only one example at test time and thus not a complete mini-batch to compute mean and variance.

We use an estimate: exp. weighted average across mini-batches


### 3.3. C. Multi-class classification / Softmax regression

Softmax activation function:
1. <img src="https://render.githubusercontent.com/render/math?math=t=e^{z^{[L]}}">
2. <img src="https://render.githubusercontent.com/render/math?math=a^{L}=\frac{t}{\sum_{j=i}^{C}t_{j}}">

With C being the number of classes (size of the output layer L)

N.B. this softmax version takes a vector and outputs a same shape vector.

*softmax* opposes to *hard max* that would put 1 for the maximum and 0 for the rest.

If C=2, softmax reduces to logistic regression.

**Loss function**

<img src="https://render.githubusercontent.com/render/math?math=$L(\hat{y}, y) = - \sum_{j=1}^{C} y_j*log(\hat{y}_j)">

Init of backpropagation: <img src="https://render.githubusercontent.com/render/math?math=dz^{[L]}=\hat y-y">
