# Structuring Machine Learning Projects

- [1. Orthogonalization](#1-orthogonalization)
- [2. Setting up your goal](#2-setting-up-your-goal)
  - [2.1. Single evaluation metric and satisficing metric](#21-single-evaluation-metric-and-satisficing-metric)
  - [2.2. Dev/Test set](#22-devtest-set)
- [3. Comparing to human-level performance](#3-comparing-to-human-level-performance)
- [4. Error Analysis](#4-error-analysis)
- [5. Mismatched training and dev/test set](#5-mismatched-training-and-devtest-set)
  - [5.1. Bias and variance with mismatched data distributions](#51-bias-and-variance-with-mismatched-data-distributions)
  - [5.2. Adressing data mismatch](#52-adressing-data-mismatch)
- [6. Learning from multiple tasks](#6-learning-from-multiple-tasks)
  - [6.1. Transfer learning](#61-transfer-learning)
  - [6.2. Multi-task learning](#62-multi-task-learning)
- [7. End-to-end deep learning](#7-end-to-end-deep-learning)

## 1. Orthogonalization

Identify what to tune to achieve one effect

## 2. Setting up your goal

Metrics and dev/test needs to be adjusted when your model ends up not working on real cases.

Set metrics first even if you change it later, since it will allow to iterate much more rapidly.

### 2.1. Single evaluation metric and satisficing metric

* Optimizing metric: e.g. maximize accuracy
* Satisficing metric: e.g. running time < 100ms

Generally, N metrics: 
* 1 optimizing
* N-1 satisficing

If you evaluation metric does not select the best model in your opinion, you need to find a better metric.

### 2.2. Dev/Test set

Pay attention that they have the same distribution.

Test set: might not be compulsory. Anyway chose size to allow for high enough confidence (which might differ a lot from a classic 60/20/20 split)

## 3. Comparing to human-level performance

*Bayes optimal error*: best possible error rate (that can be proxied with human-level error)
*Human-level error*: typically the best error rate achieved by human. Depending on what you are trying to achieve several values can be considered. (if you want to approximate bayes optimal error, takes the best you can find, but else you might consider taking the average human-level performance)


Progress often slows down after reaching human-level performance:
* often HL performance is close to Bayes optimal error
* you can get labeled data from humans
* gain insight from manual error analysis

**Avoidable bias**

Focus on bias or variance reduction can depend on how far you are from Bayes error (=avoidable bias).


## 4. Error Analysis

Get insights on your model's errors, and try to see if there are predominant causes that resolving would largely help reducing error. Since resolving an issue accounting for only 10% of your errors will only reduce your model error by that amount.

**Incorrectly labeled examples**

DL algorithms are quite robust to random (non systematic) errors in training set 

For dev/test set: account for it in error analysis. However as long as it does not prevent you to correctly evaluate your model it might not be the best area to focus. In any cases:
* apply same procedure to both dev and test sets
* train and dev/test may end be slightly different (and that's ok, as long as dev =~= test set)

**Build first system quickly, then iterate**

As long as you don't have a first system with a set of metrics, you can't evaluate:
* bias/variance
* errors

However if starting a system well known, it might be OK to start with a more complex system.

## 5. Mismatched training and dev/test set

Usually better to use specific data for your problem that are more close to the real problem you want to solve for you dev and test set, than mixing it with the rest and randomly split the data.

### 5.1. Bias and variance with mismatched data distributions

Keep a training-dev set, so that you have an additional dev set but with a similar distribution as the training set.

You can then analyze your error between:
* avoidable bias: difference between human level and error on examples trained on
* variance: difference between example trained on and not trained on errors
* data mismatch: difference between training-dev and dev/test error
  
|   |General speech recognition | Rearview mirror speech data |
|---|---|---|
|Human level | 4% |  6% |
|Error on examples trained on | *Training error*: 7% |  6% |
|Error on examples not trained on | *Training-dev error*: 10% | *dev/test error*: 6% |

### 5.2. Adressing data mismatch

* Manual error analysis to try to understand difference between training and dev/test
* Make them more similar:
  * artifical data synthetis (for instance add car noise to existing audio)
    * Be careful not to overfit to a small set you synthesized that does not reflect test set 

## 6. Learning from multiple tasks

### 6.1. Transfer learning

Retrain only the last layer of a network that has previously been trained on a similar problem (applicable if you have a smaller new data set).

Pre-training/fine-tuning: retrain the whole network but starting with an already trained NN on a similar problem (applicable if you have a new large dataset).

Also possible to add several new layers to replace the last layer.

### 6.2. Multi-task learning

Similar to softmax but you can a simultaneaous classes for a Y  (e.g. multiple objects recognized in a single image).

The idea is that it might perform better than multiple trained softmax algorithms.

If your data set is not entirely labeled (for instance unsure whether a sign post is in the image): the log loss only takes into account properly labeled classes when calculating error.

Makes sense:
* training on a set of task that could benefir from having shared lower-level features
* amount of data between each task is similar
* can train a big enough NN to do well on all tasks

## 7. End-to-end deep learning

Recently deep learning systems have managed to replace multiple systems.

However deep learning requires a large amount of data and may times you can have many data for smaller/simpler problems that for the full problem youa re trying to resolve.

**Key question**: Do we have enough data to learn a function of the required complexity to map x to y
