# Overview

This project contains the implementation of a dense neural net. 

# Installation

Open a shell and paste the following lines:

```bash
git clone https://gitlab2.informatik.uni-wuerzburg.de/s353743/neuralnet
cd neuralnet
pip install -r requirements.txt
```

# Usage

## Creation of the neural net

To create a neural net, a list of `layer` and a `lossLayer` is required. These are taken by the constructor of the neural net.

```python
from neuralnet import *

denseLayer = Dense(None, 16, 8)
sigmoid = Sigmoid()
lossLayer = CategoricalCrossEntropy()
neuralNet = NeuralNet([denseLayer, sigmoid], lossLayer)
```

The order of the layer in the list is important! For normal a activation layer (sigmoid, tanh) should follow a normal, dense layer. As you can see, the Dense Layer takes three parameters. The second one describes the number of input neurons (16), while the third one describes the number of output neurons (8). The corresponding weights are set randomly.

---

The second parameter (set to 'None' in this instance) will be described later on (subsection `Optimizer`).

It is also possible to initialize a dense layer by directly setting the weights. In this case the second parameter describes the weights, while the third parameter sets the bias values of the layer. 

```python
weights = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([0.5, 0.5])

denseLayer = Dense(None, weights, bias)
```

## Training 

After the neural net was created it can be trained, by using the function `train`. This function includes both function `forward` and function `backward`, which also can be calld independently.

```python
train(	x, 		# input data -> np.ndarray
        y, 		# ground truth
        lr,		# learning rate
        
        amountTestData = 0.1, 	        # how much data from the given data(x) shall be used to test the performance
        batch_size = -1, 		# size of batches. If set to -1 all data is gathered in one batch
        iterations = 1, 		# amount of iteration over the whole data
        shuffleData = True,		# shuffling data
        
        storePerformanceHistory = False,	        # stores information about performance 
                                                        # (F1, Accuracy, Precision, Recall)
        gatherPerformanceHistoryPerBatch = False,	# stores performance information after each batch
                                                        # else information is stored after each iteration
        printPerformanceStats = False			# if set to 'True', prints 'Accuracy' and 'Loss' into 
                                                        # the command line
        )
```

## Performance evaluation

The function `train` returns a dictionary of performance values. This dictionary contains following keys and the corresponding values (if `storePerformanceHistory = True`):

```python
'loss' -> loss values gathered per iteration or per batch
'accuracy' -> overall accuracy of the model
'f1' -> f1 score of the indiviual ground_truths
'recall' -> recall of the individual ground_truths
'precision' -> precision of the individual ground_truths 
```

If you'd like to visualize the gathered data, you can use the build-in function `visualizeHistory`. 
For example:

```python
visualizeHistory(performanceValues["precision"], "precision", labels)
visualizeHistory(performanceValues["recall"], "recall", labels)
visualizeHistory(performanceValues["f1"], "f1", labels)
...
```

This creates diagrams by using matplotlib. Any value of `performanceValues` can be displayed by using this format.

## Optimizer

Optimizer allow for a better training performance. A dense layer takes an optimizer for its first parameter. 
Lets take a look at the optimizer `Adam`.

```python
adam = Adam(0.1, 0.2, 0.01)
```

The first and second parameter describe the variables $$\beta_1$$ and $$\beta_2$$ from the corresponding formular from the site 
https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c.
The third value correspons to the parameter $$\lambda$$ of the `L2-Regularization` 
(https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261).  
If no optimizer was set, the normal gradient descent will be used.

# Examples

A complete example of MNIST data classification can be found here: https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/lectures/machine-learning/student-material/ws20/Team-01/neuralnet/-/tree/master/example
