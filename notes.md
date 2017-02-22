# Machine Learning

### Validation

1. Cross-validation. Splitting training data into `n` folds in order to validate the model's accuracy on the different
folds and average the results. The model is trained on `n - 1 ` folds each time and validated over the last fold. If the
number of validation examples is low, it is better to use cross-validation. Otherwise, single 10-50% validation split is
preferred.
![](/crossval.jpeg)

## Neural networks

### Optimizers
1. Gradient descent - first-order optimization algorithm to find a local minimum of the function. Based on taking iterative steps towards the opposite direction of the function's gradient at the current position.
2. Batch gradient descent - gradient of the cost function is calculated by averaging gradients of cost function applied to each individual training example. 
3. Stochastic gradient descent - gradient of the cost function is estimated by computing it over a relatively small sample of data. One epoch of training means that all training inputs were used (our network has seen all the training data).
4. Hessian technique - descent step is calculated by multiplying gradient on the inverted Hessian matrix of second derivatives (which contains more information about gradient). This leads to fewer steps needed to achieve local minimum. However, this requires heavy matrix calculations.
5. Momentum-based gradient descent - some sort of an accumulation of the descent is introduced. Much like the real velocity. There is also a friction factor (called momentum coefficient). 
6. Adam ??

### Activations
1. Perceptron - the activation is whether 0 or 1
2. Sigmoid - 1 / (1 + exp(-x)) [0; 1]
2. Tanch - (1+tanh(z/2))/2 [-1; 1] Allows negative activations. May perform better if model has negative inputs
3. ReLU - max(0, w⋅x+b). - Never saturates while z>0. Stops learning if z<0. Known to perform better in image classification.