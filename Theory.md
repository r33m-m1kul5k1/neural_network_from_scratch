
learning rate scheduler - changes the learning rate according to the state of the loss function. examples 
* `CosineAnnealingLR` scheduler in PyTorch follows a cosine function to decrease the learning rate from a maximum value to a minimum value over a number of cycles.
	* decay rate - how fast the learning rate will slow down
* `ReduceLROnPlateau` scheduler in PyTorch reduces the learning rate by a factor when the validation loss stops improving for a certain number of epochs. This way, the scheduler can avoid getting stuck in a plateau or a local minimum, and explore lower regions of the loss function.
optimizer 
- `SGD`
https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py
```python
def backprop(self, x, y):
	"""Returns the nabla of the baises and weights
	Notes:
	- self.weights is a matrix (num_layers, Nl+1, Nl)
	- self.baises is a vector (num_layers, Nl, 1)
	- an activation and a Z is a vector (Nl, 1)
	"""
	nabla_b = [np.zeros(b.shape) for b in self.biases]
	nabla_w = [np.zeros(w.shape) for w in self.weights]
	# feedforward
	activation = x
	activations = [x] # list to store all the activations, layer by layer
	zs = [] # list to store all the z vectors, layer by layer
	for b, w in zip(self.biases, self.weights):
		z = np.dot(w, activation)+b
		zs.append(z)
		activation = sigmoid(z)
		activations.append(activation)
	
	
	# delta is a vector (Nl, 1)
	delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) 
	nabla_b[-1] = delta
	# Nl+1x1 * 1xNl => Nl+1xNl each row (d1*pa1, d2*pa1, d3*pa1...) is the nebla to a specific neuron in the previous layer
	nabla_w[-1] = np.dot(delta, activations[-2].transpose())

	for l in xrange(2, self.num_layers):
		""" NlxNl+1 * Nl+1x1 => Nlx1 
		when transposing the weights each rwo describes weights from the same source.
		multiplied by all the deltas and summed ie: (w00*d0 + w10*d1)
												    (w01*d0 + w11*d1)
		handels the case where one previous activation value influcences multiple activation values on the next layer 
		"""
		delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(zs[-l])
		nabla_b[-l] = delta
		nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
	return (nabla_b, nabla_w)

def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives partial C_x
        partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
```
- `Adam` & `AdamW`