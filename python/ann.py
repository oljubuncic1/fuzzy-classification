import numpy as np
import math
import random
import logging

logging.basicConfig(level=logging.DEBUG)

def sig(x):
	return 1 / (1 + math.exp(-x))

def sig_prim(x):
	s = sig(x)
	return s * (1 - s)

def forward(net, x):
	weights = net[0]
	biases = net[1]

	layer_outputs = [x]

	y = x
	i = 0
	for w in weights:
		v_sig = np.vectorize(sig)
		y = v_sig(np.dot(w, y) + biases[i])
		layer_outputs.append(np.transpose(y))
		i += 1

	return layer_outputs

def backpropagate(net, x, y):
	alpha = 0.1

	layer_outputs = forward(net, x)

	weights = net[0]
	biases = net[1]

	L = len(weights)
	delta = [np.ndarray((b.shape)) for b in biases]

	v_sig = np.vectorize(sig)
	v_sig_prim = np.vectorize(sig_prim)
	delta[L - 1] = (layer_outputs[-1] - y) * v_sig_prim( np.dot(weights[L - 1], layer_outputs[-2]) )

	for l in reversed(range(L - 1)):
		delta[l] = (
			np.dot( np.transpose(weights[l + 1]), layer_outputs[l + 2] ) * 
				v_sig_prim( np.dot(weights[l], layer_outputs[l]) )
		)

	# update weights
	for l in range(L):
		w = weights[l]
		for j in range(w.shape[0]):
			for k in range(w.shape[1]):
				w[j, k] = w[j, k] - alpha * layer_outputs[l][k] * delta[l][j]

	for l in range(L):
		biases[l] = biases[l] - alpha * delta[l]

	return [ weights, biases ]

def ann(layer_desc):
	scaling_factor = 1

	weights = []
	biases = []

	for i in range(len(layer_desc) - 1):
		curr_l = layer_desc[i]
		next_l = layer_desc[i + 1]

		weights.append(
			scaling_factor * np.random.rand(next_l, curr_l)
		)
		biases.append( np.array([scaling_factor * random.random() for i in  range(next_l)]) )

	return [weights, biases]

from inspect import ismethod

class Test:

	def test_net(self):
		n = ann([3, 2, 4])

		self.should_eq("weigts and biases", len(n), 2)
		
		weights = n[0]

		self.should_eq("layer count", len(weights), 2)
		self.should_eq("first layer shape", weights[0].shape, (2, 3))
		self.should_eq("second layer shape", weights[1].shape, (4, 2))

		biases = n[1]
		self.should_eq("biases size", len(biases), 2)
		self.should_eq("first layer biases", len(biases[0]), 2)
		self.should_eq("second layer biases", len(biases[1]), 4)

	def test_forward(self):
		weights = [
			np.array( [ [1 ,1], [1, 1], [1, 1] ] ),
			np.array( [ [1 ,1, 1], [1, 1, 1] ] )
		]

		biases = [
			np.transpose( np.array([1, 1, 1]) ),
			np.transpose( np.array([1, 1]) ),
		]

		net = [weights, biases]

		x = np.transpose( [1, 1] )

		layer_outputs = forward(net, x)
		y = layer_outputs[-1]

		self.should_eq("output one", round(y[0], 2), 0.98)
		self.should_eq("output two", round(y[1], 2), 0.98)

	def test_backpropagate(self):
		weights = [
			np.array( [ [1.0 ,1.0], [1.0, 1.0], [1.0, 1.0] ] ),
			np.array( [ [1.0, 1.0, 1.0], [1.0, 1.0, 1.0] ] )
		]

		biases = [
			np.transpose( np.array([1, 1, 1]) ),
			np.transpose( np.array([1, 1]) ),
		]

		net = [weights, biases]
		x = np.transpose( [1, 1] )
		y = np.array( [1, 0] )

		backpropagate(net, x, y)

	def should_eq(self, name, val1, val2):
		BEGIN_PASS = '\033[92m'
		BEGIN_FAIL = '\033[91m'
		END = '\033[0m'

		if val1 == val2:
			print(
				"\t{" + name + "}" + BEGIN_PASS + " pass --> " + 
				str(val1) + " = " + str(val2) + END
			)
		else:
			print(
				"\t{" + name + "}" + BEGIN_FAIL + " fail --> " + 
				str(val1) + " != " + str(val2) + END
			)

	def main(self):
		for name in dir(self):
			attribute = getattr(self, name)
			if ismethod(attribute) and attribute.__name__.startswith('test'):
				print(attribute.__name__)
				attribute()

		print("")
		print("")
		print("")

def test():
	test_suite = Test()
	test_suite.main()
