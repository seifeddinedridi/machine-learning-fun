import math

import numpy as np
from numpy import random

np.random.seed(1337)


# load the mnist dataset
def fetch(url):
    import requests, gzip, os, numpy
    fp = os.path.join(os.getcwd(), "tmp", url)
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)
    return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()


class NNLayer(object):
    def __init__(self, size, activation_func='relu'):
        # self.W = np.random.randn(size[0], size[1]) * math.sqrt(2.0 / size[1])
        self.W = np.random.uniform(-1.0, 1.0, size) / np.sqrt(size[0] * size[1]).astype(np.float32)
        self.bias = np.zeros(size[0])
        self.activation_func = activation_func
        self.X = np.zeros(size[1])
        self.Z = np.zeros(size[0])
        self.Sigma = np.zeros(size[0])
        self.dL_dW = np.zeros(size)
        self.dL_dB = np.zeros(size[0])

    def forward(self, X, delta_weight=None):
        self.X = X
        weights = self.W
        if delta_weight is not None:
            weights += delta_weight
        self.Z = np.dot(weights, self.X) + self.bias
        self.Sigma = self.apply_activation(self.Z)
        return self.Sigma

    def apply_activation(self, Z):
        if self.activation_func == 'relu':
            return np.maximum(0.0, Z)
        elif self.activation_func == 'softmax':
            exps = np.exp(Z - np.max(Z))
            return exps / np.sum(exps)
        else:
            return 0.0

    def softmax_grad(self, X):
        # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
        s = X.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    def backward(self, dL_dSigmas):
        # This is a batch operation that is done for all the neurons in the current layer
        dL_dX = np.zeros(self.X.shape[0])

        dZ_dX = self.W
        dZ_dW = self.X
        dZ_dB = np.ones(self.bias.shape)

        dSigma_dZ = []
        if self.activation_func == 'relu':
            # Todo (Check correctness)
            dSigma_dZ = np.diag(np.heaviside(self.Z, 1.0))
        elif self.activation_func == 'softmax':
            # Todo (Check correctness)
            dSigma_dZ = self.softmax_grad(self.Sigma)

        dSigma_dX = np.dot(dSigma_dZ, dZ_dX)
        dSigma_dW = np.dot(dSigma_dZ, np.vstack([dZ_dW] * self.W.shape[0]))
        dSigma_dB = np.dot(dSigma_dZ, dZ_dB)

        dL_dX = np.dot(dL_dSigmas, dSigma_dX)
        dL_dW = np.multiply(dL_dSigmas[:, np.newaxis], dSigma_dW)
        self.dL_dW += dL_dW
        self.dL_dB += np.dot(dL_dSigmas, dSigma_dB)

        # for neuron_index, dL_dSigma in enumerate(dL_dSigmas):
        #     dL_dXi = dL_dSigma * dSigma_dX[neuron_index]
        #     dL_dWi = dL_dSigma * dSigma_dW[neuron_index]
        #     dL_dX += dL_dXi
        #
        #     self.dL_dW[neuron_index] += dL_dWi
        #     self.dL_dB[neuron_index] += dL_dSigma * dSigma_dB[neuron_index]

        return dL_dX

    def update(self, learning_rate):
        self.W -= self.dL_dW * learning_rate
        self.bias -= self.dL_dB * learning_rate
        # Reinitialize accumulated derivatives `dL_dW` and `dL_dB`
        self.dL_dW = np.zeros(self.dL_dW.shape)
        self.dL_dB = np.zeros(self.dL_dB.shape)

    def get_neuron_gradient_vector(self, neuron_index):
        return self.dL_dW[neuron_index]


class VanillaFCNN(object):

    def __init__(self, training_X, training_Y):
        self.layers = []
        self.training_X = training_X
        self.training_Y = training_Y

    def addLayer(self, layer):
        self.layers.append(layer)

    def forward(self, X, delta_weight=None, target_layer_idx=-1):
        for layer_idx, layer in enumerate(self.layers):
            if target_layer_idx == layer_idx:
                X = layer.forward(X, delta_weight)
            else:
                X = layer.forward(X)
        return X

    def cross_entropy_loss(self, predictions_Y):
        predictions_count = predictions_Y.shape[0]
        l = predictions_Y[range(predictions_count), self.training_Y.argmax(axis=1)]
        with np.errstate(divide='ignore'):
            log_likelihood = -np.where(l > 0.0000001, np.log(l), np.log(np.full(l.shape, 0.00001)))
            log_likelihood[np.isneginf(log_likelihood)] = -1e9
        return np.sum(log_likelihood) / predictions_count

    def train(self, epochs=1, learning_rate=0.001):
        loss = 0.0
        for t in range(epochs):
            predictions_Y = np.zeros(self.training_Y.shape, dtype=float)
            for input_index, Xtr in enumerate(self.training_X):
                predictions_Y[input_index] = self.forward(Xtr)

                # Backpropagation
                # Start with the derivative of the Loss function wrt. last layer's output
                dL_dX = predictions_Y[input_index] - self.training_Y[input_index]
                for layer in reversed(self.layers):
                    dL_dX = layer.backward(dL_dX)

            # Update the weights
            for layer in self.layers:
                layer.update(learning_rate)

            loss += self.cross_entropy_loss(predictions_Y)
            accuracy = np.mean(np.argmax(predictions_Y, axis=1) == np.argmax(self.training_Y, axis=1))
            print('Epoch %d / %d, Loss = %f, Accuracy = %.2f' % (t + 1, epochs, loss / (t + 1), accuracy.mean()))

        return loss / epochs

    def get_neuron_gradient_vector(self, layer_index, neuron_index):
        return self.layers[layer_index].get_neuron_gradient_vector(neuron_index)

    def compute_analytical_gradient(self, loss, layer_index, neuron_index, delta):
        print('Base Loss value: ', loss)

        dW = np.zeros(self.layers[layer_index].W.shape)
        dL_dW = np.zeros(self.layers[layer_index].W.shape[1])

        # Given a selected neuron, we are differentiating wrt. each of its input connections
        for input_connection_index in range(self.layers[layer_index].W.shape[1]):
            dW[neuron_index][input_connection_index] = delta

            predictions_Y = np.zeros(self.training_Y.shape, dtype=float)
            for input_index, Xtr in enumerate(self.training_X):
                predictions_Y[input_index] = self.forward(Xtr, dW, layer_index)

            delta_loss = self.cross_entropy_loss(predictions_Y) - loss
            dL_dW[input_connection_index] = delta_loss / (delta * self.training_X.shape[0])

            dW[neuron_index][input_connection_index] = 0.0

        return dL_dW


def generateTrainingDataset(inputSize, outputSize, datasetSize):
    xTrs = np.random.random((datasetSize, inputSize))
    yTrs = np.zeros((datasetSize, outputSize))
    batch_size = int(datasetSize / 3)
    for y in range(datasetSize):
        idx = random.choice(range(outputSize))
        yTrs[y, idx % outputSize] = 1.0
        # yTrs[y, 0] = 1.0
    return xTrs, yTrs


def create_neural_network():
    inputShapeSize = 784
    outputShapeSize = 10

    # Generate training dataset
    trainingDatasetSize = 100
    # X_train, Y_train = generateTrainingDataset(inputShapeSize, outputShapeSize, trainingDatasetSize)

    X_train = fetch("train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 784))[0:trainingDatasetSize, ]
    Y_train = fetch("train-labels-idx1-ubyte.gz")[8:][0:trainingDatasetSize]
    X_test = fetch("t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 784))
    Y_test = fetch("t10k-labels-idx1-ubyte.gz")[8:]

    # Convert Y_train and Y_test into hot vectors
    Y_train_1hv = np.zeros((Y_train.size, Y_train.max() + 1))
    Y_train_1hv[np.arange(Y_train.size), Y_train] = 1

    fcnn = VanillaFCNN(X_train, Y_train_1hv)
    fcnn.addLayer(NNLayer((128, inputShapeSize), 'relu'))
    fcnn.addLayer(NNLayer((outputShapeSize, 128), 'softmax'))
    return fcnn

fcnn = create_neural_network()
delta = 0.001
epochs = 1000
loss = fcnn.train(epochs, delta)
print('Estimated loss = %f' % loss)

# analytical_gradient = fcnn.compute_analytical_gradient(loss, 1, 0, delta)
# print('Analytical gradient = {}, vector size = {}:\n '.format(analytical_gradient, analytical_gradient.shape))
# backprop_gradient = fcnn.get_neuron_gradient_vector(1, 0)
# print('Gradient of the Loss computed using backprop = {}, vector size = {}\n '.format(backprop_gradient, backprop_gradient.shape))
