import numpy as np
from numpy import random
from tqdm import trange

np.random.seed(1337)


# load the mnist dataset
def fetch(url):
    import requests, gzip, os, hashlib, numpy
    fp = os.path.join("C:\\temp", hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)
    return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()


def softmax_grad(X):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = X.reshape(-1, 1)
    return np.diagflat(X) - np.dot(s, s.T)


def cross_entropy_loss(predictions_Y, train_Y):
    predictions_count = predictions_Y.shape[0]
    scores = predictions_Y[range(predictions_count), train_Y]
    with np.errstate(divide='ignore'):
        log_likelihood = -np.where(scores > 0.0000001, np.log(scores), np.log(np.full(scores.shape, 0.00001)))
        log_likelihood[np.isneginf(log_likelihood)] = -1e9
    return np.sum(log_likelihood) / predictions_count


class NNLayer(object):
    def __init__(self, size, activation_func='relu'):
        # self.W = np.random.randn(size[0], size[1]) * math.sqrt(2.0 / size[1])
        self.W = np.random.uniform(-1.0, 1.0, size) / np.sqrt(size[0] * size[1]).astype(np.float32)
        self.bias = np.random.random(size[0]) * 0.01
        self.activation_func = activation_func
        self.X = np.zeros(size[1])
        self.Z = np.zeros(size[0])
        self.Sigma = np.zeros(size[0])
        self.dL_dW = np.zeros(size)
        self.dL_dB = np.zeros(size[0])

    def forward(self, X):
        self.X = X
        self.Z = np.dot(self.W, self.X) + self.bias
        self.Sigma = self.apply_activation(self.Z)
        return self.Sigma

    def apply_activation(self, Z):
        if self.activation_func == 'relu':
            return np.maximum(0.0, Z)
        elif self.activation_func == 'cross_entropy_softmax':
            exps = np.exp(Z - np.max(Z))
            return exps / np.sum(exps)
        else:
            return 0.0

    def backward(self, dL_dSigmas):
        # This is a batch operation that is done for all the neurons in the current layer
        dZ_dX = self.W
        dZ_dW = self.X
        dZ_dB = np.ones(self.bias.shape)

        if self.activation_func == 'cross_entropy_softmax':
            # dL_dSigmas actually refers to the derivative of the loss function wrt. the softmax input which is Z
            # So it is actually dL_dZ rather than dL_dSigmas
            dL_dZ = dL_dSigmas
        elif self.activation_func == 'relu':
            dL_dZ = dL_dSigmas
            dL_dZ[self.Z <= 0] = 0.0
        else:
            raise NameError('Unsupported activation function ' + self.activation_func)

        dL_dW = np.multiply(dL_dZ[:, np.newaxis], dZ_dW)
        # dL_dB = dL_dZ * dZ_dB.sum(axis=0)
        dL_dB = dL_dZ * dZ_dB
        dL_dX = np.dot(dL_dZ, dZ_dX)

        assert dL_dX.shape == self.X.shape
        assert self.dL_dW.shape == dL_dW.shape
        assert self.dL_dB.shape == dL_dB.shape
        self.dL_dW += dL_dW
        self.dL_dB += dL_dB
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

    def __init__(self, training_X, training_Y, input_output):
        self.layers = []
        self.training_X = training_X
        self.training_Y = training_Y
        self.input_output = input_output

    def addLayer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def train(self, max_epoch, learning_rate, batch_size):
        loss = 0.0
        t = trange(max_epoch)
        for i in t:
            # Create batch
            sample = np.random.randint(0, self.training_X.shape[0], size=batch_size)
            batch_train_X = self.training_X[sample].reshape((-1, 28 * 28))
            batch_train_Y = self.training_Y[sample]

            predictions_Y = np.zeros((batch_train_Y.shape[0], self.input_output[1]), dtype=np.float32)
            for input_index, Xtr in enumerate(batch_train_X):
                predictions_Y[input_index] = self.forward(Xtr)

                # Backpropagation
                # Start with the derivative of the Loss function wrt. the softmax function's input
                dL_dZ = predictions_Y[input_index].copy()
                dL_dZ[batch_train_Y[input_index]] -= 1.0

                for layer in reversed(self.layers):
                    dL_dZ = layer.backward(dL_dZ)

            # Update the weights
            for layer in self.layers:
                layer.update(learning_rate)

            loss += cross_entropy_loss(predictions_Y, batch_train_Y)
            accuracy = np.mean(np.argmax(predictions_Y, axis=1) == batch_train_Y)
            t.set_description(
                'Epoch %d / %d, Loss = %f, Accuracy = %.2f' % (i + 1, max_epoch, loss / (i + 1), accuracy.mean()))

    def get_neuron_gradient_vector(self, layer_index, neuron_index):
        return self.layers[layer_index].get_neuron_gradient_vector(neuron_index)

    # def compute_analytical_gradient(self, loss, layer_index, neuron_index, delta):
    #     print('Base Loss value: ', loss)
    #
    #     dW = np.zeros(self.layers[layer_index].W.shape)
    #     dL_dW = np.zeros(self.layers[layer_index].W.shape[1])
    #
    #     # Given a selected neuron, we are differentiating wrt. each of its input connections
    #     for input_connection_index in range(self.layers[layer_index].W.shape[1]):
    #         dW[neuron_index][input_connection_index] = delta
    #
    #         # TODO fixme: This code is broken
    #         predictions_Y = np.zeros(self.training_Y.shape, dtype=float)
    #         for input_index, Xtr in enumerate(self.training_X):
    #             predictions_Y[input_index] = self.forward(Xtr, dW, layer_index)
    #
    #         delta_loss = self.cross_entropy_loss(predictions_Y) - loss
    #         dL_dW[input_connection_index] = delta_loss / (delta * self.training_X.shape[0])
    #
    #         dW[neuron_index][input_connection_index] = 0.0
    #
    #     return dL_dW


def create_neural_network():
    X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28 * 28)) / 255.0
    Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28 * 28)) / 255.0
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    inputShapeSize = 784
    outputShapeSize = 10
    fcnn = VanillaFCNN(X_train, Y_train, (inputShapeSize, outputShapeSize))
    fcnn.addLayer(NNLayer((128, inputShapeSize), 'relu'))
    fcnn.addLayer(NNLayer((outputShapeSize, 128), 'cross_entropy_softmax'))
    return fcnn


network_parameters = {'delta': 0.001, 'epochs': 1000, 'batch_size': 128}

fcnn = create_neural_network()
fcnn.train(network_parameters['epochs'], network_parameters['delta'], network_parameters['batch_size'])

# analytical_gradient = fcnn.compute_analytical_gradient(loss, 1, 0, delta)
# print('Analytical gradient = {}, vector size = {}:\n '.format(analytical_gradient, analytical_gradient.shape))
# backprop_gradient = fcnn.get_neuron_gradient_vector(1, 0)
# print('Gradient of the Loss computed using backprop = {}, vector size = {}\n '.format(backprop_gradient, backprop_gradient.shape))
