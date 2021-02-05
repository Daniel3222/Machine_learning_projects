import pickle
import numpy as np

#datapath='cifar10.pkl',
#datapath = 'D:/PycharmProjects/HW3/svhn.pkl',
class NN(object):
    def __init__(self,
                 hidden_dims=(512, 120,120,120,120,120,120),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=0.03,
                 batch_size=100,
                 seed=0,
                 activation="relu",
                 init_method="glorot",
                 normalization=False
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            self.train = np.load(datapath)
            self.valid = np.load(datapath)
            self.test = np.load(datapath)
            if normalization:
                self.normalize()
        else:
            self.train, self.valid, self.test = None, None, None

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            # initialize bias
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))
            # weights
            bound = np.sqrt(6.0 / (all_dims[layer_n-1] + all_dims[layer_n]))
            self.weights[f"W{layer_n}"] = np.random.uniform(-bound, bound, size=(all_dims[layer_n-1], all_dims[layer_n]))

    def relu(self, x, grad=False):
        if grad:
            # WRITE CODE HERE
            return np.greater(x, 0).astype(float)
        # WRITE CODE HERE
        else:
            return np.maximum(0,x)

    def sigmoid(self, x, grad=False):
        if grad:
            # WRITE CODE HERE
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
            # WRITE CODE HERE
        else:
            return 1/(1+np.exp(-x))

    def tanh(self, x, grad=False):
        if grad:
            # WRITE CODE HERE
            t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
            return 1 - t ** 2
        # WRITE CODE HERE
        else:
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def leakyrelu(self, x, grad=False):
        alpha = 0.01
        if grad:
            # WRITE CODE HERE
            dx = np.ones_like(x)
            dx[x < 0] = alpha
            return dx
        else:
        # WRITE CODE HERE
            leaky_way1 = np.where(x > 0, x, x * 0.01)
            return leaky_way1

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            return self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            return self.tanh(x, grad)
        elif self.activation_str == "leakyrelu":
            return self.leakyrelu(x, grad)
        else:
            raise Exception("invalid")

     # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    def softmax(self, x):
        z = x - np.max(x, axis=-1, keepdims=True)
        numerator = np.exp(z)
        denominator = np.sum(numerator, axis=-1, keepdims=True)
        softmax = numerator / denominator
        return softmax

    def forward(self, x):
        cache = {"Z0": x}
        # WRITE CODE HERE

        for layer_n in range(1,self.n_hidden + 2):
            if layer_n != self.n_hidden + 1:
                cache[f"A{layer_n}"] = np.matmul(cache[f"Z{layer_n-1}"], self.weights[f"W{layer_n}"]) + self.weights[f"b{layer_n}"]
                cache[f"Z{layer_n}"] = self.activation(cache[f"A{layer_n}"])
            else :
                cache[f"A{layer_n}"] = np.matmul(cache[f"Z{layer_n - 1}"], self.weights[f"W{layer_n}"]) + self.weights[
                    f"b{layer_n}"]
                cache[f"Z{layer_n}"] = self.softmax(cache[f"A{layer_n}"])
        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = dict()
        grads[f"dA{self.n_hidden + 1}"] = output-labels
        for layer_n in range(self.n_hidden + 1, 0, -1):
            # ici calculer gradients of the cost
            grads[f"dW{layer_n}"] = (np.matmul(cache[f"Z{layer_n - 1}"].T, grads[f"dA{layer_n}"])) / self.batch_size
            grads[f"db{layer_n}"] = np.sum(grads[f"dA{layer_n}"], axis=0, keepdims=True) / self.batch_size
            if layer_n > 1:  # ici calculer gradients of the loss
                grads[f"dZ{layer_n - 1}"] = np.matmul(grads[f"dA{layer_n}"], self.weights[f"W{layer_n}"].T)
                grads[f"dA{layer_n - 1}"] = np.multiply(grads[f"dZ{layer_n - 1}"], self.activation(cache[f"A{layer_n - 1}"], grad=True))
        return grads


    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            self.weights[f"W{layer}"] -= grads[f"dW{layer}"] * self.lr
            self.weights[f"b{layer}"] -= grads[f"db{layer}"] * self.lr


    def one_hot(self, y):
        return np.eye(self.n_classes)[y]

    # https://stackoverflow.com/questions/49473587/why-is-my-implementations-of-the-log-loss-or-cross-entropy-not-producing-the-s
    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        N = prediction.shape[0]
        ce = -np.sum(labels * np.log(prediction)) / N
        return ce

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train['arr_0'], self.train['arr_1']

        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                # WRITE CODE HERE
                self.update(self.backward(self.forward(minibatchX), minibatchY))


            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)
        print(self.train_logs)
        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        # WRITE CODE HERE
        return self.compute_loss_and_accuracy(X_test,y_test)[0], self.compute_loss_and_accuracy(X_test,y_test)[1]


    def normalize(self):
        mean = np.mean(self.train[0], axis=(0))
        std = np.std(self.train[0], axis=(0))
        self.train = ((self.train[0] - mean) / std, self.train[1])
        self.valid = ((self.valid[0] - mean) / std, self.valid[1])
        self.test = ((self.test[0] - mean) / std, self.test[1])


NeuronN = NN(datapath = 'D:/PycharmProjects/data_comp_2/ift3395-6390-quickdraw/train.npz', normalization=False)
NeuronN.train_loop(30)

