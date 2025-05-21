import numpy as np
from tensorflow.keras.datasets import mnist


# 激活函数 sigmoid 及其导数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # 每层偏置是列向量
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        a = a.reshape(-1, 1)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    @staticmethod
    def cost_derivative(output_activation, y):
        expected = np.zeros_like(output_activation)
        expected[y] = 1.0
        return output_activation - expected

    def backprop(self, x, y):
        x = x.reshape(-1, 1)
        y_vec = np.zeros((self.sizes[-1], 1))
        y_vec[y] = 1.0

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = (activations[-1] - y_vec) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(pred == y) for pred, y in test_results)

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                correct = self.evaluate(test_data)
                print(f"周期 {j + 1}: {correct}/{n_test} ({correct / n_test * 100:.2f}%)")
            else:
                print(f"周期 {j + 1} 完成")


def load_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    training_data = [(x.reshape(784, 1) / 255.0, y) for x, y in zip(train_images, train_labels)]
    test_data = [(x.reshape(784, 1) / 255.0, y) for x, y in zip(test_images, test_labels)]
    return training_data, test_data, test_data


if __name__ == "__main__":
    training_data, test_data, validation_data = load_mnist_data()
    net = Network([784, 30, 10])
    net.SGD(training_data, epochs=10, mini_batch_size=10, eta=1.0, test_data=test_data)

    print("在验证集上评估：")
    val_total = len(validation_data)
    val_correct = net.evaluate(validation_data)
    print(f"验证准确率：{val_correct}/{val_total} ({val_correct / val_total * 100:.2f}%)")
