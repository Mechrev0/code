from planar_utils import load_planar_dataset, sigmoid
from testCases import *

np.random.seed(1)

X, Y = load_planar_dataset()

# print(Y.shape)
# print(np.squeeze(Y).shape)
# plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
#
# print(f'The shape of X is: {X.shape}')
# print(f'The shape of Y is: {Y.shape}')
# print(f'I have m = {X.shape[1]} training examples!')
#
# # logistic regression classifier
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T, Y.T)
#
# # Plot the decision boundary for logistic regression
# plot_decision_boundary(lambda x: clf.predict(x), X, np.squeeze(Y))
# plt.title("Logistic Regression")
#
# # Print accuracy
# LR_predictions = clf.predict(X.T)
# print('Accuracy of logistic regression: %d ' % float(
#     (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
#       '% ' + "(percentage of correctly labelled datapoints)")
#

def layer_sizes(X, Y):
    """

    :param X:input dataset of shape (input size, number of examples)
    :param Y:labels of shape (output size, number of examples)
    :return:
        n_x -- the size of the input layer
        n_h -- the size of the hidden layer
        n_y -- the size of the output layer
    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    """

    :param n_x:size of the input layer
    :param n_h:size of the hidden layer
    :param n_y:size of the output layer
    :return:
        params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return params


def forward_prpagation(X, parameters):
    """

    :param X:input data of size (n_x, m)
    :param parameters: python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    :return:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    return A2, cache


def comput_cost(A2, Y, parameters):
    """

    :param A2:he sigmoid output of the second activation, of shape (1, number of examples)
    :param Y:"true" labels vector of shape (1, number of examples)
    :param parameters: python dictionary containing your parameters W1, b1, W2 and b2
    :return:
        cost -- cross-entropy cost given equation
    """
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -1 / m * np.sum(logprobs)

    cost = np.squeeze(cost)

    assert (isinstance(cost, float))
    return cost


def backward_propagation(parameters, cache, X, Y):
    """

    :param parameters:python dictionary containing our parameters
    :param cache:a dictionary containing "Z1", "A1", "Z2" and "A2".
    :param X:input data of shape (2, number of examples)
    :param Y:"true" labels vector of shape (1, number of examples)
    :return:
        grads -- python dictionary containing your gradients with respect to different parameters

    """

    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "dW2": dW2, "db1": db1, "db2": db2}
    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    dW2 = grads['dW2']
    db1 = grads['db1']
    db2 = grads['db2']

    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    b1 -= learning_rate * db1
    b2 -= learning_rate * db2

    parameters = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}
    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        A2, cache = forward_prpagation(X, parameters)

        cost = comput_cost(A2, Y, parameters)

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads, learning_rate=1.2)

        if print_cost and i % 1000 == 0:
            print(f'cost after iteration {i}: {cost}')

    return parameters


def predict(parameters, X):
    A2, cache = forward_prpagation(X, parameters)
    predictions = (A2 > 0.5)

    return predictions

#
# parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
#
# plot_decision_boundary(lambda x: predict(parameters, x.T), X, np.squeeze(Y))
# plt.title("Decision Boundary for hidden layer size" + str(4))
#
# predictions = predict(parameters, X)
#
# print(f'Acc: {float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / Y.size * 100)}%')
#
#
# plt.figure(figsize=(16, 32))
#
# hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20]
#
# for i, n_h in enumerate(hidden_layer_sizes):
#     plt.subplot(5, 2, i+1)
#     plt.title('Hidden layer of size %d'%n_h)
#     parameters = nn_model(X, Y, n_h, num_iterations=5000)
#     plot_decision_boundary(lambda x:predict(parameters, x.T), X, np.squeeze(Y))
#     predictions = predict(parameters, X)
#     print(f'Acc: {float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / Y.size * 100)}%')
