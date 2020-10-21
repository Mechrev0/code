# import matplotlib.pyplot as plt
# import sklearn.linear_model

# from planar_utils import load_planar_dataset, plot_decision_boundary, sigmoid
# from testCases import *

# np.random.seed(1)

# X, Y = load_planar_dataset()

# # print(Y.shape)
# # print(np.squeeze(Y).shape)
# plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)

# print(f'The shape of X is: {X.shape}')
# print(f'The shape of Y is: {Y.shape}')
# print(f'I have m = {X.shape[1]} training examples!')

# # logistic regression classifier
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T, Y.T)

# # Plot the decision boundary for logistic regression
# plot_decision_boundary(lambda x: clf.predict(x), X, np.squeeze(Y))
# plt.title("Logistic Regression")

# # Print accuracy
# LR_predictions = clf.predict(X.T)
# print('Accuracy of logistic regression: %d ' % float(
#     (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
#       '% ' + "(percentage of correctly labelled datapoints)")


# def layer_sizes(X, Y):
#     """

#     :param X:input dataset of shape (input size, number of examples)
#     :param Y:labels of shape (output size, number of examples)
#     :return:
#         n_x -- the size of the input layer
#         n_h -- the size of the hidden layer
#         n_y -- the size of the output layer
#     """
#     n_x = X.shape[0]
#     n_h = 4
#     n_y = Y.shape[0]
#     return n_x, n_h, n_y


# def initialize_parameters(n_x, n_h, n_y):
#     """

#     :param n_x:size of the input layer
#     :param n_h:size of the hidden layer
#     :param n_y:size of the output layer
#     :return:
#         params -- python dictionary containing your parameters:
#                     W1 -- weight matrix of shape (n_h, n_x)
#                     b1 -- bias vector of shape (n_h, 1)
#                     W2 -- weight matrix of shape (n_y, n_h)
#                     b2 -- bias vector of shape (n_y, 1)
#     """
#     np.random.seed(2)

#     W1 = np.random.randn(n_h, n_x) * 0.01
#     b1 = np.zeros((n_h, 1))
#     W2 = np.random.randn(n_y, n_h) * 0.01
#     b2 = np.zeros((n_h, 1))

#     assert (W1.shape == (n_h, n_x))
#     assert (b1.shape == (n_h, 1))
#     assert (W2.shape == (n_y, n_h))
#     assert (b2.shape == (n_h, 1))

#     params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

#     return params


# def forward_prpagation(X, parameters):
#     """

#     :param X:input data of size (n_x, m)
#     :param parameters: python dictionary containing your parameters:
#                     W1 -- weight matrix of shape (n_h, n_x)
#                     b1 -- bias vector of shape (n_h, 1)
#                     W2 -- weight matrix of shape (n_y, n_h)
#                     b2 -- bias vector of shape (n_y, 1)
#     :return:
#         A2 -- The sigmoid output of the second activation
#         cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
#     """
#     W1 = parameters["W1"]
#     b1 = parameters["b1"]
#     W2 = parameters["W2"]
#     b2 = parameters["b2"]

#     Z1 = np.dot(W1, X) + b1
#     A1 = np.tanh(Z1)
#     Z2 = np.dot(W2, A1) + b2
#     A2 = sigmoid(Z2)

#     assert (A2.shape == (1, X.shape[1]))

#     cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

#     return A2, cache


# def comput_cost(A2, Y, parameters):
#     """

#     :param A2:he sigmoid output of the second activation, of shape (1, number of examples)
#     :param Y:"true" labels vector of shape (1, number of examples)
#     :param parameters: python dictionary containing your parameters W1, b1, W2 and b2
#     :return:
#         cost -- cross-entropy cost given equation
#     """
#     m = Y.shape[1]
#     logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
#     cost = -1 / m * np.sum(logprobs)

#     cost = np.squeeze(cost)

#     assert (isinstance(cost, float))
#     return cost

print("hello python")