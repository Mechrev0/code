from __future__ import print_function


# 我们可以定义一个模型，这个模型继承自nn.Module类。如果需要定义一个比Sequential模型更加复杂的模型，就需要定义nn.Module模型。


# import torch
# 创建一个未初始化的矩阵
# x = torch.empty(5, 3).to("cuda:0")
# print(x)
# # print(torch.__version__)
# # print(torch.cuda.is_available())
#
# # 构建一个随机初始化的矩阵:
# x = torch.rand(5, 3).to("cuda:0")
# print(x)
#
# # 构建一个全部为0，类型为long的矩阵:
# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)
#
# # 从数据直接直接构建tensor:
# x = torch.tensor([5, 3])
# print(x)
#
# # 也可以从一个已有的tensor构建一个tensor。这些方法会重用原来tensor的特征，例如，数据类型，除非提供新的数据。
# x = x.new_ones(5, 3, dtype=torch.double)
# print(x)
#
# x = torch.randn_like(x, dtype=torch.float)
# print(x)
#
# print(x.size())
#
# y = torch.rand(5, 3)
# print(x + y)
# y.add_(x)
# print(y)
# x = torch.randn(4, 4)
# print(x)
# x = x.reshape(-1, 2)
# print(x)
# x = x.reshape(8, -1)
# print(x)
# x = torch.randn(1)
# print(x)
# print(x.item())
# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)
#
# a = torch.from_numpy(b)
# print(a)
# x = torch.randn(4, 3)
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     y = torch.ones_like(x, dtype=torch.float, device=device)
#     x = x.to(device)
#     z = x + y
#     print(z)
# 热身: 用numpy实现两层神经网络
# 一个全连接ReLU神经网络，一个隐藏层，没有bias。用来从x预测y，使用L2 Loss。
# batch_size, dim_input, dim_hidden, dim_output = 64, 1000, 100, 10
#
# dtype = torch.float
# device = torch.device("cuda")
#
# x = torch.randn(dim_input, batch_size, dtype=dtype, device=device)
# y = torch.randn(dim_output, batch_size, dtype=dtype, device=device)
#
# W1 = torch.randn(dim_hidden, dim_input, dtype=dtype, device=device, requires_grad=True)
# W2 = torch.zeros(dim_output, dim_hidden, dtype=dtype, device=device, requires_grad=True)
#
# learning_rate = 1e-6
# for i in range(1000):
#     h = torch.mm(W1, x)
#     h_relu = h.clamp(min=0)
#     y_hat = torch.mm(W2, h_relu)
#
#     loss = (y_hat - y).pow(2).sum().item()
#     print(i, loss)
#
#     grad_y_pred = 2.0 * (y_hat - y)
#     grad_w2 = grad_y_pred.mm(h_relu.t())
#     grad_h_relu = W2.t().mm(grad_y_pred)
#     grad_h = grad_h_relu.clone()
#     grad_h[h < 0] = 0
#     grad_w1 = grad_h.mm(x.t())
#
#     W1 -= learning_rate * grad_w1
#     W2 -= learning_rate * grad_w2
# for i in range(10000):
#     # h = torch.mm(W1, x)
#     # h_relu = h.clamp(min=0)
#     # y_hat = torch.mm(W2, h_relu)
#
#     # 前向传播:通过Tensor预测y；这个和普通的神经网络的前向传播没有任何不同，
#     # 但是我们不需要保存网络的中间运算结果，因为我们不需要手动计算反向传播。
#     y_hat = W2.mm(W1.mm(x).clamp(min=0))
#
#     loss = (y_hat - y).pow(2).sum()
#     print(i, loss.item())
#
#     # PyTorch给我们提供了autograd的方法做反向传播。如果一个Tensor的requires_grad=True，
#     # backward会自动计算loss相对于每个Tensor的gradient。在backward之后，
#     # w1.grad和w2.grad会包含两个loss相对于两个Tensor的gradient信息。
#     loss.backward()
#
#     # 我们可以手动做gradient descent(后面我们会介绍自动的方法)。
#     # 用torch.no_grad()包含以下statements，因为w1和w2都是requires_grad=True，
#     # 但是在更新weights之后我们并不需要再做autograd。
#     # 另一种方法是在weight.data和weight.grad.data上做操作，这样就不会对grad产生影响。
#     # tensor.data会我们一个tensor，这个tensor和原来的tensor指向相同的内存空间，
#     # 但是不会记录计算图的历史。
#     with torch.no_grad():
#         W1 -= learning_rate * W1.grad
#         W2 -= learning_rate * W2.grad
#
#         W1.grad.zero_()
#         W2.grad.zero_()
#     # grad_y_pred = 2.0 * (y_hat - y)
#     # grad_w2 = grad_y_pred.mm(h_relu.t())
#     # grad_h_relu = W2.t().mm(grad_y_pred)
#     # grad_h = grad_h_relu.clone()
#     # grad_h[h < 0] = 0
#     # grad_w1 = grad_h.mm(x.t())
#     #
#     # W1 -= learning_rate * grad_w1
#     # W2 -= learning_rate * grad_w2
# import torch.nn as nn
#
# batch_size, dim_input, dim_hidden, dim_output = 64, 1000, 100, 10
#
# # dtype = torch.float
# device = torch.device("cuda")
#
# x = torch.randn(batch_size, dim_input, device=device)
# y = torch.randn(batch_size, dim_output, device=device)
#
# # Use the nn package to define our model as a sequence of layers. nn.Sequential
# # is a Module which contains other Modules, and applies them in sequence to
# # produce its output. Each Linear Module computes output from input using a
# # linear function, and holds internal Tensors for its weight and bias.
# model = nn.Sequential(
#     nn.Linear(dim_input, dim_hidden),
#     nn.ReLU(),
#     nn.Linear(dim_hidden, dim_output)
# )
# # W1 = torch.randn(dim_hidden, dim_input, dtype=dtype, device=device, requires_grad=True)
# # W2 = torch.zeros(dim_output, dim_hidden, dtype=dtype, device=device, requires_grad=True)
#
# model = model.to("cuda")
# learning_rate = 1e-4
# # The nn package also contains definitions of popular loss functions; in this
# # case we will use Mean Squared Error (MSE) as our loss function.
# loss_fn = nn.MSELoss(reduction='sum')
#
# for t in range(1446):
#     # Forward pass: compute predicted y by passing x to the model. Module objects
#     # override the __call__ operator so you can call them like functions. When
#     # doing so you pass a Tensor of input data to the Module and it produces
#     # a Tensor of output data.
#     y_pred = model(x)
#
#     # Compute and print loss. We pass Tensors containing the predicted and true
#     # values of y, and the loss function returns a Tensor containing the
#     # loss.
#     loss = loss_fn(y_pred, y)
#     print(t, loss.item())
#
#     # Zero the gradients before running the backward pass.
#     model.zero_grad()
#
#     # Backward pass: compute gradient of the loss with respect to all the learnable
#     # parameters of the model. Internally, the parameters of each Module are stored
#     # in Tensors with requires_grad=True, so this call will compute gradients for
#     # all learnable parameters in the model.
#     loss.backward()
#
#     # Update the weights using gradient descent. Each parameter is a Tensor, so
#     # we can access its gradients like we did before.
#     with torch.no_grad():
#         for param in model.parameters():
#             param -= learning_rate * param.grad
# 这一次我们不再手动更新模型的weights,而是使用optim这个包来帮助我们更新参数。
# optim这个package提供了各种不同的模型优化方法，包括SGD+momentum, RMSProp, Adam等等。
# import torch
# import torch.nn as nn
#
# batch_size, dim_input, dim_hidden, dim_output = 64, 1000, 100, 10
#
# device = torch.device("cuda")
# x = torch.randn(batch_size, dim_input, device=device)
# y = torch.randn(batch_size, dim_output, device=device)
#
# model = nn.Sequential(
#     nn.Linear(dim_input, dim_hidden, bias=False),
#     nn.ReLU(),
#     nn.Linear(dim_hidden, dim_output, bias=False)
# )
#
# model = model.to("cuda")
#
# loss_fn = nn.MSELoss(reduction="sum")
#
# learning_rate = 1e-4
#
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# for i in range(500):
#     y_pred = model(x)
#
#     loss = loss_fn(y_pred, y)
#     print(i, loss.item())
#
#     optimizer.zero_grad()
#
#     loss.backward()
#
#     optimizer.step()

#
# class TwoLayerNet(nn.Module):
#
#     def _forward_unimplemented(self, *input: Any) -> None:
#         pass
#
#     def __init__(self, dim_input, dim_hidden, dim_output):
#         super(TwoLayerNet, self).__init__()
#         self.linear1 = nn.Linear(dim_input, dim_hidden, bias=False)
#         self.linear2 = nn.Linear(dim_hidden, dim_output, bias=False)
#
#     def forward(self, x):
#         h_relu = self.linear1(x).clamp(min=0)
#         y_hat = self.linear2(h_relu)
#         return y_hat
#
#
# batch_size, dim_input, dim_hidden, dim_output = 64, 1000, 100, 10
#
# device = torch.device("cuda")
# x = torch.randn(batch_size, dim_input, device=device)
# y = torch.randn(batch_size, dim_output, device=device)
#
# model = TwoLayerNet(dim_input, dim_hidden, dim_output).to("cuda")
# criterion = nn.MSELoss(reduction="sum")
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# for i in range(500):
#     y_pred = model(x)
#
#     loss = criterion(y_pred, y)
#     print(i, loss.item())
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()


# FizzBuzz是一个简单的小游戏。游戏规则如下：从1开始往上数数，
# 当遇到3的倍数的时候，说fizz，
# 当遇到5的倍数，说buzz，当遇到15的倍数，就说fizzbuzz，其他情况下则正常数数。
# 我们可以写一个简单的小程序来决定要返回正常数值还是fizz, buzz 或者 fizzbuzz。

# One-hot encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz"]
def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    if i % 5 == 0:
        return 2
    if i % 3 == 0:
        return 1
    return 0


def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


import numpy as np
import torch

NUM_DIGITS = 10


# Represent each input by an array of its binary digits.
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])


trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

NUM_HIDDEN = 100
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, 4)
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

batch_size = 128
for epoch in range(10000):
    for start in range(0, len(trX), batch_size):
        end = start + batch_size
        batchX = trX[start:end]
        batchY = trY[start:end]

        y_pred = model(batchX)
        loss = criterion(y_pred, batchY)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Find loss on training data
    loss = criterion(model(trX), trY).item()
    print('Epoch:', epoch, 'Loss:', loss)

testX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])
with torch.no_grad():
    testY = model(testX)
predictions = zip(range(1, 101), list(testY.max(1)[1].data.tolist()))

print([fizz_buzz_decode(i, x) for (i, x) in predictions])

print(np.sum(testY.max(1)[1].numpy() == np.array([fizz_buzz_encode(i) for i in range(1, 101)])))
# testY.max(1)[1].numpy() == np.array([fizz_buzz_encode(i) for i in range(1,101)])
