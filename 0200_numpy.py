# import numpy as np
#
# array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int)
# a2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int)
#
# print(array.any() < 6)
# a4 = [i for i in array.all() if i < 6]
# print(a4)
# a3 = np.array([i for i in array if i < 6])
# print(a3)
#
# import numpy as np
#
# a = np.array([[1, 1], [0, 1]])
# b = np.arange(4).reshape((2, 2))
#
# print(a * b)
# print(a.dot(b))
# print(np.sum(b, axis=1))
# print(np.max(b, axis=0))
# # print(np.min(b, axis=0))
#
# import numpy as np
#
# A = np.arange(2, 14).reshape(3, 4)
# # print(A)
# # print(np.argmax(A))
# # print(np.argmin(A))
# # print(np.mean(A, axis=1))
# # print(np.cumsum(A))
# # B = np.zeros((3, 4), dtype=np.int).reshape(1, -1)
# # B[0][1] = 1
# # print(B)
# # print(B.cumsum())
# print(A)
# # print(np.transpose(A).dot(A))
# print(np.clip(A, 3, 7))

# import numpy as np
#
# A = np.arange(12).reshape(3, 4)
# print(A)
# # print(np.vsplit(A, 2))
# print(np.hsplit(A, 2))
# # print(np.array_split(A, 2, axis=0))
# # A = np.ones((3, 2))
# # B = np.zeros((3, 2))
# # print(A)
# # print(B)
# # print(np.vstack((A, B)))
# # print(np.hstack((A, B)))
# # print(A.reshape(3, -1))
# # print(np.concatenate((A, B), axis=1))
# # print(np.concatenate((A, B), axis=0))
# # print(A)
# # # print(A[2][1][::-1])
# # for row in A:
# #     print(row)
# #
# # for row in np.transpose(A):
# #     print(row)
# #
# # print(A.flatten())

import numpy as np

# a = np.arange(4)
# b = a
# c = a
# d = b
# e = a.copy()
#
# a[0] = 4
# print(a)
# print(b)
# print(c)
# print(d)
# print(e)

# randMat = np.mat(np.random.random((3, 4)))
# print(randMat)
# print(randMat.I)

print(np.random.randn(3, 4))
a = np.arange(12).reshape(3, 4)
print(a)
print(np.sum(a, axis=1))