import torch

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])

print("Shape of Matrix 1: ", m1.shape)  # -> torch.Size([row, column])
print("Shape of Matrix 2: ", m2.shape)

print(m1.matmul(m2))                    # matrix multiplication

print(m1*m2)                            # element-wise multiplication
print(m1.mul(m2))

t1 = torch.FloatTensor([1, 2])
t2 = torch.FloatTensor([[1, 2], [3, 4]])
print(t1.mean())                        # get average value of elements
print(t2.mean())                        # 10/4 = 2.5

print(t2.mean(dim=0))                   # remove first dimension(row) by average value
print(t2.mean(dim=1))                   # remove second dimension(column) by average value

print(t1.sum())
print(t2.sum())
print(t2.sum(dim=0))                    # remove first dimension by addition
print(t2.sum(dim=1))                    # remove second dimension by addition

t = torch.FloatTensor([[3, 2],
                       [1, 4]])
print(t.max())                          # return 4
print(t.max(dim=0))                     # return [3, 4] and argmax [0, 1]
print(t.max(dim=1))                     # return [3, 4] and argmax [0, 1]

print("only max value: ", t.max(dim=0)[0])
print("only argmax value: ", t.max(dim=0)[1])

