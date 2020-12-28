import torch

m1=torch.FloatTensor([[1, 2], [3, 4]])
m2=torch.FloatTensor([[1], [2]])

print("Shape of Matrix 1: ", m1.shape)  # -> torch.Size([row, column])
print("Shape of Matrix 2: ", m2.shape)

print(m1.matmul(m2))                    # matrix multiplication

print(m1*m2)                            # element-wise multiplication
print(m1.mul(m2))