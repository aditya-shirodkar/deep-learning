# tutorial credits: Aladdin Persson https://www.youtube.com/@AladdinPersson
# basically PyTorch works a lot like numpy and is generally very "pythonic," as you shall see

import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# create a tensor

print("\n~~~~INITIALISATION~~~~\n")
t1 = torch.tensor(
    [[1, 2, 3], [4, 5, 6]], dtype=torch.float, device=device, requires_grad=True
)
print("Tensor:")
print(t1)
print("Shape: ", t1.shape)

# other ways to initialise
print(torch.empty(size=(3, 3)))
print(torch.zeros((3, 3)))
print(torch.rand((3, 3)))
print(torch.ones((3, 3)))
print(torch.eye(3, 3))  # identity matrix
print(torch.arange(start=0, end=5, step=1))
print(torch.linspace(start=0.1, end=1, steps=10))
print(torch.empty(size=(3, 3)).normal_(mean=0, std=1))
print(torch.empty(size=(3, 3)).uniform_(0, 1))
print(torch.diag(torch.ones(3)))  # identity matrix

# converting tensors to different types
print("\n~~~~TYPE CONVERSION~~~~\n")
t2 = torch.arange(4)  # default dtype int64 i.e. long
print(t2)
print(t2.dtype)
print(t2.bool())
print(t2.short())
print(t2.long())
print(t2.float())
print(t2.half())  # float16; use only with a good GPU
print(t2.double())

# arrays to tensors
print("\n~~~~ARRAYS TO TENSORS~~~~\n")
a1 = np.zeros((5, 5))
t3 = torch.from_numpy(a1)
print(a1)
print(t3)
print(t3.numpy())

# tensor maths
print("\n~~~~TENSOR MATHS~~~~\n")
t4 = torch.tensor([1, 2, 3])
t5 = torch.tensor([4, 5, 6])

t6 = torch.empty(3)
torch.add(t4, t5, out=t6)
print(t6)
print(torch.add(t4, t5))
print(t4 + t5)
print(t4 - t5)
print(torch.true_divide(t4, t5))  # element-wise division if tensors are of equal shape

t7 = torch.zeros(3)
print(t7.add_(t4))  # inplace
t7 += t4  # inplace
print(t7)

# element-wise exponentiation
print(t4.pow(2))
print(t4**2)

# element-wise comparisons
print(t4 > 0)
print(t4 < 0)

# matrix multiplication
t8 = torch.rand((2, 6))
t9 = torch.rand((6, 3))
print("M1 = ", t8)
print("M2 = ", t9)
print("M1 @ M2 = ")
print(torch.mm(t8, t9))
print(t8.mm(t9))  # not inplace without the _

# matrix exponentiation
t10 = torch.rand(5, 5)
print("M = ", t10)
print("M @ M @ M = ", t10.matrix_power(3))

# element-wise multiplication
print("M1 = ", t4)
print("M2 = ", t5)
print("M1 * M2 (element-wise) = ", t4 * t5)

# dot product
print("M1.M2 = ", torch.dot(t4, t5))

# batch matrix multiplication, useful for NNs such as GPTs
batch = 32
p = 10
q = 20
r = 30

t11 = torch.rand((batch, p, q))
t12 = torch.rand((batch, q, r))
print("The dimensions are:")
print("batch = ", batch)
print("p = ", p)
print("q = ", q)
print("r = ", r)

print(f"M1 = {t11} of dimensions (batch, p, q)")
print(f"M2 = {t12} of dimensions (batch, q, r)")
print(
    f"Batch multiplication: BMM(M1, M2) = {torch.bmm(t11, t12)} of dimensions (batch, p, r)"
)

# other useful tensor operations; all can also be done as t.fn(whatever)
t15 = torch.sum(
    t4, dim=0
)  # sum over first (0th) dimension (only dimension in this case)
values, indices = torch.max(t4, dim=0)  # max in dimension
values, indices = torch.min(t4, dim=0)  # min in dimension
abs_t4 = torch.abs(t4)  # absolute value for each element
x1 = torch.argmax(t4, dim=0)  # index of max in dimension
x2 = torch.argmin(t4, dim=0)  # index of min in dimension
mean_t4 = torch.mean(t4.float(), dim=0)  # calculating the means requires floats
print(torch.eq(t4, t5))  # element-wise equality check
sorted_t4 = torch.sort(t4, dim=0, descending=False)  # ascending order sort
x3 = torch.clamp(t4, min=0)  # all values lower than min are set to min
x4 = torch.clamp(
    t4, min=0, max=10
)  # additionally, all values higher than max are set to 10

# boolean operations
t16 = torch.tensor([1, 0, 0, 1, 0], dtype=torch.bool)
print("M = ", t16)
print("any(M) = ", torch.any(t16))
print("all(M) = ", torch.all(t16))

# broadcasting (also works with numpy)
print("\n~~~~BROADCASTING~~~~\n")
t13 = torch.rand(5, 5)
t14 = torch.rand(1, 5)
print("M1 = ", t13)
print("M2 = ", t14)
# all columns of the first dimension in t14 will be replicated across 5 dimensions for the subtraction to be legitimate
print("M1 - M2 = ", t13 - t14)

# tensor indexing
print("\n~~~~INDEXING~~~~\n")

batch_size = 10
features = 25
t17 = torch.rand((batch_size, features))
print("M = ", t17)
print("Shape of M: ", t17.shape)
print("Shape of 0th dimension of M: ", t17[0].shape)
print("Shape of 1st dimension of M: ", t17[1].shape)
print("All elements in the 0th dimension of M: ", t17[:, 0])
print("All elements in the 1st dimension of M: ", t17[0, :])
print("Shape of all elements in the 0th dimension of M: ", t17[:, 0].shape)
print("Shape of all elements in the 1st dimension of M: ", t17[0, :].shape)
print("First six features in the 5th batch of M: ", t17[5, :6])
print("Number of dimensions of M: ", t17.ndimension())
print("Number of elements in M: ", t17.numel())
# generally, most numpy/pandas indicing rules apply to pytorch

# tensor reshaping
print("\n~~~~RESHAPING DIMENSIONS~~~~\n")
t18 = torch.arange(9)
print("M = ", t18)
print(
    "M -> 3x3 = ", t18.view(3, 3)
)  # almost the same as reshape but involves pointers (see contiguous())
print("or", t18.reshape(3, 3))  # safer
print("M -> 3x3 transposed = ", t18.reshape(3, 3).t())

t19 = torch.rand((1, 2))
t20 = torch.rand((1, 2))
print("M1 = ", t19)
print("M2 = ", t20)
print("M1 concatenated with M2 on the 0th dimension: ", torch.cat((t19, t20), dim=0))
print("M1 concatenated with M2 on the 1st dimension: ", torch.cat((t19, t20), dim=1))

batch = 8
t21 = torch.rand((batch, 1, 3))
print("M = ", t21)
print("M flattened on the non-batch (non-0th) dimensions: ", t21.view(batch, -1))
print("or (using reshape): ", t21.reshape(batch, -1))
print(
    "or, not quite (using permutations): ", t21.permute(0, 2, 1)
)  # better for more dimensions

t22 = torch.arange(10)
print("M = ", t22)
print("M unsqueezed on the 0th dimension: ", t22.unsqueeze(0))
print("M unsqueezed on the 1st dimension: ", t22.unsqueeze(1))
print(
    "M unsqueezed on the 0th dimension, and then on the 1st dimension: ",
    t22.unsqueeze(0).unsqueeze(1),
)
print(
    "M unsqueezed on the 0th dimension, then on the 1st dimension, and then squeezed on the 1st dimension: ",
    t22.unsqueeze(0).unsqueeze(1).squeeze(1),
)
