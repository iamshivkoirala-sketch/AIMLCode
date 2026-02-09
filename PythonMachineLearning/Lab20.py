import torch

# 0D Tensor (Scalar)
scalar_0d = torch.tensor(5.0)
print(f"0D Tensor:\n{scalar_0d}")
print(f"Shape: {scalar_0d.shape}\n")
# Output Shape: torch.Size([])

# 1D Tensor (Vector)
vector_1d = torch.tensor([1, 2, 3])
print(f"1D Tensor:\n{vector_1d}")
print(f"Shape: {vector_1d.shape}\n")
# Output Shape: torch.Size([3])

# 2D Tensor (Matrix)
matrix_2d = torch.tensor([
    [1, 2],
    [3, 4]
])
print(f"2D Tensor:\n{matrix_2d}")
print(f"Shape: {matrix_2d.shape}\n")
# Output Shape: torch.Size([2, 2])

# 3D Tensor (Cube)
cube_3d = torch.one_
