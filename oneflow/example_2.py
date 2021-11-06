"""
import oneflow as flow

#import oneflow.nn as nn



def index_add(x, dim, index, tensor, alpha=1):
    pass


indices = flow.Tensor([[1],[3],[5]])
updates = flow.Tensor([-1., -2., -3.])


print(flow.scatter_nd(indices,updates,[8]))
"""
import oneflow as flow
import numpy as np
input = flow.ones((3,5))*2
index = flow.tensor(np.array([[0,1,2],[0,1,3]], ), dtype=flow.int32)
src = flow.Tensor(np.array([[0,10,20,30,40],[50,60,70,80,90]]))
out = flow.scatter_add(input, 1, index, src)

print(out)
