#encoding=UTF-8
'''
Created on 2023年8月16日

@author: yangjinfeng
'''
import torch

a = torch.tensor([1., 2., 3.], requires_grad=True)
print(a)
print("a.grad a.grad_fn : ",a.grad,a.grad_fn)

b = torch.tensor([4., 5., 6.], requires_grad=True)
print(b)
print("b.grad b.grad_fn : ",b.grad,b.grad_fn)

z = a * b
print(z)
print("z.grad z.grad_fn : ",z.grad,z.grad_fn)

loss = z.sum()*2
# print(loss)

loss.backward()
print(loss)