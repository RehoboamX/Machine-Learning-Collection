"""
Walk through of a lot of different useful Tensor Operations, where we
go through what I think are four main parts in:

1. Initialization of a Tensor
2. Tensor Mathematical Operations and Comparison
3. Tensor Indexing
4. Tensor Reshaping

But also other things such as setting the device (GPU/CPU) and converting
between different types (int, float etc) and how to convert a tensor to an
numpy array and vice-versa.

Programmed by Aladdin Persson
* 2022-07-13: Initial coding
"""

import torch

# ================================================================= #
#                        Initializing Tensor                        #
# ================================================================= #

device = "cuda" if torch.cuda.is_available() else "cpu"
#use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)

# tensor的一些属性
print(f"Information about tensor: {my_tensor}")     # 打印tensor中的data，device和requires_grad
print(f"Type of tensor: {my_tensor.dtype}")         # 打印tensor的数据类型，这里为float32
print(f"Device tensor is on: {my_tensor.device}")   # 打印cpu或cuda （cuda后面的数字为gpu的序号）
print(f"Shape of tensor: {my_tensor.shape}")        # 打印tensor的形状，这里为2x3
print(f"Requires gradient: {my_tensor.requires_grad}")  # 返回True会被追溯梯度，称此种节点为叶子节点

# 其他常用的Tensors初始化方法
x = torch.empty(size=(3, 3))    # 返回一个未初始化的tensor（其值为0或接近0的极小值）
x = torch.zeros((3, 3))         # 全0张量
x = torch.rand((3, 3))          # 返回一个0~1的均匀分布（randn为正态分布）
x = torch.ones((3, 3))          # 全1张量
x = torch.eye(5, 5)             # eye（I）：单位矩阵

x = torch.arange(start=0, end=5, step=1)            # start和end是前闭后开的，输出[0, 1, 2, 3, 4]，也可以用torch.arange(5)
x = torch.linspace(start=0.1, end=1, steps=10)      # 返回在区间start和end上均匀间隔的step个点（闭区间），输出[0.1, 0.2, ..., 1]
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1) # 初始化为均值mean，标准差为std的正态分布
x = torch.empty(size=(1, 5)).uniform_(0, 1)         # 初始化为0~1的均匀分布
x = torch.diag(torch.ones(3))                       # torch.diag取矩阵的对角线元素
print(x)

# 初始化和转化Tensors到其他数据类型的方法（int, float, double）
tensor = torch.arange(4)    # [0, 1, 2, 3], 默认初始化为int64的数据类型
print(tensor.bool())        # int64 --> bool
print(tensor.short())       # int64 --> int32
print(tensor.long())        # int64 --> int64 （常用）
print(tensor.half())        # int64 --> float16, 在一些特定GPU上有时被用到
print(tensor.float())       # int64 --> float32 （常用）
print(tensor.double())      # int64 --> float64

# Numpy Arrays 和 Tensors之间的互相转化
import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array) # np array --> tensor
np_array_back = tensor.numpy()      # tensor --> np array， np_array_back应该和np_array一样（可能有数值上的四舍五入）


# =============================================================================== #
#                        Tensor Math & Comparison Operations                      #
# =============================================================================== #

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# -- 加法 --
z1 = torch.empty(3)
torch.add(x, y, out=z1) # 法1

z2 = torch.add(x, y)    # 法2
z = x + y               # 推荐法3（最简单直接）

# -- 减法 --
z = x - y

# -- 除法 --
z = torch.true_divide(x, y) # 如果x和y形状相同，则做元素级别的除法

# -- Inplace操作 --
t = torch.zeros(3)
t.add_(x)   # 当函数后面接着_时表示函数为inplace操作，会在原始的t上改变，而不生成一个新的copy变量，提高运算效率
t += x      # 同上（注：t = t + x 并非inplace操作，会生成一个copy）

# -- 指数运算 --
z = x.pow(2)    # 元素级别的指数操作
z = x ** 2      # 同上

# -- 比较运算 --
z = x > 0   # 元素级别比较，输出布尔数据
z = x < 0

# -- 矩阵乘法 --
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)   # 2x3
x3 = x1.mm(x2)  # 同上

# -- 矩阵的指数运算 --
matrix_exp = torch.rand(5, 5)
matrix_exp.matrix_power(3)  # 对矩阵的指数操作，输入要为方阵

# -- 元素级别乘法 --
z = x * y
print(z)

# -- 点乘（向量元素级别乘法再累加） --
z = torch.dot(x, y)
print(z)

# -- 批量矩阵乘法 --
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) # torch.bmm做批量的矩阵乘法，输出shape（batch, n, p）

# -- 广播机制 --
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2     # python中有广播机制会将x2复制5行与x1进行运算
z = x1 ** x2

# 其他有用的tensor操作
sum_x = torch.sum(x, dim=0)             # dim=0为最粗的粒度，dim=-1为最细的粒度（一般为最基本的行向量中的维度）
values, indices = torch.max(x, dim=0)   # values传回x中第0维度最大的数值，indices传回x中第0维度最大值的位置。
values, indices = torch.min(x, dim=0)   # 与上面类似 （注：以上操作可由 x.max(dim=0) 或 x.min(dim=0) 的形式代替，下面介绍的很多函数都有这两种表示方法）
abs_x = torch.abs(x)                    # 元素级别取绝对值
z = torch.argmax(x, dim=0)              # 与torch.max类似，只是其仅返回最大值所在位置indices
z = torch.argmin(x, dim=0)              # 与上面类似
mean_x = torch.mean(x.float(), dim=0)   # 计算均值的函数torch.mean要求其输入张量必须为浮点类型的
z = torch.eq(x, y)                      # 元素级别的比较x和y是否相等，输出布尔值
sorted_y, indices = torch.sort(y, dim=0, descending=False)  # 将张量y在第0维度上的数值以升序重新排列

z = torch.clamp(x, min=0, max=10)       # 将x中所有小于0的值置0，所有大于10的值置10
z = torch.clamp(x, min=0)               # 相当于Relu函数

x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x)    # 做或运算，任何一个元素为True则结果返回True
z = torch.all(x)    # 做与运算，所有元素均为True才返回True


# ============================================================= #
#                        Tensor Indexing                        #
# ============================================================= #

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0])     # 打印第一个样本的特征值，同x[0, :]

print(x[:, 0])  # 打印所有样本的第一个特征值，shape：[10]

print(x[2, 0:10])   # 打印第三个样本的前十个特征值，shape：[10] （注： 0：10 --> [0, 1, 2, ..., 9]）

x[0, 0] = 100   # 对某特征值重新赋值

# 复杂一些的的Indexing用法
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])   # 打印x中的第3，第6和第9个值

x = torch.arange(15).reshape(3, 5)
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols])    # 打印x中第2行第5列和第1行第1列的两个值，注意行和列相对应取而不交叉取

# 更高级的Indexing用法
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])     # 打印x中小于2或大于8的值
print(x[(x < 8) & (x > 2)])     # 打印x中小于8且大于2的值
print(x[x.remainder(3) == 1])   # 打印x中所有除以3余1的值

# 其他有用的操作
print(torch.where(x > 5, x, x*2))   # 如果x中的值>5成立，则进行前面保留x的操作；若不成立，则进行后面x乘2的操作
print(torch.tensor([0,0,1,2,2,3,4,4]).unique()) # 打印tensor中所有存在的值，重复的仅输出1次
print(x.ndimension())   # 打印张量x的维度数
print(x.numel())        # 打印x中的总元素数


# ============================================================= #
#                        Tensor Reshaping                       #
# ============================================================= #

x = torch.arange(9)

x_3x3 = x.view(3, 3)    # 法1 view只适合对满足连续性条件（contiguous）的tensor进行操作，而reshape同时还可以对不满足连续性条件的tensor进行操作，具有更好的鲁棒性。
x_3x3 = x.reshape(3, 3) # 法2 view能干的reshape都能干，如果view不能干就可以用reshape来处理。 详见https://www.jb51.net/article/236201.htm

y = x_3x3.t()   # torch.t()表示取转置，转置后的张量则不满足连续性条件，
print(y.is_contiguous())    # 会打印False，不能直接view操作，但可以reshape
print(y.contiguous().view(9))   # 直接采用y.view操作会报错，此时应采用以上或用reshape操作来生成一个copy

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=0))   # torch.cat为拼接操作，在dim=0（最粗粒度）的维度上拼接几个tensor

z = x1.view(-1)     # 将x1铺平
print(z.shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)   # x张量batch的维度保留，其余维度铺平
print(z.shape)

z = x.permute(0, 2, 1)  # 将x的维度1和2调换，相当于batchsize维度不变，其余两维度的张量做转置
print(z.shape)

x = torch.arange(10)        # 此时x为1维tensor，形状为[10]
print(x.unsqueeze(0).shape) # unsqueeze(0)操作给x添加了新的0维度，形状为[1,10]
print(x.unsqueeze(1).shape) # unsqueeze(1)操作给x添加了新的1维度，形状为[10,1]

x = torch.arange(10).unsqueeze(0).unsqueeze(1)  # 形状[1,1,10]

z = x.squeeze(1)    # 当tensor某维度上的形状为1时，可以通过squeeze操作压缩掉此维度来实现降维
print(z.shape)      # z为x去掉其dim=1的结果，形状[1,10]