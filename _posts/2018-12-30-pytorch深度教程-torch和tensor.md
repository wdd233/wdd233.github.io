---
layout: post
title:  "pytorch深度指南-torch与Tensor常用操作方法"
date:   2018-12-30 22:05:58 +0800
categories: jekyll update
---
# pytorch深度指南-torch与Tensor常用操作方法



```python
import torch
```

`torch.Tensor`会继承某些`torch`的某些数学运算，例如sort, min/max....不需要调用相应的`torch.funciton`进行处理,下文中如果是`torch/Tensor`即表示该函数可以直接对self的tensor使用，也可以使用torch给的相应函数接口

## torch/Tensor.reshape(input, shape) → Tensor
指定tensor新的shape，reshape过后不会更该数据量和数据格式，只是对数据的shape做了调整,因此要保证reshape前后元素个数一致。
**参数：**

    input(tensor) - 输入tensor
    shape(tuple or *size) - 新的shape 
    如果还剩下一个维度，很好，你可以直接使用-1来替代，表示这个维度中应该有的元素数量
## torch/Tensor.view()
改变形状

rned tensor will be a view of input. Otherwise, it will be a copy. Contiguous inputs and inputs with compatible strides can be reshaped without copying, but you should not depend on the copying vs. viewing behavior.

>二者区别：当tensor都为contiguous类型(邻近模式)时，两个函数并无差异，使用原来的数据内容，不会复制一份新的出来；如果tensor不是，例如经过了transpose或permute之后，需要contiguous然后再使用view


```python
torch.arange(10).reshape((2,5))
```
    tensor([[ 0.,  1.,  2.,  3.,  4.],
            [ 5.,  6.,  7.,  8.,  9.]])

```python
torch.arange(10).reshape(5, -1)#可以看到新的shape是(5,2),使用-1自动求剩余最后一个维度
```

    tensor([[ 0.,  1.],
            [ 2.,  3.],
            [ 4.,  5.],
            [ 6.,  7.],
            [ 8.,  9.]])

### torch.index_select(input, dim, index, out=None)
指定在哪个轴上，index是多少，注意index要是一个tensor，
dim和index确定了一个维度的坐标$(dim_1=index_i,....dim_n=index)$


```python
a = torch.randn([3,4])
print(a)
a.index_select(0,torch.tensor([2])).reshape(2,2)#
```
    tensor([[ 0.0334,  0.9123, -1.2300, -0.9336],
            [-0.8364,  0.6584,  0.6878, -2.5896],
            [ 0.1862, -0.3752,  0.4150, -1.4008]])

    tensor([[ 0.1862, -0.3752],
            [ 0.4150, -1.4008]])

### torch.linespace(start, end, steps=100, out=None)
返回一个一维tensor，包含**step**个元素，每个元素之间等间距，依次递增$\frac{end-start}{step}$


```python
torch.linspace(0,1,10, dtype=torch.float32)
```
    tensor([ 0.0000,  0.1111,  0.2222,  0.3333,  0.4444,  0.5556,  0.6667,
             0.7778,  0.8889,  1.0000])

### Tensor.repeat(input, *size)
把输入的input当做一个基本模块m，扩张成*size的tensor，其中每个元素为m，最后返回的Tensor的shape为(*)
*size必须比input的维度要高，

```
if input和size维度相等：
input = (a,b)
*size = (c,d)
out = (a×c, b×d)

size > input
input = (a,b)
*size = (c,d,e)
out = (c, d×a,e×b)
```
```python
torch.Tensor.repeat(torch.tensor([2,3]),2,4,3).shape
```
    torch.Size([2, 4, 6])

## torch/Tensor.nonzero()
返回一个包含输入input中非零元素索引的张量，输出张量中非零元素的索引
若输入input有`n`维，则输出的索引张量output形状为`z * n`, 这里z是输入张量input中所有非零元素的个数


```python
a = torch.eye(3,3)
print('a:', a)
a.nonzero()#所有非0的坐标，注意返回的每个元素都是一个完整的坐标index～
```

    a: tensor([[ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.]])


    tensor([[ 0,  0],
            [ 1,  1],
            [ 2,  2]])

```python
b = torch.arange(8).reshape(2,2,2)
print('b:', b)
b.nonzero()#z=7,n=3,输出(7*3)矩阵
```

    b: tensor([[[ 0.,  1.],
             [ 2.,  3.]],
    
            [[ 4.,  5.],
             [ 6.,  7.]]])

    tensor([[ 0,  0,  1],
            [ 0,  1,  0],
            [ 0,  1,  1],
            [ 1,  0,  0],
            [ 1,  0,  1],
            [ 1,  1,  0],
            [ 1,  1,  1]])

## torch/Tensor.chunk(tensor, chunks, dim=0)
将一个tensor在指定维度上分成chunks个数据块，为cat的逆过程,最后一个块可能会小一些，返回的是一个元组，每个元素都是其中一块
**参数:**
    tensor (Tensor) – 输入Tensor
    chunks (int) – 分成几块的数量
    dim (int) – 沿着哪个维度进行切分
>可以看成`torch.cat()`的逆运算过程


```python
c = torch.randn(7,4)
c.chunk(chunks=4,dim=0)#，在dim=0上划分，返回的是一个元组，共分为4块
```
    (tensor([[ 0.0722,  0.2573, -0.2504,  1.7426],
             [-3.3098,  0.8971,  0.0274, -0.2365]]),
     tensor([[ 1.1137,  0.3401, -1.4004,  1.2823],
             [-0.5553,  0.8056, -0.1764,  0.7666]]),
     tensor([[ 1.3739, -1.4938,  1.8446,  0.4783],
             [-1.5898, -0.2520, -0.4873,  1.6098]]),
     tensor([[ 0.5421, -0.2846,  0.1441, -0.5456]]))

## `torch/Tensor.clamp(input, min, max, out=None)`

功能：将tensor所有元素裁剪到指定范围[min, max]并返回

```python
c = torch.tensor([1,2,3,4,5,6,7,8])
c = c.clamp(3,6)#裁剪并返回
c
```
    tensor([ 3,  3,  3,  4,  5,  6,  6,  6])

## torch/Tensor.sort(input, dim=None, descending=False, out=None)
沿着指定维度排序,如果没有指定维度，默认最后一个维度,descending选择升序还是降序,返回一个元组(排序好的tensor和相应的索引)


```python
a = torch.randint(0,10,(2,6)).squeeze().int()
a
```
    tensor([[ 0,  9,  7,  6,  3,  0],
            [ 8,  1,  8,  3,  6,  4]], dtype=torch.int32)

```python
a.sort(dim=0)#返回排序好的值和索引
```

    (tensor([[ 0,  1,  7,  3,  3,  0],
             [ 8,  9,  8,  6,  6,  4]], dtype=torch.int32),
     tensor([[ 0,  1,  0,  1,  0,  0],
             [ 1,  0,  1,  0,  1,  1]]))
## torch/Tensor.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
返回在指定维度上k个最大的元素，默认最后一个维度，返回值是一个元组包括值和索引(values, indices)
**参数：**

    input (Tensor) – 输入tensor
    k (int) – k个最值元素
    dim (int) – 沿着哪个维度排许
    largest (bool) – 选取最大元素还是最小元素
    sorted (bool) – 返回元素是否有序

```python
torch.randn(12).topk(4)#返回topK数值和位置索引
```




    (tensor([ 2.0778,  1.6199,  0.8411,  0.2411]), tensor([ 0,  1,  9,  4]))



## torch.split(tensor, split_size_or_sections, dim=0)


## torch/Tensor.flatten(input, start_dim=0, end_dim=-1) → Tensor
将一个tensor按照存储位置顺序展平


```python
a = torch.randn(3,3)
print(a)
# torch.flatten(a)
# a.flatten()

```

    tensor([[-1.6724, -0.0280,  0.0100],
            [ 0.3437,  0.9472, -0.3655],
            [-0.3134, -2.6223, -0.7097]])


## torch.numel()
返回tensor包含所有元素的数量

## torch.stack(seq, dim=0, out=None) → Tensor

在一个新的维度上，拼接原有的tensor,注意该操作会产生新的维度，待组合的tensor要有相同的size

**参数：**

    seq (sequence of Tensors) – 待拼接的tensor，要以seq形式
    dim (int) – dimension to insert. Has to be between 0 and the number of dimensions of concatenated tensors (inclusive)



## torch/Tensor.squeeze(input, dim=None, out=None) → Tensor
去除tensor中维度=1所在的那个维度，很多时候，为了保持tensor的shape对齐会提升维度，提升一些无关紧要的1维,然而在使用的时候我们不希望这些无关紧要的维度干扰我们的操作。

例如，如果输出shape：(A×1×B×1×D),经过squeeze后(A×B×D)
>注意：返回的tensor与输入共享存储空间，改变其中任何一个都会改变相应其他那个



```python
a = torch.arange(10).reshape(1,1,2,5)
print('a:', a)
b = a.squeeze() 
b += 2
a#可以看到虽然对b操作，但是原始a的值也发生了变化
```

    a: tensor([[[[ 0.,  1.,  2.,  3.,  4.],
              [ 5.,  6.,  7.,  8.,  9.]]]])

    tensor([[[[  2.,   3.,   4.,   5.,   6.],
              [  7.,   8.,   9.,  10.,  11.]]]])


## torch/Tensor.unsqueeze()

## torch.transpose(input, dim0, dim1) → Tensor
轴/坐标索引 交换

## torch.where(condition, x, y) → Tensor
逐个位置元素判断，返回一个tensor可能来自源数据x或者y。

    **注意：**
    条件,x,y都必须是可广播的
    **参数:**	
    
        condition (ByteTensor) – When True (nonzero), yield x, otherwise yield y
        x (Tensor) – values selected at indices where condition is True
        y (Tensor) – values selected at indices where condition is False

    Returns:	
    返回的tensor的shape与条件,x和y都是一致的！！)


```python
x = torch.randn(3,2)
y = torch.ones(3,2)
print('x:', x)
print('y', y)
torch.where(x > 0 , x, y)
```

    x: tensor([[ 0.8968,  3.3511],
            [-0.8421, -1.1878],
            [-0.0903,  0.6168]])
    y tensor([[ 1.,  1.],
            [ 1.,  1.],
            [ 1.,  1.]])





    tensor([[ 0.8968,  3.3511],
            [ 1.0000,  1.0000],
            [ 1.0000,  0.6168]])



## Tensor.type(torch.type)tensor数据类型转换
`Tensor.int()`将数据转为int

`Tensor.long()`将数据转为longInt

`Tensor.byte()`将数据转换为无符号整型

`Tensor.float()`将数据转换为float类型

`Tensor.double()`将数据转换为double类型


```python
a = torch.tensor([1,2,3,4])
a.long().type()
```
### torch.LongTensor

```python
a.float().type()
```
### torch.FloatTensor
```python
a.byte().type()
```
### torch.ByteTensor
## Tensor.cuda(device, non_blocking=False)
Returns a copy of this object in CUDA memory.
返回一个CUDA内存数据的副本,将输入数据存入cuda

```python
a = torch.arange(1200, dtype=torch.float64).cuda()
```

## Tensor.clone()
Returns a copy of the self tensor. The copy has the same size and data type as self.
返回`self`tensor的一个拷贝，与原数据有相同的size和数据类型
>clone()得到的Tensor不仅拷贝了原始的value，而且会计算梯度传播信息，copy_()只拷贝数值


```python
a = torch.tensor([1,2,3,4,5], dtype=torch.float32, requires_grad=True)
b = a.clone()#a经过克隆得到b，
c = (b ** 2).mean()#
c.backward()
print('a_grad:', a.grad)#但是梯度传播没有因此中断
```

    a_grad: tensor([ 0.4000,  0.8000,  1.2000,  1.6000,  2.0000])


## Tensor.copy_(src, non_blocking=False)
只拷贝`src`的数据到`self`tensor，并返回`self`，`src`和`self`可以有不同的数据类型和不同的设备上.<br>
**参数：**
    src (Tensor) – 源数据
    non_blocking (bool) – 如果是True，copy操作跨CPU和GPU，但可能会出现异步问题

```python
a = torch.tensor([1,2,3,4,5], dtype=torch.float32, requires_grad=True)
b = (a ** 2).mean()
b.backward()
print('a_grad:', a.grad)
c = torch.zeros(5)
c.copy_(a)
print('c_grad:', c.grad)
```

    a_grad: tensor([ 0.4000,  0.8000,  1.2000,  1.6000,  2.0000])
    c_grad: None

## Tensor.contiguous()
将tensor改为连续存储模式

### Tensor.fill_()内容填充
tensor内部全部填充value元素

### Tensor.zero_()

### Tensor.normal_(mean, std, out=None)

```python
a = torch.randn(3,3)
a.fill_(3)#指定内容填充

```

    tensor([[ 3.,  3.,  3.],
            [ 3.,  3.,  3.],
            [ 3.,  3.,  3.]])


```python
a.zero_()
```
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])

```python
a.normal_()
```

    tensor([[ 0.1044, -0.1353, -0.7540],
            [-0.6852,  0.2683,  0.8297],
            [ 2.8920,  0.2846, -1.1742]])
## torch生成常用矩阵
### torch.zeros(*size)
生成全零矩阵
### torch.ones(*size)
生成全1矩阵
### torch.eye(*size)
生成对角矩阵，对角线元素全为1
### torch.arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
返回指定区间，指定步长的1-D tensor

```python
torch.zeros(3,3)
```
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])

```python
torch.ones(3,3)
```
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])
```python
torch.eye(3,3)
```
    tensor([[ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.]])

```python
torch.arange(0, 10, 2)#区间是左壁右开，不包含end
```
    tensor([ 0.,  2.,  4.,  6.,  8.])
    
### torch/Tensor.round(input, out=None) → Tensor
对tensor内所有元素取整

```python
a = torch.randint(0,10,(2,2))
a.round()
```
    tensor([[ 7.,  5.],
            [ 1.,  5.]])     
## torch.randomtorch自带随机模块
生成随机数的功能模块
### torch.random.manual_seed(seed)
设置随机数种子
### torch.randint(low, high, size)
返回一个服从uniform分布的int随机矩阵
### torch.randn(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
$\text{out}_{i} \sim \mathcal{N}(0, 1)$
生成一个服从(0,1)标准正态分布的矩阵,传入size使用参数收集机制。
### torch.randperm(n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False) → LongTensor

生成一个`0→n-1`随机序列
**参数：**	

    n (int) – 生成序列最大上界(不包含n)


```python
torch.randint(0, 10, (2,2))#使用参数收集机制
```
    tensor([[ 6.,  9.],
            [ 3.,  8.]])

```python
torch.randn(2,2)#使用参数收集机制，服从搞死分布
```
    tensor([[-0.7565,  1.2032],
            [-0.7030,  1.7259]])

```python
torch.randperm(4)#返回一个生成的序列
```
    tensor([ 0,  1,  2,  3])

```python
torch.arange(0,10,1).type(torch.int32)
```
    tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9], dtype=torch.int32)

