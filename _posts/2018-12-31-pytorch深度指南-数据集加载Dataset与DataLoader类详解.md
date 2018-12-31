---
layout: post
title:  "pytorch深度指南-数据集加载Dataset与DataLoader类详解"
date:   2018-12-31 15:17:58 +0800
categories: jekyll update
---
# 写在前面
我个人理解，pytorch在加载数据集分为两个步骤:  
_第一步：_是根据用户自己的情况读取数据，数据的组织形式不一样，其中用到的读取方式自然各异，最终用于训练的数据应该是`Tensor`类型。因此`Dataset`类的作用相当于是一个与真实数据集的交互的方法接口。  
_第二步:_读完数据后需要送入网络训练模型，此时可能需要指定随机、batchsize等参数，pytorch为我们提供了功能强大的`DataLoader`类。可以使用多线程调用之前写好的`Dataset`方法加快数据读取  
**总结：**
* `Dataset`负责确定读取数据的具体操作方法，需要继承重写
* `DataLoader`指定了训练时batchsize，是否随机等参数，接收之前写好的Dataset对象


```python
import torch
import torchvision.transforms as transform#transform后面有s
from torch.utils.data import Dataset, DataLoader#DataLoader,注意L大写
```

## torch.utils.Dataset介绍

`torch.utils.data.Dataset`是一个Pytorch用来表示数据集的抽象类.我们用这个类来处理自己数据的时候必须要继承Dataset,然后需要重写下面的函数:  
* `__len__(self)`:DataLoader数据集才能知道大小
* `__getitem__(self, item)`:这样DataLoader才能取到每个元素


```python
class Trainset(Dataset):
    def __init__(self, data_list, transform=None, loader=default_loader):
        imgs = []#tmp variable don't use self to bind

        for index, row in data_list.iterrows():
            imgs.append((row['img_path'], row['label']))#add to list with tuple
        self.imgs = imgs
        self.transfom = transform
        self.loader = loader
    def __getitem__(self, item):
        filename, label = self.imgs[item]
        img = self.loader(filename)
        if self.transfom:
            img = self.transfom(img)#因为DatLoader是从getitem进行取元素的,返回的元素即为DataLoader取到的元素
        return img, label
    def __len__(self):
        return len(self.imgs)

```

>通常使用PIL,cv2,skimage读取的图片都是(H,W,C)格式,如果要放入dataloader,必须要将其转换为(C,H,W)类型,对于`pil.Image`和`ndarray`可以使用`transform.ToTensor()`进行转换

## torchvision.transforms的用法
有一个好用的组合方式`transform.Compose([transform.fun1(),....])`,可以将多种变换顺序组合到一起,里面是一个可迭代的列表,也可以是元组<br>
Compose([])的解析方法<br>
```
for t in self.transforms:#transforms在定义的时候是一个list,从里面进行迭代,不断突破自我
    img = t(img)#这里的img不一定就是图像哦,可能是PIL,也可能是tensor输入,得看上一步的操作
return img
```
因为是按照列表中的循序进行迭代,处理数据格式流要指定.

以下的每个方法都是**可调用**的(声明完之后要call才能正常使用哦)先声明方法,call的时候将数据传入,一下函数只能对PIL.Image图像进行操作,返回也是PIL!!!<br>
`RandomCrop((size,..))`随机裁剪<br>
`CenterCrop((size))`中心裁剪指定尺寸大小的图片<br>
`Resize((size))`调整图像尺寸,使用的差值方法默认为BILINER<br>
`Normalize(mean, std)`mean和std都是每个通道的mean和std<br>
`RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0)`仿射变换<br>
`FiveCrop((size))`将给定的pil图片裁剪成五分,四个角落和一个中心<br>
`TenCrop((size))`将给定的pil图片裁剪为10份<br>

`ToTensor()`将PIL,numpy转换为Tensor,
`Lambda(function)`对图像使用一个lambda函数
`TenCrop(size)`这个函数返回一个tuple,包含了分割后的十张PIL图片,可以使用torch.stack()合并起来

## `torchvision.data.DataLoader`
`DataLoader`作为一个Loader,支持多线程操作,加速数据的读取工作,这个`DataLoader`类定位是数据的读取,里面有很多选项是用来指定batch_size,shuffle,num_workers等参数,只是根据index从DataSet中取出数据而已,数据内容是什么形式`DataLoader`并不关心
> DataLoader 不支持索引


```python
class Trainset(Dataset):
    def __init__(self, data_list, data_sour_path= '',transform=None, num_classes=28):
        assert isinstance(data_list, pd.DataFrame)
        self.imgs_path_list = [(v['Id'], v['Target']) for k,v in data_list[['Id', 'Target']].iterrows()]
        self.transform = transform
        self.num_classes = num_classes
        self.data_sour_path = data_sour_path
    def __getitem__(self, item):
        name, label = self.imgs_path_list[item]
        image = self.openRGBY(name)
        if self.transform is not None:
            image = self.transform(image)
        k = list(map(int, label.split(' ')))
        return image, np.eye(self.num_classes)[k].sum(axis=0)
    def __len__(self):
        return len(self.imgs_path_list)
    def openRGBY(self, img_name, ext= '.png'):
        colors = ['red', 'green', 'blue', 'yellow']
        flag = cv2.IMREAD_GRAYSCALE
        img_path = os.path.join(self.data_sour_path, img_name)
        img = [cv2.imread(img_path + '_' + c + ext, flag).astype(np.float32) for c in colors]
        return np.stack(img, axis=-1)

class TestSet(Trainset):
    def __init__(self, data_csv, data_sour_path= '', transform=None, num_classes=28):
        assert isinstance(data_csv, pd.DataFrame)
        # self.imgs_path_list = [(v['Id'], v['Target']) for k,v in data_list[['Id', 'Target']].iterrows()]
        self.imgs_path_list = data_csv['Id'].tolist()
        self.transform = transform
        self.num_classes = num_classes
        self.data_sour_path = data_sour_path
    def __getitem__(self, item):
        name = self.imgs_path_list[item]
        image = self.openRGBY(name)
        if self.transform is not None:
            image = self.transform(image)
        return image, name
```


```python
test_csv = pd.read_csv('test_sample.csv')
test_dataset = TestSet(test_csv)
```


```python
test_loader = DataLoader(test_dataset, batch_size=12)
```


```python
train_list = Trainset(data_list=label_list, transform=transform.Compose([transform.Resize((400, 400)), transform.ToTensor()]))
train_data = DataLoader(train_list, batch_size=12)
```

## TODO
