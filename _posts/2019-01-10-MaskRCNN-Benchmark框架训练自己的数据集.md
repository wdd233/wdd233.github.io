# MaskRCNN-Benchmark框架训练自己的数据集

Facebook AI 开源了Faster-RCNN和Mask RCNN的Pytorch1.0实现基准  

相比Detectron和mmdetection， MaskRCNN-Benchmark性能相当，并拥有更快的训练速度和更低的GPU显存占用
* 超越Detectron的两倍
* 节省内存


目前只包含了Mask RCNN和Faster RCNN两种检测模式，

基础安装环境：  
* Pytorch1.0, torchvision  
* cocoapi
* yacs
* matplotlib
* GCC>=4.9
* opencv for 摄像头
>如果缺少上述某一个库，可以使用conda 或者pip安装即可
## 安装方法
`python setup.py build develop`在./执行这一条语句，要确保之前安装正确

## mask-rcnn文件架构
```python
(maskrcnn_benchmark) $tree -L # 列出文件目录树状图
.
├── configs
│   ├── e2e_faster_rcnn_R_101_FPN_1x.yaml #训练和验证要用到的faster r-cnn模型配置文件
│   ├── e2e_mask_rcnn_R_101_FPN_1x.yaml #训练和验证要用到的mask r-cnn模型配置文件
│   └── quick_schedules
├── CONTRIBUTING.md
├── datasets
│   └── coco
│       ├── annotations
│  		│  ├── instances_train2014.json #训练集标注文件
│  		│  └── instances_val2014.json #验证集标注文件
│       ├── train2014  #存放训练集图片
│       └── val2014  #存放验证集图片
├── maskrcnn_benchmark
│   ├── config
│   │   ├── defaults.py #masrcnn_benchmark默认配置文件,启动时会读取该配置文件,configs目录下的模型配置文件进行参数合并
│   │   ├── __init__.py
│   │   ├── paths_catalog.py #在訪文件中配置训练和测试集的路径
│   │   └── __pycache__
│   ├── csrc
│   ├── data
│   │   ├── build.py #生成数据集的地方
│   │   ├── datasets #訪目录下的coco.py提供了coco数据集的访问接口
│   │   └── transforms
│   ├── engine
│   │   ├── inference.py #验证引擎
│   │   └── trainer.py #训练引擎
│   ├── __init__.py
│   ├── layers
│   │   ├── batch_norm.py
│   │   ├── __init__.py
│   │   ├── misc.py
│   │   ├── nms.py
│   │   ├── __pycache__
│   │   ├── roi_align.py
│   │   ├── roi_pool.py
│   │   ├── smooth_l1_loss.py
│   │   └── _utils.py
│   ├── modeling
│   │   ├── backbone
│   │   ├── balanced_positive_negative_sampler.py
│   │   ├── box_coder.py
│   │   ├── detector
│   │   ├── __init__.py
│   │   ├── matcher.py
│   │   ├── poolers.py
│   │   ├── __pycache__
│   │   ├── roi_heads
│   │   ├── rpn
│   │   └── utils.py
│   ├── solver
│   │   ├── build.py
│   │   ├── __init__.py
│   │   ├── lr_scheduler.py #在此设置学习率调整策略
│   │   └── __pycache__
│   ├── structures
│   │   ├── bounding_box.py
│   │   ├── boxlist_ops.py
│   │   ├── image_list.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── segmentation_mask.py
│   └── utils
│       ├── c2_model_loading.py
│       ├── checkpoint.py #检查点
│       ├── __init__.py
│       ├── logger.py #日志设置
│       ├── model_zoo.py
│       ├── __pycache__
│       └── README.md
├── output #我自己设定的输出目录
├── tools
│   ├── test_net.py #验证入口
│   └── train_net.py #训练入口
└── TROUBLESHOOTING.md


```

## 数据准备
maskrcnn-benchmark为coco量身打造的，
coco数据集有三种标注类型： **object instances(目标实例)， object keypoints(关键点检测)和image captions(看图说话)**， 使用json文件存储。
json标注文件
可以继续沿用coco数据集默认的名字

### 配置文件
主要涉及到的配置文件有三个：
* 模型配置文件,定义了网络模型的具体结构`(./configs)`
* 数据路径配置文件，`(./maskrnn_benchmark/config/paths_catalog.py)`仔细观察，和上面路径不一样哦！
* Mask RCNN框架配置文件，`maskrcnn_benchmark/config/defaults.py`

### 网络结构配置文件YAML
模型配置文件在启动时由--config-file参数指定，在config子目录默认提供了mask_rcnn和faster_rcnn框架不同骨干网络基于YAML格式的配置文件

```python
MODEL:
  META_ARCHITECTURE: "GeneralizedRCNN"
  WEIGHT: "catalog://ImageNetPretrained/MSRA/R-101"
  BACKBONE:
    CONV_BODY: "R-101-FPN"
    OUT_CHANNELS: 256
  RPN:
    USE_FPN: True #是否使用FPN,也就是特征金字塔结构,选择True将在不同的特征图提取候选区域
    ANCHOR_STRIDE: (4, 8, 16, 32, 64) #ANCHOR的步长
    PRE_NMS_TOP_N_TRAIN: 2000 #训练时,NMS之前的候选区数量
    PRE_NMS_TOP_N_TEST: 1000 #测试时,NMS之后的候选区数量
    POST_NMS_TOP_N_TEST: 1000
    FPN_POST_NMS_TOP_N_TEST: 1000
  ROI_HEADS:
    USE_FPN: True
  ROI_BOX_HEAD:
    POOLER_RESOLUTION: 7
    POOLER_SCALES: (0.25, 0.125, 0.0625, 0.03125)
    POOLER_SAMPLING_RATIO: 2
    FEATURE_EXTRACTOR: "FPN2MLPFeatureExtractor"
    PREDICTOR: "FPNPredictor"
  ROI_MASK_HEAD:
    POOLER_SCALES: (0.25, 0.125, 0.0625, 0.03125)
    FEATURE_EXTRACTOR: "MaskRCNNFPNFeatureExtractor"
    PREDICTOR: "MaskRCNNC4Predictor"
    POOLER_RESOLUTION: 14
    POOLER_SAMPLING_RATIO: 2
    RESOLUTION: 28
    SHARE_BOX_FEATURE_EXTRACTOR: False
  MASK_ON: False #默认是True,我这里改为False,因为我没有用到语义分割的功能
DATASETS:
  TRAIN: ("coco_2014_train",) #注意这里的训练集和测试集的名字,
  TEST: ("coco_2014_val",) #它们和paths_catalog.py中DATASETS相对应
DATALOADER:
  SIZE_DIVISIBILITY: 32
SOLVER:
  BASE_LR: 0.01 #起始学习率,学习率的调整有多种策略,訪框架自定义了一种策略
  WEIGHT_DECAY: 0.0001
  #这是什么意思呢?是为了在不同的迭代区间进行学习率的调整而设定的.以我的数据集为例,
  #我149898张图,计划是每4个epoch衰减一次,所以如下设置.
  STEPS: (599592, 1199184) 
  MAX_ITER: 1300000 #最大迭代次数

```
### 框架使用配置文件defaults.py
此处需要安装yacs，将下面的设置编译为cfg
`form maskrcnn_benchmark.config import cfg`


```python
import os
from yacs.config import CfgNode as CN
_C = CN()
_C.MODEL = CN()
_C.MODEL.RPN_ONLY = False
_C.MODEL.MASK_ON = False
_C.MODEL.DEVICE = "cuda" 
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
_C.MODEL.WEIGHT = ""
_C.INPUT = CN()
_C.INPUT.MIN_SIZE_TRAIN = 800  #训练集图片最小尺寸
_C.INPUT.MAX_SIZE_TRAIN = 1333 #训练集图片最大尺寸
_C.INPUT.MIN_SIZE_TEST = 800
_C.INPUT.MAX_SIZE_TEST = 1333
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
_C.INPUT.PIXEL_STD = [1., 1., 1.]
_C.INPUT.TO_BGR255 = True
_C.DATASETS = CN()
_C.DATASETS.TRAIN = () #在模型配置文件中已给出
_C.DATASETS.TEST = ()
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4 #数据生成启线程数
_C.DATALOADER.SIZE_DIVISIBILITY = 0
_C.DATALOADER.ASPECT_RATIO_GROUPING = True
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.CONV_BODY = "R-50-C4"
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2
_C.MODEL.BACKBONE.OUT_CHANNELS = 256 * 4
_C.MODEL.RPN = CN()
_C.MODEL.RPN.USE_FPN = False
_C.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
_C.MODEL.RPN.ANCHOR_STRIDE = (16,)
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
_C.MODEL.RPN.STRADDLE_THRESH = 0
_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
_C.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
_C.MODEL.RPN.NMS_THRESH = 0.7
_C.MODEL.RPN.MIN_SIZE = 0
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.USE_FPN = False
_C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
_C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
_C.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
_C.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
_C.MODEL.ROI_HEADS.NMS = 0.5
_C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100
_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (1.0 / 16,)
#数据集类别数,默认是81,因为coco数据集为80+1(背景),我的数据集只有4个类别,加上背景也就是5个类别
_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 5
_C.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
_C.MODEL.ROI_MASK_HEAD.RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
_C.MODEL.RESNETS = CN()
_C.MODEL.RESNETS.NUM_GROUPS = 1
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True
_C.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
_C.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"
_C.MODEL.RESNETS.RES5_DILATION = 1
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000 #最大迭代次数
_C.SOLVER.BASE_LR = 0.02 #初始学习率,这个通常在模型配置文件中有设置
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500 #预热迭代次数,预热迭代次数内(小于訪值)的学习率比较低
_C.SOLVER.WARMUP_METHOD = "constant" #预热策略,有'constant'和'linear'两种
_C.SOLVER.CHECKPOINT_PERIOD = 2000 #生成检查点(checkpoint)的步长
_C.SOLVER.IMS_PER_BATCH = 1 #一个batch包含的图片数量
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
_C.TEST.IMS_PER_BATCH = 1
_C.OUTPUT_DIR = "output" #主要作为checkpoint和inference的输出目录
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")


```

### 数据集路径配置文件**path_catalog**
```python
class DatasetCatalog(object):
    DATA_DIR = "datasets"#数据集根路径
 
    DATASETS = {
        "coco_2014_train": (
            "coco/train2014", #这里是訪数据集的主目录,称其为root,訪root会和标注文件中images字段中的file_name指定的路径进行拼接得到图片的完整路径
            "coco/annotations/instances_train2014.json", # 标注文件路径
        ),
        "coco_2014_val": (
            "coco/val2014", #同上
            "coco/annotations/instances_val2014.json" #同上
        ),
    }
 
    @staticmethod
    def get(name):
        if "coco" in name: #e.g. "coco_2014_train"
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs[0]),
                ann_file=os.path.join(data_dir, attrs[1]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


```

## 开始maskrcnn在coco上炼丹

### 在COCO数据集上训练
如果你对COCO数据集不太清晰，可以看这篇文章[COCO数据集标注方式](https://zhuanlan.zhihu.com/p/29393415)  
首先要正确安装好maskrcnn_benchmark，同时要下载COCO数据集，然后建一个COCO数据的软连接(源数据集的快捷方式)
```
cd maskrcnn-benchmark
mkdir -p datasets/coco
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2014 datasets/coco/train2014
ln -s /path_to_coco_dataset/test2014 datasets/coco/test2014
ln -s /path_to_coco_dataset/val2014 datasets/coco/val2014
```
或者你也可以到`maskrcnn_benchmark/config/paths_catalog.py`中修改对应的数据集路径

### 启动训练
确保已经build并且配置好数据集
使用`tools/train_net.py`是训练主程序
`tools/test_net.py`是测试程序
下面是用conda创建环境后训练的demo:  
```python
#进入maskrcnn-benchmark目录下，激活maskrcnn_benchmark虚拟环境
$ cd maskrcnn-benchmark
$ source activate pytorch
#指定模型配置文件,执行训练启动脚本
(pytorch)$python tools/train_net.py --config-file configs/adas_e2e_mask_rcnn_R_101_FPN_1x.yaml

```




### 多卡训练
使用`torch.distributed.launch`进行并行训练(为每个GPU开辟进程，支持多借点多卡训练)
```python
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS /path_to_maskrcnn_benchmark/tools/train_net.py --config-file "path/to/config/file.yaml"
```



## 核心代码讲解

```python

```


