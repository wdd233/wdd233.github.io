

## 安装cuda-8.0
1. 官网下载：https://developer.nvidia.com/cuda-toolkit-archive  

   具体过程如下：
   进入cuda_8.0.61_375.26_linux.run所在目录，执行下面命令：  
   `$ sudo sh cuda_8.0.61_375.26_linux.run`

    有可能要加 --override 参数：这个参数会覆盖原来的驱动，如果没出现问题可以不需要这个参数。  
    `sh cuda_8.0.61_375.26_linux.run --override`

   **安装过程中....请按下面要求选择：**  
   启动安装程序，一直按空格到最后，输入accept接受条款  
   输入n--不安装nvidia图像驱动，之前已经安装过了！  
   输入y--安装cuda 8.0工具  
   回车确认--cuda默认安装路径：/usr/local/cuda-8.0  
   输入y--用sudo权限运行安装，输入密码  
   输入y或者n--安装或者不安装指向/usr/local/cuda的符号链接  
   输入y--安装CUDA 8.0 Samples，以便后面测试  
   回车确认--CUDA 8.0 Samples默认安装路径：/home/pawn（pawn是我的用户名），该安装路径测试完可以删除！  

   ----------------------------------------------------------------------------------------

2. 添加环境变量
` $ sudo gedit ~/.bashrc`
   在最后添加下面内容：
   ```
export PATH=/usr/local/cuda-8.0/bin:$PATH`
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
```
   保存退出后使其立刻生效
   $ source ~/.bashrc 

3. 查看cuda版本
   $ nvcc --version
   最后进入cuda自带的例子中看cuda是否已正确配置

4. 测试CUDA的sammples
   `$ cd /usr/local/cuda-8.0/samples/1_Utilities/deviceQuery    #由自己电脑目录决定`  
   `$ sudo make -j16`
   ------------------------------------------------------------------------------------------------  
   这里可能出现gcc、g++版本过高的问题。（没出错，下面的过程不需要做！！）  
   具体解决方法见：（https://blog.csdn.net/lee_j_r/article/details/52693724）  
   ubuntu16.04 默认安装的gcc版本为gcc-5.4，  
 （可用gcc --version查看）有时可能需要低版本的，所以我们先安装gcc-4.8.5 或gcc-4.7.4   
   sudo apt-get install -y gcc-4.8 g++-4.8  
   sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 20 
   sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 20 
   cd /usr/bin  
   sudo cp gcc gcc_backup 
   sudo cp g++ g++_backup 
   sudo rm gcc g++ 
   sudo ln -s gcc-4.8 gcc 
   sudo ln -s g++-4.8 g++


   sudo rm gcc g++ 
   sudo ln -s gcc-5 gcc 
   sudo ln -s g++-5 g++
   
   $ ./deviceQuery

   当结果显示最后部分有GeForce GTX 1080Result = PASS，说明cuda已经安装成功！

5. 如果需要删除cuda，就执行命令： sudo rm -r cuda-8.0
 

## 安装cudnn-8.0
   https://blog.csdn.net/QLULIBIN/article/details/80728355
   http://blog.csdn.net/jzrita/article/details/72887677      装显卡cudnn  caffe的依耐和python 

1. 拷贝下载文件和解压 
   $ cp  cudnn-8.0-linux-x64-v5.1.solitairetheme8 cudnn-8.0-linux-x64-v5.1.tgz
   $ tar -xvf cudnn-8.0-linux-x64-v5.1.tgz            #解压格式
   
2. 完成后解压，得到一个 cuda 文件夹，该文件夹下include 和 lib64 两个文件夹
   $ cd cuda/include                                  #命令行进入 cuda/include 路径下
   $ sudo cp cudnn.h /usr/local/cuda/include/         #复制头文件
   $ cd cuda/lib64                                    #命令行进入 cuda/lib64 路径下
   $ sudo cp lib* /usr/local/cuda/lib64/              #复制动态链接库
   
3. 建立软链接
   $ cd /usr/local/cuda/lib64/
   $ sudo rm -rf libcudnn.so libcudnn.so.5            #删除原有动态文件
   $ sudo ln -s libcudnn.so.5.1.10 libcudnn.so.5      #生成软衔接
 请具体见路径“/usr/local/cuda/lib64”下的“libcudnn.so.5.1.10”文件编号，不正确设置的话，将影响caffe安装！！
>软连接就是win下的快捷方式

  

# 安装opencv3.1环境

   http://blog.csdn.net/xierhacker/article/details/53035989
   https://www.cnblogs.com/go-better/p/7161006.html

   首先肯定是先安装依赖了，官方列出了一些：
```
   $ sudo apt-get install build-essential
   $ sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
   $ sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
   $ sudo apt-get install --assume-yes libopencv-dev libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libqt4-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils unzip 
    注意assume-yes前面是2个“-”
    $sudo apt-get install ffmpeg libopencv-dev libgtk-3-dev python-numpy python3-numpy libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libv4l-dev libtbb-dev qtbase5-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev
 ```

   反正不管了，全部都装上去。

   -------------------------------------------------------------------------------------------------------------------------------------
   opencv3.1 安装方法1---建议首选！！！！
   安装参考方法：https://blog.csdn.net/u012841667/article/details/53501879 

   1. 进入官网: http://opencv.org/releases.html, 选择3.1.0版本的source下载 opencv 
   具体步骤请参考：https://blog.csdn.net/baobei0112/article/details/77996369    
                  https://blog.csdn.net/yhaolpz/article/details/71375762  和上面网站内容一样的！

   2. 解压拷贝到需要安装的Ubuntu目录：unzip opencv-3.1.0.zip
   3. 进入到你下载的那个opencv文件夹，这时候建立一个build的文件夹，用来接收cmake之后的文件
      $ sudo mkdir build   
   4. 进入到build里面，运行这句命令
      $ cd build    
      $ sudo cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
        caffe-ssd测试，OpenCV编译需要选择开启Qt支持，开启gtk支持！！！！
        sudo cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_QT=ON -D WITH_GTK=NO -D CMAKE_INSTALL_PREFIX=/usr/local ..
      上述步骤参考网站！！https://blog.csdn.net/huangkangying/article/details/53406370  

   ------------------------------------------------------------------------------------
       cmake过程中，网络不给力的话，将出现 ICV: Downloading ippicv_linux_20151201.tgz......之后就不动了。按下步骤可解决：
   ubuntu下安装opencv3.1出现-- ICV: Downloading ippicv_linux_20151201.tgz...之后就会提示很多错误。
   1）需要自己下载ippicv_linux_20151201.tgz
   2）然后在OpenCV源代码的根目录下创建目录：linux-808b791a6eac9ed78d32a7666804320e
   具体路径：opencv-3.1.0/3rdparty/ippicv/downloads/linux-808b791a6eac9ed78d32a7666804320e
   3）将下载后的ippicv_linux_20151201.tgz文件拷进去。
   4）然后再次cmake就OK 了！！
  ------------------------------------------------------------------------------------

      cmake过程中，网络不给力的话，将出现 ICV: Downloading ippicv_2017u2_lnx_intel64_20170418.tgz...之后就不动了。按下步骤可解决：
      （1）下载ippicv_2017u2_lnx_intel64_20170418.tgz文件
            ippicv_2017u2_lnx_intel64_20170418.tgz 的github网址 ：    
            https://github.com/opencv/opencv_3rdparty/tree/ippicv/master_20170418/ippicv

     （2） 修改  ippicv.cmake 第47行
       sudo gedit /home/liu/opencv-3.4.0/3rdparty/ippicv/ippicv.cmake   --打开ippicv.cmake文件找到第47行
        原来： "https://raw.githubusercontent.com/opencv/opencv_3rdparty/${IPPICV_COMMIT}/ippicv/"
        修改：“/home/liu/下载”      ---指定ippicv_2017u2_lnx_intel64_20170418.tgz 文件所在路径 
    （3）然后再次cmake就OK 了！！
 ------------------------------------------------------------------------------------

  5. 编译OpenCV:
      $ sudo make -j8
      $ sudo make install
        你能够在/usr/local中找到你新安装的opencv的一些头文件和库了。
       这里要说明一下，要是中途出现了一些问题是与cuda有关的，打开opencv下面那个cmakelist文件把with_cuda设置为OFF，如下图，之后再cmake，再编译。

   6. 进入测试OPENCV环节
    ------------------------------------------------------------------------------------------------------------------------------------
   opencv3.1 安装方法2

   1. 建立一个OpenCV 工作目录
   在你喜欢的地方建立一个工作目录，随便什么名字，就在home目录下面建立了一个OpenCV的目录.
   $ mkdir OpenCV 

   2. 进入这个工作目录（OpenCV）然后用git克隆官方的项目（下载接受会需要一点时间，等待）
   $ cd Opencv
   $ git clone https://github.com/opencv/opencv.git
   $ git clone https://github.com/opencv/opencv_contrib.git

   克隆好了之后，你就会看见你的工作目录（OpenCV）下面有了两个项目的文件夹opencv了。进入到你下载的那个opencv文件夹，这时候建立一个build的文件夹，用来接收cmake之后的文件。
   $ sudo mkdir build
   进入到build里面，运行这句命令（直接复制就行）
   $ cd build   
   3. 编译

      $ sudo cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
        这里要解释一下，后面是一个空格加上两个点，不要搞错了。

      $ sudo make -j8
        -j8表示并行计算，根据自己电脑的配置进行设置，配置比较低的电脑可以将数字改小或不使用，直接输make。

      ============================================================================================

      如果安装过程到90%左右报错的话就按下面的方法修改：

      进入opencv-3.1.0/modules/cudalegacy/src/目录，修改graphcuts.cpp文件，将：
        #include "precomp.hpp"
        #if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)
      改为
        #include "precomp.hpp"
        #if !defined (HAVE_CUDA) || defined (CUDA_DISABLER) || (CUDART_VERSION >= 8000) 
      然在执行sudo make -j8 编译就可以了！！！
      ============================================================================================
      以上只是将opencv编译成功，还没将opencv安装，需要运行下面指令进行安装：
      $ sudo make install

查看linux下的opencv安装库：    pkg-config opencv --libs
查看linux下的opencv安装版本：pkg-config opencv --modversion
查看linux下的opencv安装路径：sudo find / -iname "*opencv*"
 

   5. 进入测试OPENCV环节
----------------------------------------------------------------------------------------------------------------------------------------
 测试OPENCV步骤如下：
===mkdir DisplayImage 
===cd DisplayImage
===gedit DisplayImage.cpp   #编辑测试代码
----------------------------------------------------------------------
//测试代码

  

----------------------------------------------------------------------
   $ gedit CMakeLists.txt   #编写make文件
----------------------------------------------------------------------
cmake_minimum_required(VERSION 3.5) 
project(DisplayImage) 
find_package(OpenCV REQUIRED) 
add_executable(DisplayImage DisplayImage.cpp) 
target_link_libraries(DisplayImage ${OpenCV_LIBS}) 

SET( CMAKE_CXX_FLAGS "-std=c++11 -O3") 
----------------------------------------------------------------------
   $ cmake . 
   $ make  
   $ ./DisplayImage 1.jpg
   $ ./DisplayImage 2.jpg 

# 安装caffe

    https://blog.csdn.net/jzrita/article/details/72887677   --caffe依赖
    https://blog.csdn.net/baobei0112/article/details/77996369   --安装 caffe在第9步 

    Caffe是由BVLC开发的一个深度学习框架，主要由贾扬清在UC Berkeley攻读PhD期间完成。参考官网上的教程以及Github上针对Ubuntu15.04和16.04的教程。从官方下载caffe源包caffe-master。

 

1. 安装库文件：
    $ sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libboost-all-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler

2. 安装依赖：  

    $ sudo apt-get install -y build-essential cmake git pkg-config
    $ sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler   libatlas-base-dev libgflags-dev libgoogle-glog-dev liblmdb-dev
    $ sudo apt-get install --no-install-recommends libboost-all-dev 

3. Python接口依赖： 

    $ sudo apt-get install  the python-dev
    $ sudo apt-get install -y python-pip
    $ sudo apt-get install -y python-dev
    $ sudo apt-get install -y python-numpy python-scipy      # (Python 2.7 development files)
    $ sudo apt-get install -y python3-dev
    $ sudo apt-get install -y python3-numpy python3-scipy    # (Python 3.5 development files)


4. 从官方下载caffe源包caffe-master
       https://github.com/BVLC/caffe
       下载后解压到你需要的安装目录，进入caffe文件：cd caffe或cd caffe-master，执行下面语句！

    -------------------------------------------------------------------------------------

    注意执行命令前需要做下面命令否则提示：
```
liuq@liuq:~/caffe-master$ sudo make clean
make: /usr/local/MATLAB/R2014a/bin/mexext：命令未找到
liuq@liuq:~/caffe-master$ make all -j8
make: /usr/local/MATLAB/R2014a/bin/mexext：命令未找到
```
```
        ===========解决方法！如下===============
将Makefile 和 Makefile.config拷贝到caffe-master文件夹替换原文件
方法1：将Makefile 和 Makefile.config 两个文件复制到caffe文件夹中替换已有的两个文件！
方法2：直接修改 Makefile 和 Makefile.config 两个文件中的相关部分！
```
    -------------------------------------------------------------------------------------

     $ make clean
    $ make all -j8
    在上一步成功安装 caffe 之后，就可以通过 caffe 去做训练数据集或者预测各种相关的事了。

5. 安装 pycaffe notebook 接口环境
    $ cd caffe              #如当前不再caffe目录，那就要进入caffe路径！如果当前已在此目录那就不需执行cd caffe！
    $ make pycaffe    #或执行:make pycaffe -j8 

6. caffe配置matlab接口,可以在matlab安装后在来配置都可以！
    $ make matcaffe  
    #或执行:make matcaffe -j8 编译成功后生成文件./matlab/+caffe/private/caffe_.mexa64 供matlab使用  

7.  bashrc中添加环境变量（caffe中的python接口）
   $ sudo gedit ~/.bashrc
    export PYTHONPATH=/home/liu/caffe-master/python:$PYTHONPATH
   $ source ~/.bashrc
    查看环境变量路径：echo $PYTHONPATH

## 测试Caffe
    1. MNIST训练之前将Caffe的环境搭好了，现在用MNIST这个数据集进行测试，继续在$CAFFE_ROOT下进行操作。
    ```
    $ ./data/mnist/get_mnist.sh
    $ ./examples/mnist/create_mnist.sh 
    ```
    2. 经过上述操作./examples/mnist/路径下会有mnist_test_lmdb和mnist_train_lmdb两个文件夹，分别是测试和训练数据。 
    （1）如果要指定选择CPU或GPU训练,需要修改lenet_solver.prototxt文件
    `$ sudo gedit ./examples/mnist/lenet_solver.prototxt`  
 在最后一句话为设置为：solver_mode: CPU，指定在CPU上进行训练
 在最后一句话为设置为：solver_mode: GPU，指定在GPU上进行训练  

 （2）如果你还要指定GPU运行的话，需要修改train_lenet.sh文件
```
   ./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt $       #默认选择的是0号GPU
 改为：./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt --gpu 1 $ #选择1号GPU 
   $ cd caffe            #进入caffe路径
   $ ./examples/mnist/train_lenet.sh
```
网友其他安装caffe参考： https://blog.csdn.net/Rasin_Wu/article/details/80294822 
# ======================caffe常用问题及其解决方法！！=====================

ERROR提示1
liu@liu:~/caffe-master$ python

>>> import caffe


>>> quit()
解决办法：
liu@liu:~/caffe-master$ python2


>>> import caffe
>>> quit()

caffe 官网 (http://caffe.berkeleyvision.org/ ) 上也提示说, 只是较好的支持 caffe 2.7 版本；

对于其他的版本，需要自己进行摸索咯。如下所示，如果输入python显示的版本是python3就会出错！！
liu@liu:~/caffe-master$ python

 

ERROR提示2
liuq@dl:~/caffe-master$ python2
Python 2.7.12 (default, Dec 4 2017, 14:50:18) 
[GCC 5.4.0 20160609] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import caffe


解决办法：skimage(scikit-image)

python-skimage包依赖于matplotlib，scipy，pil，numpy和six。
首先安装依赖包：
sudo apt-get install python-matplotlib python-numpy python-pil python-scipy
sudo apt-get install build-essential cython
安装skimage包：sudo apt-get install python-skimage 

 
基于python的图片处理包还是很多的，比如PIL,Pillow, opencv, scikit-image等。其中，PIL和Pillow只提供最基础的数字图像处理，功能有限；opencv本质上是一个c++库，只是提供了python的接口，官方更新速度非常慢，而且安装很不容易。综合来看，scikit-image是基于scipy的一款图像处理包，它功能非常齐全，同时将图片作为numpy数组进行处理，几乎集合了matlab的所有图像处理功能，在数据的处理方式，函数名字等方面对matlab的模仿姿势非常到位。
更为重要的是，作为python的一个图像处理包，这个包是完全开源免费的，而且可以依托于python强大的功能，与tensorflow等软件配合使用于主流的深度学习等领域。因此，用scikit-image进行图像处理是一个非常好的选择，matlab能干的事情它都能干，matlab不能干的事情他也可以，个人认为在数字图像处理领域，它是完全可以代替matlab的。?
 ??

ERROR提示3
运行程序时，显示No module named model_libs


解决方法：主要是由于Python 路径错误
    使用命令：echo $PYTHONPATH 弹出当前python路径，看看是不是caffe自己的python接口，即/home/用户名/你的 caffe 路径/python。
    如果不是，采用 export PYTHONPATH=/home/用户名/你的caffe 路径/python即可。
    如果不对的话就按下面方法，添加caffe中的python环境变量！！
   $ sudo gedit ~/.bashrc
   export PYTHONPATH=/home/liu/caffe-ssd/python:$PYTHONPATH
   $ source ~/.bashrc 

ERROR提示4
import caffe 不成功, 报错“No module named google.protobuf.internal”
解决办法：
   $ pip install protobuf 
   $ sudo apt-get install python-protobuf ?? 

运行程序时，显示ImportError: No module named _caffe
解决：执行sudo make pycaffe
运行程序时，显示ImportError: No module named _tkinter, please install the python-tk package ubuntu运行tkinter错误
解决:  apt-get install python-tk

运行程序时，显示No module named easydict
解决方法:  pip install easydict
