---
layout: post  
title: "可视化版本管理SourceTree教程"  
date: 2019-1-2 21:08 +800  
categories: jekyll update  
---

# SourceTree使用教程
* 检出分支
<<<<<<< HEAD
![分支](/img/sourcetree_checkout.png)  

=======
>>>>>>> featrue_1
通常项目是三种分支类型：
 * master:主分支，稳定版本；
 * develop: 日常开发分支，每个项目版本稳定之后，合并到master分支；
 * fix： 线上紧急bug,从master创建fix分支，修复测试通过后，合并到master主分支


<<<<<<< HEAD
![检出](/img/sourcetree_branch.png)
#### 暂存、丢弃、移除的区别
![out](/img/sourcetree_drop.png)
=======

#### 暂存、丢弃、移除的区别
>>>>>>> featrue_1
* 暂存文件： 打钩为咱村的文件，会将文件放在已暂存文件筐中
* 丢弃文件： 放弃该次的更改，会改变workspace的文件内容，表示此次工作区文件更改不被认可
* 移除文件： 指的是从工作区删除文件
>例如：你在txt文件中添加了几句话，用丢弃文件，表示删除你刚才增加的那两句话；如果选择移除文件，表示将该txt在工作目录中删除

将修改的文件放入暂存区之后，可以在本地提交(commit)

### 本地提交后，先拉取(pull)，后(push)，养成良好习惯
虽说SourceTree中不拉取，直接推送，工具会检测如果代码冲突会让你修改冲突后再提交
* 拉取(pull)： 把项目中这个分支的远程仓库拉到本地合并处理
* 推送(push)： 把本地这个分支合并后的代码，推送到远程仓库
* 抓取(fetch)：仅仅把远程仓库拿下来，不与本地仓库合并


### 发生冲突与解决
1. 多人修改同一文件，提交代码或合并会造成冲突；
2. 提示文件冲突，取文件状态中找带感叹号的文件，
3. 出现冲突后回到工作区域(workspace), 对比暂存区和工作区内容，放弃其中某一个更改  

### 代码回退 和 将master重置到这次提交 之间的区别
代码回退： 撤销此次提交的更改 (git reset)撤销上一次的commit
将master重置到这次提交： **将版本变为这个选中版本**
区别：
A->B->C 回滚
## 注意
双击每一个分支就是检出(checkout)，前面会有一个空心的⭕️， 创建分支是以当前圆圈所在位置开辟分支，合并也是遵循该原则  
创建完分支之后
![]

>回滚提交会取消上一次的commit，算一次revert，同时这次revert也会被提交，可以看到文件夹中之前上传的的文件没了
![回滚图片](/img/revert.png)


## 分支合并
先checkout到目的分支,如果打算合并branch1到branch2,先checkout到branch2,然后在branch1的哪个地方右键合并到当前
![xs](/img/sourcetree_mergeBranch_2.png)
![xs](/img/sourtree_mergeBranch.png)