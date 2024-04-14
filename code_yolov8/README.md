



# 2. 环境安装（已安装好的忽略）

博客中的讲解，只是示例，具体的安装版本以下面提供的为准（安装流程不管哪个版本都是一样的）：
    
    1. python版本： 3.8.10
    
    2. cuda版本：安装哪个版本同自己的电脑显卡有关
        CUDA10.2
        CUDA11.1（建议）
        CUDA11.3

    3. torch版本：需要同安装的cuda进行匹配
        CUDA10.2 安装：torch1.9.0==cuda10.2
        CUDA11.1 安装：torch1.9.0==cuda11.1 （建议）
        CUDA11.3 安装：torch1.10.0==cuda11.3

    4. 其他的第三方库版本见：requirements.txt


## 具体安装步骤：

下载百度网盘链接：

    链接：https://pan.baidu.com/s/1Cd-9cQhKsKDSv9YK7y8RQg?pwd=lqgy 
    提取码：lqgy
    --来自百度网盘超级会员V6的分享

里面包含所有的需要的环境，下载完成后按以下步骤完成安装：

###（1） Python环境安装 

安装 python3.8.10 版本，文件在python3.8.10文件夹中，双击python-3.8.10-amd64.exe即可，若不会可参考下面的博客内容，

参考博客：https://blog.csdn.net/qq_28949847/article/details/132891691

###（2） CUDA、cudnn环境安装

安装哪个版本依据自己的电脑硬件确定，如何查询支持那个版本，看下面的参考博客链接

参考博客：https://blog.csdn.net/qq_28949847/article/details/125081074

CUDA文件夹中一共提供了10.2、11.1、11.3 三个版本的CUDA，并且里面也提供了相对应的 torch-GPU版 whl文件
    
###（3） torch-GPU版安装

whl文件 同 cuda文件在同一个文件夹下，在当前目录下打开命令提示符窗口，直接执行下面命令(以torch-1.9.0+cu111-cp38-cp38-win_amd64.whl为例)

    pip install torch-1.9.0+cu111-cp38-cp38-win_amd64.whl
    
    pip install torchvision-0.10.0+cu111-cp38-cp38-win_amd64.whl

具体命令只需替换后面的文件名即可。

参考博客：https://blog.csdn.net/qq_28949847/article/details/132911284

###（4）第三方依赖包安装

资源中提供的Lib文件夹是所有的第三方依赖包，打开 命令提示符窗口，cd到当前项目路径下，

直接运行：

    pip install --no-index --find-links=whl -r requirements.txt

自动安装环境。

就此，所有环境已经完全安装完成，直接执行 main.py 即可完成测试效果。

### （5）pycharm安装

参考博客：https://blog.csdn.net/qq_28949847/article/details/132921748?spm=1001.2014.3001.5502




# 3. yolov5 目标检测训练
## 训练步骤
    
# 以下都是以 yolov5_62为 root路径, 自己训练的步骤

1. 准备数据集（以VOC.yaml数据集为例）

2. 使用datasets文件夹下的voc2v5.py 将xml文件转为txt文件

3. 参照datasets文件夹下创建的示例，创建文件夹结构，并将数据集放入在对应的文件夹下

4. 修改 data文件夹下的VOC.yaml
    # 数据集路径
    path: D:/lg/BaiduSyncdisk/project/person_code/project_self/Yolov5_OCtrack/datasets/airplane
    train: # train数据集
      - train
    val: # val 数据集
      - train
    test: # test 数据集
      - train

    # 修改为自己的类
    names:
      0: airplane


5. 修改 train.py文件夹下的参数
    parser.add_argument('--weights', type=str,
                        default='D:/lg/BaiduSyncdisk/project/person_code/project_self/Yolov5_OCtrack/OCTrack_yolov5/weights/yolov5s.pt',
                        help='initial weights path')
    parser.add_argument('--cfg', type=str, default='./models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/VOC.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')

    ....

6. 点击运行，即可训练

7. 模型保存在 runs 文件夹下
