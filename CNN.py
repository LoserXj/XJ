import torch 
from Focal_loss import focal_loss
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        '''
        一般来说，卷积网络包括以下内容：
        1.卷积层
        2.神经网络
        3.池化层
        '''
        self.conv1=nn.Sequential(
            nn.Conv2d(              #--> (3,256,256)
                in_channels=3,      #传入的图片是几层的，灰色为1层，RGB为三层
                out_channels=16,    #输出的图片是几层
                kernel_size=5,      #代表扫描的区域点为5*5
                stride=1,           #就是每隔多少步跳一下
                padding=2,          #边框补全，其计算公式=（kernel_size-1）/2=(5-1)/2=2
            ),    # 2d代表二维卷积           --> (16,256,256)
            nn.ReLU(),              #非线性激活层
            nn.MaxPool2d(kernel_size=2),    #设定这里的扫描区域为2*2，且取出该2*2中的最大值          --> (16,128,128)
        )
 
        self.conv2=nn.Sequential(
            nn.Conv2d(              #       --> (16,128,128)
                in_channels=16,     #这里的输入是上层的输出为16层
                out_channels=32,    #在这里我们需要将其输出为32层
                kernel_size=5,      #代表扫描的区域点为5*5
                stride=1,           #就是每隔多少步跳一下
                padding=2,          #边框补全，其计算公式=（kernel_size-1）/2=(5-1)/2=
            ),                      #   --> (32,128,128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    #设定这里的扫描区域为2*2，且取出该2*2中的最大值     --> (32,64,64)，这里是三维数据
        )
        
        self.conv3=nn.Sequential(
            nn.Conv2d(              #       --> (32,64,64)
                in_channels=32,     #这里的输入是上层的输出为32层
                out_channels=128,    #在这里我们需要将其输出为64层
                kernel_size=5,      #代表扫描的区域点为5*5
                stride=1,           #就是每隔多少步跳一下
                padding=2,          #边框补全，其计算公式=（kernel_size-1）/2=(5-1)/2=
            ),                      #   --> (128,64,64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    #设定这里的扫描区域为2*2，且取出该2*2中的最大值          --> (128,32,32)
        )
        
        # self.out=nn.Linear(128*32*32,128*32*32)       #注意一下这里的数据是二维的数据
 
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)     #（batch,32,7,7）
        x=self.conv3(x)
        #然后接下来进行一下扩展展平的操作，将三维数据转为二维的数据
        x=x.view(x.size(0),-1)    #(batch ,32 * 7 * 7)
        # output=self.out(x)
        out = x.reshape(-1,256*256,2)
        return out
    

class CNNArea(nn.Module):
    def __init__(self):
        super(CNNArea,self).__init__()
        '''
        一般来说，卷积网络包括以下内容：
        1.卷积层
        2.神经网络
        3.池化层
        '''
        self.conv1=nn.Sequential(
            nn.Conv2d(              #--> (3,256,256)
                in_channels=3,      #传入的图片是几层的，灰色为1层，RGB为三层
                out_channels=16,    #输出的图片是几层
                kernel_size=5,      #代表扫描的区域点为5*5
                stride=1,           #就是每隔多少步跳一下
                padding=2,          #边框补全，其计算公式=（kernel_size-1）/2=(5-1)/2=2
            ),    # 2d代表二维卷积           --> (16,256,256)
            nn.ReLU(),              #非线性激活层
            nn.MaxPool2d(kernel_size=2),    #设定这里的扫描区域为2*2，且取出该2*2中的最大值          --> (16,128,128)
        )
 
        self.conv2=nn.Sequential(
            nn.Conv2d(              #       --> (16,128,128)
                in_channels=16,     #这里的输入是上层的输出为16层
                out_channels=32,    #在这里我们需要将其输出为32层
                kernel_size=5,      #代表扫描的区域点为5*5
                stride=1,           #就是每隔多少步跳一下
                padding=2,          #边框补全，其计算公式=（kernel_size-1）/2=(5-1)/2=
            ),                      #   --> (32,128,128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    #设定这里的扫描区域为2*2，且取出该2*2中的最大值     --> (32,64,64)，这里是三维数据
        )
        
        self.conv3=nn.Sequential(
            nn.Conv2d(              #       --> (32,64,64)
                in_channels=32,     #这里的输入是上层的输出为32层
                out_channels=64,    #在这里我们需要将其输出为64层
                kernel_size=5,      #代表扫描的区域点为5*5
                stride=1,           #就是每隔多少步跳一下
                padding=2,          #边框补全，其计算公式=（kernel_size-1）/2=(5-1)/2=
            ),                      #   --> (64,64,64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    #设定这里的扫描区域为2*2，且取出该2*2中的最大值          --> (64,32,32)
        )
        
        self.out=nn.Linear(64*32*32,256)       
        self.out1 = nn.Linear(256,1)
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)     #（batch,32,7,7）
        x=self.conv3(x)
        #然后接下来进行一下扩展展平的操作，将三维数据转为二维的数据
        x=x.view(x.size(0),-1)    #(batch ,32 * 7 * 7)
        out=self.out(x)
        out =self.out1(out)
        return out