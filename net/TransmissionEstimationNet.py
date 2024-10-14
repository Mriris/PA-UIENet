import torch
import torch.nn as nn
import torch.nn.functional as F

from LAB import LAB  # 假设这些模块定义了需要的功能
from LCH import LCH
from block import Block
from CMSFFT import CMSFFT
from IntmdSequential import IntmdSequential
from PositionalEncoding import PositionalEncoding
from SGFMT import SGFMT
from Transformer import Transformer
from Ushape_Trans import UshapeTrans
from utils import Utils


class TransmissionEstimationNet(nn.Module):
    def __init__(self):
        super(TransmissionEstimationNet, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # 定义反卷积层（上采样）
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1)
        # 定义批量归一化层
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(64)
    def forward(self, x):
        # 编码阶段
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        # 解码阶段
        y1 = F.relu(self.bn5(self.deconv1(x4) + x3))  # 跳跃连接
        y2 = F.relu(self.bn6(self.deconv2(y1) + x2))  # 跳跃连接
        y3 = F.relu(self.bn7(self.deconv3(y2) + x1))  # 跳跃连接
        y4 = torch.sigmoid(self.deconv4(y3))  # 输出为[0, 1]范围
        return y4


# 实例化并测试网络
model = TransmissionEstimationNet()

# 假设输入是一张3通道的RGB图像，尺寸为256x256
input_tensor = torch.randn(1, 3, 256, 256)

# 前向传播
output_tensor = model(input_tensor)

# 打印输出尺寸
print(output_tensor.size())
