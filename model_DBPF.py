import torch
import torch.nn as nn
from pvtv2 import pvt_v2_b3
import torch.nn.functional as F
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool, _ = torch.max(x, dim=2, keepdim=True)
        max_pool, _ = torch.max(max_pool, dim=3, keepdim=True)

        out = avg_pool + max_pool
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out * x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(torch.max(x, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
        avg_pool = F.interpolate(avg_pool, size=max_pool.shape[2:], mode='bilinear', align_corners=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        if concat.size(1) != 2:
            concat = concat[:, :2, :, :]
        out = self.conv1(concat)
        out = self.sigmoid(out)
        return out * x

class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=True, act=True):
        super().__init__()

        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)  # 先进行通道注意力处理
        x = self.spatial_attention(x)  # 然后进行空间注意力处理
        return x

class ResidualBlockWithCBAM(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

        self.cbam = CBAM(out_c)  # 加入CBAM

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.shortcut(inputs)
        x = self.relu(x1 + x2)
        x = self.cbam(x)  # 在残差块之后加入CBAM
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlockWithCBAM(in_c+out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, scale):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlockWithCBAM(in_c, out_c)

    def forward(self, inputs):
        x = self.up(inputs)
        x = self.r1(x)
        return x

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):  # 输出通道数改为2
        super(AttUNet, self).__init__()
        
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)   # 256*256*3 -> 256*256*64
        self.Conv2 = conv_block(ch_in=64, ch_out=128)      # 256*256*64 -> 256*256*128
        self.Conv3 = conv_block(ch_in=128, ch_out=256)     # 256*256*128 -> 256*256*256
        self.Conv4 = conv_block(ch_in=256, ch_out=512)     # 256*256*256 -> 256*256*512
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)    # 256*256*512 -> 256*256*1024

        self.Up5 = up_conv(ch_in=1024, ch_out=512)         # 256*256*1024 -> 256*256*512
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512) # 512 + 512 -> 256*256*512

        self.Up4 = up_conv(ch_in=512, ch_out=256)          # 256*256*512 -> 256*256*256
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)  # 256 + 256 -> 256*256*256

        self.Up3 = up_conv(ch_in=256, ch_out=128)          # 256*256*256 -> 256*256*128
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)  # 128 + 128 -> 256*256*128

        self.Up2 = up_conv(ch_in=128, ch_out=64)           # 256*256*128 -> 256*256*64
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)   # 64 + 64 -> 256*256*64

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)  # 256*256*64 -> 256*256*2
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        unet_output = self.sigmoid(d1)  # 输出层使用sigmoid激活

        return unet_output

class PVTFormer(nn.Module):
    def __init__(self, unet_model):
        super().__init__()

        """ Encoder 1"""
        self.backbone = pvt_v2_b3()  ## [64, 128, 320, 512]
        path = "./pvt_v2_b3.pth"
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        """ Channel Reduction """
        self.c1 = Conv2D(64, 64, kernel_size=1, padding=0)
        self.c2 = Conv2D(128, 64, kernel_size=1, padding=0)
        self.c3 = Conv2D(320, 64, kernel_size=1, padding=0)  # 320 -> 64

        # 解码器定义
        self.d1 = DecoderBlock(64, 64)  
        self.d2 = DecoderBlock(64, 64)
        self.d3 = UpBlock(64, 64, 4)

        # UP块定义
        self.u1 = UpBlock(64, 64, 4)
        self.u2 = UpBlock(64, 64, 8)
        self.u3 = UpBlock(64, 64, 16)

        # 残差块定义
        self.r1 = ResidualBlockWithCBAM(64 * 5, 64)  
        # 输出层定义
        self.y = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        # 更灵活的 UNet 输出通道调整
        self.unet_channel_adjust = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs, unet_output=None):
        # 确保 unet_output 的维度正确
        if unet_output is None:
            unet_output = self.unet_model(inputs)
        
        # 确保 unet_output 是 4D 张量且通道数为 1
        if unet_output.dim() == 3:
            unet_output = unet_output.unsqueeze(1)
        

        """ Encoder """
        pvt1 = self.backbone(inputs)
        e1 = pvt1[0]     
        e2 = pvt1[1]     
        e3 = pvt1[2]     

        # 三通道数都减少到64
        c1 = self.c1(e1)
        c2 = self.c2(e2)
        c3 = self.c3(e3)  

        # 解码器处理
        d1 = self.d1(c3, c2)  
        d2 = self.d2(d1, c1)  
        d3 = self.d3(d2)   

        # UP块处理
        u1 = self.u1(c1)
        u2 = self.u2(c2)
        u3 = self.u3(c3)

        # 调整 UNet 输出的通道数
        unet_output_adjusted = self.unet_channel_adjust(unet_output)

        # UP块和解码器的特征图连接
        x = torch.cat([d3, u1, u2, u3, unet_output_adjusted], axis=1)  
        x = self.r1(x)
        y = self.y(x)
        return y


# 测试模型
if __name__ == "__main__":
    # 测试 UNet
    unet_model = AttUNet()
    x = torch.randn((4, 3, 256, 256))  # 输入尺寸为 (batch_size, channels, height, width)
    unet_output = unet_model(x)

    # 测试 PVTFormer
    pvt_model = PVTFormer(unet_model)
    pvt_output = pvt_model(x, unet_output)
    print("PVTFormer output shape:", pvt_output.shape)