from torch import nn
from models.densenet3d import densenet59
from models.densenet import densenet121,densenet169,densenet201,densenet161
import torch.nn.functional as F
from models.layers import SaveFeatures,UnetBlock_,UnetBlock,UnetBlock3d_,UnetBlock3d
import torch
from functools import partial




nonlinearity = partial(F.relu, inplace=True)

class Projector(nn.Module):
    def __init__(self, in_dim, out_dim, downsample=True):
        super(Projector, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.downsample = downsample
        self.conv1 = nn.Conv2d(self.in_dim, self.in_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1, stride=1)

    def forward(self, x):
        if self.downsample:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x

def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

class DenseUnet_2d_pro(nn.Module):

    def __init__(self, out_ch,densenet='densenet161'):
        super().__init__()

        if densenet == 'densenet121':
            base_model = densenet121
        elif densenet == 'densenet169':
            base_model = densenet169
        elif densenet == 'densenet201':
            base_model = densenet201
        elif densenet == 'densenet161':
            base_model = densenet161
        else:
            raise Exception('The Densenet Model only accept densenet121, densenet169, densenet201 and densenet161')

        layers = list(base_model(pretrained=True).children())
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers[0]

        self.sfs = [SaveFeatures(base_layers[0][2])]
        self.sfs.append(SaveFeatures(base_layers[0][4]))
        self.sfs.append(SaveFeatures(base_layers[0][6]))
        self.sfs.append(SaveFeatures(base_layers[0][8]))

        self.up1 = UnetBlock_(2208, 2112, 768)
        self.up2 = UnetBlock(768, 384, 768)
        self.up3 = UnetBlock(384, 96, 384)
        self.up4 = UnetBlock(96, 96, 96)

        self.conv1 = nn.Conv2d(96, 64, kernel_size=3, padding=1)


        self.projector = Projector(96, 128)

        self.side1 = nn.Conv2d(768, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(384, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(96, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(4 * out_ch, out_ch, 1)
        #  init my layers
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.side1.weight)
        nn.init.xavier_normal_(self.side2.weight)
        nn.init.xavier_normal_(self.side3.weight)
        nn.init.xavier_normal_(self.side4.weight)


    def forward(self, x, dropout=True):
        x = F.relu(self.rn(x))  # 10,2208,16,16
        x1 = self.up1(x, self.sfs[3].features)  # 10,768,32,32
        x2 = self.up2(x1, self.sfs[2].features)  # 10,384,64,64
        x3 = self.up3(x2, self.sfs[1].features)  # 10,96,128,128
        x4 = self.up4(x3, self.sfs[0].features)  # 10,96,256,256

        fea_map = self.projector(x4)



        x_fea = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)  # 10,96,512,512
        x_fea = self.conv1(x_fea)  # 10,64,512,512
        d4 = self.side4(x_fea)

        d3 = self.side3(x3)
        d3 = _upsample_like(d3, d4)

        d2 = self.side2(x2)
        d2 = _upsample_like(d2, d4)

        d1 = self.side1(x1)
        d1 = _upsample_like(d1, d4)
        if dropout:
            d4 = F.dropout2d(d4, p=0.3)  # 10,64,512,512
            d3 = F.dropout2d(d3, p=0.3)
            d2 = F.dropout2d(d2, p=0.3)
            d1 = F.dropout2d(d1, p=0.3)


        d0 = self.outconv(torch.cat((d1, d2, d3, d4), 1))


        return [d0,d1,d2,d3,d4,fea_map]

    def close(self):
        for sf in self.sfs: sf.remove()

class DenseUnet_2d_new(nn.Module):

    def __init__(self, densenet='densenet161'):
        super().__init__()

        if densenet == 'densenet121':
            base_model = densenet121
        elif densenet == 'densenet169':
            base_model = densenet169
        elif densenet == 'densenet201':
            base_model = densenet201
        elif densenet == 'densenet161':
            base_model = densenet161
        else:
            raise Exception('The Densenet Model only accept densenet121, densenet169, densenet201 and densenet161')

        layers = list(base_model(pretrained=True).children())
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers[0]

        self.sfs = [SaveFeatures(base_layers[0][2])]
        self.sfs.append(SaveFeatures(base_layers[0][4]))
        self.sfs.append(SaveFeatures(base_layers[0][6]))
        self.sfs.append(SaveFeatures(base_layers[0][8]))


        self.up1 = UnetBlock_(2208,2112,768)
        self.up2 = UnetBlock(768,384,768)
        self.up3 = UnetBlock(384,96, 384)
        self.up4 = UnetBlock(96,96, 96)


        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 2, kernel_size=1, padding=0)

        self.projector = Projector(96, 128)




        #  init my layers
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

    def forward(self, x, dropout=True):
        x = F.relu(self.rn(x))#10,2208,16,16
        x = self.up1(x, self.sfs[3].features)#10,768,32,32
        x = self.up2(x, self.sfs[2].features)#10,384,64,64
        x = self.up3(x, self.sfs[1].features)#10,96,128,128
        x = self.up4(x, self.sfs[0].features)#10,96,256,256

        fea_map = self.projector(x)

        # x_fea = self.deconv(x)
        x_fea = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)#10,96,512,512
        x_fea = self.conv1(x_fea)#10,64,512,512
        if dropout:
            x_fea = F.dropout2d(x_fea, p=0.3)#10,64,512,512
        x_fea = F.relu(self.bn1(x_fea))#10,64,512,512
        x_out = self.conv2(x_fea)#10,2,512,512

        return [x_out,fea_map]

    def close(self):
        for sf in self.sfs: sf.remove()

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class mini_DACblock(nn.Module):
    def __init__(self, channel):
        super(mini_DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        # self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate1(x)))
        # dilate4_out = nonlinearity(self.conv1x1(self.dilate3(x)))
        # out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        out = x + dilate1_out + dilate2_out + dilate3_out
        return out

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')
        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes,ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)



class DenseUnet_2d_ce(nn.Module):

    def __init__(self, densenet='densenet161'):
        super().__init__()

        if densenet == 'densenet121':
            base_model = densenet121
        elif densenet == 'densenet169':
            base_model = densenet169
        elif densenet == 'densenet201':
            base_model = densenet201
        elif densenet == 'densenet161':
            base_model = densenet161
        else:
            raise Exception('The Densenet Model only accept densenet121, densenet169, densenet201 and densenet161')

        layers = list(base_model(pretrained=True).children())
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers[0]

        self.sfs = [SaveFeatures(base_layers[0][2])]
        self.sfs.append(SaveFeatures(base_layers[0][4]))
        self.sfs.append(SaveFeatures(base_layers[0][6]))
        self.sfs.append(SaveFeatures(base_layers[0][8]))


        self.up1 = UnetBlock_(2212,2112,768)
        self.up2 = UnetBlock(768,384,768)
        self.up3 = UnetBlock(384,96, 384)
        self.up4 = UnetBlock(96,96, 96)


        self.spp = SPPblock(2208)
        self.cam = ChannelAttention(2212)

        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 2, kernel_size=1, padding=0)


        #  init my layers
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

    def forward(self, x, dropout=True):
        x = F.relu(self.rn(x))#10,2208,16,16


        x = self.spp(x)#10,2212,16,16
        attention_value = self.cam(x)  # 10,2208,16,16
        x = x.mul(attention_value)
        x = self.up1(x, self.sfs[3].features)#10,768,32,32
        x = self.up2(x, self.sfs[2].features)#10,384,64,64
        x = self.up3(x, self.sfs[1].features)#10,96,128,128
        x = self.up4(x, self.sfs[0].features)#10,96,256,256


        # x_fea = self.deconv(x)
        x_fea = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)#10,96,512,512
        x_fea = self.conv1(x_fea)#10,64,512,512
        if dropout:
            x_fea = F.dropout2d(x_fea, p=0.3)#10,64,512,512
        x_fea = F.relu(self.bn1(x_fea))#10,64,512,512
        x_out = self.conv2(x_fea)#10,2,512,512

        return x_out

    def close(self):
        for sf in self.sfs: sf.remove()

if __name__ == '__main__':

    x = torch.randn(10, 3, 512, 512)
    u2net = DenseUnet_2d_ce()
    result = u2net(x)

