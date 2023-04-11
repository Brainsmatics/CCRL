from torch import nn
from models.densenet import densenet121, densenet169, densenet201, densenet161

import torch.nn.functional as F
import torch
from models.DenceNet_pro import *

class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class UnetBlock_(nn.Module):
    def __init__(self, up_in1, up_in2, up_out):
        super().__init__()
        self.x_conv = nn.Conv2d(up_in1, up_out, kernel_size=3, padding=1)
        self.x_conv_ = nn.Conv2d(up_in2, up_in1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(up_out)
        nn.init.xavier_normal_(self.x_conv.weight)
        nn.init.xavier_normal_(self.x_conv_.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, up_p, x_p):
        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        x_p = self.x_conv_(x_p)
        cat_p = torch.add(up_p, x_p)
        cat_p = self.x_conv(cat_p)
        cat_p = F.relu(self.bn(cat_p))

        return cat_p


class UnetBlock(nn.Module):
    def __init__(self, up_in1, up_out, size):
        super().__init__()
        self.x_conv = nn.Conv2d(up_in1, up_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(up_out)
        nn.init.xavier_normal_(self.x_conv.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, up_p, x_p):
        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        cat_p = torch.add(up_p, x_p)
        cat_p = self.x_conv(cat_p)
        cat_p = F.relu(self.bn(cat_p))

        return cat_p


class Extractor(nn.Module):
    def __init__(self, densenet='densenet161'):
        super(Extractor, self).__init__()
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
        layers = list(base_model(pretrained=False).children())
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
    def forward(self, x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)#10,2567,128,128
        x = self.up4(x, self.sfs[0].features)
        return x

    def close(self):
        for sf in self.sfs: sf.remove()


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 2, kernel_size=1, padding=0)

        #  init my layers
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

    def forward(self, x, dropout=True):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        if dropout:
            x = F.dropout2d(x, p=0.3)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)

        return x


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim, downsample=True):#256,128
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

    def forward(self, x, dropout=False):
        x = F.relu(self.rn(x))#10,2208,16,16

        x = self.spp(x)#10,2212,16,16
        attention_value = self.cam(x)  # 10,2208,16,16
        x = x.mul(attention_value)
        x = self.up1(x, self.sfs[3].features)#10,768,32,32
        x = self.up2(x, self.sfs[2].features)#10,384,64,64
        x = self.up3(x, self.sfs[1].features)#10,96,128,128
        x = self.up4(x, self.sfs[0].features)#10,96,256,256

        return x

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder = Extractor(), pred_dim=512):

        super(SimSiam, self).__init__()


        self.encoder = base_encoder

        # build a 3-layer projector
        self.projector = nn.Sequential(nn.Conv2d(96, 96, kernel_size=1, stride=1),
                                        nn.BatchNorm2d(96),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Conv2d(96, 128, kernel_size=1, stride=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Conv2d(128, 96, kernel_size=1, stride=1),
                                        nn.BatchNorm2d(96, affine=False)) # output layer


        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Conv2d(96, pred_dim, kernel_size=1, stride=1),
                                        nn.BatchNorm2d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Conv2d(pred_dim, 96, kernel_size=1, stride=1))# output layer

    def forward(self, x1, x2):
        x1 = self.encoder(x1)#10,96,128,128
        x2 = self.encoder(x2)

        z1 = self.projector(x1) # NxC
        z2 = self.projector(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

class Network(nn.Module):

    def __init__(self, base_encoder = Extractor(), base_classifier = Classifier()):

        super(Network, self).__init__()


        self.encoder = base_encoder
        self.classifier = base_classifier


    def forward(self, x):
        x = self.encoder(x)

        x = self.classifier(x)

        return x

class ConNet(nn.Module):

    def __init__(self, base_encoder = Extractor(), base_classifier = Classifier(),pred_dim=512):

        super(ConNet, self).__init__()


        self.encoder = base_encoder
        self.classifier = base_classifier
        # build a 3-layer projector
        self.projector = nn.Sequential(nn.Conv2d(96, 96, kernel_size=1, stride=1),
                                       nn.BatchNorm2d(96),
                                       nn.ReLU(inplace=True),  # first layer
                                       nn.Conv2d(96, 128, kernel_size=1, stride=1),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=True),  # second layer
                                       nn.Conv2d(128, 96, kernel_size=1, stride=1),
                                       nn.BatchNorm2d(96, affine=False))  # output layer

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Conv2d(96, pred_dim, kernel_size=1, stride=1),
                                       nn.BatchNorm2d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Conv2d(pred_dim, 96, kernel_size=1, stride=1))  # output layer

    def forward(self, x,dropout=False):
        x = self.encoder(x)

        z = self.projector(x)  # NxC

        p = self.predictor(z)  # NxC

        x = self.classifier(x)

        return x,p,z.detach()




if __name__ == "__main__":
    net = ConNet(base_encoder=Extractor(),base_classifier=Classifier())
    data = torch.randn((2,3, 224, 224))
    x,p,z = net(data)
    print(x.shape)
    print(p.shape)
    print(z.shape)


