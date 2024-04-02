import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18,resnet34
from torch.nn import init

class Backbone_resnet(nn.Module):
    def __init__(self,backbone):
        super(Backbone_resnet, self).__init__()

        if backbone == 'resnet18':
            self.net = resnet18(pretrained=True)
            del self.net.avgpool
            del self.net.fc
        elif backbone == 'resnet34':
            self.net = resnet34(pretrained=True)
            del self.net.avgpool
            del self.net.fc
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def forward(self,x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        c1 = self.net.layer1(x)
        c2 = self.net.layer2(c1)
        c3 = self.net.layer3(c2)
        c4 = self.net.layer4(c3)
        return c1, c2, c3, c4

# class eca_layer(nn.Module):
#     """Constructs a ECA module.
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#
#     def __init__(self, k_size=3):
#         super(eca_layer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x: input features with shape [b, c, h, w]
#         b, c, h, w = x.size()
#
#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)
#
#         # Two different branches of ECA module
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#
#         # Multi-scale information fusion
#         y = self.sigmoid(y)
#
#         return x * y.expand_as(x)

# #特征融合，将“+”和“cat”的语义融合
# class MyFFM_song(nn.Module):
#     def __init__(self, inc,outch):
#         super(MyFFM_song, self).__init__()
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.conv_c2 = nn.Sequential(
#             nn.Conv2d(inc, inc // 2, kernel_size=1),
#             nn.BatchNorm2d(inc//2),
#             nn.ReLU(True),
#             nn.Conv2d(inc//2, inc, kernel_size=1),
#             nn.BatchNorm2d(inc),
#         )
#
#         self.conv_cx2 = nn.Sequential(
#             nn.Conv2d(inc, 2 * inc, kernel_size=1),
#             nn.BatchNorm2d(2 * inc),
#             nn.ReLU(True),
#             nn.Conv2d(2 * inc, inc, kernel_size=1),
#             nn.BatchNorm2d(inc),
#         )
#
#         self.sigmoid = nn.Sigmoid()
#
#         self.conv1x1 = nn.Sequential(
#             nn.Conv2d(inc, 1, kernel_size=1),
#         )
#
#         self.conv3x3_2 = nn.Sequential(
#             nn.Conv2d(1, 1, kernel_size=3, padding=1),
#             nn.BatchNorm2d(1),
#             nn.ReLU(True),
#             nn.Conv2d(1, 1, kernel_size=3, padding=1),
#             nn.BatchNorm2d(1),
#
#
#         )
#
#         self.conv3x3_1 = nn.Sequential(
#             nn.Conv2d(1, 1, kernel_size=3, padding=1),
#             #nn.BatchNorm2d(1),
#
#         )
#         self.conv_last = nn.Sequential(
#             nn.Conv2d(inc,outch,kernel_size=3,padding=1),
#
#         )


    # def forward(self, x):
    #     input = x
    #     x = self.avgpool(x)
    #     branch1 = self.sigmoid(self.conv_c2(x))
    #     branch2 = self.sigmoid(self.conv_cx2(x))
    #
    #     score = branch1 + branch2
    #     out1 = torch.mul(input, score)
    #
    #     y = self.conv1x1(input)
    #     y1 = self.sigmoid(self.conv3x3_1(y))
    #     y2 = self.sigmoid(self.conv3x3_2(y))
    #
    #     score_ = y1 + y2
    #     out2 = input * score_
    #
    #     return self.conv_last(out1 + out2 + input)
# class FFM_yin(nn.Module):
#     def __init__(self,in_ch,out_ch):
#         super(FFM_yin, self).__init__()
#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#
#         # self.conv1x1 = nn.Sequential(
#         #     nn.Conv2d(in_ch, out_ch, kernel_size=1),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(out_ch, out_ch, kernel_size=1),
#         #     nn.Sigmoid(),
#         # )
#
#         self.conv3x3 = nn.Sequential(
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1,bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(in_ch),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#
#         )
#
#         self.init_weight()
#
#     def forward(self,x1,x2):
#         """x1 _ cat , x2 _ sub"""
#         x1 = self.avgpool(x1)
#         x1_ = self.conv1(x1)
#         # x1_ = self.conv1x1(x1)
#         x2_ = self.conv3x3(x2)
#         out = torch.mul(x1_,x2_)
#         return out + x2
#
#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if not ly.bias is None: nn.init.constant_(ly.bias, 0)
#
# #define MCRM mmodule
# class double_conv(nn.Module):
#     '''(conv => BN => ReLU) * 2'''
#
#     def __init__(self, in_ch, out_ch, stride=1, first=False):
#         super(double_conv, self).__init__()
#         self.stride = stride
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
#             nn.BatchNorm2d(in_ch // 2),
#             nn.ReLU(True),
#         )
#
#         self.conv_1 = nn.Sequential(
#             nn.Conv2d(in_ch // 2, in_ch // 2, 3, padding=1, stride=stride),
#             nn.BatchNorm2d(in_ch // 2),
#             nn.ReLU(True),
#         )
#
#         self.conv_2 = nn.Sequential(
#             nn.Conv2d(in_ch // 2, in_ch // 2, 3, padding=1),
#             nn.BatchNorm2d(in_ch // 2),
#             nn.ReLU(True),
#         )
#
#         self.conv_3 = nn.Sequential(
#             nn.Conv2d(in_ch // 2, in_ch // 2, 3, padding=1),
#             nn.BatchNorm2d(in_ch // 2),
#             nn.ReLU(True),
#             nn.Conv2d(in_ch // 2, in_ch // 2, 3, padding=1),
#             nn.BatchNorm2d(in_ch // 2),
#             nn.ReLU(True),
#         )
#
#         self.conv_final = nn.Sequential(
#             nn.Conv2d(in_ch // 2, out_ch, 1),
#             nn.BatchNorm2d(out_ch),
#             eca_layer(),
#         )
#
#         if in_ch != out_ch or first == True:
#             self.identity = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(out_ch),
#             )
#         if in_ch == out_ch and first == False:
#             self.identity = nn.Identity()
#
#         if self.stride == 1:
#             self.shortcut = nn.Identity()
#         if self.stride == 2:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=1, stride=2),
#                 nn.BatchNorm2d(in_ch // 2),
#                 nn.ReLU(True),
#             )
#
#         self.relu = nn.ReLU(True)
#
#     def forward(self, x):
#         input = x
#         x_ = self.conv(x)
#         x = self.conv_1(x_)
#         x1 = x
#         x2 = self.conv_2(x)
#         x3 = self.conv_3(x + x2)
#         x = self.shortcut(x_) + x1 + x2 + x3
#         x = self.conv_final(x)
#
#         identity = self.identity(input)
#         out = self.relu(x + identity)
#         return out

class first_conv(nn.Module):
    def __init__(self):
        super(first_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # double_conv(64, 64, stride=2, first=True),
        )
    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    def __init__(self, inc, outc):
        super(down, self).__init__()
        self.double_conv = nn.Conv2d(inc, outc, kernel_size=2)
    def forward(self, x):
        return self.double_conv(x)

 #res到最后一层的融合操作
# class PPM(nn.Module):
#     def __init__(self,in_dim=1024,bins=[1,2,3,6]):
#         super(PPM, self).__init__()
#         self.features = []
#         for bin in bins:
#             self.features.append(nn.Sequential(
#                 nn.AdaptiveAvgPool2d(bin),
#                 nn.Conv2d(in_dim,in_dim,kernel_size=1,bias=False),
#                 nn.BatchNorm2d(in_dim),
#                 nn.ReLU(inplace=True),
#             ))
#         self.features = nn.ModuleList(self.features)
#
#     def forward(self,x):
#         x_size = x.size()
#         out = []
#         for f in self.features:
#             out.append(F.interpolate(f(x),x_size[2:],mode='bilinear',align_corners=True))
#         return out[0] + out[1] + out[2] + out[3]

#double_3*3conv(用户上采样)
# class double_33con(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(double_33con, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#attention module
class Spatial_Attention(nn.Module):
    """空间注意力模块"""
    def __init__(self,kernel_size = 7):
        super(Spatial_Attention, self).__init__()
        assert kernel_size in (3,7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2,1,kernel_size ,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self,x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out,_ = torch.max(x,dim=1,keepdim=True)
        out = torch.cat([avg_out,max_out],dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class Channel_Attention(nn.Module):
    def __init__(self,in_ch,ration = 16):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_ch,in_ch//ration,kernel_size=1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_ch//ration,in_ch,kernel_size=1,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class attention(nn.Module):
    def __init__(self, in_ch):
        super(attention, self).__init__()
        self.sa = Spatial_Attention()
        self.ca = Channel_Attention(in_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch*2,in_ch,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.ca(x)
        x1_ = torch.mul(x,x1)
        x2 = self.sa(x1_)
        x2_ = torch.mul(x,x2)
        x2_ = self.sigmoid(x2_)
        x3 = torch.mul(x2_,x2)
        out = torch.cat([x3,x],dim=1)
        out = self.conv(out) + x

        return out



class ChannelExchange(nn.Module):
    def __init__(self, p=2):
        super(ChannelExchange, self).__init__()
        self.p = p

    def forward(self, x1, x2):
        N, C, H, W = x1.shape
        exchange_mask = torch.arange(C) % self.p == 0
        exchange_mask = exchange_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((N, -1, H, W))
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask] = x1[~exchange_mask]
        out_x2[~exchange_mask] = x2[~exchange_mask]
        out_x1[exchange_mask] = x2[exchange_mask]
        out_x2[exchange_mask] = x1[exchange_mask]
        return out_x1, out_x2


class Mynet(nn.Module):
    def __init__(self, num_classes):
        super(Mynet, self).__init__()
        self.net = Backbone_resnet(backbone='resnet34')
        self.first_conv = first_conv()

        # self.mcrm1 = double_conv(64,64)
        # self.mcrm2 = double_conv(128, 128)
        # self.mcrm3 = double_conv(256, 256)
        # self.mcrm4 = double_conv(512, 512)
        # self.mcrm5 = double_conv(1024, 1024)
        # self.mcrm6 = double_conv()
        self.att1 = attention(192)
        self.att2 = attention(384)
        self.att3 = attention(768)
        self.att4 = attention(1536)

        # self.ffm = MyFFM_song(1536,512)

        self.down1_2 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(2, stride=2, padding=0)
        )
        self.down2_3 = nn.Sequential(
            nn.Conv2d(384, 768, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(2, stride=2, padding=0)
        )
        self.down3_4 = nn.Sequential(
            nn.Conv2d(768, 1536, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(2, stride=2, padding=0)
        )
        self.channel_exchange = ChannelExchange(2)
        # self.up4 = nn.Sequential(
        #     nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
        # )
        self.up3 = nn.Sequential(
            nn.Conv2d(1536, 768, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # self.ppm = PPM()
        self.outconv = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(0.5),
            nn.Conv2d(64,2,kernel_size=1),
        )
    def forward(self,x1,x2):
        x1_layer1, x1_layer2, x1_layer3, x1_layer4 = self.net(x1)
        x2_layer1, x2_layer2, x2_layer3, x2_layer4 = self.net(x2)
        
        # layer4_out1,layer4_out2 = self.channel_exchange(x1_layer4, x2_layer4)
        #
        # layer4_changed1 =  layer4_out1+layer4_out2
        #
        #
        # layer4_changed2 = torch.cat([layer4_out1,layer4_out2],dim=1)
        #
        # layer4_changed = torch.cat([layer4_changed1,layer4_changed2],dim=1)
        # layer4_changed_ppm = self.ppm(layer4_changed)

        # 减法支路
        sub_layer1 = torch.abs(x1_layer1 - x2_layer1)  # abs用于计算绝对值

        sub_layer2 = torch.abs(x1_layer2 - x2_layer2)

        sub_layer3 = torch.abs(x1_layer3 - x2_layer3)

        sub_layer4 = torch.abs(x1_layer4 - x2_layer4)

        cat_layer1 = torch.cat([x1_layer1, x2_layer1], dim=1)

        cat_layer2 = torch.cat([x1_layer2, x2_layer2], dim=1)

        cat_layer3 = torch.cat([x1_layer3, x2_layer3], dim=1)

        cat_layer4 = torch.cat([x1_layer4, x2_layer4], dim=1)

        layer1_out = torch.cat([sub_layer1,cat_layer1],dim=1)
        layer2_out = torch.cat([sub_layer2, cat_layer2], dim=1)
        layer3_out = torch.cat([sub_layer3, cat_layer3], dim=1)
        layer4_out = torch.cat([sub_layer4, cat_layer4], dim=1)

        c_layer1 = self.att1(layer1_out)
        c_layer1_2= self.down1_2(c_layer1)

        c_layer2 = self.att2(layer2_out+c_layer1_2)
        c_layer3 = self.att3(layer3_out+self.down2_3 (c_layer2))
        c_layer4 = self.att4(layer4_out+self.down3_4 (c_layer3))

        u4 =   c_layer4
        u4_ = F.upsample(u4, scale_factor=2, mode='bilinear', align_corners=True)
        u3 = self.up3(u4_) + c_layer3
        u3_ = F.upsample(u3,scale_factor=2, mode='bilinear', align_corners=True)
        u2 = self.up2(u3_) + c_layer2
        u2_ = F.upsample(u2, scale_factor=2, mode='bilinear', align_corners=True)
        u1 = self.up1(u2_) + c_layer1
        u1_ = F.upsample(u1, scale_factor=4, mode='bilinear', align_corners=True)
        return self.outconv(u1_)



if __name__ == '__main__':
    x1 = torch.rand(1, 3, 256, 256)
    x2 = torch.rand(1, 3, 256, 256)
    model = Mynet(2)

    from flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number

    model_eval = add_flops_counting_methods(model)
    model_eval.eval().start_flops_count()
    out = model_eval(x1, x2)
    print('Flops: {}'.format(flops_to_string(model.compute_average_flops_cost())))
    print('Params: ' + get_model_parameters_number(model))
    print('Output shape: {}'.format(list(out.shape)))
    total_paramters = sum(p.numel() for p in model.parameters())
    print('Total paramters: {}'.format(total_paramters))