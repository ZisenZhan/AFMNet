# 学习单位: 南京信息工程大学
# 学   生: 詹子森
# 开发时间: 2024/1/17 14:41
# 学习单位: 南京信息工程大学
# 学   生: 詹子森
# 开发时间: 2023/4/21 19:16
# 学习单位: 南京信息工程大学
# 学   生: 詹子森
# 开发时间: 2023/4/21 19:11
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18,resnet34,resnet50
from torch.nn import init
from dpcd_parts import DPFA

import torchvision.models as models


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
        elif backbone == 'resnet50':
            self.net = resnet50(pretrained=True)
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


class Double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Double_conv1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double_conv1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class Spatial_Attention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size=7):
        super(Spatial_Attention, self).__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class Channel_Attention(nn.Module):
    def __init__(self, in_ch, ration=16):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, in_ch // ration, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_ch // ration, in_ch, kernel_size=1, bias=False)
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
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.ca(x)
        x1_ = torch.mul(x, x1)
        x2 = self.sa(x1_)
        x2_ = torch.mul(x, x2)
        x2_ = self.sigmoid(x2_)
        x3 = torch.mul(x2_, x2)
        out = torch.cat([x3, x], dim=1)
        out = self.conv(out) + x

        return out


class FFM(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(FFM, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # self.conv1x1 = nn.Sequential(
        #     nn.Conv2d(out_ch, out_ch, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_ch, out_ch, kernel_size=1),
        #     nn.Sigmoid(),
        # )

        # self.conv3x3 = nn.Sequential(
        #     nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1,bias=False),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True),
        # )

        self.conv1 = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

        )

        self.conv1_ = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),)

        self.dwconv = nn.Sequential(
            # DW卷积的深度卷积部分
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # DW卷积的逐点卷积部分
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        self.init_weight()

    def forward(self,x1,x2):
        """x1 _ cat , x2 _ sub"""
        x1_ = self.conv1(x1) # 2C————C # [b,c,h,w]
        x1_a = self.avgpool(x1_) # [b,c,1,1]
        x1_b = self.conv1_(x1_a)  # [b,c,1,1]
        x1_out = torch.mul(x1_, x1_b) # [b,c,h,w]

        x2_ = self.conv1_(x2)
        x2_out = self.dwconv(x2_)

        out = x1_out + x2_out
        out = self.conv1_(out)

        return out  #[64,128,128]

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class MY_NET(nn.Module):
    def __init__(self,num_classes):
        super(MY_NET, self).__init__()

        self.net = Backbone_resnet(backbone='resnet34')

        channles = [32, 64, 128, 256, 512]
        self.en_d = 32
        self.mid_d = self.en_d * 2      #64
        # self.swa = NeighborFeatureAggregation(channles, self.mid_d)

        """上采样"""

        self.up_2 = nn.Sequential(
            Double_conv(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )  # 上采样二倍
        self.up_3 = nn.Sequential(
            Double_conv(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )  # 上采样二倍
        self.up_4 = nn.Sequential(
            Double_conv(512, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )  # 上采样二倍

        self.up4 = nn.Sequential(
            Double_conv(512, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), ) #上采样二倍

        self.up3 = nn.Sequential(
            Double_conv(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )

        self.up2 = nn.Sequential(
            Double_conv(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )

        self.up1 = nn.Sequential(
            Double_conv(64, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )


        self.dpfa1 = DPFA(64)
        self.dpfa2 = DPFA(128)
        self.dpfa3 = DPFA(256)
        self.dpfa4 = DPFA(512)

        self.ffm1 = FFM(128, 64)
        self.ffm2 = FFM(256, 128)
        self.ffm3 = FFM(512, 256)
        self.ffm4 = FFM(1024, 512)

        self.layer1_1x1 = Double_conv1x1(64, 64)
        self.layer2_1x1 = Double_conv1x1(128, 128)
        self.layer3_1x1 = Double_conv1x1(256, 256)
        self.layer4_1x1 = Double_conv1x1(512, 512)


        self.output_aux_3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(0.2),  ##防止模型过拟合
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

        self.output_aux_2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(0.2),  ##防止模型过拟合
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

        self.output_aux_1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(0.2),  ##防止模型过拟合
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

        self.output = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # nn.Dropout2d(0.3),
            nn.Conv2d(16, num_classes, kernel_size=1, bias=False),
            # nn.Conv2d(16, num_classes, kernel_size=1),
        )

        self.fc1  = nn.Conv2d(1024, 512, kernel_size=1)
        self.fc2 = nn.Conv2d(512, 256, kernel_size=1)
        self.fc3 = nn.Conv2d(256, 128, kernel_size=1)
        self.fc4 = nn.Conv2d(128, 64, kernel_size=1)
        self.fc = nn.Conv2d(64, 2, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

        self.att1 = attention(64)
        self.att2 = attention(128)
        self.att3 = attention(256)
        self.att4 = attention(512)


    def forward(self,x1,x2):
        h, w = x1.shape[2:]
        ##x_layer1:[4,64,64,64] x_layer2:[4,128,32,32] x_layer3:[4,256,16,16] x_layer4:[4,512,8,8]


        x1_layer1,x1_layer2,x1_layer3,x1_layer4 = self.net(x1)
        x2_layer1,x2_layer2,x2_layer3,x2_layer4 = self.net(x2)

        x1_layer1_, x2_layer1_ = self.dpfa1(x1_layer1,x2_layer1)
        x1_layer2_, x2_layer2_ = self.dpfa2(x1_layer2, x2_layer2)
        x1_layer3_, x2_layer3_ = self.dpfa3(x1_layer3, x2_layer3)
        x1_layer4_, x2_layer4_ = self.dpfa4(x1_layer4, x2_layer4)

        """cat_branch"""
        cat_layer1 = torch.cat([x1_layer1,x2_layer1],dim=1) #[4,128,128,128]

        cat_layer2 = torch.cat([x1_layer2,x2_layer2],dim=1) #[4,256,64,64]

        cat_layer3 = torch.cat([x1_layer3,x2_layer3],dim=1) #[4,512,32,32]

        cat_layer4 = torch.cat([x1_layer4,x2_layer4],dim=1)#[4,1024,16,16]


        """sub_branch"""
        sub_layer1_ = torch.abs(x1_layer1_ - x2_layer1_)  # 逐元素做差，然后取绝对值。[4,64,64,64]

        sub_layer2_ = torch.abs(x1_layer2_ - x2_layer2_)  # [4,128,32,32]

        sub_layer3_ = torch.abs(x1_layer3_ - x2_layer3_)  # [4,256,16,16]

        sub_layer4_ = torch.abs(x1_layer4_ - x2_layer4_)  # [4,512,8,8]

        "FFM融合 cat & sub"
        ffm_layer1 = self.ffm1(cat_layer1, sub_layer1_) # [4,64,64,64]
        ffm_layer2 = self.ffm2(cat_layer2, sub_layer2_) # [4,128,32,32]
        ffm_layer3 = self.ffm3(cat_layer3, sub_layer3_) # [4,256,16,16]
        ffm_layer4 = self.ffm4(cat_layer4, sub_layer4_) # [4,512,8,8]

        up4 = self.up4(self.att4(ffm_layer4))  # [4,256,16,16]
        out3 = self.att3(ffm_layer3) + up4  # [4,256,16,16]
        up3 = self.up3(out3)  # [4,128,32,32]
        out2 = self.att2(ffm_layer2) + up3  ##[4,128,32,32]
        up2 = self.up2(out2)  # [4,64,64,64]

        out1 = self.att1(ffm_layer1) + up2  # [4,64,64,64]

        out = self.up1(out1) # [4,32,128,128]

        output  = self.output(out)  # [4,2,128,128]

        output  = F.interpolate(output,size=(h,w),mode='bilinear', align_corners=True)


        return output




if __name__ == '__main__':
    x1 = torch.rand(4,3,256,256)
    x2 = torch.rand(4,3,256,256)
    model = MY_NET(num_classes=2)
    out = model(x1,x2)
    print('out',out.shape)
    print('output:', out.shape)

    from flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number

    model_eval = add_flops_counting_methods(model)
    model_eval.eval().start_flops_count()
    output, output1, output2, output3 = model_eval(x1, x2)
    print('Flops: {}'.format(flops_to_string(model.compute_average_flops_cost())))
    print('Params: ' + get_model_parameters_number(model))
    print('Output shape: {}'.format(list(out.shape)))
    total_paramters = sum(p.numel() for p in model.parameters())
    print('Total paramters: {}'.format(total_paramters))

    import time

    if torch.cuda.is_available():
        model = model.cuda()  # .half()  #HALF seems to be doing slower for some reason
        x1 = x1.cuda()  # .half()
        x2 = x2.cuda()

    time_train = []
    i = 0
    # model.load_state_dict(torch.load("../Testmodel_List/KR94187_Portrait_98/result/Dnc_C3Portrait/model_266.pth",
    #                       map_location=torch.device(device='cpu')))
    # 0.273
    while (i < 20):
        # for step, (images, labels, filename, filenameGt) in enumerate(loader):

        start_time = time.time()

        inputs1 = torch.autograd.Variable(x1)
        inputs2 = torch.autograd.Variable(x2)
        with torch.no_grad():
            outputs = model(x1, x2)

        # preds = outputs.cpu()
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)

        if i != 0:  # first run always takes some time for setup
            fwt = time.time() - start_time
            time_train.append(fwt)
            print("Forward time per img (b=%d): %.3f (Mean: %.3f)" % (
                1, fwt / 1, sum(time_train) / len(time_train) / 1))

        time.sleep(1)  # to avoid overheating the GPU too much
        i += 1

# print('output:',output.shape,'output1:',output1.shape,'output2:',output2.shape,'output3:',output3.shape)
    #
    #
    # from flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number
    # model_eval = add_flops_counting_methods(model)
    # model_eval.eval().start_flops_count()
    # output,output1,output2,output3 = model_eval(x1, x2)
    # print('Flops: {}'.format(flops_to_string(model.compute_average_flops_cost())))
    # print('Params: ' + get_model_parameters_number(model))
    # print('Output shape: {}'.format(list(output.shape)),'Output1 shape: {}'.format(list(output1.shape)),'Output2 shape: {}'.format(list(output2.shape)),'Output3 shape: {}'.format(list(output3.shape)))
    # total_paramters = sum(p.numel() for p in model.parameters())
    # print('Total paramters: {}'.format(total_paramters))
    #
    # import time
    # if torch.cuda.is_available():
    #     model = model.cuda()  # .half()  #HALF seems to be doing slower for some reason
    #     x1 = x1.cuda()  # .half()
    #     x2 = x2.cuda()
    #
    # time_train = []
    # i = 0
    # # model.load_state_dict(torch.load("../Testmodel_List/KR94187_Portrait_98/result/Dnc_C3Portrait/model_266.pth",
    # #                       map_location=torch.device(device='cpu')))
    # # 0.273
    # while (i < 20):
    #     # for step, (images, labels, filename, filenameGt) in enumerate(loader):
    #
    #     start_time = time.time()
    #
    #     inputs1 = torch.autograd.Variable(x1)
    #     inputs2 = torch.autograd.Variable(x2)
    #     with torch.no_grad():
    #         outputs = model(x1,x2)
    #
    #     # preds = outputs.cpu()
    #     # if torch.cuda.is_available():
    #     #     torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
    #
    #     if i != 0:  # first run always takes some time for setup
    #         fwt = time.time() - start_time
    #         time_train.append(fwt)
    #         print("Forward time per img (b=%d): %.3f (Mean: %.3f)" % (
    #            1, fwt / 1, sum(time_train) / len(time_train) /1))
    #
    #     time.sleep(1)  # to avoid overheating the GPU too much
    #     i += 1
    #

# 学习单位: 南京信息工程大学
# 学   生: 任鸿金
# 开发时间: 2024/1/22 20:10
