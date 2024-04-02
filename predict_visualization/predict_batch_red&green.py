import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from PIL import Image
import cv2
from xiaorong4 import MY_NET
# from Model_Yin import MY_NET
# from model_others.ChangNet import ChangNet
import warnings
warnings.filterwarnings('ignore')
from torchvision.models import vgg16
n_class = 2
# class vgg16_base(nn.Module):
#     def __init__(self):
#         super(vgg16_base, self).__init__()
#         features = list(vgg16(pretrained=True).features)[:30]
#         self.features = nn.ModuleList(features).eval()
#
#     def forward(self, x):
#         results = []
#         for ii, model in enumerate(self.features):
#             x = model(x)
#             if ii in {3, 8, 15, 22, 29}:
#                 results.append(x)
#         return results

# vgg = vgg16_base()
# 加载模型
# net = torch.load(r'C:\PycharmProjects\pytorch-from book\codes-My\data\DAMCFNet_My\CDD\epoch\epoch_196.pth').cuda()
# net = torch.load(r'C:\PycharmProjects\pytorch-from book\codes-My\data\DAMCFNet_My\BTCDD\epoch\epoch_197.pth').cuda()
# net = torch.load(r'C:\PycharmProjects\pytorch-from book\codes-My\data\DAMCFNet_My\LEVIR\epoch\epoch_198.pth').cuda()
# net = torch.load(r'C:\PycharmProjects\pytorch-from book\codes-paper2\results\xiaorong_res18\xiaorong1\F1_0.7724_epoch_11.pth')
# net = torch.load(r'C:\PycharmProjects\pytorch-from book\codes-My\model_others\data\results_BTCDD\Unet\epoch\epoch_199.pth').cuda()
checkpoint = torch.load(r"D:\desk\codes\LEVIR-CD\aux2\F1_0.9079_iou_0.8313_epoch_199.pth",map_location=torch.device('cpu'))

net = MY_NET(2).cpu()
net.load_state_dict(checkpoint)
net.eval()  # 将网络设置为评估模式


# 定义颜色
COLORS = {
    0: (0, 0, 0),  # 黑色背景
    1: (255, 255, 255),  # 白色前景
    2: (0, 0, 255),  # 红色误报
    3: (0, 255, 0)  # 绿色
}



# 定义create_visual_anno函数
def create_visual_anno(label):
    label = label.squeeze()
    H, W = label.shape[:2]
    image = np.zeros((H, W, 3), dtype=np.uint8)
    for key in COLORS:
        if key == 0:
            continue
        color = COLORS[key]
        indices = np.where(label == int(key))
        image[indices[0], indices[1], :] = color
    return image

# 预测文件夹中的所有图片
img_folder = r'D:\desk\codes\LEVIR-CD\test'
img_list = os.listdir(os.path.join(img_folder, 'A'))


for i in range(len(img_list)):
    before = tf.to_tensor(Image.open(os.path.join(img_folder, 'A', img_list[i]))).unsqueeze(dim=0).cpu()
    after = tf.to_tensor(Image.open(os.path.join(img_folder, 'B', img_list[i]))).unsqueeze(dim=0).cpu()
    # change = tf.to_tensor(Image.open(os.path.join(img_folder, 'OUT', img_list[i]))).cuda()
    change_filename = img_list[i].replace('.jpg', '.png')
    # change_filename = img_list[i].replace('.png', '.jpg')
    # change_filename = img_list[i]
    change = tf.to_tensor(Image.open(os.path.join(img_folder, 'OUT', change_filename))).cpu()

    # pred,aux1,aux2,aux3 = net(before, after)
    pred= net(before, after)
    # label_pred = F.softmax(pred, dim=1).max(dim=1)[1].data.cpu().numpy().astype(np.int)
    label_pred = torch.argmax(pred, dim=1).cpu().numpy().astype(int)
    label_true = change.data.cpu().numpy()
    label_true_ = label_true.astype(int)

    # 计算true positive、false positive、false negative
    true_positive = np.logical_and(label_pred == 1, label_true_ == 1)
    false_positive = np.logical_and(label_pred == 1, label_true_ == 0)
    false_negative = np.logical_and(label_pred == 0, label_true_ == 1)



    # 创建可视化标注图
    visual_anno = np.zeros_like(label_true_)
    visual_anno[true_positive] = 1
    visual_anno[false_positive] = 2
    visual_anno[false_negative] = 3
    visual_anno_img = create_visual_anno(visual_anno)

    # 保存预测结果
    cv2.imwrite(os.path.join(r'D:\desk\codes\predict_LEVIR\ours', img_list[i]), visual_anno_img)
    # result_filename = change_filename.replace('.png', '_predict.png')
    # result_path = os.path.join(
    #     r'C:\PycharmProjects\pytorch-from book\codes-paper2\results\xiaorong_res18\xiaorong1\predict', result_filename)
    # cv2.imwrite(result_path, visual_anno_img)