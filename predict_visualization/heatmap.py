# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 加载模型文件
# model = torch.load(r'C:\PycharmProjects\pytorch-from book\codes-My\xiaorong\xiaorong_3\epoch\epoch_199.pth')
#
# # 设置图片路径和变换
# img1_path = r'C:\PycharmProjects\pytorch-from book\codes-My\data\results_BTCDD\predict_BTCDD_batch\A\33.jpg'
# img2_path = r'C:\PycharmProjects\pytorch-from book\codes-My\data\results_BTCDD\predict_BTCDD_batch\B\33.jpg'
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])
#
# # 读取图片并应用变换
# img1 = Image.open(img1_path)
# img1_tensor = transform(img1)
# img2 = Image.open(img2_path)
# img2_tensor = transform(img2)
#
# # 添加批处理维度，以匹配模型的输入
# img1_tensor = img1_tensor.unsqueeze(0)
# img2_tensor = img2_tensor.unsqueeze(0)
#
# # 将图像输入到模型中，并获得输出
# with torch.no_grad():
#     output = model(img1_tensor, img2_tensor)
#
# # 将输出的张量转换为热力图
# heatmap = output.squeeze().numpy()
#
# # 可视化热力图
# plt.imshow(heatmap, cmap='jet')
# plt.axis('off')
# plt.show()


import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from xiaorong3 import MY_NET

# 加载模型
model = torch.load(r"D:\desk\codes\LEVIR-CD\xiaorong3\F1_0.8750_epoch_33.pth")
net = MY_NET(2).cpu()
net.load_state_dict(model, strict=False)
net.eval()  # 将网络设置为评估模式


# 加载两张图像
image1 = Image.open(r'D:\desk\codes\LEVIR-CD\test\A\1051.png').convert("RGB")
image2 = Image.open(r'D:\desk\codes\LEVIR-CD\test\B\1051.png').convert("RGB")

# image1=image1[:, :, ::-1]
# image2=image2[:, :, ::-1]
# 转换图像为 PyTorch 张量，并归一化
tensor1 = torch.tensor(np.array(image1)).float() / 255
tensor2 = torch.tensor(np.array(image2)).float() / 255


# tensor1 = torch.tensor(np.array(image1[:, :, ::-1]).copy()).float() / 255
# tensor2 = torch.tensor(np.array(image2[:, :, ::-1]).copy()).float() / 255

# 将张量调整为模型所需的形状
tensor1 = tensor1.permute(2, 0, 1).unsqueeze(0).to('cpu')
tensor2 = tensor2.permute(2, 0, 1).unsqueeze(0).to('cpu')



# 使用模型进行预测
with torch.no_grad():
    output = net(tensor1, tensor2)
    # output, aux1, aux2, aux3 = net(tensor1, tensor2)   ####     ,out1,out2,out3         ####
    # output,out1,out2,out3 = model(tensor1, tensor2)

# 将输出转换为热力图，并可视化
heatmap = output[0, 0].cpu().numpy()
plt.imshow(heatmap, cmap="jet_r")
plt.axis("off")
plt.savefig(r'D:\desk\codes\LEVIR-CD\xiaorong3_1051.png', bbox_inches='tight')
plt.show()

