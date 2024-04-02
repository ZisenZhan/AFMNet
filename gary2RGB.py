import numpy as np

'''
输入anno为单张单通道索引图，类型为array
输出彩色RGB图像
'''
def create_visual_anno(anno):
    """"""
    assert np.max(anno) <= 1
    label2color_dict = {
        0: [0, 0, 0],
        1: [255,255,255],

    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno



def create_visual_anno_1(anno):
    assert np.max(anno) <= 3
    label2color_dict = {
        0: [0, 0, 0],    # 黑色表示背景
        1: [255, 255, 255],  # 白色表示变化区域
        2: [255, 0, 0],   # 红色表示误报
        3: [0, 255,0],   # 绿色表示漏报
    }
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno


def create_visual_anno_2(anno, gt_anno):
    # 确保标签的值小于或等于1
    assert np.max(anno) <= 1

    label2color_dict = {
        0: [0, 0, 0],  # 黑色表示背景
        1: [255, 255, 255],  # 白色表示变化区域
        2: [255, 0, 0],  # 红色表示误报
        3: [0, 255, 0],  # 绿色表示漏报
    }

    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)

    # 计算误检区域和漏检区域的标签
    fp = np.logical_and(anno == 1, gt_anno == 0)
    fn = np.logical_and(anno == 0, gt_anno == 1)

    # 为误检和漏检区域分别分配标签
    anno[fp] = 2
    anno[fn] = 3

    # 根据标签字典分配颜色
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            if anno[i, j] in label2color_dict:
                color = label2color_dict[anno[i, j]]
            else:
                color = [0, 0, 255]  # 蓝色表示未知标签
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno

