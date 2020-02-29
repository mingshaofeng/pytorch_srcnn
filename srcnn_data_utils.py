import torch.utils.data as data
#创建数据集，作用: (1) 创建数据集,有__getitem__(self, index)函数来根据索引序号获取图片和标签, 有__len__(self)函数来获取数据集的长度.
#其他的数据集类必须是torch.utils.data.Dataset的子类,比如说torchvision.ImageFolder.
from os import listdir
from os.path import join
from PIL import Image, ImageFilter
#PIL是python的图像处理库


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])
'''
endswith()函数,描述：判断字符串是否以指定字符或子字符串结尾。语法：str.endswith("suffix", start, end) 或

str[start,end].endswith("suffix")    用于判断字符串中某段字符串是否以指定字符或子字符串结尾。

—> bool    返回值为布尔类型（True,False）
extension:扩展名
'''

def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')

    #y, _, _ = img.split()
    #return y
    return img
'''
将图片转化为YCbCr格式
我们知道PIL中有九种不同模式。分别为1，L，P，RGB，RGBA，CMYK，YCbCr，I，F。
模式“YCbCr”为24位彩色图像，它的每个像素用24个bit表示。
YCbCr其中Y是指亮度分量，Cb指蓝色色度分量，而Cr指红色色度分量。人的肉眼对视频的Y分量更敏感，
因此在通过对色度分量进行子采样来减少色度分量后，肉眼将察觉不到的图像质量的变化。
'''

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform
#__getitem__(self, index)函数来根据索引序号获取图片和标签
    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = input.filter(ImageFilter.GaussianBlur(2))
            #将图片进行高斯模糊

            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target
# 有__len__(self)函数来获取数据集的长度.
    def __len__(self):
        return len(self.image_filenames)
