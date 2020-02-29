from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
#相关：urllib是python内置的http请求库，本文介绍urllib三个模块：
# 请求模块urllib.request、异常处理模块urllib.error、url解析模块urllib.parse。
import tarfile
#TarFile类对于就是tar压缩包实例. 其由member块组成, member块则包括header块和data块. 每个member以TarInfo对象形式描述.
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale
#Transfoms 是很常用的图片变换方式，可以通过compose将各个变换串联起来
from srcnn_data_utils import DatasetFromFolder


def download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")
#如果不存在这个数据，我们将到别的地方下载
    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Scale(crop_size),
        #Scale(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])

#训练数据集设置
def get_training_set(upscale_factor):

    root_dir = download_bsd300()
    
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))

#测试数据集设置
def get_test_set(upscale_factor):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))