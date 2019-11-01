from exercise1 import imread, imshow
from scipy import ndimage as ndi
import numpy as np


def square_se(width, dtype=np.uint8):
    """创建矩形结构元"""
    return np.ones((width, width), dtype=dtype)


def erode(img, selem):
    """腐蚀操作"""
    selem = np.array(selem)
    out = np.empty_like(img)
    ndi.grey_erosion(img, footprint=selem, output=out)
    return out


def dilate(img, selem):
    """膨胀操作"""
    selem = np.array(selem)
    out = np.empty_like(img)
    ndi.grey_dilation(img, footprint=selem, output=out)
    return out


def open_operate(img, selem):
    """开操作"""
    img = erode(img, selem)
    img = dilate(img, selem)
    return img


def close_operate(img, selem):
    """闭操作"""
    img = dilate(img, selem)
    img = erode(img, selem)
    return img


if __name__ == "__main__":
    filename = input('请输入要打开的图片路径：')
    img = imread(filename, as_gray=True)
    opt = input('1.膨胀 2.腐蚀 3.开 4.闭：')

    se = square_se(35)  # 创建结构元

    if opt == '1':
        img = dilate(img, se)
    elif opt == '2':
        img = erode(img, se)
    elif opt == '3':
        img = open_operate(img, se)
    elif opt == '4':
        img = close_operate(img, se)

    imshow(img)
