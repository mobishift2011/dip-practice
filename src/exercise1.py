# -*-conding:utf8-*-
import numpy as np
from skimage import io, util


def imread(fname, as_gray=False):
    """根据文件名打开图像, 每个像素以8位保存，即灰度范围0-255"""
    img = io.imread(fname, as_gray=as_gray)
    return util.img_as_ubyte(img)


def imread_gray(fname):
    """打开一张灰度图"""
    return imread(fname, True)


def imread_color(fname):
    """打开一张彩色图"""
    return imread(fname, False)


def imshow(img, **kwargs):
    """显示图像"""
    io.imshow(img, **kwargs)
    io.show()


def imsave(fname, img):
    """保存一张图片，保存文件名为fname"""
    io.imsave(fname, img)


def immake(shape, dtype=np.uint8):
    return np.zeros(shape, dtype=dtype)


def scale(img, n):
    """
    缩放图像，n为放大或缩小的倍数
    缩小一半scale(img, 0.5)
    放大一倍scale(img, 2)
    """
    shape = img.shape
    rows, cols = shape
    arr = np.array([
        [img[int(i/n), int(j/n)] for j in range(0, int(cols*n))]
        for i in range(0, int(rows*n))
    ])
    return np.array(arr)


def degray(img, level):
    """
    降灰度级
    level: 灰度级数，如降为8级则level为8
    """
    if level < 1 or level > 255:
        return

    rows, cols = img.shape
    base = int(255 / level)

    for i in range(rows):
        for j in range(cols):
            img[i, j] = int(img[i, j] / base)*base

    return img
