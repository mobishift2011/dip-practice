# -*-conding:utf8-*-
from exercise1 import imread, imshow, immake
from skimage.filters import laplace
import numpy as np
import random


def _rand_pos(shape):
    x = np.random.randint(0, shape[0])
    y = np.random.randint(0, shape[1])
    return x, y


def add_salt_noise(img, percent=0.05):
    """
    添加密度为percent的椒盐噪声
    """
    shape = img.shape
    rows, cols = shape
    n = int(percent * rows * cols)

    for i in range(n):
        x, y = _rand_pos(shape)
        val = np.random.randint(0, 2) * 255
        if img.ndim == 2:
            img[x, y] = val
        elif img.dim == 3:
            for z in range(0, 3):
                img[x, y, z] = val

    return img


def add_gaussian_noise(img, percent=0.05):
    """
    添加密度为percent的高斯噪声
    """
    shape = img.shape
    rows, cols = shape
    n = int(percent * rows * cols)

    for i in range(n):
        x, y = _rand_pos(shape)
        val = img[x, y] + random.gauss(20, 40)
        if val < 0:
            val = 0
        elif val > 255:
            val = 255
        if img.ndim == 2:
            img[x, y] = val
        elif img.dim == 3:
            for z in range(0, 3):
                img[x, y, z] = val

    return img


def linear_filter(img):
    """3*3九点均值滤波"""
    operator = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    size = 9
    dim = len(operator)
    shape = img.shape
    new_img = immake(shape)  # 构造新像素矩阵
    rows, cols = shape
    for i in range(rows):
        for j in range(cols):
            r = 0
            for m in range(dim):
                # 超出边界行的元素按边界值处理
                x = max(i+m-1, 0)
                x = min(x, rows-1)
                for n in range(dim):
                    # 超出边界列的元素按边界值处理
                    y = max(j+n-1, 0)
                    y = min(y, cols-1)
                    # 与相应模板值做卷积
                    w = operator[m][n]
                    r += w * img[x, y]

            new_img[i, j] = r/size

    return new_img


def median_filter(img):
    """九点中值滤波"""
    dim = 3
    n = dim * dim
    mid_pos = int((n+1)/2)
    margin = int((dim-1)/2)
    shape = img.shape
    new_img = immake(shape)  # 构造新像素矩阵
    rows, cols = shape

    for i in range(0, rows):
        for j in range(0, cols):
            # 边缘像素不处理
            if i <= margin or i >= rows-margin or j <= margin or j >= cols-margin:
                new_img[i, j] = img[i, j]
                continue

            # 邻域像素排序后取中值
            arr = [
                img[x, y]
                for x in range(i-margin, i+margin+1)
                for y in range(j-margin, j+margin+1)
            ]
            arr.sort()
            new_img[i, j] = arr[mid_pos]

    return new_img


def grad_sharpen(img):
    """梯度法锐化滤波"""
    shape = img.shape
    new_img = immake(shape)
    rows, cols = shape
    for i in range(0, rows):
        for j in range(0, cols):
            grad = 0
            if i < rows - 1:
                grad += abs(int(img[i, j]) - int(img[i+1, j]))
            if j < cols - 1:
                grad += abs(int(img[i, j]) - int(img[i, j+1]))

            new_img[i, j] = grad

    return new_img


def roberts_sharpen(img):
    """Roberts交叉差分锐化滤波"""
    shape = img.shape
    new_img = immake(shape)
    rows, cols = shape
    for i in range(0, rows-1):
        for j in range(0, cols-1):
            grad = abs(int(img[i, j]-int(img[i+1, j+1]))) + \
                abs(int(img[i+1, j])-int(img[i, j+1]))
            new_img[i, j] = grad

    return new_img


def laplacian_sharpen(img):
    """3*3中心点为-8的掩模拉普拉斯锐化滤波"""
    operator = [
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1],
    ]
    shape = img.shape
    new_img = immake(shape)
    rows, cols = shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            s = 0
            for m in range(0, 3):
                for n in range(0, 3):
                    w = operator[m][n]
                    x = i-1+m
                    y = j-1+n
                    s += w * img[x][y]

            new_img[i, j] = abs(int(s))

    return new_img


def laplacian_enhance(img):
    operator = [
        [1, 1, 1],
        [1, -9, 1],
        [1, 1, 1],
    ]
    shape = img.shape
    new_img = immake(shape)
    rows, cols = shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            s = 0
            for m in range(0, 3):
                for n in range(0, 3):
                    w = operator[m][n]
                    x = i-1+m
                    y = j-1+n
                    s += w * img[x][y]

            new_img[i, j] = abs(int(s))

    return new_img


if __name__ == "__main__":
    # filename = input('请输入要打开的图片路径：')
    # img = imread(filename, as_gray=True)
    img = imread('~/Desktop/coins.png', as_gray=True)

    opt = input(
        '1.加椒盐噪声 2.加高斯噪声 3.均值平滑 4.中值平滑 5.梯度锐化 6.Roberts锐化 7.拉普拉斯锐化 8.拉普拉斯锐化增强：')

    if opt == '1':
        add_salt_noise(img)
    elif opt == '2':
        add_gaussian_noise(img)
    elif opt == '3':
        img = linear_filter(img)
    elif opt == '4':
        img = median_filter(img)
    elif opt == '5':
        img = grad_sharpen(img)
    elif opt == '6':
        img = roberts_sharpen(img)
    elif opt == '7':
        img = laplacian_sharpen(img)
    elif opt == '8':
        img = laplacian_enhance(img)

    imshow(img)
