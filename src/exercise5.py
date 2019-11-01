from exercise1 import imread, imshow, immake
import cv2
import numpy as np
import scipy.stats
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


# 算数均值滤波
def arithmeticMeanOperator(roi):
    return np.mean(roi)


def arithmeticMeanAlogrithm(image):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            new_image[i-1, j -
                      1] = arithmeticMeanOperator(image[i-1:i+2, j-1:j+2])
    new_image = (new_image-np.min(image))*(255/np.max(image))
    return new_image.astype(np.uint8)


def arithmeticMean(image):
    img = arithmeticMeanAlogrithm(image)
    return img


# 几何均值滤波
def geometricMeanOperator(roi):
    roi = roi.astype(np.float64)
    p = np.prod(roi)
    return p**(1/(roi.shape[0]*roi.shape[1]))


def geometricMeanAlogrithm(image):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            new_image[i-1, j -
                      1] = geometricMeanOperator(image[i-1:i+2, j-1:j+2])
    new_image = (new_image-np.min(image))*(255/np.max(image))
    return new_image.astype(np.uint8)


def gemotricMean(img):
    img = geometricMeanAlogrithm(img)
    return img


# 谐波均值
def HMeanOperator(roi):
    roi = roi.astype(np.float64)
    if 0 in roi:
        roi = 0
    else:
        roi = scipy.stats.hmean(roi.reshape(-1))
    return roi


def HMeanAlogrithm(image):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            new_image[i-1, j-1] = HMeanOperator(image[i-1:i+2, j-1:j+2])
    new_image = (new_image-np.min(image))*(255/np.max(image))
    return new_image.astype(np.uint8)


def HMean(image):
    img = HMeanAlogrithm(image)
    return img


# 逆谐波均值
def IHMeanOperator(roi, q):
    roi = roi.astype(np.float64)
    return np.mean((roi)**(q+1))/np.mean((roi)**(q))


def IHMeanAlogrithm(image, q):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            new_image[i-1, j-1] = IHMeanOperator(image[i-1:i+2, j-1:j+2], q)
    new_image = (new_image-np.min(image))*(255/np.max(image))
    return new_image.astype(np.uint8)


def IHMean(image, q):
    img = IHMeanAlogrithm(image, q)
    return img


if __name__ == "__main__":
    filename = input('请输入要打开的图片路径：')
    img = imread(filename, as_gray=True)
    opt = input('1.加椒盐噪声 2.加高斯噪声 3.几何均值 4.算术均值 5.谐波 6.逆谐波：')

    if opt == '1':
        img = add_salt_noise(img)
    elif opt == '2':
        img = add_gaussian_noise(img)
    elif opt == '3':
        img = gemotricMean(img)
    elif opt == '4':
        img = arithmeticMean(img)
    elif opt == '5':
        img = HMean(img)
    elif opt == '6':
        img = IHMean(img, 2)

    imshow(img)
