from exercise1 import imread, imshow
from skimage import img_as_float, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt


def imfft(img):
    """傅里叶变换"""
    img = img_as_float(img)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift


def imifft(img):
    """傅里叶反变换"""
    return np.fft.ifft2(np.fft.ifftshift(img))


def fftshow(img):
    """显示傅里叶变换后频谱图"""
    img = np.log(np.abs(img))

    plt.subplot(111)
    plt.imshow(img, 'gray')
    plt.show()
    return img


def fft_smooth(img, val):
    """
    频率域低通平滑
    1.傅里叶变换
    2.根据阈值val(eg. 5, 50, 150)过滤，保留低通分量
    3.傅里叶反变换
    4.超出数值部分像素处理
    5.转换回255灰度级
    """
    img = imfft(img)
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.zeros(img.shape)
    mask[crow-val:crow+val, ccol-val:ccol+val] = 1
    img = np.abs(imifft(img*mask))
    rows, cols = img.shape
    for row in range(rows):
        for col in range(cols):
            pix = img[row, col]
            if pix > 1:
                img[row, col] = 1
            elif pix < -1:
                img[row, col] = -1
    img = img_as_ubyte(img)
    return img


def fft_sharpen(img, val):
    """
    频率域高通锐化
    1.傅里叶变换
    2.根据阈值val(eg. 5, 50, 150)过滤，保留高通分量
    3.傅里叶反变换
    4.超出数值部分像素处理
    5.转换回255灰度级    
    """
    img = imfft(img)
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.zeros(img.shape)
    mask[crow-val:crow+val, ccol-val:ccol+val] = 1
    mask = 1-mask
    img = np.abs(imifft(img*mask))
    rows, cols = img.shape
    for row in range(rows):
        for col in range(cols):
            pix = img[row, col]
            if pix > 1:
                img[row, col] = 1
            elif pix < -1:
                img[row, col] = -1
    img = img_as_ubyte(img)
    return img


if __name__ == "__main__":
    filename = input('请输入要打开的图片路径：')
    img = imread(filename, as_gray=True)
    opt = input('1.傅立叶变换，显示频谱 2.频率域平滑 3.频率域锐化：')

    if opt == '1':
        img = imfft(img)
        fftshow(img)
    else:
        i = input('请输入截至频率(eg. 5，50，150)')
        if opt == '2':
            img = fft_smooth(img, int(i))
        elif opt == '3':
            img = fft_sharpen(img, int(i))

    imshow(img)
