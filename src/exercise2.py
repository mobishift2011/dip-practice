from exercise1 import *
from skimage import io, exposure
from collections import Counter


def log_enhance(img):
    """对数变换"""
    max_val = np.log(1+img.max())
    c = 255/max_val if max_val else 0
    img = c * np.log(1+img)
    img = np.array(img, dtype=np.uint8)
    return img


def exp_enhance(img, g=0.5):
    """幂次变换, 指数g默认为0.5"""
    img = (img/255) ** g
    c = 255/img.max()
    img *= c
    img = np.array(img, dtype=np.uint8)
    return img


def linear_convert(img, origin=(50, 150), ext=(0, 255)):
    """线性变换到整个灰度级"""
    rows, cols = img.shape

    for i in range(rows):
        for j in range(cols):
            val = img[i, j]
            if val < origin[0]:
                img[i, j] = val*ext[0]/origin[0]
            elif val > origin[1]:
                img[i, j] = (val-origin[1])*(255-ext[1]) / \
                    (255-origin[1]) + ext[1]
            else:
                img[i, j] = (val-origin[0])*(ext[1]-ext[0]) / \
                    (origin[1]-origin[0]) + ext[0]

    return img


def equalize_hist(img, nbins=256):
    """
    直方图均衡
    1.统计灰度值
    2.计算概率密度
    3.计算累计分布
    4.扩展求整int[(max-min)*cumulative_p+0.5]
    5.映射关系
    """
    arr = img.flatten()
    r = Counter()

    # 统计原始图像灰度级个数
    for v in arr:
        r[v] += 1

    # 计算概率密度，并直接求出累计分布函数
    rows, cols = img.shape
    total = rows * cols
    reduced_val = 0
    cumulative_dist = [0] * nbins

    for k in range(nbins):
        reduced_val += r[k]
        cumulative_dist[k] = reduced_val / total

    # 扩展取整
    sk_map = {}
    for k in range(nbins):
        sk_map[k] = int((nbins-1)*cumulative_dist[k] + 0.5)

    # 根据灰度映射关系修改灰度值
    for i in range(rows):
        for j in range(cols):
            img[i, j] = sk_map[img[i, j]]

    return img


if __name__ == "__main__":
    filename = input('请输入要打开的图片路径：')
    img = imread(filename, as_gray=True)
    opt = input('1.对数变换 2.幂次变换 3.线性变换(拉伸) 4.直方图均衡：')

    if opt == '1':
        img = log_enhance(img)
    elif opt == '2':
        img = exp_enhance(img)
    elif opt == '3':
        i = input("请输入希望拉伸的像素最小和最大值，以逗号分割，eg. 10,100：")
        img = linear_convert(img, origin=tuple(map(int, i.split(','))))
    elif opt == '4':
        img = equalize_hist(img)

    imshow(img)
