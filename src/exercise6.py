from exercise1 import imread, imshow, immake
import numpy as np
from skimage import util


def rgb2hsi(rgb):
    """rgb转hsi"""
    arr = util.img_as_float(rgb)
    out = np.empty_like(arr)

    # -- V channel
    out_v = arr.max(-1)

    # -- S channel
    delta = arr.ptp(-1)
    # Ignore warning for zero divided by zero
    old_settings = np.seterr(invalid='ignore')
    out_s = delta / out_v
    out_s[delta == 0.] = 0.

    # -- H channel
    # red is max
    idx = (arr[:, :, 0] == out_v)
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]

    # green is max
    idx = (arr[:, :, 1] == out_v)
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]

    # blue is max
    idx = (arr[:, :, 2] == out_v)
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]
    out_h = (out[:, :, 0] / 6.) % 1.
    out_h[delta == 0.] = 0.

    np.seterr(**old_settings)

    # -- output
    out[:, :, 0] = out_h
    out[:, :, 1] = out_s
    out[:, :, 2] = out_v

    # remove NaN
    out[np.isnan(out)] = 0

    return out


def hsi2rgb(hsv):
    """hsi转rgb"""
    arr = util.img_as_float(hsv)

    hi = np.floor(arr[:, :, 0] * 6)
    f = arr[:, :, 0] * 6 - hi
    p = arr[:, :, 2] * (1 - arr[:, :, 1])
    q = arr[:, :, 2] * (1 - f * arr[:, :, 1])
    t = arr[:, :, 2] * (1 - (1 - f) * arr[:, :, 1])
    v = arr[:, :, 2]

    hi = np.dstack([hi, hi, hi]).astype(np.uint8) % 6
    out = np.choose(hi, [np.dstack((v, t, p)),
                         np.dstack((q, v, p)),
                         np.dstack((p, v, t)),
                         np.dstack((p, q, v)),
                         np.dstack((t, p, v)),
                         np.dstack((v, p, q))])

    return out


def pick_red(img, s=0.5):
    """
    提取饱和度在s=50%以上的红色色系图像像素
    """
    img_hsv = rgb2hsi(img)

    mark_h = img_hsv[:, :, 0] < 0.8  # 筛选出非红色色调的像素
    mark_s = img_hsv[:, :, 1] < 0.5  # 筛选出饱和度过低的像素
    img_hsv[mark_h] = [0, 0, 0]      # 将筛选出的不合条件的像素归0
    img_hsv[mark_s] = [0, 0, 0]
    return hsi2rgb(img_hsv)


if __name__ == "__main__":
    filename = input('请输入要打开的图片路径：')
    img = imread(filename, as_gray=False)
    img = pick_red(img)
    imshow(img)
