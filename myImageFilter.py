import numpy as np

def myImageFilter(img0, h):
    pad_h = h.shape[0] // 2
    pad_w = h.shape[1] // 2
    img_padded = np.pad(img0, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    img1 = np.zeros_like(img0)

    for i in range(img0.shape[0]):
        for j in range(img0.shape[1]):
            img1[i, j] = np.sum(img_padded[i:i + h.shape[0], j:j + h.shape[1]] * h)

    return img1