import numpy as np
from myImageFilter import myImageFilter

def myEdgeFilter(img0, sigma):
    hsize = int(2 * np.ceil(3 * sigma) + 1)
    x = np.linspace(-3 * sigma, 3 * sigma, hsize)
    gauss_1d = np.exp(-0.5 * (x / sigma)**2)
    gauss_1d /= gauss_1d.sum()
    gaussian_kernel = np.outer(gauss_1d, gauss_1d)

    img_smoothed = myImageFilter(img0, gaussian_kernel)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    grad_x = myImageFilter(img_smoothed, sobel_x)
    grad_y = myImageFilter(img_smoothed, sobel_y)

    magnitude = np.hypot(grad_x, grad_y)
    angle = np.arctan2(grad_y, grad_x) * (180 / np.pi)

    suppressed_magnitude = np.zeros_like(magnitude)
    rows, cols = magnitude.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle_rounded = (np.round(angle[i, j] / 45) * 45) % 180
            if angle_rounded == 0 and magnitude[i, j] < max(magnitude[i, j - 1], magnitude[i, j + 1]):
                suppressed_magnitude[i, j] = 0
            elif angle_rounded == 45 and magnitude[i, j] < max(magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]):
                suppressed_magnitude[i, j] = 0
            elif angle_rounded == 90 and magnitude[i, j] < max(magnitude[i - 1, j], magnitude[i + 1, j]):
                suppressed_magnitude[i, j] = 0
            elif angle_rounded == 135 and magnitude[i, j] < max(magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]):
                suppressed_magnitude[i, j] = 0
            else:
                suppressed_magnitude[i, j] = magnitude[i, j]

    return suppressed_magnitude
