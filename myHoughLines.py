import numpy as np
import cv2  # For cv2.dilate function

def myHoughLines(H, nLines):
    
    # Normalize and apply dilation for peak detection
    H_norm = cv2.normalize(H, None, 0, 255, cv2.NORM_MINMAX)  # As professor allowed to use this normalize function
    H_uint8 = np.uint8(H_norm)

    kernel = np.ones((7, 7), np.uint8)  # Kernel for better peak suppression
    H_dilated = cv2.dilate(H_uint8, kernel)

    # Find local maxima in Hough space
    local_maxima = (H_uint8 == H_dilated)
    peaks = np.argwhere(local_maxima)
    peak_values = H[peaks[:, 0], peaks[:, 1]]

    sorted_indices = np.argsort(peak_values)[::-1]
    selected_peaks = peaks[sorted_indices]

    selected_rhos = []
    selected_thetas = []
    
    min_rho_diff = 10 
    min_theta_diff = np.pi / 90  

    for rho_idx, theta_idx in selected_peaks:
        rho = rho_idx  # Index corresponds to rho scale
        theta = theta_idx  # Index corresponds to theta scale

        # Check if this line is too close to a previously selected one
        is_duplicate = False
        for prev_rho, prev_theta in zip(selected_rhos, selected_thetas):
            if abs(rho - prev_rho) < min_rho_diff and abs(theta - prev_theta) < min_theta_diff:
                is_duplicate = True
                break

        if is_duplicate:
            continue  # Skip duplicate lines

        selected_rhos.append(rho)
        selected_thetas.append(theta)

        # Stop when the required number of lines is reached
        if len(selected_rhos) >= nLines:
            break

    return np.array(selected_rhos), np.array(selected_thetas)