import numpy as np

def myHoughTransform(img_threshold, rhoRes, thetaRes):
    height, width = img_threshold.shape
    diag_len = int(np.ceil(np.sqrt(height**2 + width**2)))
    rhos = np.arange(-diag_len, diag_len + 1, rhoRes)
    thetas = np.arange(0, np.pi, thetaRes)
    H = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    
    # Get indices of edge pixels
    y_idxs, x_idxs = np.nonzero(img_threshold)
    
    # Precompute cosine and sine for each theta
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    rhos_vals = x_idxs[:, np.newaxis] * cos_t[np.newaxis, :] + y_idxs[:, np.newaxis] * sin_t[np.newaxis, :]

    rho_indices = np.round((rhos_vals - rhos[0]) / rhoRes).astype(np.int32)
    
    rows = rho_indices.flatten()
    cols = np.tile(np.arange(len(thetas)), rho_indices.shape[0])
    np.add.at(H, (rows, cols), 1)
    
    return H, rhos, thetas
