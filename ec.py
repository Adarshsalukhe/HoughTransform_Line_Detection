import cv2
import numpy as np
import os

from myEdgeFilter import myEdgeFilter
from myHoughTransform import myHoughTransform
from myHoughLineSegments import myHoughLineSegments, detectGreenLines

# Directory setup
datadir = 'ec'
resultsdir = 'ec_results'

if not os.path.exists(resultsdir):
    os.makedirs(resultsdir)

# Parameters
sigma = 1.2
threshold = 0.3
rhoRes = 2
thetaRes = np.pi / 180
nLines = 20

# Get list of valid images first
valid_files = [f for f in os.listdir(datadir) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Found {len(valid_files)} images to process...")

for file in valid_files:
    print(f"Processing {file}...")
    file_name = os.path.splitext(file)[0]
    
    # Read image
    img = cv2.imread(os.path.join(datadir, file))
    if img is None:
        print(f"Failed to load {file}")
        continue

    # Convert to grayscale and normalize in one step
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = (img / 255.0).astype(np.float32)  # More efficient than np.float32(img) / 255.0

    # Edge detection
    img_edge = myEdgeFilter(img, sigma)
    img_threshold = np.float32(img_edge > threshold)

    # Hough Transform
    img_hough, rhoScale, thetaScale = myHoughTransform(img_threshold, rhoRes, thetaRes)
    
    # Create output image more efficiently
    img_lines = np.broadcast_to(img[..., None], (*img.shape, 3)).copy()

    # Process segments in batches
    green_segments = np.array(detectGreenLines(img_threshold, img_hough, rhoScale, thetaScale, nLines))
    red_segments = np.array(myHoughLineSegments(img_threshold, img_hough, rhoScale, thetaScale, nLines))

    # Draw lines efficiently
    if len(green_segments) > 0:
        for x1, y1, x2, y2 in green_segments:
            cv2.line(img_lines, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
    
    if len(red_segments) > 0:
        for x1, y1, x2, y2 in red_segments:
            cv2.line(img_lines, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

    # Prepare all outputs first
    edge_out = 255 * np.sqrt(img_edge / np.maximum(img_edge.max(), 1e-10))
    threshold_out = 255 * img_threshold
    hough_out = 255 * img_hough / np.maximum(img_hough.max(), 1e-10)
    lines_out = 255 * img_lines

    # Save all results
    cv2.imwrite(os.path.join(resultsdir, f'{file_name}_01edge.png'), edge_out)
    cv2.imwrite(os.path.join(resultsdir, f'{file_name}_02threshold.png'), threshold_out)
    cv2.imwrite(os.path.join(resultsdir, f'{file_name}_03hough.png'), hough_out)
    cv2.imwrite(os.path.join(resultsdir, f'{file_name}_04lines.png'), lines_out)

print("Processing complete! Check the ec_results directory for output images.")