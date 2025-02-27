import cv2
import numpy as np
import os

from myEdgeFilter import myEdgeFilter
from myHoughTransform import myHoughTransform
from myHoughLineSegments import myHoughLineSegments, detectGreenLines

datadir = 'data'
resultsdir = 'results'

if not os.path.exists(resultsdir):
    os.makedirs(resultsdir)

# Parameters
sigma = 1.2
threshold = 0.3
rhoRes = 2
thetaRes = np.pi / 180
nLines = 15

for file in os.listdir(datadir):
    if file.endswith(('.jpg', '.png')):
        file_name = os.path.splitext(file)[0]
        img = cv2.imread(os.path.join(datadir, file))

        if img is None:
            continue

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = np.float32(img) / 255.0

        # Edge detection
        img_edge = myEdgeFilter(img, sigma)
        img_threshold = np.float32(img_edge > threshold)

        # Hough Transform
        img_hough, rhoScale, thetaScale = myHoughTransform(img_threshold, rhoRes, thetaRes)
        
        # Create output image
        img_lines = np.dstack([img, img, img])


        green_segments = detectGreenLines(img_threshold, img_hough, rhoScale, thetaScale, nLines)
        for x1, y1, x2, y2 in green_segments:
            cv2.line(img_lines, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            
        red_segments = myHoughLineSegments(img_threshold, img_hough, rhoScale, thetaScale, nLines)
        for x1, y1, x2, y2 in red_segments:
            cv2.line(img_lines, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        # Save results
        cv2.imwrite(os.path.join(resultsdir, f'{file_name}_01edge.png'), 
                    255 * np.sqrt(img_edge / img_edge.max()))
        cv2.imwrite(os.path.join(resultsdir, f'{file_name}_02threshold.png'), 
                    255 * img_threshold)
        cv2.imwrite(os.path.join(resultsdir, f'{file_name}_03hough.png'), 
                    255 * img_hough / img_hough.max())
        cv2.imwrite(os.path.join(resultsdir, f'{file_name}_04lines.png'), 
                    255 * img_lines)