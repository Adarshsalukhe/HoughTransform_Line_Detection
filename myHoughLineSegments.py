import numpy as np

def myHoughLineSegments(img_threshold, H, rhoScale, thetaScale, nLines):
    height, width = img_threshold.shape
    segments = []
    
    peaks = []
    H_copy = H.copy()
    min_peak_value = np.max(H_copy) * 0.3 
    
    for _ in range(nLines):
        max_idx = np.argmax(H_copy)
        peak_value = H_copy.flat[max_idx]
        
        if peak_value < min_peak_value:
            break
            
        rho_idx, theta_idx = np.unravel_index(max_idx, H_copy.shape)
        peaks.append((rho_idx, theta_idx))
        
        r_start = max(0, rho_idx - 3)
        r_end = min(H_copy.shape[0], rho_idx + 4)
        t_start = max(0, theta_idx - 3)
        t_end = min(H_copy.shape[1], theta_idx + 4)
        H_copy[r_start:r_end, t_start:t_end] = 0
    
    # Process each peak
    window_size = 3   # Window size kept to 3x3 for simplicity and getting accurate outputs
    min_segment_length = 15
    gap_threshold = 10 
    
    for rho_idx, theta_idx in peaks:
        rho = rhoScale[rho_idx]
        theta = thetaScale[theta_idx]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        if abs(cos_t) < 0.001:
            x0 = int(rho/sin_t)
            if 0 <= x0 < width:
                points = []
                for y in range(0, height-1):
                    if 0 <= x0 < width and img_threshold[y, x0] > 0:
                        points.append((x0, y))
                    elif points:
                        if len(points) >= min_segment_length:
                            segments.append((points[0][0], points[0][1], 
                                          points[-1][0], points[-1][1]))
                        points = []
                
                if len(points) >= min_segment_length:
                    segments.append((points[0][0], points[0][1], 
                                   points[-1][0], points[-1][1]))
        
        elif abs(sin_t) < 0.001:
            y0 = int(rho/cos_t)
            if 0 <= y0 < height:
                points = []
                for x in range(0, width-1):
                    if 0 <= y0 < height and img_threshold[y0, x] > 0:
                        points.append((x, y0))
                    elif points:
                        if len(points) >= min_segment_length:
                            segments.append((points[0][0], points[0][1], 
                                          points[-1][0], points[-1][1]))
                        points = []
                
                if len(points) >= min_segment_length:
                    segments.append((points[0][0], points[0][1], 
                                   points[-1][0], points[-1][1]))
        
        else:
            points = []
            x0, y0 = rho * cos_t, rho * sin_t
            
            for t in range(-max(width, height), max(width, height)):
                x = int(x0 - t * sin_t)
                y = int(y0 + t * cos_t)
                
                if 0 <= x < width and 0 <= y < height:
                    if img_threshold[y, x] > 0:
                        points.append((x, y))
                    elif points:
                        if len(points) >= min_segment_length:
                            segments.append((points[0][0], points[0][1], 
                                          points[-1][0], points[-1][1]))
                        points = []
                        
            if len(points) >= min_segment_length:
                segments.append((points[0][0], points[0][1], 
                               points[-1][0], points[-1][1]))
    
    merged_segments = []
    used = set()
    
    for i, (x1, y1, x2, y2) in enumerate(segments):
        if i in used:
            continue
            
        current_segment = [x1, y1, x2, y2]
        used.add(i)
        
        while True:
            found_merge = False
            for j, (x3, y3, x4, y4) in enumerate(segments):
                if j in used or j == i:
                    continue

                d1 = np.sqrt((x2-x3)**2 + (y2-y3)**2)
                d2 = np.sqrt((x2-x4)**2 + (y2-y4)**2)
                d3 = np.sqrt((x1-x3)**2 + (y1-y3)**2)
                d4 = np.sqrt((x1-x4)**2 + (y1-y4)**2)
                
                min_dist = min(d1, d2, d3, d4)
                if min_dist < gap_threshold:
                    points = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                    points.sort(key=lambda p: (p[0], p[1]))
                    current_segment = [points[0][0], points[0][1], 
                                     points[-1][0], points[-1][1]]
                    used.add(j)
                    found_merge = True
                    break
            
            if not found_merge:
                break
        
        merged_segments.append(current_segment)
    
    return merged_segments

# Function for GreenLines

def detectGreenLines(img_threshold, H, rhoScale, thetaScale, nLines):
    height, width = img_threshold.shape
    segments = []

    H_copy = H.copy()
    peaks = []
    min_peak_value = np.max(H_copy) * 0.3
    
    for _ in range(nLines):
        if np.max(H_copy) < min_peak_value:
            break
            
        max_idx = np.argmax(H_copy)
        rho_idx, theta_idx = np.unravel_index(max_idx, H_copy.shape)
        peaks.append((rho_idx, theta_idx))
        
        r_start = max(0, rho_idx - 4)
        r_end = min(H_copy.shape[0], rho_idx + 5)
        t_start = max(0, theta_idx - 4)
        t_end = min(H_copy.shape[1], theta_idx + 5)
        H_copy[r_start:r_end, t_start:t_end] = 0
    
    for rho_idx, theta_idx in peaks:
        rho = rhoScale[rho_idx]
        theta = thetaScale[theta_idx]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        x0 = cos_t * rho
        y0 = sin_t * rho
        
        points = []
        for t in range(-max(width, height), max(width, height)):
            x = int(x0 - t * sin_t)
            y = int(y0 + t * cos_t)
            
            if 0 <= x < width and 0 <= y < height:
                y1, y2 = max(0, y-1), min(height, y+2)
                x1, x2 = max(0, x-1), min(width, x+2)
                if np.any(img_threshold[y1:y2, x1:x2]):
                    points.append((x, y))

        if len(points) > 0:
            current_segment = [points[0]]
            max_gap = 5 
            min_length = 20
            
            for i in range(1, len(points)):
                dx = points[i][0] - points[i-1][0]
                dy = points[i][1] - points[i-1][1]
                gap = np.sqrt(dx*dx + dy*dy)
                
                if gap <= max_gap:
                    current_segment.append(points[i])
                else:
                    if len(current_segment) >= min_length:
                        segments.append((current_segment[0][0], current_segment[0][1],
                                      current_segment[-1][0], current_segment[-1][1]))
                    current_segment = [points[i]]
            
            if len(current_segment) >= min_length:
                segments.append((current_segment[0][0], current_segment[0][1],
                              current_segment[-1][0], current_segment[-1][1]))
    
    return segments