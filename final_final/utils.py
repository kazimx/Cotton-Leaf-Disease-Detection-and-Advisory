import cv2, numpy as np

def load_image_bgr(path):
    # Clean up stray spaces and newlines
    path = str(path).strip()
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image from {path}")
    return img

def green_leaf_mask(image_bgr):
    """
    Generate a refined binary mask for green cotton leaves.
    Keeps only the largest connected green region to avoid background noise.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # HSV range for green leaves (tune if needed)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological cleanup (remove noise, fill gaps)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours and keep the largest one (main leaf area)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        refined_mask = np.zeros_like(mask)
        cv2.drawContours(refined_mask, [largest], -1, 255, -1)
        return refined_mask

    return mask  # fallback if no contours found

def rasterized_union_area(mask_shape, boxes_xyxy):
    H, W = mask_shape[:2]
    m = np.zeros((H, W), dtype=np.uint8)
    for (x1,y1,x2,y2) in boxes_xyxy:
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
        y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
        if x2 > x1 and y2 > y1:
            m[y1:y2+1, x1:x2+1] = 1
    return int(m.sum()), m
