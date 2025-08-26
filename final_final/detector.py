from ultralytics import YOLO
import numpy as np, pandas as pd
from utils import load_image_bgr, green_leaf_mask, rasterized_union_area

DEFAULT_CLASS_NAMES = ['curl_stage1', 'curl_stage2', 'healthy', 'leaf_enation', 'sooty']

def load_model(model_path: str):
    return YOLO(model_path)

def run_detection(image_path: str, model, class_names=None):
    img_bgr = load_image_bgr(image_path)
    result = model.predict(source=img_bgr, verbose=False)[0]
    sev_df, debug = compute_severity_from_result(img_bgr, result, class_names)

    # 🔥 Ensure sev_df is always JSON-like
    if isinstance(sev_df, pd.DataFrame):
        return sev_df.to_dict(orient="records")  # list of dicts
    elif isinstance(sev_df, list):  # already JSON-like
        return sev_df
    else:
        return []


def compute_severity_from_result(image_bgr, result, class_names=None, exclude_labels=('healthy',)):
    if class_names is None:
        if hasattr(result, 'names') and isinstance(result.names, dict):
            class_names = [result.names[i] for i in range(len(result.names))]
        else:
            class_names = DEFAULT_CLASS_NAMES

    H, W = image_bgr.shape[:2]
    leaf_mask = green_leaf_mask(image_bgr)
    leaf_area = int((leaf_mask > 0).sum()) or H * W

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else np.empty((0,4))
    cls  = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else np.empty((0,), dtype=int)

    per_class_boxes = {}
    for (x1,y1,x2,y2), c in zip(xyxy, cls):
        per_class_boxes.setdefault(c, []).append((x1,y1,x2,y2))

    rows, debug = [], {}
    for c, bboxes in per_class_boxes.items():
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        if name in (exclude_labels or ()):
            continue

        _, union_mask = rasterized_union_area((H, W), bboxes)
        diseased_inside_leaf = int((union_mask.astype(bool) & (leaf_mask > 0)).sum())
        sev_pct = round(100.0 * diseased_inside_leaf / max(1, leaf_area), 2)

        stage = None
        if "stage" in name:
            try: stage = name.split("stage", 1)[1].strip("_- ")
            except: stage = None

        rows.append({
            "disease": name,
            "stage": stage,
            "n_boxes": len(bboxes),
            "severity_pct": sev_pct,
            "area_px": diseased_inside_leaf
        })
        debug[name] = {"class_id": c, "boxes": bboxes, "leaf_area_px": leaf_area}

        # 🔹 Save debug masks
        import cv2, os
        os.makedirs("debug_masks", exist_ok=True)
        cv2.imwrite(f"debug_masks/{name}_leaf_mask.png", leaf_mask)
        cv2.imwrite(f"debug_masks/{name}_union_mask.png", (union_mask*255).astype("uint8"))

    severity_df = (
        pd.DataFrame(rows)
        .sort_values("severity_pct", ascending=False)
        .reset_index(drop=True)
    )

    return severity_df.to_dict(orient="records"), debug

