# utils/annotator_helpers.py
import json
from matplotlib.path import Path as MplPath

def parse_points(click_str=None, polygon_str=None):
    """Parse click and polygon strings into usable JSON objects."""
    click_points, poly_points = [], []
    try:
        if click_str:
            click_points = json.loads(click_str)
        if polygon_str:
            poly_points = json.loads(polygon_str)
            if isinstance(poly_points, list) and isinstance(poly_points[0], str):
                poly_points = [json.loads(p) for p in poly_points]
    except Exception as e:
        print("Error parsing click/polygon:", e)
    return click_points, poly_points


def get_nearest_box(x, y, boxes):
    min_dist = float('inf')
    best_idx = -1
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = (cx - x) ** 2 + (cy - y) ** 2
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    return best_idx


def point_in_polygon(x, y, polygon):
    path = MplPath([(pt['x'], pt['y']) for pt in polygon])
    return path.contains_point((x, y))


def filter_boxes_from_click_polygon(click_points, poly_points, boxes, logits, phrases):
    results = []
    added_idxs = set()

    for pt in click_points:
        idx = get_nearest_box(pt['x'], pt['y'], boxes)
        if idx != -1 and idx not in added_idxs:
            results.append({
                "bbox": list(map(int, boxes[idx])),
                "label": phrases[idx],
                "confidence": float(logits[idx])
            })
            added_idxs.add(idx)

    if poly_points:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if point_in_polygon(cx, cy, poly_points) and i not in added_idxs:
                results.append({
                    "bbox": list(map(int, boxes[i])),
                    "label": phrases[i],
                    "confidence": float(logits[i])
                })
                added_idxs.add(i)

    return results
