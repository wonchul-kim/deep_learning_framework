import numpy as np

def get_intersection_area(box1, box2):
    """
    - box1, box2: tuple of (x1, y1, x2, y2)
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    x1 = max(b1_x1, b2_x1)
    y1 = max(b1_y1, b2_y1)
    x2 = min(b1_x2, b2_x2)
    y2 = min(b1_y2, b2_y2)

    return max(0, (x2 - x1))*max(0, (y2 - y1))