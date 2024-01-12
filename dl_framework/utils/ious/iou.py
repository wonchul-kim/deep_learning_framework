from functional import *
from vis import *

box1 = (10, 20, 50, 60)
box2 = (30, 40, 80, 70)

# Visualize the bounding boxes
# visualize_bounding_boxes(box1, box2)
intersection_area = get_intersection_area(box1, box2)
print(intersection_area)



def bbox_