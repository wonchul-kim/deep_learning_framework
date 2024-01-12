import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_bounding_boxes(box1, box2):
    fig, ax = plt.subplots(1)

    rect1 = patches.Rectangle((box1[0], box1[1]), box1[2] - box1[0], box1[3] - box1[1], linewidth=2, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((box2[0], box2[1]), box2[2] - box2[0], box2[3] - box2[1], linewidth=2, edgecolor='b', facecolor='none')

    ax.add_patch(rect1)
    ax.add_patch(rect2)

    ax.set_xlim(min(box1[0], box2[0]) - 5, max(box1[2], box2[2]) + 5)
    ax.set_ylim(min(box1[1], box2[1]) - 5, max(box1[3], box2[3]) + 5)

    plt.gca().set_aspect('equal', adjustable='box')  # Make the axes equal for a better visualization
    plt.show()
