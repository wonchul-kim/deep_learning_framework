import matplotlib.pyplot as plt
from torchvision.io import read_image


def vis_image(image_fn, mask_fn, filename):

    image = read_image(image_fn)
    mask = read_image(mask_fn)

    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.title("Image")
    plt.imshow(image.permute(1, 2, 0))
    plt.subplot(122)
    plt.title("Mask")
    plt.imshow(mask.permute(1, 2, 0))   
    plt.savefig(filename)