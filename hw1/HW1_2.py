import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import utils

def gaussian_pyramid(input_image, level):
    """
    Args:
        input_image (numpy array): input array
        level (int): level of pyramid

    Return:
        Gaussian pyramid (list of numpy array)
    """
    # Your code
    # Note that elements in the list must be arranged in descending order in image resolution (from big image to small image)

    Gp=list()
    Gp.append(input_image)
    aa = input_image.copy()
    for i in range(0, level):
        aa = utils.down_sampling(aa)
        Gp.append(aa)
    return Gp


def laplacian_pyramid(gaussian_pyramid):
    """
    Args:
        gaussian_pyramid (list of numpy array): result from the gaussian_pyramid function

    Return:
        laplacian pyramid (list of numpy array)
    """

    # Your code
    # Note that elements in the list must be arranged in descending order in image resolution (from big image to small image).

    gp=gaussian_pyramid.copy()
    lp=list()
    for i in range(0,len(gp)-1):
        a=utils.up_sampling(gp[i + 1])
        b=utils.safe_subtract(gp[i],a)
        lp.append(b)
    lp.append(gp[len(gp)-1])
    return lp

def blend_images(image1, image2, mask, level):
    """
    Args:
        image1 (numpy array): background image
        image2 (numpy array): object image
        mask (numpy array): mask
        level (int): level of pyramid
    Return:
        blended image (numpy array)
    """
    # Your code
    l1=laplacian_pyramid(gaussian_pyramid(utils.safe_subtract(image1,mask),level))
    l2=laplacian_pyramid(gaussian_pyramid(image2,level))
    b = utils.up_sampling(utils.safe_add(l1[level], l2[level]))
    for i in range(1,level+1):
        a=utils.safe_add(l1[level-i],l2[level-i])
        b=utils.safe_add(a,b)
        if i!=level:
            b=utils.up_sampling(b)
    return b


if __name__ == '__main__':
    hand = np.asarray(Image.open(os.path.join('images', 'hand.jpeg')).convert('RGB'))
    flame = np.asarray(Image.open(os.path.join('images', 'flame.jpeg')).convert('RGB'))
    mask = np.asarray(Image.open(os.path.join('images', 'mask.jpeg')).convert('RGB'))

    logdir = os.path.join('results', 'HW1_2')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    level = 3


    plt.figure()
    plt.imshow(Image.open(os.path.join('images', 'direct_concat.jpeg')))
    plt.axis('off')
    plt.savefig(os.path.join(logdir, 'direct.jpeg'))
    plt.show()

    ret = gaussian_pyramid(hand, level)
    if ret is not None:
        plt.figure()
        for i in range(len(ret)):
            plt.subplot(1, len(ret), i + 1)
            plt.imshow(ret[i].astype(np.uint8))
            plt.axis('off')
        plt.savefig(os.path.join(logdir, 'gaussian_pyramid.jpeg'))
        plt.show()

        ret = laplacian_pyramid(ret)
        if ret is not None:
            plt.figure()
            for i in range(len(ret)):
                plt.subplot(1, len(ret), i + 1)
                plt.imshow(ret[i].astype(np.uint8))
                plt.axis('off')
            plt.savefig(os.path.join(logdir, 'laplacian_pyramid.jpeg'))
            plt.show()

    ret = blend_images(hand, flame, mask, level)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'blended.jpeg'))
        plt.show()
