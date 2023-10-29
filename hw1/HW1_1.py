import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def reflect_padding(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of three int [height, width]): filter size (e.g. (3,3))
    Return:
        padded image (numpy array)
    """
    # Your code
    h = int((size[0]-1)/2)
    w = int((size[1]-1)/2)
    aa=np.zeros((input_image.shape[0]+2*h, input_image.shape[1]+2*w, 3))
    for k in range(0, 3):
        for i in range(h, h + input_image.shape[0]):
            for j in range(w, w + input_image.shape[1]):
                aa[i][j][k] = input_image[i-h][j-w][k]
    for k in range(0, 3):
        for i in range(0, h):
            for j in range(w, w + input_image.shape[1]):
                aa[i][j][k] = aa[2*h-i][j][k]
    for k in range(0, 3):
        for i in range(h + input_image.shape[0], 2*h + input_image.shape[0]):
            for j in range(w, w + input_image.shape[1]):
                aa[i][j][k] = aa[2*(h + input_image.shape[0]-1)-i][j][k]
    for k in range(0, 3):
        for i in range(0, 2*h + input_image.shape[0]):
            for j in range(0, w):
                aa[i][j][k] = aa[i][2*w-j][k]
    for k in range(0, 3):
        for i in range(0, 2*h + input_image.shape[0]):
            for j in range(w + input_image.shape[1], 2*w + input_image.shape[1]):
                aa[i][j][k] = aa[i][2*(w + input_image.shape[1]-1)-j][k]
    return aa

def convolve(input_image, Kernel):
    """
    Args:
        input_image (numpy array): input array
        Kernel (numpy array): kernel shape of (height, width)
    Return:
        convolved image (numpy array)
    """
    # Your code
    # Note that the dimension of the input_image and the Kernel are different.
    # shape of input_image: (height, width, channel)
    # shape of Kernel: (height, width)
    # Make sure that the same Kernel be applied to each of the channels of the input_image
    input_image=reflect_padding(input_image, Kernel.shape)
    a=input_image.shape[0]
    b=input_image.shape[1]
    c=input_image.shape[2]
    h=Kernel.shape[0]
    w=Kernel.shape[1]
    aa=np.zeros((a-h+1,b-w+1,c))
    for m in range(0, c):
        for i in range(0, a - h + 1):
            for j in range(0, b - w + 1):
                for k in range(0, h):
                    for l in range(0, w):
                        aa[i][j][m] += input_image[i + k][j + l][m] * Kernel[k][l]
    return aa


def median_filter(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,3))
    Return:
        Median filtered image (numpy array)
    """
    input_image=reflect_padding(input_image, size)
    for s in size:
        if s % 2 == 0:
            raise Exception("size must be odd for median filter")
    a = input_image.shape[0]
    b = input_image.shape[1]
    c = input_image.shape[2]
    h=size[0]
    w=size[1]
    aa = np.zeros((a-h+1,b-w+1,c))
    q=list()
    for m in range(0,c):
        for i in range(0, a - h + 1):
            for j in range(0, b - w + 1):
                for k in range(0, h):
                    for l in range(0, w):
                        q.append(input_image[i + k][j + l][m])
                q.sort()
                aa[i][j][m] = q[int((h * w - 1) / 2)]
                q.clear()
    return aa


def gaussian_filter(input_image, size, sigmax, sigmay):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,.3))
        sigmax (float): standard deviation in X direction
        sigmay (float): standard deviation in Y direction
    Return:
        Gaussian filtered image (numpy array)
    """
    # Your code
    kk=np.zeros((size[0],size[1]))
    a=int((size[0]-1)/2)
    b=int((size[1]-1)/2)
    for i in range(-a, a):
        for j in range(-b, b):
            kk[a+i][b+j] = np.exp(-(i*i/(2*sigmax*sigmax)+j*j/(2*sigmay*sigmay)))
    kk /= kk.sum()
    return convolve(input_image,kk)


if __name__ == '__main__':
    #image = np.asarray(Image.open(os.path.join('images', 'baboon.jpeg')).convert('RGB'))
    image = np.asarray(Image.open(os.path.join('images', 'gaussian_noise.jpeg')).convert('RGB'))
    #image = np.asarray(Image.open(os.path.join('images', 'salt_and_pepper_noise.jpeg')).convert('RGB'))

    logdir = os.path.join('results', 'HW1_1')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    kernel_1 = np.ones((5,5)) / 25.
    sigmax, sigmay = 5, 5

    ret = reflect_padding(image.copy(), kernel_1.shape)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'reflect.jpeg'))
        plt.show()

    ret = convolve(image.copy(), kernel_1)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'convolve.jpeg'))
        plt.show()

    ret = median_filter(image.copy(), kernel_1.shape)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'median.jpeg'))
        plt.show()

    ret = gaussian_filter(image.copy(), kernel_1.shape, sigmax, sigmay)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'gaussian.jpeg'))
        plt.show()


