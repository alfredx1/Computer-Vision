from PIL import Image
import math
import glob
import numpy as np
import os
from math import pi

# parameters

datadir = './data'
resultdir='./results'

# you can calibrate these parameters
sigma=2
GaussianKernelSize=7 #변수는 홀수만 가능!!
highThreshold=0.15
lowThreshold=0.04
rhoRes=1.8
thetaRes=math.pi/200
nLines=20
rhoMaxNum=1200 #bigger than diagonal length
offset=3


def ConvFilter(Igs, G):
    # TODO ...padding 한 이후에 convolution!!
    h = int((G.shape[0]-1)/2)
    w = int((G.shape[1]-1)/2)
    a = Igs.shape[0]
    b = Igs.shape[1]
    pad = np.zeros((a+2*h, b+2*w))
    Iconv = np.zeros((a,b))
    for i in range(h, h+a):
        for j in range(w, w+b):
            pad[i][j] = Igs[i-h][j-w]
    for i in range(0, h):
        for j in range(w, b+w):
            pad[i][j] = pad[h][j]
    for i in range(a+h, a+2*h):
        for j in range(w, b+w):
            pad[i][j] = pad[a+h-1][j]
    for i in range(0, a+2*h):
        for j in range(0, w):
            pad[i][j] = pad[i][w]
    for i in range(0, a + 2 * h):
        for j in range(b+w, b+2*w):
            pad[i][j] = pad[i][b+w-1]
    for i in range(0, a):
        for j in range(0, b):
            for k in range(0, 2*h+1):
                for l in range(0, 2*w+1):
                    Iconv[i][j] += pad[i + k][j + l] * G[k][l]
    return Iconv

def EdgeDetection(Igs, sigma, highThreshold, lowThreshold):
    # TODO ...1. 가우시안 2. 소벨필터 적용 3. Im, Io compute 4. Non maximal suppression 5. Double Thresholding
    a = int((GaussianKernelSize-1)/2)
    G = np.zeros((GaussianKernelSize,GaussianKernelSize))
    for i in range (-a,a+1):
        for j in range (-a,a+1):
            G[i+a][j+a] = np.exp(-(i*i+j*j)/(2*sigma*sigma))/(2*pi*sigma*sigma)
    smooth = ConvFilter(Igs,G)
    Sobel_X = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Sobel_Y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    Ix = ConvFilter(smooth,Sobel_X)
    Iy = ConvFilter(smooth,Sobel_Y)
    Im_0 = np.zeros((Igs.shape[0],Igs.shape[1]))
    Io = np.zeros((Igs.shape[0],Igs.shape[1]))
    for i in range (0, Igs.shape[0]):
        for j in range (0, Igs.shape[1]):
            Im_0[i][j]= math.sqrt(Ix[i][j]*Ix[i][j]+Iy[i][j]*Iy[i][j])
            if Ix[i][j]!=0:
                Io[i][j] = np.arctan(Iy[i][j]/Ix[i][j])
    Im_1 = Im_0.copy()
    # Non maximal suppression
    for i in range (1, Igs.shape[0]-1):
        for j in range(1, Igs.shape[1]-1):
            if -3*pi/8>Io[i][j]>-pi/2 or 3*pi/8<Io[i][j]<pi/2 :
                if Im_0[i][j]<Im_0[i-1][j] or Im_0[i][j]<Im_0[i+1][j]:
                    Im_1[i][j]=0
            elif -3 * pi / 8 < Io[i][j] < -pi / 8:
                if Im_0[i][j] < Im_0[i - 1][j-1] or Im_0[i][j] < Im_0[i + 1][j+1]:
                    Im_1[i][j] = 0
            elif 3 * pi / 8 > Io[i][j] > pi / 8:
                if Im_0[i][j] < Im_0[i - 1][j+1] or Im_0[i][j] < Im_0[i + 1][j-1]:
                    Im_1[i][j] = 0
            elif -pi / 8 < Io[i][j] < pi / 8 :
                if Im_0[i][j] < Im_0[i][j-1] or Im_0[i][j] < Im_0[i][j+1]:
                    Im_1[i][j] = 0
    # Double Thresholding
    Im=Im_1.copy()
    for i in range(1, Igs.shape[0]-1):
        for j in range(1, Igs.shape[1] - 1):
            if Im_1[i][j] <= lowThreshold:
                Im[i][j] = 0
            elif lowThreshold< Im_1[i][j] <= highThreshold:
                if Im_1[i][j-1] < highThreshold and Im_1[i-1][j] < highThreshold and Im_1[i][j+1] < highThreshold and Im_1[i+1][j] < highThreshold:
                    Im[i][j] = 0
            else:
                Im[i][j] = 1


    return Im, Io, Ix, Iy

def HoughTransform(Im, rhoRes, thetaRes):
    # TODO
    a = int(2*pi/thetaRes)
    H=np.zeros((rhoMaxNum, a))
    for i in range(0, Im.shape[0]):
        for j in range(0, Im.shape[1]):
            if Im[i][j] == 1:
                for k in range(0, a):
                    H[int((i*math.cos(k*thetaRes)+j*math.sin(k*thetaRes))/rhoRes+0.5)][k]+=1
    return H

def HoughLines(H,rhoRes,thetaRes,nLines):
    lRho = []
    lTheta = []
    #Non Maximal Suppression
    P=[]
    for i in range(1, H.shape[0]-1):
        for j in range(1, H.shape[1]-1):
            P.append((i,j,H[i][j]))
            for k in range(-1,2):
                for l in range(-1,2):
                    if H[i+k][j+l]>H[i][j]:
                        H[i][j]=0

    T = sorted(P, key=lambda tup: tup[2], reverse=True)
    for i in range(0, nLines):
        lRho.append(T[i][0]*rhoRes)
        lTheta.append(T[i][1]*thetaRes)
    return lRho,lTheta

def HoughLineSegments(lRho, lTheta, Im):
    l=[]
    a=[]
    for k in range(offset, nLines-offset):
        a.clear()
        for i in range(offset, Im.shape[0]-offset):
            j=int((lRho[k]-math.cos(lTheta[k])*i)/math.sin(lTheta[k])+0.5)
            if offset <= j < Im.shape[1]-offset:
                t=1
                for n in range(-offset, offset + 1):
                    for m in range(-offset, offset + 1):
                        if Im[i + n][j + m] == 1:
                            t = 0
                if t == 0:
                    a.append((i, j))
        for j in range(offset, Im.shape[1] - offset):
            i = int((lRho[k] - math.sin(lTheta[k]) * j) / math.cos(lTheta[k]) + 0.5)
            if offset <= i < Im.shape[0] - offset:
                t = 1
                for n in range(-offset, offset + 1):
                    for m in range(-offset, offset + 1):
                        if Im[i + n][j + m] == 1:
                            t = 0
                if t == 0:
                    a.append((i, j))
        if len(a)>0:
            a.sort(key=lambda tup: tup[0])
            l.append({'start':a[0], 'end':a[-1]})
    return l

def main():
    filenum=0;
    # read images
    for img_path in glob.glob(datadir+'/*.jpg'):
        # load grayscale image
        img = Image.open(img_path).convert("L")
        color = Image.open(img_path)
        Igs = np.array(img)
        Igs = Igs / 255.

        # Hough function
        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma, highThreshold, lowThreshold)
        H= HoughTransform(Im, rhoRes, thetaRes)
        lRho,lTheta =HoughLines(H,rhoRes,thetaRes,nLines)
        l = HoughLineSegments(lRho, lTheta, Im)

        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments
        Image.fromarray(Im * 255.).show()
        Image.fromarray(H).show()
        Im_houghline=np.array(color)
        for k in range(nLines):
            for i in range(0, Im.shape[0]):
                j = int((lRho[k]-math.cos(lTheta[k])*i)/math.sin(lTheta[k])+0.5)
                if 0 <= j < Igs.shape[1]:
                    Im_houghline[i, j] = (255,0,0)
        for k in range(nLines):
            for j in range(0, Im.shape[1]):
                i = int((lRho[k] - math.sin(lTheta[k]) * j) / math.cos(lTheta[k]) + 0.5)
                if 0 <= i < Igs.shape[0]:
                    Im_houghline[i, j] = (255,0,0)
        Image.fromarray(Im_houghline).show()
        Im_houghlinesegments =np.array(color)
        for k in range(len(l)):
            for i in range(l[k]['start'][0], l[k]['end'][0]+1):
                j = int((lRho[k]-math.cos(lTheta[k])*i)/math.sin(lTheta[k])+0.5)
                if 0 <= j < Igs.shape[1]:
                    Im_houghlinesegments[i, j] = (255,0,0)
        for k in range(len(l)):
            for j in range(min(l[k]['start'][1], l[k]['end'][1]), max(l[k]['start'][1], l[k]['end'][1])+ 1):
                i = int((lRho[k]-math.sin(lTheta[k])*j)/math.cos(lTheta[k])+0.5)
                if 0 <= i < Igs.shape[0]:
                    Im_houghlinesegments[i, j] = (255,0,0)
        Image.fromarray(Im_houghlinesegments).show()
        filenum+=1
        """
        Image.fromarray(Im * 255).convert('RGB').save('Im' + str(filenum) + '.jpg')
        Image.fromarray(H).convert('RGB').save('H' + str(filenum) + '.jpg')
        Image.fromarray(Im_houghline).save('Im_line' + str(filenum) + '.jpg')
        Image.fromarray(Im_houghlinesegments).save('Im_seg' + str(filenum) + '.jpg')
        """


if __name__ == '__main__':
    main()