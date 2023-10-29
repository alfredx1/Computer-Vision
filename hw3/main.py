import math
import numpy as np
from PIL import Image

def compute_h(p1, p2):
    # First Make matrix A from given coordinates A : 2N X 9 matrix
    A = np.zeros((2*p1.shape[0],9))
    for i in range (0, p1.shape[0]):
        A[2 * i][0] = p2[i][0]
        A[2 * i][1] = p2[i][1]
        A[2 * i][2] = 1
        A[2 * i][6] = -p2[i][0] * p1[i][0]
        A[2 * i][7] = -p2[i][1] * p1[i][0]
        A[2 * i][8] = -p1[i][0]
        A[2 * i + 1][3] = p2[i][0]
        A[2 * i + 1][4] = p2[i][1]
        A[2 * i + 1][5] = 1
        A[2 * i + 1][6] = -p2[i][0] * p1[i][1]
        A[2 * i + 1][7] = -p2[i][1] * p1[i][1]
        A[2 * i + 1][8] = -p1[i][1]
    U,s,V=np.linalg.svd(A)

    a = V[-1]
    H=np.zeros((3,3))
    for i in range(0,3):
        for j in range(0,3):
            H[i][j]=a[j+3*i]
    return H

def compute_h_norm(p1, p2):
    # TODO ...
    m=1600.0
    p1=p1/m
    p2=p2/m
    A=compute_h(p1, p2)
    H=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            H[i][j]=A[i][j]
    for i in range(0,2):
        H[0][i]=A[0][i]/m
        H[1][i]=A[1][i]/m
    H[2][0]=A[2][0]/m**2
    H[2][1]=A[2][1]/m**2
    H[2][2]=A[2][2]/m
    return H

def warp_image(igs_in, igs_ref, H):
    # TODO ...
    N = igs_in.shape[0]
    M = igs_in.shape[1]
    H_1 = np.linalg.inv(H)
    igs_warp=np.zeros((N,M,3))
    igs_merge=np.zeros((3400,4000,3))
    for x in range(0, N):
       for y in range(0, M):
           z = (H_1[0][0] * x + H_1[0][1] * y + H_1[0][2]) / (H_1[2][0] * x + H_1[2][1] * y + H_1[2][2])
           w = (H_1[1][0] * x + H_1[1][1] * y + H_1[1][2]) / (H_1[2][0] * x + H_1[2][1] * y + H_1[2][2])
           if 0<z<N-1 and 0<w<M-1:
               z1=z-int(z)
               w1=w-int(w)
               igs_warp[x][y][:] += ((1-z1)*(1-w1)*igs_in[int(z)][int(w)][:]+z1*w1*igs_in[int(z)+1][int(w)+1][:]+(1-z1)*(w1)*igs_in[int(z)][int(w)+1][:]+(z1)*(1-w1)*igs_in[int(z)+1][int(w)][:])

    for i in range(0, 3400):
        for j in range(0, 4000):
            x=i-1700
            y=j-2000
            z = (H_1[0][0] * x + H_1[0][1] * y + H_1[0][2]) / (H_1[2][0] * x + H_1[2][1] * y + H_1[2][2])
            w = (H_1[1][0] * x + H_1[1][1] * y + H_1[1][2]) / (H_1[2][0] * x + H_1[2][1] * y + H_1[2][2])
            if 0 < int(z) < N - 1 and 0 < int(w) < M - 1:
                z1 = z - int(z)
                w1 = w - int(w)
                igs_merge[i][j][:] += (
                            (1 - z1) * (1 - w1) * igs_in[int(z)][int(w)][:] + z1 * w1 * igs_in[int(z) + 1][int(w) + 1][
                                                                                        :] + (1 - z1) * (w1) *
                            igs_in[int(z)][int(w) + 1][:] + (z1) * (1 - w1) * igs_in[int(z) + 1][int(w)][:])
    for i in range(1700, 1700+N):
        for j in range(2000, 2000+M):
            igs_merge[i][j][:]=igs_ref[i-1700][j-2000][:]
    return igs_warp, igs_merge

def rectify(igs, p1, p2):
    # TODO ...

    H=compute_h_norm(p2, p1)
    A=igs.shape[0]
    B=igs.shape[1]
    N=1000
    M=500
    igs_rec=np.zeros((N,M,3))
    H_1 = np.linalg.inv(H)
    for x in range(0, N):
        for y in range(0, M):
            z = (H_1[0][0] * x + H_1[0][1] * y + H_1[0][2]) / (H_1[2][0] * x + H_1[2][1] * y + H_1[2][2])
            w = (H_1[1][0] * x + H_1[1][1] * y + H_1[1][2]) / (H_1[2][0] * x + H_1[2][1] * y + H_1[2][2])
            if 0 < z < A - 1 and 0 < w < B - 1:
                z1 = z - int(z)
                w1 = w - int(w)
                igs_rec[x][y][:] += (
                            (1 - z1) * (1 - w1) * igs[int(z)][int(w)][:] + z1 * w1 * igs[int(z) + 1][int(w) + 1][
                                                                                        :] + (1 - z1) * (w1) *
                            igs[int(z)][int(w) + 1][:] + (z1) * (1 - w1) * igs[int(z) + 1][int(w)][:])

    return igs_rec

def set_cor_mosaic():
    # TODO ...노가다....

    p_in = np.zeros((14,2))
    p_ref = np.zeros((14, 2))
    p_in[0][0]=190
    p_in[0][1]=1187
    p_in[1][0]=252
    p_in[1][1]=1286
    p_in[2][0]=413
    p_in[2][1]=1106
    p_in[3][0]=419
    p_in[3][1]=1285
    p_in[4][0] =407
    p_in[4][1] =1440
    p_in[5][0] =507
    p_in[5][1] =1283
    p_in[6][0] =505
    p_in[6][1] =1443
    p_in[7][0] =545
    p_in[7][1] =1243
    p_in[8][0] =545
    p_in[8][1] =1292
    p_in[9][0] =857
    p_in[9][1] =889
    p_in[10][0] =857
    p_in[10][1] =944
    p_in[11][0] =915
    p_in[11][1] =1332
    p_in[12][0] =956
    p_in[12][1] =1256
    p_in[13][0] =959
    p_in[13][1] =1284
    p_ref[0][0] =194
    p_ref[0][1] =450
    p_ref[1][0] =265
    p_ref[1][1] =538
    p_ref[2][0] =409
    p_ref[2][1] =362
    p_ref[3][0] =426
    p_ref[3][1] =538
    p_ref[4][0] =423
    p_ref[4][1] =674
    p_ref[5][0] =511
    p_ref[5][1] =538
    p_ref[6][0] =512
    p_ref[6][1] =676
    p_ref[7][0] =545
    p_ref[7][1] =497
    p_ref[8][0] =544
    p_ref[8][1] =546
    p_ref[9][0] =888
    p_ref[9][1] =117
    p_ref[10][0] =879
    p_ref[10][1] =180
    p_ref[11][0] =897
    p_ref[11][1] =581
    p_ref[12][0] =948
    p_ref[12][1] =509
    p_ref[13][0] =947
    p_ref[13][1] =538

    return p_in, p_ref

def set_cor_rec():
    # TODO ...
    c_in=np.zeros((4,2))
    c_ref=np.zeros((4,2))
    c_in[0][0]=167
    c_in[0][1]=1061
    c_in[1][0]=127
    c_in[1][1]=1401
    c_in[2][0]=854
    c_in[2][1]=1061
    c_in[3][0]=897
    c_in[3][1]=1401
    c_ref[0][0]=100
    c_ref[0][1]=10
    c_ref[1][0]=100
    c_ref[1][1]=410
    c_ref[2][0]=900
    c_ref[2][1]=10
    c_ref[3][0]=900
    c_ref[3][1]=410
    return c_in, c_ref

def main():
    ##############
    # step 1: mosaicing
    ##############

    # read images
    img_in = Image.open('data/porto1.png').convert('RGB')
    img_ref = Image.open('data/porto2.png').convert('RGB')

    # shape of igs_in, igs_ref: [y, x, 3]
    igs_in = np.array(img_in)
    igs_ref = np.array(img_ref)

    # lists of the corresponding points (x,y)
    # shape of p_in, p_ref: [N, 2]
    p_in, p_ref = set_cor_mosaic()

    # p_ref = H * p_in
    H = compute_h_norm(p_ref, p_in)
    igs_warp, igs_merge = warp_image(igs_in, igs_ref, H)

    # plot images
    img_warp = Image.fromarray(igs_warp.astype(np.uint8))
    img_merge = Image.fromarray(igs_merge.astype(np.uint8))

    # save images
    img_warp.save('porto1_warped.png')
    img_merge.save('porto_merged.png')

    ##############
    # step 2: rectification
    ##############

    img_rec = Image.open('data/iphone.png').convert('RGB')
    igs_rec = np.array(img_rec)

    c_in, c_ref = set_cor_rec()

    igs_rec = rectify(igs_rec, c_in, c_ref)

    img_rec = Image.fromarray(igs_rec.astype(np.uint8))
    img_rec.save('iphone_rectified.png')

if __name__ == '__main__':
    main()
