import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold

def lucas_kanade_affine(img1, img2, p, Gx, Gy):
    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and 
    # RectBivariateSpline. Never use OpenCV.
    t_x, t_y = img1.shape
    i_x, i_y = img2.shape
    comptemp = np.ones(img1.shape)
    Aff = np.array([[p[0] + 1, p[2], p[4]], [p[1], p[3] + 1, p[5]], [0, 0, 1]])
    row, column = np.meshgrid(range(t_y), range(t_x))
    temp = np.stack([column, row, comptemp], axis = -1)
    respond = temp @ Aff.T
    warped_img = np.zeros(img1.shape)
    warped_Gx = np.zeros(img1.shape)
    warped_Gy = np.zeros(img1.shape)
    intp_img = RectBivariateSpline(range(i_x), range(i_y), img2)
    intp_gx = RectBivariateSpline(range(i_x), range(i_y), Gx)
    intp_gy = RectBivariateSpline(range(i_x), range(i_y), Gy)
    check = np.zeros(img1.shape)
    error = np.zeros(img1.shape)
    for i in range(t_x):
        for j in range(t_y):
            if (t_y >respond[i, j, 0] >= 0) and (0 <= respond[i, j, 1] < t_x):
                warped_img[i,j]=intp_img(respond[i,j,0],respond[i,j,1])
                warped_Gx[i,j]=intp_gx(respond[i,j,0],respond[i,j,1])
                warped_Gy[i,j]=intp_gy(respond[i,j,0],respond[i,j,1])
                check[i,j]=1;
                error[i,j]=img1[i,j]-warped_img[i,j]
    check_expanded = np.expand_dims(np.expand_dims(check, axis=-1), axis=-1)
    warped_G = np.where(check_expanded, np.expand_dims(np.stack([warped_Gx,warped_Gy], axis=-1), axis=-2), 0)
    Zero=np.zeros(img1.shape)
    jacob1=np.stack([column, Zero, row, Zero, comptemp, Zero], axis=-1)
    jacob2=np.stack([Zero, column, Zero, row, Zero, comptemp], axis=-1)
    jacob=np.where(check_expanded,np.stack([jacob1, jacob2], axis=-2),0)
    A=warped_G @ jacob
    hessian = np.sum(np.sum(np.where(check_expanded, np.transpose(A, axes=(0, 1, 3, 2)) @ A, 0), axis=0), axis=0)
    error_expanded = np.expand_dims(np.expand_dims(error, axis=-1), axis=-1)
    B= np.sum(np.sum(np.where(check_expanded, np.transpose(A, axes=(0, 1, 3, 2)) @ error_expanded, 0), axis=0), axis=0)
    C= np.linalg.pinv(hessian) @ B
    dp = np.array([C[0, 0], C[1, 0], C[2, 0], C[3, 0], C[4, 0], C[5, 0]])
    ### END CODE HERE ###
    return dp

def subtract_dominant_motion(img1, img2):
    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize = 5) # do not modify this
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize = 5) # do not modify this

    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and 
    # RectBivariateSpline. Never use OpenCV.
    ti=0.001 #initial gradient
    th_hi = 0.5 * 256 # you can modify this
    th_lo = 0.3 * 256 # you can modify this
    i1= img1/255
    i2= img2/255
    Gx= Gx/np.max(Gx)
    Gy= Gy/np.max(Gy)
    p=np.zeros(6)*ti
    dp = lucas_kanade_affine(i1,i2,p,Gx,Gy)
    p=p+dp
    t_x, t_y = img1.shape
    i_x, i_y = img2.shape
    comptemp = np.ones(img1.shape)
    Aff = np.array([[p[0] + 1, p[2], p[4]], [p[1], p[3] + 1, p[5]], [0, 0, 1]])
    row, column = np.meshgrid(range(t_y), range(t_x))
    temp = np.stack([column, row, comptemp], axis=-1)
    inv_aff = temp @ np.linalg.inv(Aff).T
    intp_img = RectBivariateSpline(range(t_x), range(t_y), i1)
    warped_imgdiffer = np.zeros(img2.shape)
    for i in range(i_x):
        for j in range(i_y):
            if (t_y > inv_aff[i, j, 0] >= 0) * (0 <= inv_aff[i, j, 1] < t_x) == 1:
                warped_imgdiffer[i, j] = intp_img(inv_aff[i, j, 0], inv_aff[i, j, 1])-i2[i,j]
    movie = np.abs(warped_imgdiffer)
    movie = movie/movie.max()*255
    moving_image = movie.astype('uint8')
    
    ### END CODE HERE ###

    # Declare thresholds
    th_hi = 0.48 * 256  # you can modify this
    th_lo = 0.32 * 256  # you can modify this
    hyst = apply_hysteresis_threshold(moving_image, th_lo, th_hi)
    return hyst

if __name__ == "__main__":
    data_dir = 'data'
    video_path = 'motion.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 150/20, (636, 318))
    tmp_path = os.path.join(data_dir, "organized-{}.jpg".format(0))
    T = cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)
    for i in range(0, 50):
        img_path = os.path.join(data_dir, "organized-{}.jpg".format(i))
        I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        clone = I.copy()
        moving_img = subtract_dominant_motion(T, I)
        clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
        clone[moving_img, 2] = 522
        out.write(clone)
        T = I
    out.release()
    