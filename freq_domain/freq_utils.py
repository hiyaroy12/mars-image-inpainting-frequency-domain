import os
import os.path
import argparse
import numpy as np
import cv2
import glob
import random
import matplotlib.pyplot as plt
import torch

nx, ny = (256, 256)
x = np.linspace(-1,1,nx)
y = np.linspace(-1,1, ny)
xv, yv = np.meshgrid(x, y)



def plot_all_funcs(x_im, dx=32, half=False):
    # z = ((z+1)/2 * mask - 0.5)/0.5
    color=False
    if len(x_im.shape)>=3:
        color=True
    
    mask = np.ones_like(x_im)
    w, h = mask.shape
    x0, y0 = 0, 0
    if half:
        mask[:, 0:w//2]=0
    else: # for center
        mask[h//2- x0 - dx: h//2 - x0 + dx, w//2 - y0 - dx: w//2 - y0 + dx]=0
#     z_mask = z[:,:,0] * mask#
    if color:
        x_im = x_im[:,:,0]
    else:
        pass
        
    im_mask = (((x_im + 1)/2 * mask) - 0.5) / 0.5
        
    dct1_cv, mag_cv, phase_cv, idx_dct_cv = fft_compute(x_im, center=True) 
    min_v1 = mag_cv.min()
    max_v1 = mag_cv.max()
    
    dct1_cv_m, mag_cv_m, phase_cv_m, idx_dct_cv_m = fft_compute(im_mask, center=True) 
    min_v2 = mag_cv.min()
    max_v2 = mag_cv.max()
    
    min_v = min(min_v1, min_v2)
    max_v = max(max_v1, max_v2)
    
    mag_cv = (mag_cv - min_v)/(max_v - min_v)
    phase_cv = phase_cv/(2 * np.pi)
    
    mag_cv_m = (mag_cv_m - min_v)/(max_v - min_v)
    phase_cv_m = phase_cv_m/(2 * np.pi)
    
    plt.imshow(np.concatenate((mag_cv, mag_cv_m), axis=1), cmap='ocean')
    plt.colorbar()
    plt.title('Power spectrum')
    plt.show()

    plt.imshow(np.concatenate((phase_cv, phase_cv_m), axis=1), cmap='ocean')
    plt.colorbar()
    plt.title('Phase spectrum')
    plt.show()


    img_back = ifft_compute(mag_cv, phase_cv, idx_dct_cv, center=True, val_lims=[min_v, max_v])
    img_back_masked = ifft_compute(mag_cv_m, phase_cv_m, idx_dct_cv_m, center=True, val_lims=[min_v, max_v])

    plt.imshow(np.concatenate(((x_im+1)/2, (im_mask+1)/2), axis=1), cmap='gray')
    plt.colorbar()
    plt.title('Original image')
    plt.show()

    plt.imshow(np.concatenate(((img_back+1)/2, (img_back_masked+1)/2), axis=1), cmap='gray')
    plt.colorbar()
    plt.title('Recon image')
    plt.show()



#######################################################################
###############         grayscale utils here            ###############
#######################################################################

def fft_compute(img, center=False):
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    if center:
        dft = np.fft.fftshift(dft)
    mag = cv2.magnitude(dft[:,:,0],dft[:,:,1])
    idx = (mag==0)
    mag[idx] = 1.
    magnitude_spectrum = np.log(mag)
    phase_spectrum = cv2.phase(dft[:,:,0],dft[:,:,1])
    return dft, magnitude_spectrum, phase_spectrum, idx

def ifft_compute(magnitude, phase, idx, center=False, val_lims=None):
    min_v, max_v = val_lims
    
    recon_mag = magnitude.copy()
    recon_mag = (max_v - min_v) * recon_mag + min_v
    recon_mag[idx] = -np.inf
    
    recon_phase = phase * (2 * np.pi)
    
    real_part = np.cos(recon_phase) * np.exp(recon_mag)
    imag_part = np.sin(recon_phase) * np.exp(recon_mag)
   
    true_fft = np.concatenate((real_part[:, :, None], imag_part[:, :, None]), axis=-1)
    if center:
        true_fft = np.fft.fftshift(true_fft)
    img_back = cv2.idft(true_fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    return img_back

    
# fx=10
# fy=12
# # z = (np.cos(2*np.pi*(fx*xv + fy*yv)) + 1)/2
# z = np.cos(2*np.pi*(fx*xv + fy*yv))
def get_gray_fft_images(x_im, dx=16, half=False, return_mask=False):
    x_im = np.squeeze(x_im, axis=1)
    assert len(x_im[0].shape)<=2, "Image should be grayscale"
    h, w = x_im[0].shape
    mask = np.ones((h,w))
    
    if half:
        mask[:, 0:w//2]=0
    else:
        mask[h//2-dx: h//2+dx, w//2-dx: w//2+dx]=0
    
    N = len(x_im)
    x_masked = np.zeros((N, 1, h, w))
    x_fft = np.zeros((N, 2, h, w))
    x_masked_fft = np.zeros((N, 2, h, w))
    
    lims_list = []
    idx_list_ = []
    idx_list_m = []

    for i in range(N):
        x_ = x_im[i]
        x_m = (((x_ + 1)/2 * mask) - 0.5) / 0.5
        x_masked[i,0] = x_m
        
        _, mag_x_, phase_x_, idx_x_ = fft_compute(x_, center=True) 
        min_v1 = mag_x_.min()
        max_v1 = mag_x_.max()
        
        _, mag_x_m, phase_x_m, idx_x_m = fft_compute(x_m, center=True) 
        min_v2 = mag_x_m.min()
        max_v2 = mag_x_m.max()
        
        min_v = min(min_v1, min_v2)
        max_v = max(max_v1, max_v2)
        
        mag_x_ = (mag_x_ - min_v)/(max_v - min_v)
        phase_x_ = phase_x_/(2 * np.pi)

        mag_x_m = (mag_x_m - min_v)/(max_v - min_v)
        phase_x_m = phase_x_m/(2 * np.pi)
        
        x_fft[i,0] = mag_x_
        x_fft[i,1] = phase_x_
        
        x_masked_fft[i,0] = mag_x_m
        x_masked_fft[i,1] = phase_x_m
        
        lims_list.append([min_v, max_v])
        idx_list_.append(idx_x_)
        idx_list_m.append(idx_x_m)
    
    if return_mask:
        return x_masked, x_fft, x_masked_fft, lims_list, idx_list_, idx_list_m, mask
    else:
        return x_masked, x_fft, x_masked_fft, lims_list, idx_list_, idx_list_m
    
############################### for hirise dataset (grayscale irregular mask)##################
    
from PIL import Image
mars_mask = np.load('./data/mask_train_test.npz')
train_masks = list(mars_mask['train'])
test_masks = list(mars_mask['test'])


def get_gray_fft_images_irregular(x_im, return_mask=False): # b x 3 x h x w 
    # x_fft has first three channel as magnitude next 3 channels as phase
    N = len(x_im)
    x_im = np.squeeze(x_im, axis=1)
    assert len(x_im[0].shape)<=2, "Image should be grayscale"
    h, w = x_im[0].shape
#     mask = np.ones((h,w))
    all_masks = np.ones((N, h, w))
    

    for k, mask_file in enumerate(random.sample(train_masks, N)):
        mask = cv2.resize(np.array(mask_file),(w,h))
        mask = mask.copy()/255.
#         if len(mask.shape)<=2:
#             mask = np.concatenate([mask[...,None]]*3, axis=-1)
        all_masks[k] = 1.0 - mask
#         all_masks[k] = mask
    
    
    x_masked = np.zeros((N, 1, h, w))
    x_fft = np.zeros((N, 2, h, w))
    x_masked_fft = np.zeros((N, 2, h, w))
    mask_fft = np.zeros((N, 2, h, w))
    
    lims_list = []
    idx_list_ = []
    idx_list_m = []

    for i in range(N):
        x_ = x_im[i]
#         import ipdb; ipdb.set_trace()
#         mask = np.transpose(all_masks[i], (2,0,1))
        mask = all_masks[i]
        x_m = (((x_ + 1)/2 * mask) - 0.5) / 0.5
        x_masked[i] = x_m ###[i,0]?
        
        _, mag_x_, phase_x_, idx_x_ = fft_compute(x_, center=True) 
        min_v1 = mag_x_.min()
        max_v1 = mag_x_.max()
        
        _, mag_x_m, phase_x_m, idx_x_m = fft_compute(x_m, center=True) 
        min_v2 = mag_x_m.min()
        max_v2 = mag_x_m.max()
        
        min_v = min(min_v1, min_v2)
        max_v = max(max_v1, max_v2)
        
        mag_x_ = (mag_x_ - min_v)/(max_v - min_v)
        phase_x_ = phase_x_/(2 * np.pi)

        mag_x_m = (mag_x_m - min_v)/(max_v - min_v)
        phase_x_m = phase_x_m/(2 * np.pi)
        
        x_fft[i,0] = mag_x_
        x_fft[i,1] = phase_x_
        
        x_masked_fft[i,0] = mag_x_m
        x_masked_fft[i,1] = phase_x_m
        
        
        ##############################################################################
        _, mag_fft_mask_, phase_fft_mask_x_, _ = fft_compute(mask, center=True) 
        min_v3 = mag_fft_mask_.min()
        max_v3 = mag_fft_mask_.max()
        mag_fft_mask_ = (mag_fft_mask_ - min_v3)/(max_v3 - min_v3)
        phase_fft_mask_x_ = phase_fft_mask_x_/(2 * np.pi)
        
        mask_fft[i,0] = mag_fft_mask_
        mask_fft[i,1] = phase_fft_mask_x_
        ###############################################################################
        
        lims_list.append([min_v, max_v])
        idx_list_.append(idx_x_)
        idx_list_m.append(idx_x_m)

    if return_mask:
        return x_masked, x_fft, x_masked_fft, lims_list, idx_list_, idx_list_m, all_masks, mask_fft

################################################################################################################
    

def get_gray_images_back(x_fft, lims_list, idx_list):
    h, w = x_fft[0, 0].shape
    
    N = len(x_fft)
    x_back = np.zeros((N, 1, h, w))

    for i in range(N):
        mag_ = x_fft[i, 0]
        phase_ = x_fft[i, 1]
        idx_ = idx_list[i]
        lims_ = lims_list[i]
        
        img_back = ifft_compute(mag_, phase_, idx_, center=True, val_lims=lims_)
        x_back[i,0]=img_back
        
    return x_back
        



    
############################################################################################################
####################################      Irregular mask                ####################################
############################################################################################################
from PIL import Image
def read_file(filename):
    return [line.rstrip('\n') for line in open(filename)]
############################################################################################################
# train_masks = read_file('/home3/hiya/workspace/inpainting_fft/PConv-Keras/data/masks/irregular_mask/our_train_mask_flist.txt')
# test_masks = read_file('/home3/hiya/workspace/inpainting_fft/PConv-Keras/data/masks/test_mask/test_mask_flist.txt')
############################################################################################################

mars_mask = np.load('./data/mask_train_test.npz')

train_masks = list(mars_mask['train'])
test_masks = list(mars_mask['test'])
# np.savez('irregular_masks.npz', train_masks=train_masks, test_masks=test_masks)

def get_color_fft_images_irregular(x_im, return_mask=False): # b x 3 x h x w 
#     x_im = np.squeeze(x_im, axis=1)
    # x_fft has first three channel as magnitude next 3 channels as phase
    N = len(x_im)
    assert len(x_im[0].shape)==3, "Image should be color" #pytorch 3 x h x w
    h, w = x_im[0].shape[1:]
    all_masks = np.ones((N, h, w, 3))

    for k, mask_file in enumerate(random.sample(train_masks, N)):
        mask = cv2.resize(np.array(Image.open(mask_file)),(w,h))
        mask = mask.copy()/255.
        if len(mask.shape)<=2:
            mask = np.concatenate([mask[...,None]]*3, axis=-1)
        all_masks[k] = 1.0 - mask
    
    x_masked = np.zeros((N, 3, h, w))
    x_fft = np.zeros((N, 6, h, w))
    x_masked_fft = np.zeros((N, 6, h, w))
    mask_fft = np.zeros((N, 6, h, w))
    
    lims_list = []
    idx_list_ = []
    idx_list_m = []

    for i in range(N):
        x_ = x_im[i]
        mask = np.transpose(all_masks[i], (2,0,1))
        # import ipdb; ipdb.set_trace()
        x_m = (((x_ + 1)/2 * mask) - 0.5) / 0.5
        x_masked[i] = x_m
        
        _, mag_x_, phase_x_, idx_x_ = fft_compute_color(x_, center=True) 
        min_v1 = mag_x_.min()
        max_v1 = mag_x_.max()
        
        _, mag_x_m, phase_x_m, idx_x_m = fft_compute_color(x_m, center=True) 
        min_v2 = mag_x_m.min()
        max_v2 = mag_x_m.max()
        
        min_v = min(min_v1, min_v2)
        max_v = max(max_v1, max_v2)
        
        mag_x_ = (mag_x_ - min_v)/(max_v - min_v)
        phase_x_ = phase_x_/(2 * np.pi)

        mag_x_m = (mag_x_m - min_v)/(max_v - min_v)
        phase_x_m = phase_x_m/(2 * np.pi)
        
        x_fft[i,:3] = mag_x_
        x_fft[i,3:] = phase_x_
        
        x_masked_fft[i, :3] = mag_x_m
        x_masked_fft[i, 3:] = phase_x_m
        
        
        ##############################################################################
        _, mag_fft_mask_, phase_fft_mask_x_, _ = fft_compute_color(mask, center=True) 
        min_v3 = mag_fft_mask_.min()
        max_v3 = mag_fft_mask_.max()
        mag_fft_mask_ = (mag_fft_mask_ - min_v3)/(max_v3 - min_v3)
        phase_fft_mask_x_ = phase_fft_mask_x_/(2 * np.pi)
        
        mask_fft[i,:3] = mag_fft_mask_
        mask_fft[i,3:] = phase_fft_mask_x_
        ###############################################################################
        
        lims_list.append([min_v, max_v])
        idx_list_.append(idx_x_)
        idx_list_m.append(idx_x_m)

    if return_mask:
        return x_masked, x_fft, x_masked_fft, lims_list, idx_list_, idx_list_m, all_masks, mask_fft
    else:
        return x_masked, x_fft, x_masked_fft, lims_list, idx_list_, idx_list_m
