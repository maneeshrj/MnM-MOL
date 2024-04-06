#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 16:59:30 2021

@author: apramanik
"""


import pickle
import numpy as np
import yaml
import argparse
import os
#import h5py as h5
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

#%%
def preload(path, num_sub, num_sl, start_sub, start_sl, acc):
    print("Loading subjects...", end=" ")
    for i in range(start_sub, num_sub + start_sub):
        print(i, end=" ")
        filepath = path + '/example_fastmri_knee_sub_' + str(i) + '.pickle'
        with open(filepath, "rb") as f:
            temp_sub, temp_csm = pickle.load(f)            
        
        temp_sub = temp_sub[start_sl:start_sl + num_sl,:,:,:]
        temp_csm = temp_csm[start_sl:start_sl + num_sl,:,:,:]
        
        
        if i == start_sub:
            org_data = np.zeros((num_sub,) + temp_sub.shape, dtype=np.complex64)
            org_data[i-start_sub] = temp_sub
            csm_data = np.zeros((num_sub,) + temp_csm.shape, dtype=np.complex64)
            csm_data[i-start_sub] = temp_csm 
        
        else:
            org_data[i-start_sub] = temp_sub
            csm_data[i-start_sub] = temp_csm
    print("\nLoading masks...")   
            
    mask_filename = f'mask{acc:.1f}f_320by320.npy'
    mask = np.load(mask_filename).astype(np.complex64)  
    mask = np.expand_dims(mask, axis=0)
    mask = np.tile(mask, [org_data.shape[0],org_data.shape[1],1,1,1])
    
    org_data = torch.tensor(org_data)
    csm_data = torch.tensor(csm_data)
    mask = torch.tensor(mask)
    
    org_data = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(org_data, dim=[-1,-2]), dim=[-1,-2], norm="ortho"), dim=[-1,-2])
    for i in range(num_sub):
        maxval = org_data[i].abs().max()
        org_data[i] = org_data[i]/maxval
    org_data = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(org_data, dim=[-1,-2]), dim=[-1,-2], norm="ortho"), dim=[-1,-2])
    
    return org_data, csm_data, mask


def preprocess(org_data, csm_data, mask_data):
    print("Preprocessing...")
    org_data = torch.reshape(org_data,(org_data.shape[0]*org_data.shape[1],org_data.shape[2],org_data.shape[3],org_data.shape[4]))
    mask_data = torch.reshape(mask_data,(mask_data.shape[0]*mask_data.shape[1],mask_data.shape[2],mask_data.shape[3],mask_data.shape[4]))
    csm_data = torch.reshape(csm_data,(csm_data.shape[0]*csm_data.shape[1],csm_data.shape[2],csm_data.shape[3],csm_data.shape[4]))    
    
    us_data = org_data*mask_data
    org_data = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(org_data, dim=[-1,-2]), dim=[-1,-2], norm="ortho"), dim=[-1,-2])
    org_data = torch.sum(org_data*csm_data.conj(),dim=1,keepdim=True)
    print(f"Data builder completed\n")

    return org_data, us_data, csm_data, mask_data


def preprocess_hist(org_data, csm_data, mask_data, coil_combined=False, noise_sigma=0.):
    
    print("Preprocessing...")
    org_data = torch.reshape(org_data,(org_data.shape[0]*org_data.shape[1],org_data.shape[2],org_data.shape[3],org_data.shape[4]))
    mask_data = torch.reshape(mask_data,(mask_data.shape[0]*mask_data.shape[1],mask_data.shape[2],mask_data.shape[3],mask_data.shape[4]))
    csm_data = torch.reshape(csm_data,(csm_data.shape[0]*csm_data.shape[1],csm_data.shape[2],csm_data.shape[3],csm_data.shape[4]))    
    
    us_data = org_data*mask_data
    org_data = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(org_data, dim=[-1,-2]), dim=[-1,-2], norm="ortho"), dim=[-1,-2])
    org_data = torch.sum(org_data*csm_data.conj(),dim=1,keepdim=True)
    print(f"Data builder completed\n")

    return org_data, us_data, csm_data, mask_data


#%%dataset preparation

if __name__ == "__main__":
    
    
    path = '/Shared/lss_jcb/aniket/fastmri_mol/fastmri_50subjects_cropped'
    num_sub = 1
    num_sl = 2
    start_sub = 1
    start_sl = 20
    acc = 4.0
    org_data, csm_data, mask_data = preload(path, num_sub, num_sl, start_sub, start_sl, acc)
    org_data, us_data, csm_data, mask_data = preprocess(org_data, csm_data, mask_data)
    
    dispind = 0
    
    for i in range(0,1):
        print(org_data[i:i+1].numpy().max())
        print(org_data[i:i+1].numpy().min())
        print(us_data[i:i+1].numpy().max())
        print(us_data[i:i+1].numpy().min())
        print('shape of fsim is', org_data[i:i+1].shape)
        print('shape of usk is', us_data[i:i+1].shape)
        print('shape of csm is', csm_data[i:i+1].shape)
        print('shape of mask is', mask_data[i:i+1].shape)
        fsim = org_data[i:i+1].numpy()
        usk = us_data[i:i+1].numpy()
        csm = csm_data[i:i+1].numpy()
        mask = mask_data[i:i+1].numpy()
        usim = np.zeros_like(fsim)
        img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(usk[0], axes=[-1,-2]), axes=(-1,-2), norm="ortho"), axes=[-1,-2])
        usim[0,0] = np.sum(img*np.conj(csm[0]),axis=0).astype(np.complex64)
        fig, axes = plt.subplots(1,3)
        pos = axes[0].imshow(np.abs(fsim[dispind,0,:,:]))
        pos = axes[1].imshow(np.abs(usim[dispind,0,:,:]))
        pos = axes[2].imshow(np.abs(mask[dispind,0,:,:]))
        plt.show()
        break
    
    # with open("poisson_mask_2d_acc4_320x320.npy", "rb") as f:
    with open("mask4.0f_320by320.npy", "rb") as f:
        msk = np.load(f)
        
    print(msk.shape)
    fig, ax = plt.subplots(1,2)
    
    ax[0].imshow(np.abs(fsim[0,0]), cmap='gray')
    ax[0].axis('off')
    ax[1].imshow(np.abs(msk[0,0]), cmap='gray')
    ax[1].axis('off')