import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

# Image analysis algorithm to detect the nematic director

# Copyright (C) 2023 Francine Kolley
# The code was written for multi-channel images of insect flight muscle
# Pipeline in collaboration with Benoit Dehapiot
# Francine Kolley
# Physics of Life, Benjamin M. Friedrich group
# TU_dresden
# contact: francine.kolley@tu-dresden.de
# Latest code 07-2022

# STEPS:
# (1) run this script
# (2) run Correlation_function.m

# Clear variables and close all plots
np.random.seed(0)
plt.close('all')

# Parameters
GBlur_Sigma = 2  # Background subtraction
Mask_Thresh = 350  # Threshold to filter interesting structures for ROI
ROI_Size = 30  # Size of ROI in pixels
ROI_Thresh = 0.25  # Minimal value used; some stages increase up to 0.85
Tophat_Sigma = 28  # Tophat filtering
Steerable_Sigma = 2  # Size for Steerable filter
Pad_Angle = 6  # Window size to crop image
Pad_Corr = 3  # Pixel padding for correlation
ROI_Pad_Corr = (ROI_Size + (ROI_Size * (Pad_Corr - 1)) * 2)

# Image read in
input_figures_directory = 'insert_path/'
image_filename = 'insert_image_name.tif'

# Read in images (adjust as per your actual image reading method)
# actinin_image, actin_image, myosin_image, titin_image = read_in_images(...)
# Load RawPath (path to image file) and extract information
# Adjust accordingly to read and process the image file

# Get variables
nY, nX = Raw.shape[0], Raw.shape[1]
qLow, qHigh = np.quantile(Raw, [0.001, 0.999])

# Mask and background subtraction
choice = 1
while choice > 0:
    # Crop Raw image
    nGridY, nGridX = nY // ROI_Size, nX // ROI_Size
    Raw = Raw[:nGridY * ROI_Size, :nGridX * ROI_Size]
    nYCrop, nXCrop = Raw.shape[0], Raw.shape[1]

    # Subtract background
    Raw_BGSub = ndi.gaussian_filter(Raw, sigma=GBlur_Sigma)

    # Create Mask
    Raw_Mask = np.zeros_like(Raw_BGSub)
    Raw_Mask[Raw_BGSub < Mask_Thresh] = 0
    Raw_Mask[Raw_BGSub >= Mask_Thresh] = 1

    # Create ROIsMask
    Raw_ROIsMask = np.zeros((nGridY, nGridX))

    for i in range(nGridY):
        for j in range(nGridX):
            temp = np.mean(
                Raw_Mask[ROI_Size * i - (ROI_Size - 1):ROI_Size * i, ROI_Size * j - (ROI_Size - 1):ROI_Size * j])
            white_percentage[i, j] = np.mean(temp)  # how many pixels per ROI are white
            if np.mean(temp) > ROI_Thresh:
                if Pad_Angle <= i <= nGridY - (Pad_Angle - 1) and Pad_Angle <= j <= nGridX - (Pad_Angle - 1):
                    Raw_ROIsMask[i, j] = 1

    # Display
    plt.subplot(2, 1, 1)
    plt.imshow(Raw_Mask, cmap='gray')
    plt.title(f'Mask (thresh. = {Mask_Thresh})')

    plt.subplot(2, 1, 2)
    plt.imshow(Raw_ROIsMask, cmap='gray')
    plt.title(f'ROIsMask (ROIs size = {ROI_Size} pix.; ROIs thresh. = {ROI_Thresh};)')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()

    # Dialog box
    choice = input('What next? (Modify Parameters / Proceed): ')
    if choice.lower() == 'modify parameters':
        Mask_Thresh = int(input('Mask_Thresh: '))
        ROI_Thresh = float(input('ROI_Thresh: '))
    else:
        break

# Top hat transformation
Raw_Tophat = ndi.white_tophat(Raw_BGSub, structure=np.ones((Tophat_Sigma, Tophat_Sigma)))

# Measure local nematic order
choice = 1
while choice > 0:
    # Steerable Filter
    RawRSize = np.array(Image.resize(Raw, (nGridY, nGridX)))
    MaskRSize = np.array(Image.resize(Raw_Mask, (nGridY, nGridX)))

    res, _, _, rot = steerableDetector(RawRSize, 2, Steerable_Sigma, 180)
    for i in range(rot.shape[2]):
        temp = rot[:, :, i]
        temp[MaskRSize == 0] = np.nan
        rot[:, :, i] = temp

    # Make AngleMap
    AngleMap = np.zeros((nGridY, nGridX))

    for i in range(nGridY):
        for j in range(nGridX):
            if Raw_ROIsMask[i, j] == 1:
                Crop = rot[i - (Pad_Angle - 1):i + (Pad_Angle - 1), j - (Pad_Angle - 1):j + (Pad_Angle - 1), :]
                idxMax = np.nanmean(Crop, axis=(0, 1))
                M, I = np.nanmax(idxMax), np.nanargmax(idxMax)
                AngleMap[i, j] = I

    # Display
    plt.subplot(3, 1, 1)
    plt.imshow(Raw, vmin=qLow, vmax=qHigh)
    plt.title('Raw')

    plt.subplot(3, 1, 2)
    plt.imshow(res, vmin=np.min(res), vmax=np.max(res))
    plt.title(f'Steerable filter (sigma = {Steerable_Sigma} pix.)')

    Raw_new = np.ones_like(Raw)
    plt.subplot(3, 1, 3)
    plt.imshow(Raw_new, vmin=qLow, vmax=qHigh)
    for i in range(nGridY):
        for j in range(nGridX):
            if Raw_ROIsMask[i, j] == 1:
                Angle = AngleMap[i, j]
                k = 0
                xi = ROI_Size * j + ROI_Size * np.array([-1, 1]) * np.cos((90 - Angle) * -1 * np.pi / 180) + k * np.cos(
                    (Angle) * np.pi / 180)
                yi = ROI_Size * i + ROI_Size * np.array([-1, 1]) * np.sin((90 - Angle) * -1 * np.pi / 180) + k * np.sin(
                    (Angle) * np.pi / 180)
                plt.plot(xi, yi, 'c')
    plt.title('Nematic')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()

    # Dialog box
    choice = input('What next? (Modify Parameters / Proceed): ')
    if choice.lower() == 'modify parameters':
        Steerable_Sigma = int(input('Steerable_Sigma: '))
    else:
        break

# Save all parameters needed for the next script
Parameters = np.array([
    GBlur_Sigma,
    Mask_Thresh,
    ROI_Size,
    ROI_Thresh,
    Tophat_Sigma,
    Steerable_Sigma
])

Parameters = np.vstack((
    GBlur_Sigma,
    Mask_Thresh,
    ROI_Size,
    ROI_Thresh,
    Tophat_Sigma,
    Steerable_Sigma
)).T

Parameters = pd.DataFrame(Parameters, columns=['Value'], index=[
    'GBlur_Sigma',
    'Mask_Thresh',
    'ROI_Size',
    'ROI_Thresh',
    'Tophat_Sigma',
    'Steerable_Sigma'
])

# Save parameters to a file for use in the next script
Parameters.to_csv('parameters.csv')
