
import numpy as np
from scipy.ndimage import gaussian_filter
from spectrum.correlation import xcorr

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from IPython import embed

import os

from tifffile import TiffFile

import steerable

from skimage.morphology import disk, white_tophat
from skimage.transform import resize



def choose_image(tif, n_channels=4):
    """
    Displays an interactive window to choose a frame from a sequence of images.

    Parameters:
    tif: tif file.

    Returns:
    None
    """
    # Get the number of slices
    n = len(tif.pages)
    num_slices = n/n_channels
    if n%4 != 0:
        raise ValueError('The number of frames is not a multiple of 4.')
    
    
    
    # Create the interactive window to display the frames of the image
    fig, ax = plt.subplots()
    if num_slices > 1:
        slice_slider_ax = plt.axes([0.2, 0.06, 0.65, 0.03])  
        slice_slider = Slider(slice_slider_ax, 'slice', 0, num_slices- 1, valinit=0, valstep=1)
    
    channel_slider_ax = plt.axes([0.2, 0.02, 0.65, 0.03])  
    channel_slider = Slider(channel_slider_ax, 'channel', 0, n_channels-1, valinit=0, valstep=1)
    # Display the initial frame
    image_0 = tif.pages[0].asarray()
    image_display = ax.imshow(image_0)
    def update_image_selection(val):
        channel_index = int(channel_slider.val)
        if num_slices > 1:
            slice_index = int(slice_slider.val)
        else:
            slice_index = 0
        image = np.asarray(tif.pages[4 * slice_index + channel_index].asarray())
        # reset the color limits
        image_display.set_clim(vmin=image.min(), vmax=image.max())
        # update the displayed image
        image_display.set_data(image)
        fig.canvas.draw_idle()

    # Connect the sliders to the update function
    if num_slices > 1:
        slice_slider.on_changed(update_image_selection)
    channel_slider.on_changed(update_image_selection)
    plt.show()
    
def choose_parameter_mask(raw, mask_thresh, roi_thresh, roi_size, gblur_sigma, pad_angle):
    """
    Allows the user to interactively choose parameter values for mask generation.

    Parameters:
        raw : The raw image data.
        mask_thresh (float): The threshold value for generating the mask.
        roi_thresh (float): The threshold value for generating the region of interest (ROI) mask.
        roi_size (int): The size of the ROI.
        gblur_sigma (float): The standard deviation for Gaussian blurring.
        pad_angle (float): The angle for padding the image.

    Returns:
    - None
    """
    fig = plt.figure()
    
    mask_thresh_slider_ax = plt.axes([0.2, 0.02, 0.65, 0.03])
    mask_thresh_frame_slider = Slider(mask_thresh_slider_ax, 'mask threshhold', 0, 1000, valinit=mask_thresh, valstep=50)
    
    roi_thresh_slider_ax = plt.axes([0.2, 0.08, 0.65, 0.03])
    roi_thresh_frame_slider = Slider(roi_thresh_slider_ax, 'roi threshhold', 0, 0.85, valinit=roi_thresh, valstep=0.1)
    
    roi_size_slider_ax = plt.axes([0.2, 0.14, 0.65, 0.03])
    roi_size_frame_slider = Slider(roi_size_slider_ax, 'roi size', 5, 50, valinit=roi_size, valstep=5)
    
    gblur_sigma_slider_ax = plt.axes([0.2, 0.2, 0.65, 0.03])
    gblur_sigma_frame_slider = Slider(gblur_sigma_slider_ax, 'gblur sigma', 0.5, 4, valinit=gblur_sigma, valstep=0.5)
    
    _, mask, roi_mask, _ = mask_background(raw, mask_thresh, roi_thresh, roi_size, gblur_sigma, pad_angle)
    mask_display, roi_mask_display, ax1, ax2 = plot_mask_background(mask, mask_thresh, roi_mask, roi_size, roi_thresh, gblur_sigma, fig)
    
    def update_image(val, param):
        nonlocal mask_thresh, roi_thresh, roi_size, gblur_sigma
        if param == 'mask_thresh':
            mask_thresh = val
        elif param == 'roi_thresh':
            roi_thresh = val
        elif param == 'roi_size':
            roi_size = val
        elif param == 'gblur_sigma':
            gblur_sigma = val
        _, mask, roi_mask, _ = mask_background(raw, mask_thresh, roi_thresh, roi_size, gblur_sigma, pad_angle)
        mask_display.set_data(mask)
        roi_mask_display.set_data(roi_mask)
        ax1.set_title(fr'mask (thresh. = {mask_thresh}), $\sigma_{{gb}}$ = {gblur_sigma})')
        ax2.set_title(f'ROIMask (size = {roi_size}, thresh. = {roi_thresh})')
        fig.canvas.draw_idle()

    mask_thresh_frame_slider.on_changed(lambda val: update_image(val, 'mask_thresh'))
    roi_thresh_frame_slider.on_changed(lambda val: update_image(val, 'roi_thresh'))
    roi_size_frame_slider.on_changed(lambda val: update_image(val, 'roi_size'))
    gblur_sigma_frame_slider.on_changed(lambda val: update_image(val, 'gblur_sigma'))
    plt.show()
    

def mask_background(raw, mask_thresh, roi_thresh, roi_size, gblur_sigma, pad_angle):
    """
    Masks the background of an image and creates a binary mask for regions of interest (ROIs).

    Parameters:
        raw (ndarray): The raw image.
        mask_thresh (float): The threshold value for creating the binary mask.
        roi_thresh (float): The threshold value for determining ROIs.
        roi_size (int): The size of each ROI.
        gblur_sigma (float): The standard deviation for Gaussian blur.
        pad_angle (int): The padding angle for excluding ROIs near the image edges.

        Returns:
        raw_crop (ndarray): The cropped raw image.
        raw_mask (ndarray): The binary mask.
        raw_roi_mask (ndarray): The mask for ROIs.
        raw_bgsub (ndarray): The background-subtracted image.
    """
    
    nY, nX = raw.shape
    # Crop raw image
    nGridY = int(np.floor(nY / roi_size))
    nGridX = int(np.floor(nX / roi_size))
    raw_crop = raw[:nGridY * roi_size, :nGridX * roi_size]
    nYCrop, nXCrop = raw_crop.shape
    
    # Subtract background
    raw_bgsub = gaussian_filter(raw_crop, sigma=gblur_sigma)  # Gaussian blur

    # Create Mask
    # Binary mask; I<Threshold --> Black, I>Threshold --> White
    raw_mask = raw_bgsub.copy()
    raw_mask[raw_mask < mask_thresh] = 0
    raw_mask[raw_mask >= mask_thresh] = 1
    #embed()

    # Create ROIsMask
    raw_roi_mask = np.zeros((nGridY, nGridX))
    white_percentage = np.zeros((nGridY, nGridX))
    for i in range(nGridY):
        for j in range(nGridX):
            temp = raw_mask[i*roi_size:(i+1)*roi_size, j*roi_size:(j+1)*roi_size].mean()
            white_percentage[i, j] = temp  # how many pixels per ROI are white
            if temp > roi_thresh:
                if i >= pad_angle - 1 and i <= nGridY - pad_angle and j >= pad_angle - 1 and j <= nGridX - pad_angle:
                    raw_roi_mask[i, j] = 1
    return raw_crop, raw_mask, raw_roi_mask, raw_bgsub

def plot_mask_background(raw_mask, mask_thresh, raw_roi_mask, roi_size, roi_thresh, gblur_sigma, fig=None):
    """
    Plots the mask and ROIMask images.

    Parameters:
        raw_mask (numpy.ndarray): The raw mask image.
        mask_thresh (float): The threshold value for the mask image.
        raw_roi_mask (numpy.ndarray): The raw ROIMask image.
        roi_size (int): The size of the ROI.
        roi_thresh (float): The threshold value for the ROIMask image.
        fig (matplotlib.figure.Figure, optional): The figure object to plot the images on. If not provided, a new figure will be created.

    Returns:
        tuple: A tuple containing the two image objects (pic1, pic2) if fig is provided, otherwise None.
    """
    return_fig = True
    if not fig:
        fig = plt.figure()
        return_fig = False
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title(fr'mask (thresh. = {mask_thresh}), $\sigma_{{gb}}$ = {gblur_sigma})')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title(f'ROIMask (size = {roi_size}, thresh. = {roi_thresh})')

    pic1 = ax1.imshow(raw_mask, cmap='gray')
    pic2 = ax2.imshow(raw_roi_mask, cmap='gray')
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5)
    
    if return_fig:
        return pic1, pic2, ax1, ax2
    else:
        plt.show()
        
def change_tophat_sigma(raw_bgsub, tophat_sigma, roi_size):
    """
    Allows the user to interactively change the value of sigma for the top-hat filter.

    Parameters:
        raw_bgsub (ndarray): The background-subtracted image.
        tophat_sigma (float): The sigma value for the top-hat filter.

    Returns:
        None
    """
    fig = plt.figure()
    # plot raw_bgsub in one subplot
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(raw_bgsub, cmap='gray')
    ax1.set_title('raw_bgsub')
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title(fr'raw_tophat ($\sigma_{{th}}$={tophat_sigma})')
    
    tophat_sigma_slider_ax = plt.axes([0.2, 0.02, 0.65, 0.03])
    tophat_sigma_slider = Slider(tophat_sigma_slider_ax, 'tophat sigma', 2, 2*roi_size, valinit=tophat_sigma, valstep=2)
    
    raw_tophat = white_tophat(raw_bgsub, disk(tophat_sigma))
    raw_tophat_display = ax2.imshow(raw_tophat, cmap='gray')
    
    def update_image(tophat_sigma):
        raw_tophat = white_tophat(raw_bgsub, disk(tophat_sigma))
        raw_tophat_display.set_data(raw_tophat)
        ax2.set_title(fr'raw_tophat ($\sigma_{{th}}$={tophat_sigma})')  # Set the title of the right subplot
        fig.canvas.draw_idle()
        
    tophat_sigma_slider.on_changed(lambda val: update_image(val))    
    plt.show()
    return raw_tophat, tophat_sigma

def steerable_filter(raw, raw_mask, raw_roi_mask, steerable_sigma, roi_size, pad_angle, nX, nY):
    """
    Apply steerable filter to the input image.

    Args:
        raw (ndarray): The input image.
        raw_mask (ndarray): The mask of the input image.
        raw_roi_mask (ndarray): The region of interest mask.
        steerable_sigma (float): The sigma value for the steerable filter.
        roi_size (int): The size of the region of interest.
        pad_angle (int): The padding angle for cropping.
        nX (int): The number of columns in the image.
        nY (int): The number of rows in the image.

    Returns:
        tuple: A tuple containing the following:
            raw_tophat (ndarray): The result of the top-hat filter.
            anglemap (ndarray): The angle map.
            res (ndarray): The response of the steerable filter.
            rot (ndarray): The angles corresponding to the detected features.
            xy_values (ndarray): The x and y values of the detected features.
    """
    
    nGridX = int(np.floor(nX / roi_size))
    nGridY = int(np.floor(nY / roi_size))
    
    raw_rsize = resize(raw, (nGridY, nGridX), order=0) # order=0 --> nearest neighbor interpolation

    raw_rsize = raw_rsize.astype(np.float64)
    mask_rsize = resize(raw_mask, (nGridY, nGridX), order=0)
    
    
    sd = steerable.Detector2D(raw_rsize, 2, steerable_sigma)
    res, _ = sd.filter()
    rot = np.transpose(sd.get_angle_response(180))
    for i in range(rot.shape[2]):
        temp = rot[:, :, i]
        temp[mask_rsize == 0] = np.nan
        rot[:, :, i] = temp
    # Make anglemap
    anglemap = np.zeros((nGridY, nGridX))
    for i in range(nGridY):
        for j in range(nGridX):
            if raw_roi_mask[i, j] == 1:
                x_idx = [i-(pad_angle-1),i+(pad_angle)]
                y_idx =  [j-(pad_angle-1),j+(pad_angle)]
                Crop = rot[x_idx[0]:x_idx[1], y_idx[0]:y_idx[1], :]
                idxMax = np.full(Crop.shape[2], np.nan)
                for k in range(Crop.shape[2]):
                    temp = Crop[:, :, k]
                    idxMax[k] = np.nanmean(temp)
                M, I = np.nanmax(idxMax), np.nanargmax(idxMax)
                anglemap[i,j] = I+1
    #np.savetxt("comparison/anglemap.csv", anglemap, delimiter=",")
    xy_values = np.zeros((nGridY*nGridX, 4))
    t=0
    for i in range(nGridY):
        for j in range(nGridX):
            if raw_roi_mask[i, j] == 1:
                Angle = anglemap[i, j]
                k = 0
                xi = roi_size*(j+1) + roi_size * np.array([-1, 1]) * np.cos((90-Angle)*-1*np.pi/180) + k*np.cos(Angle*np.pi/180)
                yi = roi_size*(i+1) + roi_size * np.array([-1, 1]) * np.sin((90-Angle)*-1*np.pi/180) + k*np.sin(Angle*np.pi/180)
                #embed()
                # Append xi and yi values to the DataFrame
                xy_values[t, 0] = xi[0]
                xy_values[t, 1] = xi[1]
                xy_values[t, 2] = yi[0]
                xy_values[t, 3] = yi[1]
                t+=1
    np.savetxt("res.csv", res, delimiter=",")
    return anglemap, res, rot, xy_values

def choose_sigma(raw, raw_mask, raw_roi_mask, steerable_sigma, roi_size, pad_angle,qlow, qhigh,nX, nY, sd_0 = None):
    """
    Function to choose the value of sigma for steerable filter.
    
    Parameters:
        raw (numpy.ndarray): Raw image data.
        raw_mask (numpy.ndarray): Mask for raw image.
        raw_roi_mask (numpy.ndarray): Mask for region of interest in raw image.
        steerable_sigma (float): Initial value of sigma for steerable filter.
        roi_size (int): Size of the region of interest.
        pad_angle (float): Angle for padding.
        qlow (float): The lower limit for the colorbar.
        qhigh (float): The upper limit for the colorbar.
        nX (int): Number of pixels in x-direction.
        nY (int): Number of pixels in y-direction.
        sd_0 (tuple, optional): Tuple containing initial values of angelmap,res,rot,xy_values. Defaults to None.
    
    Returns:
        None
    """
    fig = plt.figure()
    
    
    steerable_sigma_slider_ax = plt.axes([0.2, 0.08, 0.65, 0.03])
    steerable_sigma_slider = Slider(steerable_sigma_slider_ax, 'steerable sigma', 1, 4, valinit=steerable_sigma, valstep=0.5)
    if not sd_0:
        anglemap, res, rot, xy_values = steerable_filter(raw, raw_mask, raw_roi_mask, steerable_sigma, roi_size, pad_angle, nX, nY)
        res = pd.read_csv('res.csv', header=None).to_numpy()
    else:
        angelmap,res,rot,xy_values = sd_0
    fig, _, res1_display, res2_display, ax1, ax2, ax3 = plot_steerable_filter_results(raw, res, xy_values, steerable_sigma,qlow, qhigh, fig)
    
    def update_image(steerable_sigma):
        anglemap, res, rot, xy_values = steerable_filter(raw, raw_mask, raw_roi_mask, steerable_sigma, roi_size, pad_angle,nX, nY)
        res = pd.read_csv('res.csv', header=None).to_numpy()
        res1_display.set_data(res)   
        raw_new = np.ones((raw.shape[0], raw.shape[1]))
        res2_display.set_data(raw_new)
        ax3.imshow(raw_new, cmap='gray', vmin=np.min(res), vmax=np.max(res))
        ax1.set_title(f'Raw')
        ax2.set_title(f'Steerable filter (sigma = {steerable_sigma} pix.)')
        
        # Remove all lines from ax3
        while len(ax3.lines) > 0:
            line = ax3.lines[0]
            line.remove()
        for i in range(xy_values.shape[0]):
            ax3.plot([xy_values[i, 0], xy_values[i, 1]], [xy_values[i, 2], xy_values[i, 3]], 'c')
        fig.canvas.draw_idle()
        
    steerable_sigma_slider.on_changed(lambda val: update_image(val))    
    plt.show()
    
def plot_steerable_filter_results(raw, res, xy_values, steerable_sigma, qlow, qhigh, fig=None):
    """
    Plots the results of a steerable filter analysis.

    Args:
        raw (ndarray): The raw input image.
        res (ndarray): The filtered image.
        xy_values (ndarray): Array of xy coordinate values.
        steerable_sigma (float): The sigma value for the steerable filter.
        qlow (float): The lower limit for the colorbar.
        qhigh (float): The upper limit for the colorbar.
        fig (Figure, optional): The figure object to plot on. If not provided, a new figure will be created.

    Returns:
        tuple: A tuple containing the figure object, raw image display, filtered image display, 
               new raw image display, and the axis object for the xy plot.
    """
    return_fig = True
    if not fig:
        fig = plt.figure()
        return_fig = False
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title(f'Raw')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title(f'Steerable filter (sigma = {steerable_sigma} pix.)')
    ax3 = fig.add_subplot(1, 3, 3)
        
 

    raw_display = ax1.imshow(raw, cmap='gray', vmin=qlow, vmax=qhigh)
    res1_display = ax2.imshow(res, cmap='gray', vmin=np.min(res), vmax=np.max(res))
    
    raw_new = np.ones((raw.shape[0], raw.shape[1]))
    res2_display = ax3.imshow(raw_new, cmap='gray', vmin=qlow, vmax=qhigh)
    for i in range(xy_values.shape[0]):
        ax3.plot([xy_values[i, 0], xy_values[i, 1]], [xy_values[i, 2], xy_values[i, 3]], 'c')
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5)
    
    if return_fig:
        return fig, raw_display, res1_display, res2_display, ax1, ax2, ax3
    else:
        plt.show()
        
def intensity_profile(image, start, end, num_points):
    """
    Calculate the intensity profile along a line segment in an image.

    Parameters:
        image : The input image (grayscale).
        start (tuple): The starting point of the line segment (x, y).
        end (tuple): The ending point of the line segment (x, y).
        num_points (int): The number of equally spaced points to sample along the line segment.

    Returns:
        numpy.ndarray: The intensity profile along the line segment, normalized between 0 and 1.
    """

    # convert image numpy array
    image_data = np.array(image)
    nrows, ncols = image_data.shape[0], image_data.shape[1]

    # check if start and end points are within image
    if start[0] < 0 or start[0] > ncols or start[1] < 0 or start[1] > nrows:
        raise ValueError('start point is not within image')
    if end[0] < 0 or end[0] > ncols or end[1] < 0 or end[1] > nrows:
        raise ValueError('end point is not within image')

    if end[0] != start[0]:
        # calculate slope and y-intercept
        slope = (end[1] - start[1]) / (end[0] - start[0])
        y_intercept = start[1] - slope * start[0]
        def line(x):
            return slope * x + y_intercept

        # calculate num_points equally spaced integer (x,y) coordinates along the line
        x = np.linspace(start[0], end[0], num_points, dtype=int)
        y = np.round(line(x)).astype(int)
    else:
        y = np.linspace(start[1], end[1], num_points, dtype=int)
        x = np.full(num_points, start[0], dtype=int)

    # read out image values at the coordinates
    values = image_data[y, x]

    # get intensity values between 0 and 1
    min_value, max_value = values.min(), values.max()
    diff = max_value - min_value
    intensity = (values - min_value) / diff
    return intensity


def cross_correlation(raw, raw1, raw2, raw_roi_mask, anglemap, gblur_sigma, pixSize, nX, nY, roi_size, tophat_sigma=None):
    """
    Perform cross-correlation analysis on image data.

    Args:
        raw (ndarray): Raw image data.
        raw1 (ndarray): image of substrate 1.
        raw2 (ndarray): image of substrate 2.
        raw_roi_mask (ndarray): ROI mask for raw image.
        anglemap (ndarray): Angle map for each ROI.
        gblur_sigma (float): Standard deviation for Gaussian blur.
        tophat_sigma (float): Standard deviation for top-hat filter.
        pixSize (float): Pixel size in micrometers.
        nX (int): Number of pixels in X direction.
        nY (int): Number of pixels in Y direction.
        roi_size (int): Size of each ROI.

    Returns:
        None
    """
    
    nGridX = int(np.floor(nX / roi_size))
    nGridY = int(np.floor(nY / roi_size))
    
    # apply masks
    raw1_BGSub = gaussian_filter(raw1, sigma=gblur_sigma)
    raw2_BGSub = gaussian_filter(raw2, sigma=gblur_sigma)
    if tophat_sigma:
        selem1 = disk(tophat_sigma)
        raw1_Tophat = white_tophat(raw1_BGSub, selem1)
        selem2 = disk(tophat_sigma)
        raw2_Tophat = white_tophat(raw2_BGSub, selem2)

    # Define constants and parameters
    corr_length = 4  # [μm]
    crop_length = round(corr_length / (2 * pixSize))  # [pixel]
    max_dist = round(crop_length) + 10  # +10 pixels to corr_length to smaller deviations
    maxlag = 2 * max_dist  # Correlation length will be doubled because of mirror symmetry

    ccf_all = np.full((nGridY * nGridX, 2 * maxlag + 1), np.nan)  # Allocate memory for ACF/CCF
    MergedData = np.empty((nGridY, nGridX), dtype=object)
    z = 1  # Counting number

    fig, ax = plt.subplots(figsize=(10, 10))
    # Computation over the whole grid
    for i in range(nGridY):
        for j in range(nGridX):
            if raw_roi_mask[i, j] == 1:
                Angle = anglemap[i, j]
                ccf = np.zeros(2*maxlag+1)
                var1 = 0
                var2 = 0
                kmax = 10

                for k in range(-kmax, kmax+1, 2):
                    dir = np.array([-1, 1])
                    xi = roi_size*(j+1) + max_dist*dir*np.cos((90-Angle)*-1*np.pi/180) + k*np.cos(Angle*np.pi/180)
                    yi = roi_size*(i+1) + max_dist*dir*np.sin((90-Angle)*-1*np.pi/180) + k*np.sin(Angle*np.pi/180) # pretty much the same as in matlab (see comp.ipynb)
                    start = (xi[0], yi[0])
                    end = (xi[1], yi[1])
                    linescan1 = intensity_profile(raw1_Tophat, start,end, 2*max_dist+1)
                    linescan2 = intensity_profile(raw2_Tophat, start,end, 2*max_dist+1)
                    if len(linescan1) > 2*max_dist+1:
                        linescan1 = linescan1[:2*max_dist+1]
                        linescan2 = linescan2[:2*max_dist+1]
                        
                    linescan1_mean = np.mean(linescan1)
                    linescan1 -= linescan1_mean
                    linescan2_mean = np.mean(linescan2)
                    linescan2 -= linescan2_mean
                    
                    cc, lag = xcorr(linescan1, linescan2, norm='unbiased') # does the same as xcorr in matlab
                    cc = np.array(cc)/ (2*kmax+1)

                    ccf += cc
                    var1 += np.var(linescan1) / (2*kmax+1)
                    var2 += np.var(linescan2) / (2*kmax+1)

                ccf /= np.sqrt(var1 * var2)
                ccf_all[i*nGridX+j, :] = ccf

                maxlag_plot = len(linescan1) - 1
                lags = np.arange(-maxlag, maxlag+1) * pixSize
                ind = np.arange(maxlag_plot, min(2*maxlag_plot+1, len(ccf), len(lags)))
                ax.plot(lags[ind], ccf[ind], color='#cccaca')
    ax.set_xlabel(r'$\Delta$ x [$\mu$m]')
    ax.set_xlim(0,corr_length)
    ax.set_ylim(-0.5,1)
    ax.set_ylabel('CCF')


    # Find the valid CCFs/ACFs and calculate the mean
    ccf_all_valid = ccf_all[~np.isnan(ccf_all).any(axis=1)]
    mean_ccf = np.mean(ccf_all_valid, axis=0)
    std_mean_ccf = np.std(ccf_all_valid, axis=0, ddof=1)

    ax.plot(lags[ind], mean_ccf[ind], '-', color='#d13111', linewidth=1.8)
    ax.plot(lags[ind], mean_ccf[ind] - std_mean_ccf[ind], '--', color='#d13111', linewidth=1.8)
    ax.plot(lags[ind], mean_ccf[ind] + std_mean_ccf[ind], '--', color='#d13111', linewidth=1.8)

    plt.show()
    
def main():
    """
    This function is the main entry point of the image analysis program.
    It prompts the user for input, loads the image, applies various image processing techniques,
    and calculates the cross correlation between two substrates.
    """
    
    
    # define parameter
    # Background subtraction
    gblur_sigma = 2 

    # ROIsMask (obtained from Mask)
    roi_size = 30 # [pixels] 

    # Tophat filtering
    tophat_sigma = 28 

    # Steerable Filter
    steerable_sigma = 2 # Size for Steerable filter  

    # Mask treshold to filter interesting structures for ROI 
    mask_thresh =350 # it varies for different time points of myofibrillogenesis

    # ROIsMask (obtained from Mask)
    roi_thresh = 0.3 # minimal value used, some stages increase up to 0.85

    # Padding
    pad_angle = 6 # Window size --> we will crop image to ensure linescans are not reaching edge of image
    pad_corr = 3 # [pixel]
    roi_pad_corr = (roi_size+(roi_size*(pad_corr-1))*2)
    
    # Ask the user for the path
    exists = False
    while not exists:
        image_dir = input("Enter the path where the images are located: ")
        # check if the path exists
        if not os.path.exists(image_dir):
            print("The path you entered does not exist.\n Please try again.")
        else:
            exists = True
        
    # list all .tif files in the directory
    image_names = [file for file in os.listdir(image_dir) if file.endswith(".tif")]
    
    # user input for image selection
    print("Available images:")
    for i, image_name in enumerate(image_names):
        print(f"{i+1}. {image_name}")

    choice = int(input("Enter the number of the image you want to choose: ")) - 1
    chosen_image = image_names[choice]
    fly1_name = '2023.06.12_MhcGFPweeP26_30hrsAPF_Phallo568_647nano62actn_405nano2sls_100Xz2.5_1_2.tif'
    fly2_name = '2023.06.12_MhcGFPweeP26_24hrsAPF_Phallo568_647nano62actn_405nano2sls_100Xz2.5_1_2.tif'
    human_name = 'Trial8_D12_488-TTNrb+633-MHCall_DAPI+568-Rhod_100X_01_stitched.tif'
    print(f"You chose: {chosen_image}")
    
    if chosen_image == fly1_name or chosen_image == fly2_name:
        tmp = '''You chose a image of fly muscle.\n \t- slice 0: alpha-actinin\n\t- slice 1: actin\n\t- slice 2: myosin\n\t- slice 3: sallimus (mask channel)'''
        print(tmp)
        slice_index = 0
        mask_channel = 3
        substrate1_channel = 0
        substrate2_channel = 3
        change_th = True
    if chosen_image == human_name:
        tmp = '''You chose a image of human muscle.\n \t- slice 0: titin N-terminus (mask channel\n\t- slice 1: muscle myosin\n\t- slice 2: nuclei\n\t- slice 3: actin'''
        print(tmp)
        mask_channel = 0
        change_th = False

    path = os.path.join(image_dir, chosen_image)
    
    # load the image
    tif = TiffFile(path)
    
    # choose the frame to use as a mask
    change = True
    while change:
        choose_image(tif)
        if 'slice_index' not in locals():
            slice_index = int(input('Enter the slice index: '))
        if 'mask_channel' not in locals():
            mask_channel = int(input('Enter the index for the mask channel: '))
        image = np.asarray(tif.pages[4*slice_index+mask_channel].asarray())
        plt.imshow(image, vmin=image.min(), vmax=image.max())
        plt.title(f'mask image (slice {slice_index}, channel {mask_channel})')
        plt.show()
        proceed = input('Proceed? (Y/n)')
        if proceed == 'n' or proceed == 'N':
            change = True
        else:
            change = False
        
    raw = np.asarray(tif.pages[4*slice_index+mask_channel].asarray())
    raw_info = tif.pages[4*slice_index+mask_channel].tags
    
    nY, nX = raw.shape
    qLow = np.quantile(raw, 0.001)
    qHigh = np.quantile(raw, 0.999)
    
    # get image factor
    factor = raw_info['XResolution'].value[0]  # The Xresolution tag is a ratio, so we take the numerator
    pixSize = 1 / factor *10**(6) # Factor to go from pixels to micrometers
    
    
    change = True
    while change:
        tmp =f'''Current parameter for creation of mask and background substraction:\n\t- mask_thresh: {mask_thresh}\n\t- roi_thresh: {roi_thresh}\n\t- roi_size: {roi_size}\n\t- gblur_sigma: {gblur_sigma}'''
        print(tmp)
        choose_parameter_mask(raw, mask_thresh, roi_thresh, roi_size, gblur_sigma, pad_angle)
        change = input('\n Change parameter? (y/N)')

        if change == 'y' or change == 'Y':
            print('Enter new parameter:')
            mask_thresh = int(input('mask threshhold:'))
            roi_thresh = float(input('roi threshhold:'))
            roi_size = int(input('roi size:'))
            gblur_sigma = float(input('gblur sigma:'))
            
            raw, raw_mask, raw_roi_mask, raw_bgsub = mask_background(raw, mask_thresh, roi_thresh, roi_size, gblur_sigma, pad_angle)
            
            plot_mask_background(raw_mask, mask_thresh, raw_roi_mask, roi_size, roi_thresh, gblur_sigma)
            proceed = input('Proceed? (Y/n)')
            if proceed == 'n' or proceed == 'N':
                change = True
            else:
                change = False
        else:
            change = False
            raw, raw_mask, raw_roi_mask, raw_bgsub = mask_background(raw, mask_thresh, roi_thresh, roi_size, gblur_sigma, pad_angle)
    change = True
    while change and change_th:
        raw_tophat, tophat_sigma = change_tophat_sigma(raw_bgsub, tophat_sigma, roi_size)
        tmp = input(f'Current tophat_sigma: {tophat_sigma}\nChange tophat_sigma? (y/N)')
        if tmp == 'y' or tmp == 'Y':
            tophat_sigma = int(input('Enter new tophat_sigma:'))
            
            selem = disk(tophat_sigma)
            raw_tophat = white_tophat(raw_bgsub, selem)
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            plt.title(f'raw_bgsub')
            ax1.imshow(raw_bgsub, cmap='gray')
            ax2 = fig.add_subplot(1, 2, 2)
            plt.title(f'raw_tophat (sigma={tophat_sigma})')
            ax2.imshow(raw_tophat, cmap='gray')
            plt.show()
            
            proceed = input('Proceed? (Y/n)')
            if proceed == 'n' or proceed == 'N':
                change = True
            else:
                change = False
        else:
            change = False
    change = True
    

    while change:
        print(f'Current sigma for steerable filter: {steerable_sigma}\n Aplying steerable filter...')
        anglemap, _, rot, xy_values = steerable_filter(raw, raw_mask, raw_roi_mask, steerable_sigma, roi_size, pad_angle, nX, nY)
        res = pd.read_csv('res.csv', header=None).to_numpy()
        sd_0 = [anglemap, res, rot, xy_values]
        choose_sigma(raw, raw_mask, raw_roi_mask, steerable_sigma, roi_size, pad_angle,qLow, qHigh, nX, nY, sd_0)
        tmp = input('Change sigma for steerable filter? (y/N)')
        if tmp == 'y' or tmp == 'Y':
            steerable_sigma = float(input('Enter new sigma for steerable filter:'))
            anglemap, _, rot, xy_values = steerable_filter(raw, raw_mask, raw_roi_mask, steerable_sigma, roi_size, pad_angle, nX, nY)
            res = pd.read_csv('res.csv', header=None).to_numpy()
            plot_steerable_filter_results(raw, res, xy_values, steerable_sigma,qLow, qHigh)
            proceed = input('Proceed? (Y/n)')
            if proceed == 'n' or proceed == 'N':
                change = True
            else:
                change = False
        else:
            change = False
    # calculate cross correlation
    print('Calculating the cross correlation:')
    
    if 'substrate1_channel' not in locals():
        substrate1_channel = int(input('channel number for the first substrate: '))
        substrate2_channel = int(input('channel number for the second substrate: '))
        
    print('calculating cross correlation...')
    image1 = np.asarray(tif.pages[4*slice_index + substrate1_channel].asarray())
    image2 = np.asarray(tif.pages[4*slice_index + substrate2_channel].asarray())
    
    if change_th:
        cross_correlation(raw, image1, image2, raw_roi_mask, anglemap, gblur_sigma, pixSize, nX, nY, roi_size, tophat_sigma)
    else:
        cross_correlation(raw, image1, image2, raw_roi_mask, anglemap, gblur_sigma, pixSize, nX, nY, roi_size)
    
        
        
    
if __name__ == "__main__":
    main()
