# Provides functions for interacting with the operating system
import os  

# Provides regular expression matching operations
import re  

# Provides time-related functions
import time  

# A dict subclass for counting hashable objects
from collections import Counter  

# OpenCV library for image processing
import cv2  

# Provides an interactive shell
from IPython import embed  

# Plotting library
import matplotlib.pyplot as plt  

# For creating polygon patches in matplotlib
from matplotlib.patches import Polygon, Rectangle

# For creating slider and button widgets in matplotlib
from matplotlib.widgets import Slider, Button  

# Just-In-Time compiler for Python, used for performance optimization
from numba import jit  

# Numerical computing library
import numpy as np  

# Data analysis and manipulation library
import pandas as pd  

# For applying gaussian filter and mapping coordinates
from scipy.ndimage import gaussian_filter, map_coordinates  

# for fitting curves to data
from scipy.optimize import curve_fit
from scipy.fftpack import fft
from scipy.signal import argrelextrema, find_peaks
from scipy.stats import circmean, circstd, circvar

# For resizing images
from skimage.transform import resize  
from skimage import measure


# for detecting nuclei
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from stardist.models import StarDist2D

from csbdeep.utils import normalize

# KMeans clustering algorithm
from sklearn.cluster import KMeans  

# Cross-correlation function
from spectrum.correlation import xcorr  

# Presumably a library for steerable filters, but it's not a standard Python library
import steerable  

# For reading and writing TIFF files
from tifffile import TiffFile

def get_linear_trend(x, y, smooth=True):
    if smooth:
        # Define window size for moving average
        window_size = 5

        # Compute moving averages
        smoothed_y = np.convolve(y, np.ones(window_size)/window_size, mode='same')

        y = smoothed_y

    # Calculate the linear trend
    slope, intercept = np.polyfit(x, y, 1)

    # Calculate the linear trend
    trend = slope * x + intercept

    return trend, slope, intercept



def sine_func(x, offs, amp, f, phi):
    """
    sine function
    
    Parameter:
        - x (list): The x-coordinates.
        - offs (float): The offset of the sine wave.
        - amp (float): The amplitude of the sine wave.
        - f (float): The frequency of the sine wave.
        - phi (float): The phase of the sine wave.
    
    Returns:
        - (list): The y-coordinates of the sine wave.
    """
    return offs + amp * np.sin(2 * np.pi * f * x + phi)

def sine_fit(x, y, trend=None, guess=None, plot=False, tol=0.00025):
    """
    Estimate Parameter of a noisy sine wave by FFT and non-linear fitting.
    
    Parameter:
        - x (list): The x-coordinates of the signal.
        - y (list): The y-coordinates of the signal.
        - trend (list, optional): The trend of the signal. Defaults to None.
        - guess (list, optional): The initial guess for the Parameter (offset, amplitude, frequency, phase). Defaults to None.
        - plot (bool): Whether to plot the signal and the fit.
    
    Returns:
        - popt (list): The estimated Parameter (offset, amplitude, frequency, phase) of the sine wave.
        - pcov (list): The estimated covariance of the Parameter.
        - mse (float): The mean squared error of the fit.
    
    """
    if trend is not None:
        y = y - trend
    
    
    if guess is None:
        popt, pcov = curve_fit(sine_func, x, y)
    else:
        # Perform the fit
        popt, pcov = curve_fit(sine_func, x, y, p0=guess)
        
    if trend is not None:
        popt[0] += trend[0]
    
    # Calculate mean squared error
    mse = np.mean((y - sine_func(x, *popt)) ** 2)
    if mse > tol:
        #print(f'Warning: Mean squared error of the fit is {mse:.6f}.')
        #plot = True
        f=5
    
    
    # Plot if requested
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(x, y,'o', color='b', label='data')
        x = np.linspace(x[0], x[-1], 1000)
        plt.plot(x, sine_func(x, *popt), 'r-', label='fit')
        plt.legend()
        plt.show()
    
    return popt, pcov, mse

def crop_around_extrema(x, y, prominence, factor, fig_ax=None):
    
    # Find the indices of the minima and maxima in the dataset
    min_peak_idx, _ = find_peaks(-y, prominence=prominence)
    max_peak_idx, _ = find_peaks(y, prominence=prominence)

    # Raise an error if no prominent local extrema are found
    if len(min_peak_idx) == 0 or len(max_peak_idx) == 0:
        raise ValueError(f'no prominent local extrema for prominence {prominence} found')


    # Filter out maxima that occur before the first minimum
    max_peak_idx = max_peak_idx[max_peak_idx > min_peak_idx[0]] 

    # Initialize indices and values
    idx_1, idx_2, idx_3 = None, None, None
    val_1, val_2, val_3 = None, None, None

    # If there's only one minimum, set the indices and values accordingly
    if len(min_peak_idx) == 1:
        idx_1, idx_2 = min_peak_idx[0], max_peak_idx[0]
        val_1, val_2 = y[idx_1], y[idx_2]
        y_min = val_1    
        # determine new "maximum" position based on distance to the first minimum
        norm_tmp = np.linalg.norm([x[idx_2]-x[idx_1], val_2-val_1])            
        norm_right_tmp = [np.linalg.norm([x[idx+idx_2]-x[idx_2], val-y[idx_2]]) for idx, val in enumerate(y[idx_2:])]
        idx_3 = idx_2 + np.argmin(np.abs(np.array(norm_right_tmp)-norm_tmp))
        val_3 = y[idx_3]   
    else:
        # If there's more than one minimum, set the indices and values accordingly
        idx_1, idx_2, idx_3 = min_peak_idx[0], max_peak_idx[0], min_peak_idx[1]
        val_1, val_2, val_3 = y[idx_1], y[idx_2], y[idx_3]
        
    # If the distance between the second minimum and maximum is too large, ignore the second minimum
    if x[idx_3] - x[idx_2] > 1.5 * (x[idx_2] - x[idx_1]):
        y_min = val_1
        
        # determine new "maximum" position based on distance to the first minimum
        norm_tmp = np.linalg.norm([x[idx_2]-x[idx_1], val_2-val_1])            
        norm_right_tmp = [np.linalg.norm([x[idx_2+ idx]-x[idx_2], val-y[idx_2]]) for idx, val in enumerate(y[idx_2:])]
        idx_3 = idx_2 + np.argmin(np.abs(np.array(norm_right_tmp)-norm_tmp))
        val_3 = y[idx_3]
        
    # If the distance between the first minimum and the following maximum is too large, ignore the first minimum
    elif x[idx_2] - x[idx_1] > 1.5 * (x[idx_3] - x[idx_2]) and idx_3 != len(x)-1:
        idx_1, val_1 = None, None
        y_min = val_3
        
        # determine new "minimum" position based on distance to the first maximum
        norm_tmp = np.linalg.norm([x[idx_2]-x[idx_3], val_2-val_3])
        norm_left_tmp = [np.linalg.norm([x[idx]-x[idx_2], val-y[idx_2]]) for idx, val in enumerate(y[:idx_2+1])]
        idx_1 = np.argmin(np.abs(np.array(norm_left_tmp)-norm_tmp))
        val_1 = y[idx_1]
    y_min = min(val_1, val_3)

    # Set the maximum value
    y_max = val_2

    # Calculate the difference between the maximum and minimum values
    diff = np.abs(y_max - y_min)

    # Calculate the norm
    norm_idx = idx_1 + np.argmin(np.abs(y[idx_1:idx_2] - (val_1 + diff * factor)))
    norm = np.linalg.norm([x[norm_idx]-x[idx_1], y[norm_idx]-y[idx_1]])

    # Calculate the norms for the left and right sides
    y_norm_left = np.array([np.linalg.norm([x[idx]-x[idx_1], val-y[idx_1]])-norm for idx, val in enumerate(y[:idx_1+1])])
    #y_norm_left[y[:idx_1] > y_max] = np.nan
    #y_norm_left[y[:idx_1] < val_1] = np.nan
    y_norm_right = np.array([np.linalg.norm([x[idx_3 + idx]-x[idx_3], val-y[idx_3]])-norm for idx, val in enumerate(y[idx_3:])])
    #y_norm_right[y[idx_3:] > y_max] = np.nan
    #y_norm_right[y[idx_3:] < val_3] = np.nan
    
    # Find the start and end indices
    try:
        start_idx = np.nanargmin(np.abs(y_norm_left))
        end_idx = np.nanargmin(np.abs(y_norm_right)) + idx_3
    except ValueError:
        start_idx = idx_1
        end_idx = idx_3

    if fig_ax is not None:
        fig, ax = fig_ax
        ax.plot(x, y, 'o')
        ax.plot(x[min_peak_idx], y[min_peak_idx], 'o', color='r')
        ax.plot(x[max_peak_idx], y[max_peak_idx], 'o', color='g')
        ax.plot([x[idx_1], x[idx_2]], [y[idx_1], y[idx_2]], 'o', color='m')
        ax.plot(x[idx_3], y[idx_3], 'o', color='m')
        ax.vlines(x[start_idx], ymin=min(y), ymax=max(y), colors='k', linestyles='dashed')
        ax.vlines(x[end_idx], ymin=min(y), ymax=max(y), colors='k', linestyles='dashed')
        return start_idx, end_idx, diff,[idx_1, idx_2, idx_3], (fig, ax)
    return start_idx, end_idx, diff,[idx_1, idx_2, idx_3], None

def plot_fourier_peaks(omega_valid, amplitude_valid):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))
    
    std_omega = np.std(omega_valid)
    mean_omega = np.mean(omega_valid)
    min_omega = mean_omega - 3*std_omega
    max_omega = mean_omega + 3*std_omega
    
    n_bins = 30
    
    n, bins, patches = ax1.hist(omega_valid, bins=n_bins, range=(min_omega, max_omega), histtype='step', density=True)
    ax1.vlines(mean_omega,0, n.max(), color='red')
    ax1.set_xlabel(r'$\omega$')
    ax1.set_ylabel('Frequency')
    
    std_amp = np.std(amplitude_valid)
    mean_amp = np.mean(amplitude_valid)
    min_amp = min(np.abs(mean_amp - 3*std_amp), 0)
    min_amp = mean_amp + 3*std_amp
    
    n, bins, patches = ax2.hist(amplitude_valid, bins=n_bins, range=(min_amp, min_amp), histtype='step', density=True)
    ax2.vlines(mean_amp,0, n.max(), color='red')
    ax2.set_xlabel('Amplitude')
    ax2.set_ylabel('Frequency')
    
    plt.show()


def white_tophat_trafo(raw, tophat_sigma):
    """
    Applies a white tophat transformation to an image (remove objects smaller than the structuring element)

    Parameter:
        - raw (ndarray): The raw input image.
        - tophat_sigma (float): defines the size of the structuring element (disk with diameter 2*tophat_sigma+1).

    Returns:
        - ndarray: The transformed image.
    """
    # Create a circular structuring element equivalent to the 'disk' in skimage
    selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2*tophat_sigma+1), int(2*tophat_sigma+1)))
    
    # Apply the white tophat transformation
    return cv2.morphologyEx(raw, cv2.MORPH_TOPHAT, selem)
                

def plot_steerable_filter_results(raw, res, xy_values_k0, steerable_sigma, roi_instances=[], fig=None, load_res=True):
    """
    Plots the results of a steerable filter analysis.

    Parameter:
        - raw (ndarray): The raw input image.
        - res (ndarray): The response of the steerable filter.
        - xy_values_k0 (ndarray): Array of xy coordinate values of structures detected by the steerable filter.
        - steerable_sigma (float): The sigma value used when the steerable filter was applied.
        - roi_instances (list, optional): A list of obejects of class ROI. Defaults to [].
        - fig (Figure, optional): The figure object to plot on. If not provided, a new figure will be created.

    Returns:
        - tuple: A tuple containing the figure object, raw image display, filtered image display, 
                 new raw image display, and the axis object for the xy plot.
    """
    if load_res:
        res = pd.read_csv('res.csv', header=None).to_numpy()
    # limits of colorbar
    qlow = np.quantile(raw, 0.001)
    qhigh = np.quantile(raw, 0.999)
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
    
    raw_new = np.ones((raw.shape[0], raw.shape[1]))
    res2_display = ax3.imshow(raw, cmap='gray', vmin=qlow, vmax=qhigh)
    
    x_values = xy_values_k0[:, 0].reshape(-1, 2)
    x_start, x_end = x_values.T
    
    y_values = xy_values_k0[:, 1].reshape(-1, 2)
    y_start, y_end = y_values.T
    
    ax3.plot([x_start, x_end],[y_start, y_end], 'c')
    res1_display = ax2.imshow(res, cmap='gray', vmin=np.min(res), vmax=np.max(res))

    if len(roi_instances)>0:
        res1_display = ax2.imshow(res, cmap='gray', vmin=np.min(res), vmax=np.max(res))
        
        ax3 = plot_steerable_filter_roi(roi_instances, ax3)
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5)
    
    if return_fig:
        return fig, raw_display, res1_display, res2_display, ax1, ax2, ax3
    else:
        plt.show()

def plot_mask_background(raw_mask, raw_roi_mask, mask_thresh, roi_size, roi_thresh, gblur_sigma, roi_instances=[],nuclei_mask=None, fig=None):
    """
    Plots the mask and ROIMask images.

    Parameter:
        - raw_mask (numpy.ndarray): The raw mask image.
        - raw_roi_mask (numpy.ndarray): The raw ROIMask image.
        - mask_thresh (float): The threshold value used to create the binary mask.
        - roi_size (int): The size of the ROI.
        - roi_thresh (float): The threshold value used when creating the ROIMask image.
        - gblur_sigma (float): The sigma value used when applying a Gaussian blur to the image.
        - roi_instances (list, optional): A list of obejects of class ROI. Defaults to [].
        - fig (matplotlib.figure.Figure, optional): The figure object to plot the images on. If not provided, a new figure will be created.

    Returns:
        - tuple: A tuple containing the two image objects (pic1, pic2) if fig is provided, otherwise None.
    """
    return_fig = True
    if not fig:
        fig = plt.figure()
        return_fig = False
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_title(fr'Raw Mask (thresh. = {mask_thresh}), $\sigma_{{gb}}$ = {gblur_sigma})')
    
    ax2.set_title(f'ROIMask (size = {roi_size}, thresh. = {roi_thresh})')

    pic1 = ax1.imshow(raw_mask, cmap='gray')
  
    pic2 = ax2.imshow(raw_roi_mask, cmap='gray')
    if nuclei_mask is not None:
        contours = measure.find_contours(nuclei_mask, 0.8)

        # Plot the contours in red
        for contour in contours:
            ax1.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')

    if len(roi_instances) > 0:
        for i in range(len(roi_instances)):
            roi = roi_instances[i]
            # check if roi is instance of class ROI
            if not isinstance(roi, ROI):
                raise TypeError('roi_instances must be a list of ROI objects.')
            
            # plot the detected structures in this ROI
            polygon = Polygon(roi.corners, closed=True, edgecolor='r', fill=False, linewidth=2)
            xi,yi = roi.x_borders, roi.y_borders
            rect = [(xi[0], yi[0]), (xi[1], yi[0]), (xi[1], yi[1]), (xi[0], yi[1]), (xi[0], yi[0])]
            polygon2 = Polygon(rect, closed=True, edgecolor='b', fill=False, linewidth=2)
            ax1.add_patch(polygon)
            ax1.add_patch(polygon2)
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5)
    
    if return_fig:
        return pic1, pic2, ax1, ax2
    else:
        plt.show()
        
def choose_image(tif, n_channels=4):
    """
    Displays an interactive window to choose a frame from a sequence of images.

    Parameter:
        - tif (tifffile.TiffFile): The TIFF file containing the image sequence.
        - n_channels (int) : The number of channels in the image.

    Returns:
        - None
    """
    # Get the number of slices
    n = len(tif.pages)
    num_slices = n/n_channels
    if n%n_channels != 0:
        raise ValueError(f'The number of frames is not a multiple of {n_channels}.')
    
    
    
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

def improfile(img, xi, yi, N, order=1, mode='nearest'):
    """ 
    Calculate the intensity profile along a line in an image.
    
    Parameter:
        - img (numpy.ndarray): The image data.
        - xi (tuple): The x-coordinates defining the line.
        - yi (tuple): The y-coordinates defining the line.
        - N (int): The number of points to interpolate along the line.
        - order (int, optional): The order of the spline interpolation. Defaults to 1.
        - mode (str, optional): The mode parameter passed to map_coordinates. Defaults to 'nearest' (nearest-neighbor interpolation)
    
    """
    # Create coordinate arrays
    x = np.linspace(xi[0], xi[1], N)
    y = np.linspace(yi[0], yi[1], N)
    coords = np.vstack((y,x)) # stack them as a (2, N) array

    # Extract the intensities along the line
    intensities = map_coordinates(img, coords, order=order, mode=mode)

    return np.array(intensities, dtype=float)


def rectangle_interpolation(point1, point2, point3):
    """
    Create a rectangle from three points.

    Parameter:
        - point1 (tuple) (x1,y1)         : The first corner of the rectangle.
        - point2 (tuple) (x2,y2)         : The second corner of the rectangle .
        - point3 (tuple) (x_proj, y_proj): The third point defining the rectangle (only the projection matters).
        
                              (x1, y1)               
                      
                                 *                            
                               *     *                         
                             *           *                     
                           *                 *                 
                         *                       *             
                       *                             * (x2, x3)
                     *                             *           
                   *                             *             
                 *             ROI             *               
               *                             *                 
    (x4, y4) *                             *                   
                 *                       *                     
                     *                 *                       
                         *           *                         
                             *     *                           
                                 *                             
                             (x4, y4)*                         
                                         *
                                             *
                                      (x_proj, y_proj)
            

    Returns:
        - tuple: A tuple containing the four corner points of the rectangle.
    """
    x1, y1 = point1
    x2, y2 = point2
    x_proj, y_proj = point3

    if x1 == x2:  # The line is vertical
        x3 = x4 = x1
        y3 = y4 = y_proj
    else:
        # line through point1 and point2
        m1 = (y2 - y1) / (x2 - x1)
        n1 = y1 - m1 * x1

        if m1 == 0:  # The line is horizontal
            y3 = y4 = y2
            x3 = x4 = x_proj
        else:
            # perpendicular line trough point2
            m2 = -1 / m1
            n2 = y2 - m2 * x2

            # line through point3
            n3 = y_proj - m1 * x_proj

            # intersection point
            x3 = (n3 - n2) / (m2 - m1)
            y3 = m2 * x3 + n2

            # fourth corner
            n4 = y1 - m2 * x1
            x4 = (n4 - n3) / (m1 - m2)
            y4 = m1 * x4 + n3

    return ((x1, y1), (x2, y2), (x3, y3), (x4, y4))

#jit decorator tells Numba to compile this function.
@jit(nopython=True)
def process_mask(roi, raw_mask, mask_thresh):
    """
    Create a binary mask based on a threshold and a ROI.
    
    Parameter:
        - roi (tuple) (x1,y1,x2,y2,x3,y3,x4,y4): The four corner points of the rectangle defining the ROI.
        - raw_mask (numpy.ndarray): The empty raw mask image to save the created mask .
        - mask_thresh (float): The threshold value used to create the binary mask.
    """
    nY, nX = raw_mask.shape
    for y in range(nY):
        for x in range(nX):
            in_roi = False
            # Check if the pixel is inside the ROI
            if inside_rectangle((x, y), roi):
                in_roi = True
            # Set the pixel value to 1 if it is inside the ROI and above the threshold, otherwise set it to 0
            if in_roi and raw_mask[y, x] >= mask_thresh:
                raw_mask[y, x] = 1
            else:
                raw_mask[y, x] = 0
    return raw_mask

@jit(nopython=True)
def inside_rectangle(point, rect):
    """
    Check if a point is inside a roteted rectangle.

    Parameter:
        - point (tuple) (x,y): The point to check.
        - rect (tuple) (x1,y1,x2,y2,x3,y3,x4,y4): The four corner points of the rectangle.

    Returns:
        - bool: True if the point is inside the rectangle, False otherwise.
    """
    x, y = point
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = rect
    
    # calculate two vectors representing the sides of the rectangle
    AB = (x2-x1, y2-y1)
    AD = (x4-x1, y4-y1)
    
    # vector from point A to point P
    AP = (x-x1, y-y1)
    
    # calculate the dot products
    ABAP = AB[0]*AP[0] + AB[1]*AP[1]
    ABAB = AB[0]*AB[0] + AB[1]*AB[1]
    ADAP = AD[0]*AP[0] + AD[1]*AP[1]
    ADAD = AD[0]*AD[0] + AD[1]*AD[1]

    return (0 <= ABAP <= ABAB) and (0 <= ADAP <= ADAD)


        
                
class ImageAnalysis:
    """
    Class for performing image analysis on a given image.

    Attributes:
        Parameter:
            - gblur_sigma: The sigma value for Gaussian blurring.
            - roi_size: The size of the region of interest (ROI).
            - tophat_sigma: The sigma value for the top-hat filter.
            - steerable_sigma: The sigma value for the steerable filter.
            - mask_thresh: The threshold value for creating a binary mask (values below are set to 0 and values above to 1).
            - roi_thresh: The threshold value for creating a ROI mask.
            - pad_angle: The padding angle for creating the angle map.
            - pad_corr: The padding correction value for creating the ROI pad correction.
            - roi_pad_corr: The ROI pad correction value.
            - roi_array: The array of manually created ROIs. Containes the coordinates of the four corners of each ROI. [[(x1,x2),(y1,y2),(x3,x4),(y3,y4)],...]
            - pixSize: The pixel size.
            - nX: The number of columns in the image.
            - nY: The number of rows in the image.
            - nXCrop: The number of columns in the cropped image.
            - nYCrop: The number of rows in the cropped image.
        Images for mask creating to substract the background:
            - raw: The raw image which is the base for the mask to substract the background.
            - raw_tophat: The image after applying the top-hat filter to remove small objects.
            - raw_crop: The cropped image after masking the background.
            - raw_mask: The binary mask image.
            - raw_roi_mask: The ROI mask image.
            - raw_bgsub: The image after applying an gaussian blur.
            
        Two substrate Images to calculate the cross correlation between each other:
            - raw1: The first substrate image.
            - raw1_tophat: The image after applying the top-hat filter to the first substrate image.
            - raw2: The second substrate image.
            - raw2_tophat: The image after applying the top-hat filter to the second substrate image.
            - ccf_all_valid: The valid cross-correlation values.
            - mean_ccf: The mean cross-correlation.
            - std_mean_ccf: The standard deviation of the mean cross-correlation function.
            
        Parameter for detecting the nematic ordered structures on mask and results:
            - steerable_bool: A boolean value indicating whether the steerable filter has been applied.
            - anglemap: The angle map.
            - res: The response of the steerable filter to the masked image
            - rot:  Response of the input image to rotated versions of the filter, at 'nAngles' different angles.
            - xy_values_k0: cordinates of the start and endpoints of detected nematic ordered structures in the image.
    
    
    """

    def __init__(self, config={}):
        # general Parameter
        self.gblur_sigma = config.get('gblur_sigma', 2)
        self.roi_size = config.get('roi_size', 30)
        self.tophat_sigma = config.get('tophat_sigma', 28)
        self.steerable_sigma = config.get('steerable_sigma', 2)
        self.mask_thresh = config.get('mask_thresh', 350)
        self.roi_thresh = config.get('roi_thresh', 0.3)
        
        self.pad_angle_left = config.get('pad_angle_left', 6)
        self.pad_angle_right = config.get('pad_angle_right', 6)
        self.pad_angle_top = config.get('pad_angle_top', 6)
        self.pad_angle_bottom = config.get('pad_angle_bottom', 6)
        
        self.pad_corr = config.get('pad_corr', 3)
        self.roi_pad_corr = (self.roi_size + (self.roi_size * (self.pad_corr - 1)) * 2)
        
        # manually created larger ROI
        self.roi_array = config.get('roi_array', [])
        self.roi_instances = config.get('roi_instances', np.empty(0, dtype=ROI))

        # Images for mask
        self.raw = config.get('raw', [])
        self.raw_tophat = config.get('raw_tophat', [])
        self.raw_crop = config.get('raw_crop', [])
        self.raw_mask = config.get('raw_mask', [])
        self.raw_roi_mask = config.get('raw_roi_mask', [])
        self.raw_bgsub = config.get('raw_bgsub', [])
        self.pixSize = config.get('pixSize', 0)

        # Steerable filter results
        self.nAngles = config.get('nAngles', 180)
        self.steerable_bool = config.get('steerable_bool', False)
        self.anglemap = config.get('anglemap', [])
        self.res = config.get('res', [])
        self.rot = config.get('rot', [])
        self.xy_values_k0 = config.get('xy_values_k0', [])

        self.nX = config.get('nX', 0)
        self.nY = config.get('nY', 0)
        self.nXCrop = config.get('nXCrop', 0)
        self.nYCrop = config.get('nYCrop', 0)

        # Images for cross correlation
        self.raw1 = config.get('raw1', [])
        self.raw1_tophat = config.get('raw1_tophat', [])
        self.raw2 = config.get('raw2', [])
        self.raw2_tophat = config.get('raw2_tophat', [])

        self.ccf_all_valid = config.get('ccf_all_valid', [])
        self.mean_ccf = config.get('mean_ccf', [])
        self.std_mean_ccf = config.get('std_mean_ccf', [])
                
    def set_mask_image(self, raw, pixSize):
        """
        Set the mask image.

        Parameter:
            - raw (numpy.ndarray): The raw image data.
            - pixSize (float): The pixel size in micrometers.

        Returns:
            - None
        """
        self.raw = raw
        self.nY, self.nX = self.raw.shape
        self.pixSize = pixSize
        
    def set_roi_array(self, roi_array):
        """
        Set the ROI (Region of Interest) array.

        Parameter:
            - roi_array: The ROI array to be set.

        Returns:
            - None
        """
        self.roi_array = np.array(roi_array)
        
    def get_cross_correlation(self):
        """
        Calculate the cross-correlation of the image.

        Returns:
            - ccf_all_valid (numpy.ndarray): The cross-correlation values for all valid pixels.
            - mean_ccf (float): The mean cross-correlation value.
            - std_mean_ccf (float): The standard deviation of the mean cross-correlation value.
        """
        return self.ccf_all_valid, self.mean_ccf, self.std_mean_ccf
        
    def get_mask_Parameter(self):
        """
        Returns the mask Parameter used for image processing.

        Returns:
            - tuple: A tuple containing the mask threshold, ROI size, ROI threshold, and Gaussian blur sigma.
        """
        return self.mask_thresh, self.roi_size, self.roi_thresh, self.gblur_sigma
    
    def apply_steerable_filter(self, smaller_roi_size=None, plot_finetuning=True):
        """
        If the anglemap is empty, apply the steerable filter the instance of the class ImageAnalysis.
        If plot_finetuning is True: apply the steerable filter to each ROI (region of interest in self.raw) in self.roi_instances with a smaller ROI size (ROIs for mask creation)
        Parameter:
            - smaller_roi_size: The size of the ROIs for mask creation in the area of interests
        """
        
        if len(self.anglemap)==0 and smaller_roi_size is None:
            self.__apply_steerable_filter()
        if smaller_roi_size is None:
            smaller_roi_size = self.roi_size
            
        for i in range(len(self.roi_instances)):
            # reduce the size of the ROI to the smaller_roi_size
            self.roi_instances[i].set_roi_size(smaller_roi_size)
            
            # set the superclass of each ROI instance to self, then the ROI instance has the same attributes as self
            self.roi_instances[i].set_superclass(self)
            
            # crop the images of the superclass to the ROI
            self.roi_instances[i].crop_images_superclass()
            
            
            #TODO: maybe also change other Parameter for mask creation?
            
            # apply the steerable filter to the ROI (to the cropped images of the superclass)
            self.roi_instances[i].apply_steerable_filter_finetuning(plot=plot_finetuning)
            
    def substract_nuclei_background(self):
        if hasattr(self, 'raw_mask'):
            if hasattr(self, 'nuclei_mask'):
                self.nuclei_mask_crop = self.nuclei_mask[:self.raw_mask.shape[0], :self.raw_mask.shape[1]]
                self.raw_mask[self.nuclei_mask_crop == 1] = 0
            elif hasattr(self, 'raw_nuclei'):
                # creates a pretrained model to detect nuclei
                model = StarDist2D.from_pretrained('2D_versatile_fluo')
                labels, details = model.predict_instances(normalize(self.raw_nuclei))
                self.nuclei_mask = (labels > 0).astype(int)
                self.substract_nuclei_background()
            
    
    def mask_background(self):
        """
        This method processes an image by applying a Gaussian blur, creating a binary mask based on a threshold,
        and creating a region of interest (ROI) mask based on the percentage of white pixels in each ROI of size self.roi_size x self.roi_size.

        The method updates the following instance variables:
            - raw_crop: Cropped version of the raw image
            - nYCrop, nXCrop: Dimensions of the cropped image
            - raw_bgsub: cropped image with gaussian blur applied
            - raw_mask: Binary mask of raw_bgsub
            - raw_roi_mask: Binary mask of the ROIs of the whole image
            - raw_mask_array: Array of binary masks for each ROI
            - raw_roi_mask_array: Array of binary masks for each ROI
        """

        # Calculate the number of ROIs in the y and x directions
        self.nGridY = int(np.floor(self.nY / self.roi_size))
        self.nGridX = int(np.floor(self.nX / self.roi_size))

        # Crop the raw image to fit an integer number of ROIs
        self.raw_crop = np.copy(self.raw[:self.nGridY * self.roi_size, :self.nGridX * self.roi_size])
        self.nYCrop, self.nXCrop = self.raw_crop.shape

        # applying a Gaussian blur
        self.raw_bgsub = gaussian_filter(self.raw_crop, sigma=self.gblur_sigma)
        

        # Copy the background-subtracted image to create a mask
        self.raw_mask = self.raw_bgsub.copy()

        # If there are ROIs defined, create a mask based on whether each pixel is inside an ROI and above a threshold
        n_roi = len(self.roi_array)
        if n_roi > 0:
            raw_mask_temp = np.zeros((self.nYCrop, self.nXCrop), dtype=int)
            self.raw_mask_array = np.array([np.zeros((self.nYCrop, self.nXCrop), dtype=int) for i in range(n_roi)])
            for i in range(n_roi):
                roi = self.roi_array[i]
                # Create a copy of raw_mask
                raw_mask_copy = self.raw_mask.copy()
                # Create a binary mask for the current ROI
                self.raw_mask_array[i] = process_mask(roi, raw_mask_copy, self.mask_thresh)
                raw_mask_temp += self.raw_mask_array[i]
         
            self.raw_mask = raw_mask_temp
        else:
            # If no ROIs are defined, create a binary mask based on the threshold
            self.raw_mask = (self.raw_mask >= self.mask_thresh).astype(int)
        # Create a mask of the ROIs based on the percentage of white pixels in each ROI
        self.substract_nuclei_background()
        self.create_roi_mask()
    

        
    def create_roi_mask(self):
        """
        Create a binary mask of the ROIs based on the percentage of white pixels in each ROI.
        If the percentage of white pixels is above the threshold (self.roi_tresh), the ROI is considered valid and the corresponding pixel in the mask is set to 1.
        
        The method updates the following instance variables:
            - raw_roi_mask: Binary mask of the ROIs of the whole image
            - raw_roi_mask_array: Array of binary masks for each ROI
            
        """
        # Create a mask of the ROIs based on the percentage of white pixels in each ROI
        self.raw_roi_mask = np.zeros((self.nGridY, self.nGridX), dtype=int)
        white_percentage = np.zeros((self.nGridY, self.nGridX))
        start = time.time()
        if len(self.roi_array) != 0:
            roi_mask_temp = np.zeros((self.nGridY, self.nGridX), dtype=int)
            roi_mask_array = np.array([np.zeros((self.nGridY, self.nGridX), dtype=int) for i in range(len(self.roi_array))])
            for k in range(len(self.roi_array)):
                raw_mask = self.raw_mask_array[k]
                for i in range(self.nGridY):
                    for j in range(self.nGridX):
                        temp = raw_mask[i*self.roi_size:(i+1)*self.roi_size, j*self.roi_size:(j+1)*self.roi_size].mean()
                        white_percentage[i, j] = temp
                        if temp > self.roi_thresh and self.pad_angle_top - 1 <= i <= self.nGridY - self.pad_angle_bottom and self.pad_angle_left - 1 <= j <= self.nGridX - self.pad_angle_right:
                            roi_mask_array[k][i, j] = 1
                roi_mask_temp += roi_mask_array[k]
            self.raw_roi_mask = roi_mask_temp
            self.raw_roi_mask_array = roi_mask_array
        else:
            for i in range(self.nGridY):
                for j in range(self.nGridX):
                    temp = self.raw_mask[i*self.roi_size:(i+1)*self.roi_size, j*self.roi_size:(j+1)*self.roi_size].mean()
                    white_percentage[i, j] = temp
                    if temp > self.roi_thresh and self.pad_angle_top - 1 <= i <= self.nGridY - self.pad_angle_bottom and self.pad_angle_left - 1 <= j <= self.nGridX - self.pad_angle_right:
                        self.raw_roi_mask[i, j] = 1
        
        
        #plt.imshow(self.raw_roi_mask, cmap='gray')
        #plt.show()
        end = time.time()
        #print(f'ROI mask creation took {end - start} seconds')
                        
    
        
    def load_res_from_csv(self, path='res.csv'):
        """
        Load the response of the steerable filter from a CSV file. Neccessarry because of weird bug (res entries somehow get changed to zero even if the array is not explicitly changed)

        Parameter:
            - path (str, optional): The path to the CSV file. Defaults to 'res.csv'.
        """
        self.res = pd.read_csv(path, header=None).to_numpy()
 
        
    def __apply_steerable_filter(self, angle_array=None, save=True):
        """
        This method applies a steerable filter to the image, calculates the angle response, and creates an angle map.
        It also calculates the start xi and endposition yi of each detected structure based on the angle map.
        
        Parameter:
            - angle_array: Array of angles for the steerable filter. If None, the self.nAngles parameter is used to apply the steerable filter
            at 180/self.nAngles degree steps.
            - save: If True, the response of the steerable filter is saved to a CSV file.

        The method updates the following instance variables:
            - res: filter response
            - rot: response of the input image to rotated versions of the filter, at 'nAngles' different angles
            - anglemap: Map of the angles of the maximum response for each ROI
            - xy_values_k0: start xi and endposition yi of each detected sructure( xy_values_k0[t, 0] = xi, xy_values_k0[t, 1] = yi)
            - steerable_bool: Flag indicating that the steerable filter has been applied
        """
        
        # Resize the raw image and the mask to fit the number of ROIs
        raw_rsize = resize(self.raw, (self.nGridY, self.nGridX), order=0)  # order=0 --> nearest neighbor interpolation
        raw_rsize = raw_rsize.astype(np.float64)
        mask_rsize = resize(self.raw_mask, (self.nGridY, self.nGridX), order=0)

        
        # Apply the steerable filter
        try:
            sd = steerable.Detector2D(raw_rsize, 2, self.steerable_sigma)
        except ValueError as e:
            # Extract the suggested sigma from the error message
            match = re.search(r'Sigma must be < (\d+(\.\d+)?)', str(e))
            if match:
                suggested_sigma = float(match.group(1))
                print('Suggested sigma:', suggested_sigma)
                valid = False
                while not valid:
                    # Ask the user to enter a new sigma
                    sigma = float(input('Enter new sigma: '))
                    if sigma < suggested_sigma:
                        self.steerabel_sigma = sigma
                        valid = True
                    else:
                        print('Sigma must be smaller than the suggested sigma.')
                # Run the function again with the suggested sigma
                sd = steerable.Detector2D(raw_rsize, 2, suggested_sigma)
        
        # get the response of the filter
        res_tmp, _ = sd.filter()
        if save:
            np.savetxt('res.csv', res_tmp, delimiter=',')
        self.res = res_tmp

        
        if angle_array is None:
            # Calculate the response of the filter to rotated versions of the filter, at 'nAngles' different angles between 0 and 180 degrees
            if self.nAngles >180:
                raise ValueError(f'nAngles must be smaller than 180 for even orders M for the steerable filter (here M=2)')
            rot = np.transpose(sd.get_angle_response(self.nAngles))
            angle_step = 180/self.nAngles
            min_angle = 0
        else:
            # Calculate the response of the filter to rotated versions of the filter, at the angles in angle_array
            min_angle = min(angle_array*180/np.pi)
            angle_step = (angle_array[1]-angle_array[0])*180/np.pi
            rot = np.transpose(sd.get_angle_response_with_array(angle_array))
        for i in range(rot.shape[2]):
            # extract the response of the filter for each angle angle_array[i]
            temp = rot[:, :, i]
            
            # set the response to zero if the corresponding pixel in the mask is zero
            temp[mask_rsize == 0] = np.nan
            
            # save the response
            rot[:, :, i] = temp

        self.rot = rot
        
        
        # Create an angle map based on the maximum response for each ROI
        anglemap = np.full((self.nGridY, self.nGridX), np.nan)
        for i in range(self.nGridY):
            for j in range(self.nGridX):
                if self.raw_roi_mask[i, j] == 1:
                    # coordinates inside of one ROI
                    y_idx = [i-(self.pad_angle_bottom-1), i+self.pad_angle_top]
                    x_idx = [j-(self.pad_angle_left-1), j+self.pad_angle_right]
                    # Crop the response of the filter to the range of coordinates we are interested in and find the maximum response and the corresponding angle
                    Crop = rot[y_idx[0]:y_idx[1], x_idx[0]:x_idx[1], :]
                    idxMax = np.full(Crop.shape[2], np.nan)
                    for k in range(Crop.shape[2]):
                        temp = Crop[:, :, k]
                        idxMax[k] = np.nanmean(temp)
                    if np.all(np.isnan(idxMax)):
                        I = 0
                    else:
                        M, I = np.nanmax(idxMax), np.nanargmax(idxMax)
                    # self.rot as shape (nY, nX, nAngles) so I is the index of the angle with the maximum response. To extract the angle, we need to multiply I by angle_step and add min_angle where min_angle is the minimum angle in angle_array or 0 if we calculated the angle map for the whole range of angles from 0 to 180 degrees
                    I  = I * angle_step + min_angle
                    anglemap[i, j] = I

        # Calculate the x and y coordinates of detected structures for each ROI based on the angle map
        xy_values_k0 = np.zeros((self.nGridY*self.nGridX, 2, 2))
        t = 0
        for i in range(self.nGridY):
            for j in range(self.nGridX):
                if self.raw_roi_mask[i, j] == 1:
                    Angle = anglemap[i, j]
                    k = 0
                    xi = self.roi_size*(j+1) + self.roi_size * np.array([-1, 1]) * np.cos((90-Angle)*-1*np.pi/180) + k*np.cos(Angle*np.pi/180)
                    yi = self.roi_size*(i+1) + self.roi_size * np.array([-1, 1]) * np.sin((90-Angle)*-1*np.pi/180) + k*np.sin(Angle*np.pi/180)
                    xy_values_k0[t, 0] = xi
                    xy_values_k0[t, 1] = yi
                    t += 1

        self.xy_values_k0 = xy_values_k0
        self.anglemap = anglemap
        self.steerable_bool = True
        
    def set_subtrate1_image(self, raw_substrate1):
        """
        Set the image for substrate 1.

        Parameter:
            - raw_substrate1: The raw image data for substrate 1.

        Returns:
            - None
        """
        self.raw1 = raw_substrate1
        
    def set_substrate2_image(self, raw_substrate2):
        """
        Set the raw substrate 2 image.

        Parameter:
            - raw_substrate2: The raw substrate 2 image.

        Returns:
            - None
        """
        self.raw2 = raw_substrate2
        
    def set_nuclei_image(self, raw_nuclei):
        """
        Set image of nuclei background.

        Parameter:
            - raw_nuclei: The raw nuclei image.

        Returns:
            - None
        """
        self.raw_nuclei = raw_nuclei
        
    def calc_all_ccf(self, plot=True):
        """
        Calculate the cross-correlation between two images for this instance and in all instances of subclass ROI. The cross-correlation is calculated for each ROI and the detected structures in it.
        """
        if len(self.roi_instances) > 0:
            for i in range(len(self.roi_instances)):
                self.roi_instances[i].__calculate_own_cross_correlation(plot=False)
            self.corr_length = self.roi_instances[0].corr_length
        else:
            self.__calculate_own_cross_correlation(plot=False)
        if plot:
            self.plot_cross_correlation()
            
    def __calculate_own_cross_correlation(self, tophat=True,  plot=True):
        """
        Calculates the cross-correlation between two images for this instance. The cross-correlation is calculated for each ROI and the detected structures in it.

        TODO: Calculate cross-correlation in each ROI in self.roi_instances
        Parameter:
            - tophat: If True, the tophat filter is applied to the substrate images before calculating the cross-correlation.
            - plot: If True, the cross-correlation is plotted.
        """
        
        # check if tophat filter should be applied
        if tophat:
            self.raw1_tophat = white_tophat_trafo(self.raw1, self.tophat_sigma)
            self.raw2_tophat = white_tophat_trafo(self.raw2, self.tophat_sigma)
            img1 = self.raw1_tophat
            img2 = self.raw2_tophat
        else:
            img1 = self.raw1
            img2 = self.raw2
        #np.savetxt('img1.csv', img1, delimiter=',')
        #np.savetxt('img2.csv', img2, delimiter=',')
        
        # Define constants and Parameter TODO: maybe make them attributes of the class
        corr_length = 4  # [μm]
        crop_length = round(corr_length / (2 * self.pixSize))  # [pixel]
        max_dist = round(crop_length) + 10  # +10 pixels to corr_length to smaller deviations
        maxlag = 2 * max_dist  # Correlation length will be doubled because of mirror symmetry

        ccf_all = np.full((self.nGridY * self.nGridX, 2 * maxlag + 1), np.nan)  # Allocate memory for ACF/CCF
        
        # Computation over the whole grid
        for i in range(self.nGridY):
            for j in range(self.nGridX):
                if self.raw_roi_mask[i, j] == 1:
                    ccf = np.zeros(2*maxlag+1)
                    var1 = 0
                    var2 = 0
                    kmax = 10
                    Angle = self.anglemap[i, j]
                    for k in range(-kmax, kmax+1, 2):
                        dir = np.array([-1, 1])
                        xi = self.roi_size*(j+1) + max_dist*dir*np.cos((90-Angle)*-1*np.pi/180) + k*np.cos(Angle*np.pi/180)
                        yi = self.roi_size*(i+1) + max_dist*dir*np.sin((90-Angle)*-1*np.pi/180) + k*np.sin(Angle*np.pi/180) # pretty much the same as in matlab (see comp.ipynb)
                        # create intensity profiles along the line defined by xi and yi with 2*max_dist+1 points
                        linescan1 = improfile(img1, xi,yi, 2*max_dist+1)
                        linescan2 = improfile(img2, xi,yi, 2*max_dist+1)
                            
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
                    ccf_all[i * self.nGridX + j, :] = ccf

                    maxlag_plot = len(linescan1) - 1
                    lags = np.arange(-maxlag, maxlag+1) * self.pixSize
                    ind = np.arange(maxlag_plot, 2*maxlag_plot+1)

        # Find the valid CCFs/ACFs and calculate the mean
        self.ccf_all = ccf_all
        
        # remove nan linescans
        self.ccf_mask = ~np.isnan(ccf_all).any(axis=1)
        self.ccf_all_valid = ccf_all[self.ccf_mask]
        
        self.mean_ccf = np.mean(self.ccf_all_valid, axis=0)
        self.std_mean_ccf = np.std(self.ccf_all_valid, axis=0, ddof=1)
        self.lags = lags
        self.ind = ind
        self.corr_length = corr_length
        
        if plot:
            self.plot_cross_correlation()
            
    def plot_cross_correlation(self):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if len(self.roi_instances) > 0:
            all_ccf = []
            for i in range(len(self.roi_instances)):
                roi = self.roi_instances[i]
                all_ccf.append(roi.ccf_all)
                lags = roi.lags[roi.ind]
                self.ind_roi = roi.ind
                ax.plot(np.tile(lags, (len(roi.ccf_all), 1)).T, roi.ccf_all[:, roi.ind].T, color='#cccaca')
            all_ccf_roi = np.concatenate(all_ccf, axis=0)
            
            # Calculate the mean and standard deviation of the combined array
            self.mean_ccf_all_roi = np.nanmean(all_ccf_roi, axis=0)
            self.std_mean_ccf_all_roi = np.nanstd(all_ccf_roi, axis=0, ddof=1)
            ax.plot(lags, self.mean_ccf_all_roi[self.ind_roi], '-', color='#d13111', linewidth=1.8)
            ax.plot(lags, self.mean_ccf_all_roi[self.ind_roi] - self.std_mean_ccf_all_roi[self.ind_roi], '--', color='#d13111', linewidth=1.8)
            ax.plot(lags, self.mean_ccf_all_roi[self.ind_roi] + self.std_mean_ccf_all_roi[self.ind_roi], '--', color='#d13111', linewidth=1.8)
        else:
            lags = self.lags[self.ind]
	        #df = pd.DataFrame()
            #df['lags'] = self.lags[self.ind]
            ax.plot(np.tile(lags, (len(self.ccf_all_valid), 1)).T, self.ccf_all_valid[:, self.ind].T, color='#cccaca')
            ax.plot(lags, self.mean_ccf[self.ind], '-', color='#d13111', linewidth=1.8)
            ax.plot(lags, self.mean_ccf[self.ind] - self.std_mean_ccf[self.ind], '--', color='#d13111', linewidth=1.8)
            ax.plot(lags, self.mean_ccf[self.ind] + self.std_mean_ccf[self.ind], '--', color='#d13111', linewidth=1.8)
        
        ax.set_xlabel(r'$\Delta$ x [$\mu$m]')
        ax.set_xlim(0,self.corr_length)
        ax.set_ylim(-0.5,1)
        ax.set_ylabel('CCF')
        
        # Add mean_ccf, mean_ccf - std_mean_ccf, and mean_ccf + std_mean_ccf to the DataFrame
        #df['mean_ccf'] = self.mean_ccf[self.ind]
        #df['mean_ccf_minus_std'] = self.mean_ccf[self.ind] - self.std_mean_ccf[self.ind]
        #df['mean_ccf_plus_std'] = self.mean_ccf[self.ind] + self.std_mean_ccf[self.ind]

        # Save the DataFrame to a CSV file
        #df.to_csv('comparison/ccf_Parameter.csv', index=False)
        plt.savefig('ccf_human.png', dpi=300)
        plt.show()
    
    def calc_fourier_peak_one_ccf(self, i, plot=False, upper_limit=3):
        """
        Calculate the Fourier peak of one cross-correlation function (CCF).

        Parameters:
            - i (int): Index of the CCF to analyze.
            - plot (bool, optional): If True, plot the CCF and the fitted sine wave. Defaults to False.
            - upper_limit (float, optional): Upper limit for the lags. Defaults to 3 \mu m.

        Returns:
            - popt (array): Optimal values for the parameters of the sine function that fits the CCF.
            - pcov (2d array): The estimated covariance of popt.
            - mse (float): Mean squared error of the fit.

        This function first crops the CCF around its first minima and maxima. The cropping is done iteratively, 
        starting with a prominence percentile of 80 and decreasing by 5 each time an error occurs during cropping.
        After a successful crop, the function estimates the frequency of the CCF using FFT and fits a sine function to it.
        If the amplitude or wavelength of the fit is too large, the function increases the cropping factor that effectively enlarges the cropped region and tries again.
        If the cropping factor exceeds 0.5, the function gives up and returns None for all outputs.
        """
        if not self.ccf_mask[i]:
            return None, None, None
        ccf = self.ccf_all[i, self.ind] # y
        lags = self.lags[self.ind] # x
        # crop ccf around the first minima and maxima
        percent = 80
        factor = 1/10
        invalid_fit = True
        clear = False
        while invalid_fit:
            invalid_crop = True
            lags_crop = None
            while invalid_crop and percent > 50:
                try:  
                    prominence = np.percentile(ccf, percent)
                    if upper_limit is not None:
                        lags_tmp = lags[lags<upper_limit]
                        ccf_tmp = ccf[lags<upper_limit]
                    else:
                        lags_tmp = lags
                        ccf_tmp = ccf
                    if not plot:
                        start_idx, end_idx, diff, [idx_1, idx_2, idx_3],_= crop_around_extrema(lags_tmp, ccf_tmp, prominence, factor)
                    else:
                        if clear:
                            ax.clear()
                        else:
                            fig, ax = plt.subplots(figsize=(12, 10))
                        start_idx, end_idx, diff, [idx_1, idx_2, idx_3], (fig, ax)= crop_around_extrema(lags_tmp, ccf_tmp, prominence, factor, (fig, ax))
                    lags_crop = lags[start_idx:end_idx]
                    ccf_crop = ccf[start_idx:end_idx]
                    invalid_crop = False
                except Exception as e:
                    print(f"An error occurred during cropping for {i}th ccf: ", e)
                    percent -= 5
                    invalid_crop = True
                    clear = True
        
            if lags_crop is None:
                return None, None, None
            
            try:
                guess = None
                try:
                    # Estimate frequency using FFT
                    N = len(ccf_crop)
                    f = np.linspace(0, 1, N)  # Frequency range
                    yf = fft(ccf_crop)
                    estimate_f = f[np.argmax(np.abs(yf[1:N//2]))] 
                    
                    # Initial guess for the Parameter
                    guess = [np.mean(ccf_crop), np.std(ccf_crop), estimate_f, 0]
                except Exception as er:
                    print(f"An error occurred during FFT for {i}th ccf: ", er)
                trend, slope, intercept = get_linear_trend(lags, ccf)
                trend_crop = trend[start_idx:end_idx]
                popt, pcov, mse = sine_fit(lags_crop, ccf_crop, trend=trend_crop, guess=guess)
                
                if np.abs(popt[1])>1.3*diff/2:
                    print(f"An error occurred during sine fitting for {i}th ccf: ", f'amplitude of fit is too large ({round(np.abs(popt[1]), 2)} >> {round(diff/2, 2)})')
                    factor += 1/10
                    clear = True
                    if factor > 1/2:
                        return None, None, None
                elif np.abs(1/(popt[2]))>1.5* (lags[idx_3]-lags[idx_1]):
                    print(f"An error occurred during sine fitting for {i}th ccf: ", f'wavelength of fit is too large ({round(np.abs(1/popt[2]), 2)} >> {round(lags[idx_3]-lags[idx_1], 2)})')
                    factor += 1/10
                    clear = True
                    if factor > 1/2:
                        return None, None, None
                else:
                    invalid_fit = False
            
            except Exception as e:
                invalid_fit = False
                print(f"An error occurred during sine fitting for {i}th ccf: ", e)
                #fig, ax = plt.subplots(figsize=(12, 10))
                #ax.plot(lags, ccf,'o', color='#cccaca')
                #ax.vlines(lags_crop[0], min(ccf), max(ccf), color='red')
                #ax.vlines(lags_crop[-1], min(ccf), max(ccf), color='red')
                #plt.show()
                return None, None, None
        if plot:
            x_fit = np.linspace(lags_crop[0]-0.5 if lags_crop[0]-0.5 >0 else 0, lags_crop[-1]+0.5 if lags_crop[-1]+0.5 < lags[-1] else lags[-1], 100)
            if upper_limit is not None:
                x_upper = lags[lags>upper_limit]
                y_upper = ccf[lags>upper_limit]
                ax.plot(x_upper, y_upper, 'o', color='#1f77b4')
            y_fit = sine_func(x_fit, *popt)
            ax.plot(lags, slope*lags+intercept, color='orange', linestyle='--')
            ax.plot(x_fit, y_fit, color='r', linestyle='--')
            tmp = f'{i}th ccf: \n amp: {round(popt[1], 2)}, freq: {round(popt[2], 2)}, phase: {round(popt[3], 2)}'
            fig.text(0.5, 0.02, tmp, ha='center')
            plt.show()
        return popt, pcov, mse
        
    def calc_fourier_peaks(self, plot=False):
        if len(self.roi_instances)==0:
            self.__calc_fourier_peaks(plot)
        else:
            for i in range(len(self.roi_instances)):
                self.roi_instances[i].__calc_fourier_peaks()
            if plot:
                omega_arr = []
                amplitude_arr = []
                for i in range(len(self.roi_instances)):
                    omega_arr.extend(self.roi_instances[i].omega[self.roi_instances[i].ccf_mask & self.roi_instances[i].ccf_fit_valid])
                    amplitude_arr.extend(self.roi_instances[i].amplitude[self.roi_instances[i].ccf_mask & self.roi_instances[i].ccf_fit_valid])
                omega = np.array(omega_arr).flatten()
                amplitude = np.array(amplitude_arr).flatten()
                plot_fourier_peaks(omega, amplitude)
                    
            
        
    def __calc_fourier_peaks(self, plot=False):
        n = self.ccf_all.shape[0]
        
        omega = np.zeros(n)
        amplitude = np.zeros(n)
        mse = np.zeros(n)
        ccf_fit_valid = np.ones(n, dtype=bool)
        
        for i in range(n):
            if self.ccf_mask[i]:
                popt, _, mse_i = self.calc_fourier_peak_one_ccf(i)
                if popt is not None:
                    omega[i] = popt[2]
                    amplitude[i] = popt[1]
                    mse[i] = mse_i
                else:
                    ccf_fit_valid[i] = False
                    
        omega_valid = omega[ccf_fit_valid & self.ccf_mask]
        mse_valid = mse[ccf_fit_valid & self.ccf_mask]
        amplitude_valid = amplitude[ccf_fit_valid & self.ccf_mask]
        
        n_valid = len(omega_valid)
        print(f'fittings found for {n_valid} of {self.ccf_all_valid.shape[0]} valid ccfs')
        
        self.ccf_fit_valid = ccf_fit_valid
        self.amplitude = amplitude
        self.omega = omega
        if plot:
            plot_fourier_peaks(omega_valid, amplitude_valid)
            
    def get_amplitude_top_x(self, x=0.1):
        """
        Retrieve the minimum Fourier peak amplitude from the top x% of ROIs, ranked by their Fourier peak amplitudes.

        Parameters:
            - x: The percentage of ROIs to consider, selected based on their Fourier peak amplitudes.

        Returns:
            - min_amp: The minimum Fourier peak amplitude among the selected top x% of ROIs.
        """
        
        if not hasattr(self, 'amplitude'):
            self.calc_fourier_peaks()
            
        # get indices of top x% highest amplitudes
        if len(self.roi_instances) > 0:
            amp_valid = []
            for i in range(len(self.roi_instances)):
                roi = self.roi_instances[i]
                amp_valid_tmp = roi.amplitude[roi.ccf_mask & roi.ccf_fit_valid]
                amp_valid.extend(amp_valid_tmp)
            n = len(amp_valid)
        else:
            amp_valid = self.amplitude[self.ccf_mask & self.ccf_fit_valid]
            n = len(amp_valid)
            
        n_top = int(n*x)
        self.top_x = x
        return np.sort(amp_valid)[-n_top]
    
    def set_idx_top_x_roi(self, min_amp):
        """
        Identify the ROIs with Fourier peak amplitudes above a threshold value.
        
        Parameter:
            - min_amp: The minimum Fourier peak amplitude to consider.
        
        This method updates the following instance variables:
            - idx_top: The indices of the ROIs with Fourier peak amplitudes above the threshold value.
        """
        if len(self.roi_instances) > 0:
            for i in range(len(self.roi_instances)):
                self.roi_instances[i].set_idx_top_x_roi(min_amp)
        else:
            amp = self.amplitude
            idx = np.where(amp >= min_amp)[0]
            self.idx_top = idx
        
    def plot_top_x_roi(self, x=0.1, fig_ax=None):
        """
        Plot the top x% of ROIs, ranked by their Fourier peak amplitudes.
        
        Parameter:
            - x: The percentage of ROIs to consider, selected based on their Fourier peak amplitudes.
        """
        if not hasattr(self, 'idx_top'):
            min_amp = self.get_amplitude_top_x(x)
            self.set_idx_top_x_roi(min_amp)
        if hasattr(self, 'top_x') and self.top_x != x:
            min_amp = self.get_amplitude_top_x(x)
            self.set_idx_top_x_roi(min_amp)
        if fig_ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        else:
            fig, ax = fig_ax
        ax.imshow(self.raw, cmap='gray')
        if len(self.roi_instances) == 0:
            idx = self.idx_top
            # get rectangles of corresponding ROIs
            for n in range(len(idx)):
                i, j = np.unravel_index(idx[n], (self.nGridY, self.nGridX))
                rect = Rectangle((j*self.roi_size, i*self.roi_size), self.roi_size, self.roi_size, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        else:
        
            for i in range(len(self.roi_instances)):
                roi = self.roi_instances[i]
                idx = roi.idx_top
                # get rectangles of corresponding ROIs
                for n in range(len(idx)):
                    i, j = np.unravel_index(idx[n], (roi.nGridY, roi.nGridX))
                    rect = Rectangle((roi.nx0*self.roi_size+j*roi.roi_size, roi.ny0*self.roi_size+i*roi.roi_size), roi.roi_size, roi.roi_size, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                
        ax.set_title(f'Top {x*100}% of ROIs')
        if fig_ax is None:
            plt.show()
        else:
            return fig, ax
        
    def plot_top_x_roi_interactive(self, x=0.1):
        fig, ax = plt.subplots(figsize=(12, 10))
        fig, ax = self.plot_top_x_roi(x, (fig, ax))
        
        n_click = 0
        def onclick(event):
            nonlocal n_click, fig, ax, self
            x, y = event.xdata, event.ydata
            if 0 < x < self.nGridX*self.roi_size and 0 < y < self.nGridY*self.roi_size:
                if len(self.roi_instances)>0:
                    for i in range(len(self.roi_instances)):
                        roi = self.roi_instances[i]
                        if inside_rectangle((x, y), self.roi_array[i]):
                            
                            i = int(np.floor((y - roi.ny0*self.roi_size)/roi.roi_size))
                            j = int(np.floor((x - roi.nx0*self.roi_size)/roi.roi_size))
                            idx_n = np.ravel_multi_index((i, j), (roi.nGridY, roi.nGridX))
                            rect = Rectangle((roi.nx0*self.roi_size+j*roi.roi_size, roi.ny0*self.roi_size+i*roi.roi_size), roi.roi_size, roi.roi_size, linewidth=1, edgecolor='r', facecolor='r', alpha=0.4)
                            if n_click >0:
                                # remove last patch from ax
                                ax.patches[-1].remove()
                            ax.add_patch(rect)
                            if roi.ccf_mask[idx_n]:
                                roi.calc_fourier_peak_one_ccf(idx_n, plot=True)
                            n_click += 1
                            # Redraw the figure
                            fig.canvas.draw()
                else:
                    i = int(np.floor(y/self.roi_size))
                    j = int(np.floor(x/self.roi_size))
                    idx_n = np.ravel_multi_index((i, j), (self.nGridY, self.nGridX))
                    rect = Rectangle((j*self.roi_size, i*self.roi_size), self.roi_size, self.roi_size, linewidth=1, edgecolor='r', facecolor='r', alpha=0.4)
                    if n_click >0:
                        # remove last patch from ax
                        ax.patches[-1].remove()
                    ax.add_patch(rect)
                    if self.ccf_mask[idx_n]:
                        self.calc_fourier_peak_one_ccf(idx_n, plot=True)
                    n_click += 1
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
                            
                            
                
            
        
        
            
    def select_roi_manually(self):
        """
        Allows the user to manually select rectangluar regions of interest (ROI) in the image self.raw.
        
        This method updates the following instance variables:
            - roi_array: The array of manually created ROIs. Containes the coordinates of the four corners of each ROI. [[(x1,x2),(y1,y2),(x3,x4),(y3,y4)],...]
            - roi_instances: The array of ROI instances.
        """
        tmp = ROISelector(self.raw)
        roi_array = tmp.roi_array
        if len(roi_array) == 0:
            return
        roi_instances = np.empty(len(roi_array), dtype=object)
        for i in range(len(roi_array)):
            roi_instances[i] = ROI(roi_array[i], self, i)
            
        # check if roi_array is empty
        if len(self.roi_array) == 0:
            self.roi_array = roi_array
        else:
            self.roi_array = np.concatenate((self.roi_array, roi_array))
            
        # create ROI instances
        # check if roi_instances is empty
        if len(self.roi_instances) == 0:
            self.roi_instances = roi_instances
        else:
            self.roi_instances = np.concatenate((self.roi_instances, roi_instances))
        
        
    def get_parameter_dict(self):
        """
        Return a dictionary containing the Parameter of the ImageAnalysis instance.
        """
        return vars(self)
        
        
    def choose_parameter_mask(self):
        """
        This method allows the user to interactively choose parameter values for mask generation.
        It creates sliders for each parameter and updates the mask and ROI mask images whenever a slider value changes.

        The method updates the following instance variables:
            - mask_thresh: The threshold value for the mask.
            - roi_thresh: The threshold value for the ROI.
            - roi_size: The size of the ROI.
            - gblur_sigma: The sigma value for the Gaussian blur.
            - raw_mask: The binary mask image.
            - raw_roi_mask: The ROI mask image.
        """

        # Create a new figure
        fig = plt.figure()

        # Create sliders for each parameter
        mask_thresh_slider_ax = plt.axes([0.2, 0.02, 0.65, 0.03])
        mask_thresh_frame_slider = Slider(mask_thresh_slider_ax, 'mask threshhold', 0, 1000, valinit=self.mask_thresh, valstep=50)

        roi_thresh_slider_ax = plt.axes([0.2, 0.08, 0.65, 0.03])
        roi_thresh_frame_slider = Slider(roi_thresh_slider_ax, 'roi threshhold', 0, 0.85, valinit=self.roi_thresh, valstep=0.05)

        roi_size_slider_ax = plt.axes([0.2, 0.14, 0.65, 0.03])
        roi_size_frame_slider = Slider(roi_size_slider_ax, 'roi size', 5, 50, valinit=self.roi_size, valstep=5)

        gblur_sigma_slider_ax = plt.axes([0.2, 0.2, 0.65, 0.03])
        gblur_sigma_frame_slider = Slider(gblur_sigma_slider_ax, 'gblur sigma', 0.5, 4, valinit=self.gblur_sigma, valstep=0.5)

        # Generate the mask and ROI mask images
        self.mask_background()
        if hasattr(self, 'nuclei_mask'):
            param = [self.raw_mask, self.raw_roi_mask, self.mask_thresh, self.roi_size, self.roi_thresh, self.gblur_sigma, self.roi_instances, self.nuclei_mask, fig]
        else:
            param = [self.raw_mask, self.raw_roi_mask, self.mask_thresh, self.roi_size, self.roi_thresh, self.gblur_sigma, self.roi_instances,None, fig]
        mask_display, roi_mask_display, ax1, ax2 = plot_mask_background(*param)

        # Define a function to update the images whenever a slider value changes
        def update_image(val, param):
            if param == 'mask_thresh':
                self.mask_thresh = val
            elif param == 'roi_thresh':
                self.roi_thresh = val
            elif param == 'roi_size':
                self.roi_size = val
            elif param == 'gblur_sigma':
                self.gblur_sigma = val
            self.mask_background()
            mask_display.set_data(self.raw_mask)
            roi_mask_display.set_data(self.raw_roi_mask)
            ax1.set_title(fr'mask (thresh. = {self.mask_thresh}), $\sigma_{{gb}}$ = {self.gblur_sigma})')
            ax2.set_title(f'ROIMask (size = {self.roi_size}, thresh. = {self.roi_thresh})')
            fig.canvas.draw_idle()

        # Connect the update function to the sliders
        mask_thresh_frame_slider.on_changed(lambda val: update_image(val, 'mask_thresh'))
        roi_thresh_frame_slider.on_changed(lambda val: update_image(val, 'roi_thresh'))
        roi_size_frame_slider.on_changed(lambda val: update_image(val, 'roi_size'))
        gblur_sigma_frame_slider.on_changed(lambda val: update_image(val, 'gblur_sigma'))

        # Display the figure
        plt.show()
        
    def choose_steerable_sigma(self):
        """
        This method allows the user to interactively choose the sigma value for a steerable filter.
        It creates a slider for the sigma value and updates the images whenever the slider value changes.

        The method updates the following instance variable:
            - steerable_sigma: The sigma value for the steerable filter.
            - res: The response of the steerable filter
            - rot: Response of the input image to rotated versions of the filter, at 'nAngles' different angles.
            - xy_values_k0: cordinates of the start and endpoints of detected nematic ordered structures in the image.
            - anglemap: The angle map.
        """

        # Create a new figure
        fig = plt.figure()

        # Create a slider for the sigma value
        steerable_sigma_slider_ax = plt.axes([0.2, 0.08, 0.65, 0.03])
        steerable_sigma_slider = Slider(steerable_sigma_slider_ax, 'steerable sigma', 1, 4, valinit=self.steerable_sigma, valstep=0.5)

        # Apply the steerable filter if it hasn't been applied yet
        if not self.steerable_bool:
            self.__apply_steerable_filter()

        # Load the results from a CSV file
        self.load_res_from_csv()

        # Display the images
        fig, _, res1_display, res2_display, ax1, ax2, ax3 = plot_steerable_filter_results(self.raw, self.res, self.xy_values_k0, self.steerable_sigma,self.roi_instances, fig)
        qlow = np.quantile(self.raw, 0.001)
        qhigh = np.quantile(self.raw, 0.999)

        # Define a function to update the images whenever the slider value changes
        def update_image(val):
            self.steerable_sigma = val
            self.__apply_steerable_filter()
            self.load_res_from_csv()
            res1_display.set_data(self.res)
            raw_new = np.ones((self.nX, self.nY))
            res2_display.set_data(raw_new)
            ax3.imshow(self.raw, cmap='gray', vmin=qlow, vmax=qhigh)
            ax1.set_title(f'Raw')
            ax2.set_title(f'Steerable filter (sigma = {self.steerable_sigma} pix.)')

            # Remove all lines from ax3
            while len(ax3.lines) > 0:
                line = ax3.lines[0]
                line.remove()
            x_start, x_end = self.xy_values_k0[:, 0].T
            y_start, y_end = self.xy_values_k0[:, 1].T
            ax3.plot([x_start, x_end], [y_start, y_end], 'c')
            fig.canvas.draw_idle()

        # Connect the update function to the slider
        steerable_sigma_slider.on_changed(lambda val: update_image(val))

        # Display the figure
        plt.show()
        
def plot_steerable_filter_roi(roi_instances, ax):
    """
    Plot the ROIs in the image self.raw.
    
    Parameter:
        - roi_instances: The ROI instances to be plotted.
        - ax: The axes object to plot the ROIs on.
    
    Returns:
        - ax: The axes object with the ROIs plotted on it.
    """
    
    if len(roi_instances)==0:
        return
        
    for i in range(len(roi_instances)):
        roi = roi_instances[i]
        xy_values = roi.xy_values_superclass
        x_values = xy_values[:, 0].reshape(-1, 2)
        x_start, x_end = x_values.T
        
        y_values = xy_values[:, 1].reshape(-1, 2)
        y_start, y_end = y_values.T
        
        ax.plot([x_start, x_end],[y_start, y_end], 'orange')
    return ax
            
class ROI(ImageAnalysis):
    """
    This class represents one region of interest (ROI) which is represented by a rotated rectangle. Based on a  general orientation of structures in this ROI a more accurate orientation is detected.
     
     Instance variables:
        - corners: The four corners of the ROI.
        - angle0: The general orientation of the structures in the ROI.
    
    """
    
    def __init__(self, corner_array, image_analysis, i):
        if not isinstance(image_analysis, ImageAnalysis):
            raise TypeError('image_analysis must be an instance of ImageAnalysis')
        # run init of superclass --> copy all attributes
        super().__init__(image_analysis.get_parameter_dict())
        self.i = i
        self.corners = corner_array
        self.roi_size = 0
        self.pad_angle = 0
        self.set_superclass(image_analysis)
        self.nAngles = 30
        
    def __apply_steerable_filter(self, angle_array=None, save=True):
        return self._ImageAnalysis__apply_steerable_filter(angle_array, save)
    
    def __calculate_own_cross_correlation(self, tophat=True,  plot=True):
        return self._ImageAnalysis__calculate_own_cross_correlation(tophat, plot)
    
    def __calc_fourier_peaks(self, plot=False):
        return self._ImageAnalysis__calc_fourier_peaks(plot)
        
    def calc_area(self):
        """
        Caculates an area around the ROI (length and width are multiples of the roi_size of the superclass) which is later used to crop the images of the superclass.
        The area is defined as follows:
    
        
                 (min_x_area, max_y_area)      (x2, max_y)                   (max_x_area, max_y_area)
                            -------------------------------------------------------------                
                            |                    *                                 ^    |                ^
                            |                  *     *                             |    |                |
                            |                *           *                         |    |                | 
                            |              *                 *                     |    |                | 
                            |            *                       *                 |    |                |
                            |          *                             * (max_x, y3) |    |                |
                            |        *                             *               |    |                |
                            |      *                             *                 |dy  |                |
                            |    *             ROI             *                   |    |                |
                            |  *                             *                     |    |                |  n_roi_y * image_analysis.roi_size
                (min_x, y1) |*                             *                       |    |                |
                            |    *                       *                         |    |                |
                            |        *                 *                           |    |                |
                            |            *           *                             |    |                |
                            |                *     *                               |    |                |
                            |                    *                                 v    |                |
                            |               (x4, min_y)                                 |                |
                            |<--------------------------------------->                  |                |
                            |                   dx                                      |                |
                            |                                                           |                |
                            -------------------------------------------------------------                v
                 (min_x_area, min_y_area)                                    (max_x_area, min_y_area)    
                            
                            <----------------------------------------------------------->
                                            n_roi_x * image_analysis.roi_size
        Further Parameter:
            - dx: The width of the ROI.
        """
        
        # find min and max of x and y coordinates of the corners of the ROI
        min_x = min(x for x, y in self.corners)
        max_x = max(x for x, y in self.corners)
        min_y = min(y for x, y in self.corners)
        max_y = max(y for x, y in self.corners)
        
        self.dx, self.dy = max_x-min_x, max_y-min_y
        
        # check if part of the ROI is outside of the image
        
        if min_x<0:
            min_x = 0
        if min_y<0:
            min_y = 0
        if max_x>self.image_analysis.nX:
            max_x = self.image_analysis.nX
        if max_y>self.image_analysis.nY:
            max_y = self.image_analysis.nY
 
        
        min_x_area = int(np.floor(min_x / self.image_analysis.roi_size) * self.image_analysis.roi_size)
        diff_x = min_x - min_x_area
        
        min_y_area = int(np.floor(min_y / self.image_analysis.roi_size) * self.image_analysis.roi_size)
        diff_y = min_y - min_y_area
        
        n_roi_x = int(np.ceil((self.dx + diff_x) / self.image_analysis.roi_size))
        n_roi_y = int(np.ceil((self.dy+ diff_y) / self.image_analysis.roi_size))
        
        # slice indices for anglemap
        self.nx0 = int(min_x_area / self.image_analysis.roi_size)
        self.nx1 = int(self.nx0 + n_roi_x)
        
        self.ny0 = int(min_y_area / self.image_analysis.roi_size)
        self.ny1 = int(self.ny0 + n_roi_y)
        
        max_x_area = min_x_area + n_roi_x * self.image_analysis.roi_size
        max_y_area = min_y_area + n_roi_y * self.image_analysis.roi_size
        
        self.x_borders = [min_x_area, max_x_area]
        self.y_borders = [min_y_area, max_y_area]
        self.nX = max_x_area - min_x_area
        self.nY = max_y_area - min_y_area
        if self.roi_size != 0:
            roi_factor = self.image_analysis.roi_size / self.roi_size
            
            if min_x_area < self.image_analysis.pad_angle_left*self.image_analysis.roi_size:
                diff = self.image_analysis.pad_angle_left*self.image_analysis.roi_size - min_x_area
                self.pad_angle_left = int(np.floor(diff / self.roi_size))
            else:
                self.pad_angle_left = self.pad_angle
                
            if max_x_area > (self.image_analysis.nX - self.image_analysis.pad_angle_right*self.image_analysis.roi_size):
                diff = max_x_area - (self.image_analysis.nX - self.image_analysis.pad_angle_right*self.image_analysis.roi_size)
                self.pad_angle_right = int(np.floor(diff / self.roi_size))
            else:
                self.pad_angle_right = self.pad_angle
                
            if max_y_area > (self.image_analysis.nY - self.image_analysis.pad_angle_bottom*self.image_analysis.roi_size):
                diff = max_y_area - (self.image_analysis.nY - self.image_analysis.pad_angle_bottom*self.image_analysis.roi_size)
                self.pad_angle_bottom = int(np.floor(diff / self.roi_size))
            else:
                self.pad_angle_bottom = self.pad_angle
                
            if min_y_area < self.image_analysis.pad_angle_top*self.image_analysis.roi_size:
                diff = self.image_analysis.pad_angle_top*self.image_analysis.roi_size - min_y_area
                self.pad_angle_top = int(np.floor(diff / self.roi_size))
            else:
                self.pad_angle_top = self.pad_angle

        
            
    def set_superclass(self, image_analysis):
        """
        Set the superclass for this instance and perform initial calculations.
        """
        # Check if image_analysis is an instance of ImageAnalysis
        if not isinstance(image_analysis, ImageAnalysis):
            raise TypeError('image_analysis must be an instance of ImageAnalysis')
        
        # Set the superclass
        self.image_analysis = image_analysis
        # Perform initial calculations
        self.calc_area()
        self.crop_images_superclass()
    

    def crop_images_superclass(self):
        """
        Crop the images of the superclass according to the set borders.
        """
        # Check if raw image is set
        if len(self.image_analysis.raw)==0:
            raise ValueError('raw image of superclass is not yet set!')
        else:
            # Crop the images
            self.raw = self.image_analysis.raw[self.y_borders[0]:self.y_borders[1], self.x_borders[0]:self.x_borders[1]]
            if len(self.image_analysis.raw_tophat)>0:
                self.raw_tophat = self.image_analysis.raw_tophat[self.y_borders[0]:self.y_borders[1], self.x_borders[0]:self.x_borders[1]]
            if len(self.image_analysis.raw_mask)>0:
                self.raw_mask = self.image_analysis.raw_mask_array[self.i][self.y_borders[0]:self.y_borders[1], self.x_borders[0]:self.x_borders[1]]
            if len(self.image_analysis.raw_bgsub)>0:
                self.raw_bgsub = self.image_analysis.raw_bgsub[self.y_borders[0]:self.y_borders[1], self.x_borders[0]:self.x_borders[1]]
            if len(self.image_analysis.raw1)>0:
                self.raw1 = self.image_analysis.raw1[self.y_borders[0]:self.y_borders[1], self.x_borders[0]:self.x_borders[1]]
            if len(self.image_analysis.raw1_tophat)>0:
                self.raw1_tophat = self.image_analysis.raw1_tophat[self.y_borders[0]:self.y_borders[1], self.x_borders[0]:self.x_borders[1]]
            if len(self.image_analysis.raw2)>0:
                self.raw2 = self.image_analysis.raw2[self.y_borders[0]:self.y_borders[1], self.x_borders[0]:self.x_borders[1]]
            if len(self.image_analysis.raw2_tophat)>0:
                self.raw2_tophat = self.image_analysis.raw2_tophat[self.y_borders[0]:self.y_borders[1], self.x_borders[0]:self.x_borders[1]]
            if hasattr(self.image_analysis, 'raw_nuclei'):
                self.raw_nuclei = self.image_analysis.raw_nuclei[self.y_borders[0]:self.y_borders[1], self.x_borders[0]:self.x_borders[1]]
        
    def set_roi_size(self, roi_size):
        """
        Set the size of the ROI for this instance.
        """
        if roi_size > self.image_analysis.roi_size:
            raise ValueError('roi_size must be smaller than the image_analysis.roi_size')
        self.roi_size = roi_size
        
        self.nGridY = int(np.ceil(self.nY / self.roi_size))
        self.nGridX = int(np.ceil(self.nX / self.roi_size))

        # Crop the raw image to fit an integer number of ROIs
        self.raw_crop = np.copy(self.raw[:self.nGridY * self.roi_size, :self.nGridX * self.roi_size])
        self.nYCrop, self.nXCrop = self.raw_crop.shape

    def apply_steerable_filter_finetuning(self, plot=True):
        """
        This method applies a steerable filter to the image self.raw. It calculates the response of the filter, the response of the input image to rotated versions of the filter, and the angle map.
        
        The method updates the following instance variables:
            - res: filter response
            - rot: response of the input image to rotated versions of the filter, at 'nAngles' different angles
            - anglemap: Map of the angles of the maximum response for each ROI
            - xy_values_k0: start xi and endposition yi of each detected stucture( xy_values_k0[t, 0] = xi, xy_values_k0[t, 1] = yi)
            - steerable_bool: Flag indicating that the steerable filter has been applied
        """
        roi_mask_superclass = self.image_analysis.raw_roi_mask_array[self.i].copy()
        anglemap = self.image_analysis.anglemap.copy()
        anglemap[roi_mask_superclass == 0] = np.nan
        # TODO: add check if steerable filter has already been applied
        
        self.create_roi_mask() 
        anglemap = anglemap[self.ny0:self.ny1, self.nx0:self.nx1]
        
        # Get only valid angles
        angles_deg = anglemap[~np.isnan(anglemap)]
        angles_rad = angles_deg * np.pi / 180
        
        #np.savetxt('comparison/angles_deg.csv', anglemap_nonzero, delimiter=',')
        angles_rad, labels, mean_arr, std_arr, perc_arr = find_cluster(angles_rad, 3, min_diff_rad=30*np.pi/180, angle_max=np.pi, print_=True)
        idx = np.argmax(perc_arr)
        
        mean = mean_arr[idx]*180/np.pi
        std = std_arr[idx]*180/np.pi
        
        min_angle = mean - 3*std
        max_angle = mean + 3*std
        if max_angle-min_angle < 10:
            min_angle -=3
            max_angle +=3
        if self.nAngles < 2 * int(np.floor(max_angle - min_angle)):
            self.nAngles = 2 * int(np.floor(max_angle - min_angle))
        angle_array = np.linspace(min_angle, max_angle, self.nAngles, endpoint=True)
        print(f'applying steerable filter with {self.nAngles} angles between {round(min_angle, 2)} and {round(max_angle, 2)} degrees')
        angle_array = angle_array * np.pi / 180
        
        self.__apply_steerable_filter(angle_array)
        if plot:
            plot_steerable_filter_results(self.raw, self.res, self.xy_values_k0, self.steerable_sigma)
 
        self.xy_values_superclass = np.array([(x+self.x_borders[0], y+self.y_borders[0]) for x,y in self.xy_values_k0])
        
def find_cluster(angles_rad, n_clusters, min_diff_rad=0, angle_max=np.pi, print_=False):
    angles_rad = np.fmod(angles_rad, np.pi)
    angles_rad = angles_rad[~np.isnan(angles_rad)]
    angles_rad = angles_rad.flatten().reshape(-1, 1)
    
    points = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])
    
    kmeans = KMeans(n_clusters=n_clusters).fit(points)
    labels = kmeans.labels_
    mean_arr = np.zeros(n_clusters)
    std_arr = np.zeros(n_clusters)
    perc_arr = np.zeros(n_clusters)
    plot_ = False
    if plot_:
        for i in range(n_clusters):
            plt.plot(points[labels == i, 0], points[labels == i, 1], 'o')
        ax = plt.gca()
        ax.set_aspect(1)
        plt.show()
    for i in np.unique(labels):
        cluster_angles_rad = angles_rad[labels == i]
        mean_arr[i] = circmean(cluster_angles_rad, high=angle_max, low=0)
        std_arr[i] = circstd(cluster_angles_rad, high=angle_max, low=0)
        perc_arr[i] = len(cluster_angles_rad)/len(angles_rad)
        if print_:
            print(f'Cluster {i}: mean = {mean_arr[i]*180/np.pi :.2f}, std = {std_arr[i]*180/np.pi :.2f}, percentage = {perc_arr[i]*100 :.2f}%')

    if min_diff_rad > 0:
        # Calculate the absolute difference between each mean and all other means
        diff_matrix = np.abs(mean_arr[:, None] - mean_arr)

        # Get the upper triangle of the matrix excluding the diagonal
        
        diff_matrix = np.triu(diff_matrix, k=1)
        diff_matrix[np.tril_indices(diff_matrix.shape[0])] = np.nan
        close_clusters = np.argwhere(diff_matrix < min_diff_rad)
        
        if n_clusters > 1 and len(close_clusters) > 0:
            if print_:
                print(f'clusters are too close together, reducing number of clusters from {n_clusters} to {n_clusters-1}')
            find_cluster(angles_rad=angles_rad, n_clusters=n_clusters-1, min_diff_rad=min_diff_rad, angle_max=angle_max, print_=print_)
        else:
            return angles_rad, labels, mean_arr, std_arr, perc_arr
    return angles_rad, labels, mean_arr, std_arr, perc_arr
        
        

        
class ROISelector:
    """
    This class allows the user to interactively select regions of interest (ROIs) in an image.
    The user can add one rectangular ROI after clicking once on the "Add ROI" button. Then the next two clicks define one corner of the rectangle and the last click defines the width of the rectangle. The user can remove the last ROI by clicking on the "Remove last ROI" button.
    """

    def __init__(self, raw):
        """
        Initialize the ROISelector with an image.

        Parameter:
            - raw: The image to select ROIs from.
        """
        # Create a figure and axes for the image
        self.fig, self.ax = plt.subplots()

        # Store the image
        self.raw = raw

        # Initialize the click counter and click position
        self.click_counter = 0
        self.click_pos = np.zeros((3, 2))

        # Initialize the list of ROIs
        self.roi_array = []

        # Display the image
        self.ax.imshow(self.raw, extent=[0, self.raw.shape[1], self.raw.shape[0], 0])
        self.ax.set_xlim(0, self.raw.shape[1])
        self.ax.set_ylim(self.raw.shape[0], 0)

        # Create a button for removing the last ROI
        self.remove_roi_button_axis = plt.axes([0.8, 0.05, 0.1, 0.075])
        self.remove_roi_button = Button(self.remove_roi_button_axis, 'Remove last ROI')
        self.remove_roi_button.on_clicked(self.remove_roi_button_click)

        # Create a button for adding a new ROI
        self.add_roi_button_axis = plt.axes([0.2, 0.05, 0.1, 0.075])
        self.add_roi_button = Button(self.add_roi_button_axis, 'Add ROI')
        self.add_roi_button.on_clicked(self.add_roi_button_click)

        # Show the figure
        plt.show()

    def onclick(self, event):
        """
        Handle a click event.

        Parameter:
            - event: The click event.
        """
        # Get the click position
        x, y = event.xdata, event.ydata

        # If the click position is within the image
        if x and y and x >= 0 and x < self.raw.shape[1] and y >= 0 and y <= self.raw.shape[0]:
            # Store the click position
            self.click_pos[self.click_counter] = [x, y]

            # If this is the first or second click, plot a point at the click position
            if self.click_counter in [0, 1]:
                self.ax.plot(x, y, 'ro', markersize=2)
                if self.click_counter == 1:
                    # If this is the second click, plot a line from the first click to the second click
                    self.ax.plot(self.click_pos[0:2, 0], self.click_pos[0:2, 1], 'r-', lw=2)
                self.click_counter += 1
            else:
                # Interpolate a rectangle from the three click positions
                tmp = rectangle_interpolation(self.click_pos[0], self.click_pos[1], self.click_pos[2])

                # Add the new ROI to the list of ROIs
                self.roi_array.append(tmp)

                # Plot the new ROI
                polygon = Polygon(tmp, closed=True, edgecolor='r', fill=False, linewidth=2)
                self.ax.add_patch(polygon)
                # Disconnect the click event
                self.fig.canvas.mpl_disconnect(self.fig.canvas.mpl_connect('button_press_event', self.onclick))
                # Reset the click counter and click positions
                self.click_counter = 0
                self.click_pos = np.zeros((3, 2))

        # Redraw the figure
        self.fig.canvas.draw()

    def remove_roi_button_click(self, event):
        """
        Handle a click event on the "Remove last ROI" button.

        Parameter:
            - event: The click event.
        """
        # If there are any ROIs
        if self.ax.patches:
            # Remove the last ROI
            patch = self.ax.patches[-1]
            patch.remove()
            self.roi_array = self.roi_array[:-1]
        while len(self.ax.lines) > 0:
            line = self.ax.lines[-1]
            line.remove()

        # Redraw the figure
        self.fig.canvas.draw()

    def add_roi_button_click(self, event):
        """
        Handle a click event on the "Add ROI" button.

        Parameter:
            - event: The click event.
        """
        # Connect the click event to the onclick method
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        # Redraw the figure
        self.fig.canvas.draw()
        
def choose_image_interactive(n_channels=4):
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
        mask_channel = 3
    if chosen_image == human_name:
        tmp = '''You chose a image of human muscle.\n \t- slice 0: titin N-terminus (mask channel\n\t- slice 1: muscle myosin\n\t- slice 2: nuclei\n\t- slice 3: actin'''
        print(tmp)
        mask_channel = 0
        

    path = os.path.join(image_dir, chosen_image)
    
    # load the image
    tif = TiffFile(path)
    n_slices = int(len(tif.pages)/n_channels)
    
    # choose the slice to use and the mask channel 
    change = True
    while change:
        choose_image(tif)
        if n_slices == 1:
            slice_index = 0
        else:
            slice_index = int(input('Enter the index of the slice you want to use: '))
        if change and 'mask_channel' not in locals():
            valid = True
            while valid:
                mask_channel = int(input('Enter the index for the mask channel: '))
                if mask_channel > n_channels-1 or mask_channel < 0:
                    print('Invalid channel index. Please try again.')
                else:
                    valid = False
        tmp = input(f'Set channel of image with nuclei? (y/N)')
        if tmp == 'y' or tmp == 'Y':
            if change and 'nuclei_channel' not in locals():
                valid = True
                while valid:
                    nuclei_channel = int(input('Enter the index for the channel of image with nuclei: '))
                    if nuclei_channel > n_channels-1 or nuclei_channel < 0:
                        print('Invalid channel index. Please try again.')
                    else:
                        valid = False
        if change and 'substrate1_channel' not in locals():
            valid = True
            while valid:
                substrate1_channel = int(input('Enter the index for the channel of the first substrate: '))
                if substrate1_channel > n_channels-1 or substrate1_channel < 0:
                    print('Invalid channel index. Please try again.')
                else:
                    valid = False
        if change and 'substrate2_channel' not in locals():
            valid = True
            while valid:
                substrate2_channel = int(input('Enter the index for the channel of the second substrate: '))
                if substrate2_channel > n_channels-1 or substrate2_channel < 0:
                    print('Invalid channel index. Please try again.')
                else:
                    valid = False
        mask_image = np.asarray(tif.pages[n_channels*slice_index+mask_channel].asarray())
        sub1_image = np.asarray(tif.pages[n_channels*slice_index+substrate1_channel].asarray())
        sub2_image = np.asarray(tif.pages[n_channels*slice_index+substrate2_channel].asarray())
        if 'nuclei_channel' in locals():
            fig, ax = plt.subplots(1,4, figsize=(12, 4))
            nuclei_image = np.asarray(tif.pages[n_channels*slice_index+nuclei_channel].asarray())
            ax[3].imshow(nuclei_image, vmin=nuclei_image.min(), vmax=nuclei_image.max())
            ax[3].set_title(f'nuclei image (channel {nuclei_channel})')
        else:
            fig, ax = plt.subplots(1,3, figsize=(12, 3))
        fig.suptitle(f'slice {slice_index}')
        ax[0].imshow(mask_image, vmin=mask_image.min(), vmax=mask_image.max())
        ax[0].set_title(f'mask image (channel {mask_channel})')
        ax[1].imshow(sub1_image, vmin=sub1_image.min(), vmax=sub1_image.max())
        ax[1].set_title(f'substrate 1 image (channel {substrate1_channel})')
        ax[2].imshow(sub2_image, vmin=sub2_image.min(), vmax=sub2_image.max())
        ax[2].set_title(f'substrate 2 image (channel {substrate2_channel})')
        
        plt.show()
        
        proceed = input('Proceed? (Y/n)')
        if proceed == 'n' or proceed == 'N':
            change = True
            del mask_channel, substrate1_channel, substrate2_channel, nuclei_channel
        else:
            change = False
    if 'nuclei_channel' in locals():
        return tif, path, slice_index, mask_channel, substrate1_channel, substrate2_channel, nuclei_channel
    else:
        return tif, path, slice_index, mask_channel, substrate1_channel, substrate2_channel, None
    

def interactive_image_analysis():
    image_analysis = ImageAnalysis()
    tif, path, slice_index, mask_channel, substrate1_channel, substrate2_channel, nuclei_channel = choose_image_interactive()
    raw = np.asarray(tif.pages[4*slice_index+mask_channel].asarray())
    raw_info = tif.pages[4*slice_index+mask_channel].tags
    
    # get image factor
    factor = raw_info['XResolution'].value[0]  # The Xresolution tag is a ratio, so we take the numerator
    pixSize = 1 / factor *10**(6) # Factor to go from pixels to micrometers
    
    image_analysis.set_mask_image(raw, pixSize)
    image1 = np.asarray(tif.pages[4*slice_index + substrate1_channel].asarray())
    image2 = np.asarray(tif.pages[4*slice_index + substrate2_channel].asarray())
    image_analysis.set_subtrate1_image(image1)
    image_analysis.set_substrate2_image(image2)
    if nuclei_channel is not None:
        nuclei_image = np.asarray(tif.pages[4*slice_index + nuclei_channel].asarray())
        image_analysis.set_nuclei_image(nuclei_image)
    tmp = input(f'Configure rectangular regions of interest manually? (y/N)')
    if tmp == 'y' or tmp == 'Y':
        image_analysis.select_roi_manually()
        
    change = True
    image_analysis.mask_background()
    while change:
        tmp =f'''Current parameter for creation of mask and background substraction:\n\t- mask_thresh: {image_analysis.mask_thresh}\n\t- roi_thresh: {image_analysis.roi_thresh}\n\t- roi_size: {image_analysis.roi_size}\n\t- gblur_sigma: {image_analysis.gblur_sigma}'''
        print(tmp)
        if nuclei_channel is not None:
            param = [image_analysis.raw_mask,image_analysis.raw_roi_mask, *image_analysis.get_mask_Parameter(), image_analysis.roi_instances, image_analysis.nuclei_mask]
        else:
            param = [image_analysis.raw_mask,image_analysis.raw_roi_mask, *image_analysis.get_mask_Parameter(), image_analysis.roi_instances]
        plot_mask_background(*param)
        
        change = input('\n Change parameter? (y/N)')

        if change == 'y' or change == 'Y':
            image_analysis.choose_parameter_mask()
            proceed = input(f'New Parameter:\n\t- mask_thresh: {image_analysis.mask_thresh}\n\t- roi_thresh: {image_analysis.roi_thresh}\n\t- roi_size: {image_analysis.roi_size}\n\t- gblur_sigma: {image_analysis.gblur_sigma}\nProceed? (Y/n)')
            if proceed == 'n' or proceed == 'N':
                change = True
            else:
                change = False
        else:
            change = False
            
    
    change = True
    print(f'Current sigma for steerable filter: {image_analysis.steerable_sigma}\n Applying steerable filter...')
    image_analysis.apply_steerable_filter(plot_finetuning=False)
    plot_steerable_filter_results(image_analysis.raw, image_analysis.res, image_analysis.xy_values_k0, image_analysis.steerable_sigma)
    
    while change:
        tmp = input('Change sigma for steerable filter? (y/N)')
        if tmp == 'y' or tmp == 'Y':
            image_analysis.choose_steerable_sigma()
            
            proceed = input(f'Current sigma for steerable filter: {image_analysis.steerable_sigma}\nProceed? (Y/n)')
            if proceed == 'n' or proceed == 'N':
                change = True
            else:
                change = False
        else:
            change = False
    np.savetxt('comparison/anglemap_accum.csv', image_analysis.anglemap, delimiter=',')
    image_analysis.apply_steerable_filter()#int(image_analysis.roi_size/2)
    plot_steerable_filter_results(image_analysis.raw, image_analysis.res, image_analysis.xy_values_k0, image_analysis.steerable_sigma, image_analysis.roi_instances)
    
    
    # calculate cross correlation
    print('Calculating the cross correlation:')
    image_analysis.calc_all_ccf()
    #return image_analysis
    image_analysis.plot_top_x_roi_interactive(0.03)
    save = True
    if save:
        #embed()
        if len(image_analysis.roi_instances) > 0:
            lags = image_analysis.roi_instances[0].lags[image_analysis.roi_instances[0].ind]
            ccf_all = np.concatenate([roi.ccf_all[:,roi.ind] for roi in image_analysis.roi_instances])
            ccf_mask = np.concatenate([roi.ccf_mask for roi in image_analysis.roi_instances])
            ccf_fit_valid = np.concatenate([roi.ccf_fit_valid for roi in image_analysis.roi_instances])
        else:
            lags = image_analysis.lags[image_analysis.ind]
            ccf_all = image_analysis.ccf_all[:,image_analysis.ind]
            ccf_mask = image_analysis.ccf_mask
            ccf_fit_valid = image_analysis.ccf_fit_valid
        
        np.savetxt('comparison/lags.csv', lags, delimiter=',')
        np.savetxt('comparison/ccf_all.csv', ccf_all, delimiter=',')
        np.savetxt('comparison/ccf_mask.csv', ccf_mask, delimiter=',')
        np.savetxt('comparison/ccf_fit_valid.csv', ccf_fit_valid, delimiter=',')
    
    #embed()
def vary_roi_size(min, max, step=1):
    roi_size_arr = np.arange(min, max+step, step)
    var = np.zeros_like(roi_size_arr, dtype=float)
    for i, roi_size in enumerate(roi_size_arr):
        start = time.time()
        path = os.path.join('Data', '2023.06.12_MhcGFPweeP26_30hrsAPF_Phallo568_647nano62actn_405nano2sls_100Xz2.5_1_2.tif')
        #path = os.path.join('Data', 'Trial8_D12_488-TTNrb+633-MHCall_DAPI+568-Rhod_100X_01_stitched.tif')
        # load the image
        tif = TiffFile(path)
        n_slices = int(len(tif.pages)/4)
        mask_channel = 3#0#3#3 #0 # 3
        slice_index = 0#9#0 #9 # 0
        substrate1_channel = 3#0#3
        substrate2_channel = 3#0#3
        
        image_analysis = ImageAnalysis()
        raw = np.asarray(tif.pages[4*slice_index+mask_channel].asarray())
        raw_info = tif.pages[4*slice_index+mask_channel].tags
        
        # get image factor
        factor = raw_info['XResolution'].value[0]  # The Xresolution tag is a ratio, so we take the numerator
        pixSize = 1 / factor * 10**(6) # Factor to go from pixels to micrometers
        
        image_analysis.set_mask_image(raw, pixSize)
        image1 = np.asarray(tif.pages[4*slice_index + substrate1_channel].asarray())
        image2 = np.asarray(tif.pages[4*slice_index + substrate2_channel].asarray())
        image_analysis.set_subtrate1_image(image1)
        image_analysis.set_substrate2_image(image2)
        
        image_analysis.roi_size = roi_size
        image_analysis.mask_background()
        image_analysis.apply_steerable_filter()
        end = time.time()
        #plot_steerable_filter_results(image_analysis.raw, image_analysis.res, image_analysis.xy_values_k0, image_analysis.steerable_sigma, image_analysis.roi_instances)
        
        angles_deg = image_analysis.anglemap[~np.isnan(image_analysis.anglemap)]
        angles_rad = angles_deg * np.pi / 180
        var[i] = circvar(angles_rad, high=np.pi, low=0)
        
        print(f' {i+1}/{len(roi_size_arr)} done in {end-start  :.2f}s : roi_size = {roi_size}, var = {var[i]:.2f}')
    plt.plot(roi_size_arr, var, 'o')
    plt.show()
    
def non_interactive():
    path = os.path.join('Data', '2023.06.12_MhcGFPweeP26_30hrsAPF_Phallo568_647nano62actn_405nano2sls_100Xz2.5_1_2.tif')
    #path = os.path.join('Data', 'Trial8_D12_488-TTNrb+633-MHCall_DAPI+568-Rhod_100X_01_stitched.tif')
    # load the image
    tif = TiffFile(path)
    n_slices = int(len(tif.pages)/4)
    mask_channel = 3#0#3#3 #0 # 3
    slice_index = 0#9#0 #9 # 0
    substrate1_channel = 3#0#3
    substrate2_channel = 3#0#3
    
    image_analysis = ImageAnalysis()
    raw = np.asarray(tif.pages[4*slice_index+mask_channel].asarray())
    raw_info = tif.pages[4*slice_index+mask_channel].tags
    
    # get image factor
    factor = raw_info['XResolution'].value[0]  # The Xresolution tag is a ratio, so we take the numerator
    pixSize = 1 / factor * 10**(6) # Factor to go from pixels to micrometers
    
    image_analysis.set_mask_image(raw, pixSize)
    image1 = np.asarray(tif.pages[4*slice_index + substrate1_channel].asarray())
    image2 = np.asarray(tif.pages[4*slice_index + substrate2_channel].asarray())
    image_analysis.set_subtrate1_image(image1)
    image_analysis.set_substrate2_image(image2)
    
    
    image_analysis.mask_background()
    image_analysis.apply_steerable_filter()
    image_analysis.calc_all_ccf()
    #return image_analysis
    image_analysis.plot_top_x_roi_interactive()
    save = True
    if save:
        
        np.savetxt('comparison/lags.csv', image_analysis.lags[image_analysis.ind], delimiter=',')
        np.savetxt('comparison/ccf_all.csv', image_analysis.ccf_all[:,image_analysis.ind], delimiter=',')
        np.savetxt('comparison/ccf_mask.csv', image_analysis.ccf_mask, delimiter=',')
        np.savetxt('comparison/ccf_fit_valid.csv', image_analysis.ccf_fit_valid, delimiter=',')
    #embed()

def main():
    
    
    interactive_image_analysis()
    #non_interactive()
    #vary_roi_size(15, 58, 1)

    
 
if __name__ == "__main__":
    main()
