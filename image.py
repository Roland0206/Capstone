import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from spectrum.correlation import xcorr

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Polygon


from IPython import embed

import os
import time
from numba import jit

from tifffile import TiffFile

import steerable

from skimage.transform import resize

import cv2

def white_tophat_trafo(raw, tophat_sigma):
    """
    Applies a white tophat transformation to an image.

    Args:
        raw (ndarray): The raw input image.
        tophat_sigma (float): The sigma value for the white tophat transformation.

    Returns:
        ndarray: The transformed image.
    """
    # Create a circular structuring element equivalent to the 'disk' in skimage
    selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2*tophat_sigma+1), int(2*tophat_sigma+1)))
    
    # Apply the white tophat transformation
    return cv2.morphologyEx(raw, cv2.MORPH_TOPHAT, selem)
                

def plot_steerable_filter_results(raw, res, xy_values_k0, steerable_sigma, fig=None):
    """
    Plots the results of a steerable filter analysis.

    Args:
        raw (ndarray): The raw input image.
        res (ndarray): The filtered image.
        xy_values_k0 (ndarray): Array of xy coordinate values.
        steerable_sigma (float): The sigma value for the steerable filter.
        qlow (float): The lower limit for the colorbar.
        qhigh (float): The upper limit for the colorbar.
        fig (Figure, optional): The figure object to plot on. If not provided, a new figure will be created.

    Returns:
        tuple: A tuple containing the figure object, raw image display, filtered image display, 
               new raw image display, and the axis object for the xy plot.
    """
    res = pd.read_csv('res.csv', header=None).to_numpy()
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
    res1_display = ax2.imshow(res, cmap='gray', vmin=np.min(res), vmax=np.max(res))
    
    raw_new = np.ones((raw.shape[0], raw.shape[1]))
    res2_display = ax3.imshow(raw, cmap='gray', vmin=qlow, vmax=qhigh)
    for i in range(xy_values_k0.shape[0]):
        ax3.plot(xy_values_k0[i, 0], xy_values_k0[i, 1], 'c')
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5)
    
    if return_fig:
        return fig, raw_display, res1_display, res2_display, ax1, ax2, ax3
    else:
        plt.show()

def plot_mask_background(raw_mask, raw_roi_mask, mask_thresh, roi_size, roi_thresh, gblur_sigma,roi_instances=[], fig=None):
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
    if len(roi_instances) > 0:
        for i in range(len(roi_instances)):
            roi = roi_instances[i]
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

def improfile(img, xi, yi, N, order=1, mode='nearest'):
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

    Args:
        point1 (tuple) (x1,y1)         : The first corner of the rectangle.
        point2 (tuple) (x2,y2)         : The second corner of the rectangle .
        point3 (tuple) (x_proj, y_proj): The third point defining the rectangle (only the projection matters).
        
          (x1,y1)--------------(x2,y2)
            |                     |
            |                     |
            |                     |
          (x4,y4)--------------(x3,y3)--------(x_proj,y_proj)
            

    Returns:
        tuple: A tuple containing the four corner points of the rectangle.
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
@jit(nopython=True)
def process_mask(nYCrop, nXCrop, roi_array, raw_mask, mask_thresh):
    for y in range(nYCrop):
        for x in range(nXCrop):
            in_roi = False
            for roi in roi_array:
                if inside_rectangle((x, y), roi):
                    in_roi = True
                    break
            if in_roi and raw_mask[y, x] >= mask_thresh:
                raw_mask[y, x] = 1
            else:
                raw_mask[y, x] = 0
    return raw_mask

@jit(nopython=True)
def inside_rectangle(point, rect):
    """
    Check if a point is inside a roteted rectangle.

    Args:
        point (tuple) (x,y): The point to check.
        rect (tuple) (x1,y1,x2,y2,x3,y3,x4,y4): The four corner points of the rectangle.

    Returns:
        bool: True if the point is inside the rectangle, False otherwise.
    """
    x, y = point
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = rect

    AB = (x2-x1, y2-y1)
    AD = (x4-x1, y4-y1)
    AP = (x-x1, y-y1)

    ABAP = AB[0]*AP[0] + AB[1]*AP[1]
    ABAB = AB[0]*AB[0] + AB[1]*AB[1]
    ADAP = AD[0]*AP[0] + AD[1]*AP[1]
    ADAD = AD[0]*AD[0] + AD[1]*AD[1]

    return (0 <= ABAP <= ABAB) and (0 <= ADAP <= ADAD)

                
class ImageAnalysis:
    """
    Class for performing image analysis on a given image.

    Attributes:
        Parameters:
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
            
        Parameters for detecting the nematic ordered structures on mask and results:
            - steerable_bool: A boolean value indicating whether the steerable filter has been applied.
            - anglemap: The angle map.
            - res: The response of the steerable filter to the masked image
            - rot:  Response of the input image to rotated versions of the filter, at 'nAngles' different angles.
            - xy_values_k0: cordinates of the start and endpoints of detected nematic ordered structures in the image.
    
    
    """

    def __init__(self, config={}):
        # general parameters
        self.gblur_sigma = config.get('gblur_sigma', 2)
        self.roi_size = config.get('roi_size', 30)
        self.tophat_sigma = config.get('tophat_sigma', 28)
        self.steerable_sigma = config.get('steerable_sigma', 2)
        self.mask_thresh = config.get('mask_thresh', 350)
        self.roi_thresh = config.get('roi_thresh', 0.3)
        self.pad_angle = config.get('pad_angle', 6)
        self.pad_corr = config.get('pad_corr', 3)
        self.roi_pad_corr = (self.roi_size + (self.roi_size * (self.pad_corr - 1)) * 2)
        self.roi_array = config.get('roi_array', [])

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

        Parameters:
        raw (numpy.ndarray): The raw image data.
        pixSize (float): The pixel size in micrometers.

        Returns:
        None
        """
        self.raw = raw
        self.nY, self.nX = self.raw.shape
        self.pixSize = pixSize
        
    def set_roi_array(self, roi_array):
        """
        Set the ROI (Region of Interest) array.

        Parameters:
        - roi_array: The ROI array to be set.

        Returns:
        None
        """
        self.roi_array = np.array(roi_array)
        
    def get_cross_correlation(self):
        """
        Calculate the cross-correlation of the image.

        Returns:
            ccf_all_valid (numpy.ndarray): The cross-correlation values for all valid pixels.
            mean_ccf (float): The mean cross-correlation value.
            std_mean_ccf (float): The standard deviation of the mean cross-correlation value.
        """
        return self.ccf_all_valid, self.mean_ccf, self.std_mean_ccf
        
    def get_mask_parameters(self):
        """
        Returns the mask parameters used for image processing.

        Returns:
            tuple: A tuple containing the mask threshold, ROI size, ROI threshold, and Gaussian blur sigma.
        """
        return self.mask_thresh, self.roi_size, self.roi_thresh, self.gblur_sigma
    
    def mask_background(self):
        """
        This method processes an image by applying a Gaussian blur, creating a binary mask based on a threshold,
        and creating a region of interest (ROI) mask based on the percentage of white pixels in each ROI.

        The method updates the following instance variables:
        - raw_crop: Cropped version of the raw image
        - nYCrop, nXCrop: Dimensions of the cropped image
        - raw_bgsub: cropped image with gaussian blur applied
        - raw_mask: Binary mask of raw_bgsub
        - raw_roi_mask: Binary mask of the ROIs
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
            start = time.time()
            self.raw_mask = process_mask(self.nYCrop, self.nXCrop, self.roi_array, self.raw_mask, self.mask_thresh)
            end = time.time()
            #print(f'mask creation took {end - start} seconds')
        else:
            # If no ROIs are defined, create a binary mask based on the threshold
            self.raw_mask = (self.raw_mask >= self.mask_thresh).astype(int)
        self.create_roi_mask()

        
    def create_roi_mask(self):
        # Create a mask of the ROIs based on the percentage of white pixels in each ROI
        self.raw_roi_mask = np.zeros((self.nGridY, self.nGridX))
        white_percentage = np.zeros((self.nGridY, self.nGridX))
        start = time.time()
        for i in range(self.nGridY):
            for j in range(self.nGridX):
                temp = self.raw_mask[i*self.roi_size:(i+1)*self.roi_size, j*self.roi_size:(j+1)*self.roi_size].mean()
                white_percentage[i, j] = temp
                if temp > self.roi_thresh and self.pad_angle - 1 <= i <= self.nGridY - self.pad_angle and self.pad_angle - 1 <= j <= self.nGridX - self.pad_angle:
                    self.raw_roi_mask[i, j] = 1
        end = time.time()
        #print(f'ROI mask creation took {end - start} seconds')
                        
    def set_tophat_sigma(self, tophat_sigma):
        """
        Set the value of tophat_sigma for the image.

        Parameters:
        - tophat_sigma: The standard deviation of the Gaussian kernel used for the top-hat filtering.

        Returns:
        None
        """
        self.tophat_sigma = tophat_sigma
        
    def load_res_from_csv(self, path='res.csv'):
        """
        Load the response of the steerable filter from a CSV file. Neccessarry because of weird bug (res entries somehow get changed to zero even if the array is not explicitly changed)

        Args:
            path (str, optional): The path to the CSV file. Defaults to 'res.csv'.
        """
        self.res = pd.read_csv(path, header=None).to_numpy()
        
    def apply_steerable_filter(self, angle_array=None, save=True):
        """
        This method applies a steerable filter to the image, calculates the angle response, and creates an angle map.
        It also calculates the start xi and endposition yi of each detected sructure based on the angle map.

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
        sd = steerable.Detector2D(raw_rsize, 2, self.steerable_sigma)
        
        res_tmp, _ = sd.filter()
        if save:
            np.savetxt('res.csv', res_tmp, delimiter=',')
        self.res = res_tmp

        # Calculate the angle response of the steerable filter
        if angle_array is None:
            rot = np.transpose(sd.get_angle_response(self.nAngles))
        else:
            rot = np.transpose(sd.get_angle_response_with_array(angle_array))
        for i in range(rot.shape[2]):
            temp = rot[:, :, i]
            temp[mask_rsize == 0] = np.nan
            rot[:, :, i] = temp

        # Create an angle map based on the maximum response for each ROI
        self.rot = rot
        anglemap = np.zeros((self.nGridY, self.nGridX))
        for i in range(self.nGridY):
            for j in range(self.nGridX):
                if self.raw_roi_mask[i, j] == 1:
                    y_idx = [i-(self.pad_angle-1), i+self.pad_angle]
                    x_idx = [j-(self.pad_angle-1), j+self.pad_angle]
                    Crop = rot[y_idx[0]:y_idx[1], x_idx[0]:x_idx[1], :]
                    idxMax = np.full(Crop.shape[2], np.nan)
                    for k in range(Crop.shape[2]):
                        temp = Crop[:, :, k]
                        idxMax[k] = np.nanmean(temp)
                    M, I = np.nanmax(idxMax), np.nanargmax(idxMax)
                    anglemap[i, j] = I+1 #TODO: check if +1 is necessary 

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

        Args:
            raw_substrate1: The raw image data for substrate 1.

        Returns:
            None
        """
        self.raw1 = raw_substrate1
        
    def set_substrate2_image(self, raw_substrate2):
        """
        Set the raw substrate 2 image.

        Args:
            raw_substrate2: The raw substrate 2 image.

        Returns:
            None
        """
        self.raw2 = raw_substrate2
        
    def calculate_cross_correlation(self,tophat=False,  plot=True):
        """
        Calculates the cross-correlation between two images. The cross-correlation is calculated for each ROI and the detected structures in it.

        Args:
            plot (bool, optional): Whether to plot the cross-correlation. Defaults to True.
        """
        
        # check if tophat filter has been applied
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
        
        # Define constants and parameters
        corr_length = 4  # [μm]
        crop_length = round(corr_length / (2 * self.pixSize))  # [pixel]
        max_dist = round(crop_length) + 10  # +10 pixels to corr_length to smaller deviations
        maxlag = 2 * max_dist  # Correlation length will be doubled because of mirror symmetry

        ccf_all = np.full((self.nGridY * self.nGridX, 2 * maxlag + 1), np.nan)  # Allocate memory for ACF/CCF
        # Computation over the whole grid
        if plot:
            fig, ax = plt.subplots(figsize=(12, 10))
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
                    ccf_all[i*self.nGridX+j, :] = ccf

                    maxlag_plot = len(linescan1) - 1
                    lags = np.arange(-maxlag, maxlag+1) * self.pixSize
                    ind = np.arange(maxlag_plot, min(2*maxlag_plot+1, len(ccf), len(lags)))
                    if plot:
                        ax.plot(lags[ind], ccf[ind], color='#cccaca')
                        
                    

        # Find the valid CCFs/ACFs and calculate the mean
        self.ccf_all_valid = ccf_all[~np.isnan(ccf_all).any(axis=1)]
        self.mean_ccf = np.mean(self.ccf_all_valid, axis=0)
        self.std_mean_ccf = np.std(self.ccf_all_valid, axis=0, ddof=1)
        
        if plot:
            ax.plot(lags[ind], self.mean_ccf[ind], '-', color='#d13111', linewidth=1.8)
            ax.plot(lags[ind], self.mean_ccf[ind] - self.std_mean_ccf[ind], '--', color='#d13111', linewidth=1.8)
            ax.plot(lags[ind], self.mean_ccf[ind] + self.std_mean_ccf[ind], '--', color='#d13111', linewidth=1.8)
            
            ax.set_xlabel(r'$\Delta$ x [$\mu$m]')
            ax.set_xlim(0,corr_length)
            ax.set_ylim(-0.5,1)
            ax.set_ylabel('CCF')
            plt.savefig('ccf_human.png', dpi=300)
            plt.show()
        
    def select_roi_manually(self):
        """
        Allows the user to manually select rectangluar regions of interest (ROI) in the image.
        The selected ROI is stored in the `roi_array` attribute of the object.
        """
        tmp = ROISelector(self.raw)
        self.roi_array = tmp.roi_array
        self.roi_instances = np.empty(len(self.roi_array), dtype=object)
        for i in range(len(self.roi_array)):
            self.roi_instances[i] = ROI(self.roi_array[i], self)
            
        
        
    def get_parameter_dict(self):
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
        """

        # Create a new figure
        fig = plt.figure()

        # Create sliders for each parameter
        mask_thresh_slider_ax = plt.axes([0.2, 0.02, 0.65, 0.03])
        mask_thresh_frame_slider = Slider(mask_thresh_slider_ax, 'mask threshhold', 0, 1000, valinit=self.mask_thresh, valstep=50)

        roi_thresh_slider_ax = plt.axes([0.2, 0.08, 0.65, 0.03])
        roi_thresh_frame_slider = Slider(roi_thresh_slider_ax, 'roi threshhold', 0, 0.85, valinit=self.roi_thresh, valstep=0.1)

        roi_size_slider_ax = plt.axes([0.2, 0.14, 0.65, 0.03])
        roi_size_frame_slider = Slider(roi_size_slider_ax, 'roi size', 5, 50, valinit=self.roi_size, valstep=5)

        gblur_sigma_slider_ax = plt.axes([0.2, 0.2, 0.65, 0.03])
        gblur_sigma_frame_slider = Slider(gblur_sigma_slider_ax, 'gblur sigma', 0.5, 4, valinit=self.gblur_sigma, valstep=0.5)

        # Generate the mask and ROI mask images
        self.mask_background()
        mask_display, roi_mask_display, ax1, ax2 = plot_mask_background(self.raw_mask, self.raw_roi_mask, self.mask_thresh, self.roi_size, self.roi_thresh, self.gblur_sigma, self.roi_instances, fig)

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
        """

        # Create a new figure
        fig = plt.figure()

        # Create a slider for the sigma value
        steerable_sigma_slider_ax = plt.axes([0.2, 0.08, 0.65, 0.03])
        steerable_sigma_slider = Slider(steerable_sigma_slider_ax, 'steerable sigma', 1, 4, valinit=self.steerable_sigma, valstep=0.5)

        # Apply the steerable filter if it hasn't been applied yet
        if not self.steerable_bool:
            self.apply_steerable_filter()

        # Load the results from a CSV file
        self.load_res_from_csv()

        # Display the images
        fig, _, res1_display, res2_display, ax1, ax2, ax3 = plot_steerable_filter_results(self.raw, self.res, self.xy_values_k0, self.steerable_sigma, fig)
        qlow = np.quantile(self.raw, 0.001)
        qhigh = np.quantile(self.raw, 0.999)

        # Define a function to update the images whenever the slider value changes
        def update_image(val):
            self.steerable_sigma = val
            self.apply_steerable_filter()
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
            for i in range(self.xy_values_k0.shape[0]):
                ax3.plot(self.xy_values_k0[i, 0], self.xy_values_k0[i, 1], 'c')
            fig.canvas.draw_idle()

        # Connect the update function to the slider
        steerable_sigma_slider.on_changed(lambda val: update_image(val))

        # Display the figure
        plt.show()
        
class ROI(ImageAnalysis):
    """This class represents one region of interest (ROI). Based on a predefined general orientation of structures in this ROI a more accurate orietation is detected"""
    
    def __init__(self, corner_array, image_analysis):
        if not isinstance(image_analysis, ImageAnalysis):
            raise TypeError('image_analysis must be an instance of ImageAnalysis')
        self.image_analysis = image_analysis
        self.corners = corner_array
        self.roi_size = 0
        self.roi_array = []
        self.calc_area()
        
    def calc_area(self):
        
        # find min and max of x and y coordinates
        min_x = min(x for x, y in self.corners)
        max_x = max(x for x, y in self.corners)
        min_y = min(y for x, y in self.corners)
        max_y = max(y for x, y in self.corners)
        
        self.dx, self.dy = max_x-min_x, max_y-min_y
        
        min_x_area = int(np.floor(min_x / self.image_analysis.roi_size) * self.image_analysis.roi_size)
        diff_x = min_x - min_x_area
        
        min_y_area = int(np.floor(min_y / self.image_analysis.roi_size) * self.image_analysis.roi_size)
        diff_y = min_y - min_y_area
        
        n_roi_x = int(np.ceil((self.dx + diff_x) / self.image_analysis.roi_size))
        n_roi_y = int(np.ceil((self.dy+ diff_y) / self.image_analysis.roi_size))
        
        # slice indices for anglemap
        self.nx0 = min_x_area * self.image_analysis.roi_size
        self.nx1 = self.nx0 + n_roi_x
        
        self.ny = min_y_area * self.image_analysis.roi_size
        self.ny1 = self.ny + n_roi_y
        
        
        
        
        
        max_x_area = min_x_area + n_roi_x * self.image_analysis.roi_size
        max_y_area = min_y_area + n_roi_y * self.image_analysis.roi_size
        
        self.x_borders = [min_x_area, max_x_area]
        self.y_borders = [min_y_area, max_y_area]
        self.nX = max_x_area - min_x_area
        self.nY = max_y_area - min_y_area
        
    def calc_angle(self):
        if len(self.image_analysis.anglemap)==0:
            raise ValueError('anglemap of superclass is empty! Please apply steerable filter to superclass first and then use set_superclass() method.')
        angle = self.image_analysis.anglemap[self.ny:self.ny1, self.nx0:self.nx1]
        self.angle0 = np.nanmean(angle)
        
    def set_superclass(self, image_analysis):
        if not isinstance(image_analysis, ImageAnalysis):
            raise TypeError('image_analysis must be an instance of ImageAnalysis')
        self.image_analysis = image_analysis
        
        
        
    def set_borders(self, xi, yi):
        self.x_borders = xi
        self.y_borders = yi
        self.nX = self.x_borders[1] - self.x_borders[0]
        self.nY = self.y_borders[1] - self.y_borders[0]
        

    def crop_images(self):
        if len(self.image_analysis.raw)==0:
            raise ValueError('raw image of superclass is not yet set!')
        else:
            self.raw = self.image_analysis.raw[self.y_borders[0]:self.y_borders[1], self.x_borders[0]:self.x_borders[1]]
        if len(self.image_analysis.raw_tophat)>0:
            self.raw_tophat = self.image_analysis.raw_tophat[self.y_borders[0]:self.y_borders[1], self.x_borders[0]:self.x_borders[1]]
        if len(self.image_analysis.raw_mask)>0:
            self.raw_mask = self.image_analysis.raw_mask[self.y_borders[0]:self.y_borders[1], self.x_borders[0]:self.x_borders[1]]
        if len(self.image_analysis.rawraw_bgsub_roi_mask)>0:
            self.raw_bgsub = self.image_analysis.raw_bgsub[self.y_borders[0]:self.y_borders[1], self.x_borders[0]:self.x_borders[1]]
        if len(self.image_analysis.raw1)>0:
            self.raw1 = self.image_analysis.raw1[self.y_borders[0]:self.y_borders[1], self.x_borders[0]:self.x_borders[1]]
        if len(self.image_analysis.raw1_tophat)>0:
            self.raw1_tophat = self.image_analysis.raw1_tophat[self.y_borders[0]:self.y_borders[1], self.x_borders[0]:self.x_borders[1]]
        if len(self.image_analysis.raw2)>0:
            self.raw2 = self.image_analysis.raw2[self.y_borders[0]:self.y_borders[1], self.x_borders[0]:self.x_borders[1]]
        if len(self.image_analysis.raw2_tophat)>0:
            self.raw2_tophat = self.image_analysis.raw2_tophat[self.y_borders[0]:self.y_borders[1], self.x_borders[0]:self.x_borders[1]]
        
    def set_roi_size(self, roi_size):
        self.roi_size = roi_size
        if roi_size > self.image_analysis.roi_size:
            raise ValueError('roi_size must be smaller than the image_analysis.roi_size')
        
        self.nGridY = int(np.floor(self.nY / self.roi_size))
        self.nGridX = int(np.floor(self.nX / self.roi_size))
            
        if self.nGridY * self.roi_size < self.dy:
            yi = self.y_borders
            yi[1] += self.image_analysis.roi_size
            self.set_borders(self.x_borders, yi)
            self.crop_images()
            self.nGridY = int(np.floor(self.nY / self.roi_size))
            self.nGridX = int(np.floor(self.nX / self.roi_size))

        if self.nGridX * self.roi_size < self.dx:
            xi = self.x_borders
            xi[1] += self.image_analysis.roi_size
            self.set_borders(xi, self.y_borders)
            self.crop_images()
            self.nGridY = int(np.floor(self.nY / self.roi_size))
            self.nGridX = int(np.floor(self.nX / self.roi_size))


        # Crop the raw image to fit an integer number of ROIs
        self.raw_crop = np.copy(self.raw[:self.nGridY * self.roi_size, :self.nGridX * self.roi_size])
        self.nYCrop, self.nXCrop = self.raw_crop.shape

        # applying a Gaussian blur
        self.raw_bgsub = gaussian_filter(self.raw_crop, sigma=self.gblur_sigma)
        
    def apply_steerable_filter_finetuning(self, angle_range, nangles):
        """
        This method applies a steerable filter to the image self.raw. It calculates the response of the filter, the response of the input image to rotated versions of the filter, and the angle map.
        Parameter:
        - angle_range: Range of angles around self.angle0 to be tested (angle0-anbgle_range to angle0+angle_range) in degree
        - angle_step: Step size for the angle range in degree
        
        The method updates the following instance variables:
        - res: filter response
        - rot: response of the input image to rotated versions of the filter, at 'nAngles' different angles
        - anglemap: Map of the angles of the maximum response for each ROI
        - xy_values_k0: start xi and endposition yi of each detected stucture( xy_values_k0[t, 0] = xi, xy_values_k0[t, 1] = yi)
        - steerable_bool: Flag indicating that the steerable filter has been applied
        """

        angle_array = np.linspace(self.angle0-angle_range, self.angle0+angle_range, nangles, endpoint=True)
        # convert to radians
        angle_array = angle_array * np.pi / 180
        
        self.apply_steerable_filter(angle_array)
    
    
        
class ROISelector:
    """
    This class allows the user to interactively select regions of interest (ROIs) in an image.
    The user can add one rectangular ROI after clicking once on the "Add ROI" button. Then the next two clicks define one corner of the rectangle and the last click defines the width of the rectangle. The user can remove the last ROI by clicking on the "Remove last ROI" button.
    """

    def __init__(self, raw):
        """
        Initialize the ROISelector with an image.

        Args:
            raw: The image to select ROIs from.
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

        Args:
            event: The click event.
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

        Args:
            event: The click event.
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

        Args:
            event: The click event.
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
        substrate1_channel = 0
        substrate2_channel = 3
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
        fig, ax = plt.subplots(1,3, figsize=(12, 4))
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
            del mask_channel, substrate1_channel, substrate2_channel
        else:
            change = False
    return tif, path, slice_index, mask_channel, substrate1_channel, substrate2_channel
    

def interactive_image_analysis(image_analysis):
    tmp = input(f'Configure rectangular regions of interest manually? (y/N)')
    if tmp == 'y' or tmp == 'Y':
        image_analysis.select_roi_manually()
        
    change = True
    image_analysis.mask_background()
    while change:
        tmp =f'''Current parameter for creation of mask and background substraction:\n\t- mask_thresh: {image_analysis.mask_thresh}\n\t- roi_thresh: {image_analysis.roi_thresh}\n\t- roi_size: {image_analysis.roi_size}\n\t- gblur_sigma: {image_analysis.gblur_sigma}'''
        print(tmp)
        
        param = [image_analysis.raw_mask,image_analysis.raw_roi_mask, *image_analysis.get_mask_parameters(), image_analysis.roi_instances]
        plot_mask_background(*param)
        
        change = input('\n Change parameter? (y/N)')

        if change == 'y' or change == 'Y':
            image_analysis.choose_parameter_mask()
            proceed = input(f'New Parameters:\n\t- mask_thresh: {image_analysis.mask_thresh}\n\t- roi_thresh: {image_analysis.roi_thresh}\n\t- roi_size: {image_analysis.roi_size}\n\t- gblur_sigma: {image_analysis.gblur_sigma}\nProceed? (Y/n)')
            if proceed == 'n' or proceed == 'N':
                change = True
            else:
                change = False
        else:
            change = False
            
    
    change = True
    print(f'Current sigma for steerable filter: {image_analysis.steerable_sigma}\n Aplying steerable filter...')
    image_analysis.apply_steerable_filter()
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
    
    # calculate cross correlation
    print('Calculating the cross correlation:')
    
    
    image_analysis.calculate_cross_correlation(tophat=True, plot=True)
    #embed()

        
def main():
    image_analysis = ImageAnalysis()
    tif, path, slice_index, mask_channel, substrate1_channel, substrate2_channel = choose_image_interactive()
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
    
    interactive_image_analysis(image_analysis)
    
 
if __name__ == "__main__":
    main()
