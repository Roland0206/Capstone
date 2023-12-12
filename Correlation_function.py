import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from skimage import io, filters, morphology

# Correlation functions
# < This script computes correlation function from the linescans using the nematic director >
# Copyright (C) <2023> <Francine Kolley>
# ...

# Set parameters
nChannel1 = 1  # channel for first substrate
nChannel2 = 4  # channel for second substrate

# Meta-information
# Open Data
RawPath = str(RawPath)
info = io.imread(RawPath)

# Read necessary parameters for calculation
# get the parameters from Mask
GBlur_Sigma = Parameters[0, 0]
ROI_Size = Parameters[2, 0]
Tophat_Sigma = Parameters[4, 0]

# Read in intensities
# use same parameters as for the masks
Raw1 = io.imread(RawPath, as_gray=True)[..., nChannel1 - 1]  # substrate 1
Raw1_BGSub = filters.gaussian(Raw1, sigma=GBlur_Sigma)  # Gaussian blur #1
Raw1_Tophat = morphology.white_tophat(Raw1_BGSub, morphology.disk(Tophat_Sigma))  # Raw before

Raw2 = io.imread(RawPath, as_gray=True)[..., nChannel2 - 1]  # substrate 2
Raw2_BGSub = filters.gaussian(Raw2, sigma=GBlur_Sigma)  # Gaussian blur #2
Raw2_Tophat = morphology.white_tophat(Raw2_BGSub, morphology.disk(Tophat_Sigma))  # Raw before

# correlation length and other pre-definitions of values
# Correlation length
corr_length = 4  # [μm]
crop_length = round(corr_length / (2 * pixSize))  # [pixel]

# ...

# Computation over the whole grid
plt.figure(1)
for i in range(1, nGridY + 1):
    for j in range(1, nGridX + 1):

        if Raw_ROIsMask[i - 1, j - 1] == 1:  # If they are of interest

            Angle = AngleMap[i - 1, j - 1]  # find right angle for current window
            ccf = np.zeros(2 * maxlag + 1)  # allocate memory for ACF/CCF
            var1 = 0  # reset var1
            var2 = 0  # reset var2

            kmax = 10  # in [pixel]

            # Do linescans in both, x and y direction and compute from that
            # the 1D correlation function
            for k in range(-kmax, kmax + 1, 2):
                xi = (
                    ROI_Size * j
                    + max_dist * np.array([-1, 1]) * np.cos((90 - Angle) * -1 * np.pi / 180)
                    + k * np.cos((Angle) * np.pi / 180)
                )  # x-position
                yi = (
                    ROI_Size * i
                    + max_dist * np.array([-1, 1]) * np.sin((90 - Angle) * -1 * np.pi / 180)
                    + k * np.sin((Angle) * np.pi / 180)
                )  # y-position
                linescan1 = Raw1_Tophat[yi.astype(int), xi.astype(int)]  # get the profile for substrate 1
                linescan2 = Raw2_Tophat[yi.astype(int), xi.astype(int)]  # get the profile for substrate 2

                # subtract the mean
                linescan1_mean = np.mean(linescan1)
                linescan1 -= linescan1_mean
                linescan2_mean = np.mean(linescan2)
                linescan2 -= linescan2_mean

                # compute unbiased CCF
                ccf += correlate(linescan1, linescan2, mode='valid', method='auto') / (2 * kmax + 1)

                # variances of individual linescans (Normalization)
                var1 += np.var(linescan1) / (2 * kmax + 1)
                var2 += np.var(linescan2) / (2 * kmax + 1)

            # Normalize CCF by respective standard deviations
            ccf /= np.sqrt(var1 * var2)

            # keep a record
            ccf_all[(i - 1) * nGridX + j - 1, :] = ccf

            # plot individual ACFs/CCFs
            maxlag_plot = linescan1.shape[0] - 1
            lags = np.arange(-maxlag, maxlag + 1) * pixSize  # [um]
            ind = np.arange(maxlag_plot + 1, 2 * maxlag_plot + 2)  # start with 'lag = 0'
            plt.plot(lags[ind], ccf[ind], color='#cccaca')
            plt.xlabel('\u0394x [um]')
            plt.ylabel('CCF')
            plt.title('CCF (unbiased, normalized)')

# Find the valid CCFs/ACFs and calculate the mean
ccf_all_valid = ccf_all[~np.isnan(ccf_all).any(axis=1), :]  # delete rows with nans
mean_ccf = np.mean(ccf_all_valid, axis=0)
std_mean_ccf = np.std(ccf_all_valid, axis=0, ddof=1)  # / np.sqrt(div_factor);

plt.plot(lags[ind], mean_ccf[ind], '-', color='#d13111', linewidth=1.8)
plt.plot(lags[ind], mean_ccf[ind] - std_mean_ccf[ind], '--', color='#d13111', linewidth=1.8)
plt.plot(lags[ind], mean_ccf[ind] + std_mean_ccf[ind], '--', color='#d13111', linewidth=1.8)
plt.ylim([-0.5, 1])
plt.xlim([0, corr_length])
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.show()
