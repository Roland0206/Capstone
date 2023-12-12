%[res, theta, nms, rotations] = steerableDetector(img, M, sigma) performs edge/ridge detection through a generalization of Canny's alorithm based on steerable filters
%
% Inputs: 
%         img : input image
%           M : order of the filter, between 1 and 5
%             : Odd orders: edge detectors, M = 1 is equivalent to Canny's detector
%             : Even orders: ridge detectors
%               Higher orders provide better orientation selectivity and are less sensitive to noise,
%               at a small trade-off in computational cost.
%       sigma : standard deviation of the Gaussian kernel on which the filters are based
%   {nAngles} : optional input specifing the number of angles computed for the filterbank output. Default: 36
%
% Outputs: 
%         res : response to the filter
%       theta : orientation map
%         nms : non-maximum-suppressed response
%   rotations : response of the input to rotated versions of the filter, at 'nAngles' different angles.
%
% References:
%  [1] Jacob and Unser, IEEE Trans. Pattern Anal. Mach. Intell. 26(8), 2004.

% Author: Francois Aguet
% Adapted from the SteerableJ package, Copyright (C) 2005-2008 Francois Aguet, Biomedical Imaging Group, EPFL.

function [res, theta, nms, filterBank] = steerableDetector(img, M, sigma) %#ok<STOUT,INUSD>