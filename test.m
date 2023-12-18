%% Image_analysi_algorithm to detect the nematic director
% Copyright (C) <2023>  <Francine Kolley>
% The code was written for multi-channel images of insect flight muscle
% Pipeline in collaboration with Benoit Dehapiot
% Francine Kolley
% Physics of Life, Benjamin M. Friedrich group
% TU_dresden 
% contact: francine.kolley@tu-dresden.de
% Latest code 07-2022

% STEPS:
% (1) run this script
% (2) rund Correlation_function.m

clear all
close all

% #########################################################################
%                             PARAMETER  
% #########################################################################

% Background subtraction
GBlur_Sigma = 2; 

% Mask treshold to filter interesting structures for ROI 
Mask_Thresh =350; %it varies for different time points of myofibrillogenesis

% ROIsMask (obtained from Mask)
ROI_Size = 30; %[pixels] 
ROI_Thresh = 0.25; % minimal value used, some stages increase up to 0.85

% Tophat filtering
Tophat_Sigma = 28; 

% Steerable Filter
Steerable_Sigma = 2; % Size for Steerable filter  

% Padding
Pad_Angle = 6; % Window size --> we will crop image to ensure linescans are not reaching edge of image
Pad_Corr = 3; % [pixel]
ROI_Pad_Corr = (ROI_Size+(ROI_Size*(Pad_Corr-1))*2);


% #########################################################################
%                             IMAGE READ IN 
% #########################################################################

% Get the image file using GUI interface
[filename, filepath] = uigetfile({'*.tif'}, 'Select an Image File');
full_path = fullfile(filepath, filename);

fly1_name = '2023.06.12_MhcGFPweeP26_30hrsAPF_Phallo568_647nano62actn_405nano2sls_100Xz2.5_1_2.tif';
fly2_name = '2023.06.12_MhcGFPweeP26_24hrsAPF_Phallo568_647nano62actn_405nano2sls_100Xz2.5_1_2.tif';
human_name = 'Trial8_D12_488-TTNrb+633-MHCall_DAPI+568-Rhod_100X_01_stitched.tif';
if strcmp(filename, fly1_name) || strcmp(filename, fly2_name)
    % Fly muscle: 4 channels
    %     -1: alpha-actinin
    %     -2: actin
    %     -3: myosin
    %     -4: sallimus (mask channel)
    slice_index = 0;
    mask_channel = 3;
    substrate1_channel = 0;
    substrate2_channel = 3;
elseif strcmp(filename, human_name)
    % Human muscle: 4 channels
    %     -1: titin N-terminus (mask channel)
    %     -2: muscle myosin
    %     -3: nuclei
    %     -4: actin
    mask_channel = 0;
end
% Define the order of the different channels, number of channels
[actinin_image, actin_image, myosin_image, titin_image] = read_in_images(full_path, 4);

% Save the path for meta_data later
RawPath = full_path;

% Open Data and read in information
info = imfinfo(RawPath);
% define the slice you want to observe from the 3D image
f = uifigure('Position', [100 100 600 400]);

%Create a slider for slice selection
sld = uislider(f, 'Position', [100 50 400 3]);
sld.Limits = [1 size(titin_image, 3)]; % Set the slider limits to the number of slices
sld.Value = 1; % Set the initial slider value



% Create axes for image display
ax = axes(f, 'Position', [0.2 0.3 0.6 0.6]);

% Display the initial image
imshow(titin_image(:,:,round(sld.Value)), 'Parent', ax);

% Add a listener to the slider ValueChanged event
addlistener(sld, 'ValueChanged', @(src, event) updateImage(src, event, titin_image, ax));

% Create a button for moving to the next slice
nextBtn = uibutton(f, 'push', 'Position', [510 50 80 22], 'Text', 'Next', ...
    'ButtonPushedFcn', @(btn,event) updateSlice(sld, 1));

% Create a button for moving to the previous slice
prevBtn = uibutton(f, 'push', 'Position', [10 50 80 22], 'Text', 'Previous', ...
    'ButtonPushedFcn', @(btn,event) updateSlice(sld, -1));


slice_number = size(titin_image, 3);

if slice_number > 1
    % Ask user to choose a slice value
    slice_value = inputdlg('Choose a slice value (1 to slice_number):', 'Slice Value', [1 50]);
    slice_value = str2double(slice_value);

    % Validate the input
    while isnan(slice_value) || slice_value < 1 || slice_value > slice_number
        slice_value = inputdlg('Invalid input. Choose a slice value (1 to slice_number):', 'Slice Value', [1 50]);
        slice_value = str2double(slice_value);
    end

    % Use the chosen slice value
    slice = slice_value;
else
    slice = 1;
end


% Define the callback function for updating the image
function updateImage(src, ~, img, ax)
    slice = round(src.Value); % Get the slider value and round it to the nearest integer
    imshow(img(:,:,slice), 'Parent', ax); % Display the selected slice
end

% Define the callback function for updating the slice
function updateSlice(sld, delta)
    sld.Value = min(max(round(sld.Value) + delta, sld.Limits(1)), sld.Limits(2));
end