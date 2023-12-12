function [varargout]=read_in_images(image_path,num_channels)
%This function reads in a tiff image, and outputs the results as an image array
%for each channel, which defaults to grayscale.

%As we have variable numbers of channels in our images, also check that
%the number of channels matches the input images.
info = imfinfo(image_path);
num_images = numel(info);

%Check if the number of channels and image elements match up
if mod(num_images,num_channels) ~= 0
   error('Number of channels and number of images as input are not integer multiples of each other'); 
end

%Check number of output images matches channel numbers
if nargout ~= num_channels
   error('Number of channels and number of outputs are different'); 
end

%Presuming both error checks pass, read in the images to the passed handles.
for kk = 1:num_images
    b=mod(kk-1,num_channels)+1;
    image_index=ceil(kk/num_channels);

    varargout{b}(:,:,image_index)=imread(image_path,kk);

end

return;
end