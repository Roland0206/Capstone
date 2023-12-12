
img = double(imread('../testdata/dna.tif'));
M = 4;  % 4th order ridge detector
sigma = 2.5;

[res, orientation, nms, angleRes] = steerableDetector(img, M, sigma);

rescale = @(s) (s-min(s(:)))/(max(s(:))-min(s(:)));
composite = hsv2rgb([rescale(orientation(:)) ones(numel(res),1) rescale(res(:))]);
composite = uint8(reshape(255*composite, [size(res) 3]));

figure;
ax1 = subplot(2,2,1); imagesc(img); axis equal tight off; colormap(ax1, gray(256)); title('Input');
ax2 = subplot(2,2,2); imagesc(res); axis equal tight off; colormap(ax2, gray(256)); title('Filter response');
ax3 = subplot(2,2,3); imagesc(composite); axis equal tight off; title('Orientation');
ax4 = subplot(2,2,4); imagesc(nms); axis equal tight off; colormap(ax4, gray(256)); title('Non-maximum suppressed response');
