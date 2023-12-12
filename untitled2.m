% Assuming Raw1_Tophat, xi, yi, and max_dist are defined
% For example:
image = zeros(100, 100);
image(31:70, 31:70) = 1;
xi = [31, 71];
yi = [31, 71];
max_dist = 5;

% Get the intensity profile
profile = improfile(image, xi, yi, 2*max_dist+1);

% Plot the profile
plot(profile);