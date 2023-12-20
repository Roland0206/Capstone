
clear all
close all
xy = readmatrix('/home/roland/Schreibtisch/Capstone/comparison/xy.csv');
x =xy(:,1);
y =xy(:,2);

test = xcorr(x,y,'unbiased');
writematrix(test, '/home/roland/Schreibtisch/Capstone/comparison/xy_xcorr_mat.csv');

clean_border = readmatrix('/home/roland/Schreibtisch/Capstone/comparison/raw.csv')%imread('/home/roland/Schreibtisch/Capstone/comparison/clean_border.png');
% Define the line coordinates
x = [200,300];
y = [400, 600];%[157.9966, 182.3073];
%y = %[247.2001,109.3270];

% Get the intensity profile along the line
text = improfile(clean_border, x,y, 2*70+1);
mean(text)
text = text-mean(text)

% Display the image and the line
figure;
subplot(1,2,1);
imshow(clean_border);
hold on;
line(x, y, 'Color', 'r', 'LineWidth', 2);
hold off;

% Display the intensity profile
subplot(1,2,2);
plot(text);

% Adjust layout
set(gcf, 'Position', [100, 100, 800, 350]);