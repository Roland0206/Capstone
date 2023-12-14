
xy = readmatrix('/home/roland/Schreibtisch/Capstone/comparison/xy.csv');
x =xy(:,1);
y =xy(:,2);

test = xcorr(x,y,'unbiased');
writematrix(test, '/home/roland/Schreibtisch/Capstone/comparison/xy_xcorr_mat.csv');

