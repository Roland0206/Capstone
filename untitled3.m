test = readmatrix('comparison/RawRSize.csv');
test = test(:,1:70);
[res, ~, ~ , rot] = steerableDetector(test,2,2,180);

