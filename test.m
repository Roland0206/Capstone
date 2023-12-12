% Specify the path to the folder you want to list
folderPath = '/home/roland/Schreibtisch/Capstone/Data/';

% Use the dir function to get information about files in the folder
files = dir(folderPath);

% Loop through the 'files' structure array to display file names
for i = 1:numel(files)
    if ~files(i).isdir  % Check if it's not a directory
        fprintf('%s\n', files(i).name);
    end
end
