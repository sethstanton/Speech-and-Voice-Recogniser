
% Set the parent directory
parentDir = fullfile(pwd, 'recordings');

% Get a list of subdirectories
subDirs = dir(parentDir);
subDirs = subDirs([subDirs.isdir] & ~ismember({subDirs.name}, {'.', '..'}));

% Iterate over top-level subdirectories
for i = 1:numel(subDirs)
    % Get the current top-level subdirectory
    currentTopLevelDir = fullfile(parentDir, subDirs(i).name);
    
    % Get a list of sub-subdirectories within the current top-level subdirectory
    subSubDirs = dir(fullfile(currentTopLevelDir, '**'));
    subSubDirs = subSubDirs([subSubDirs.isdir] & ~ismember({subSubDirs.name}, {'.', '..'}));
    
    % Iterate over sub-subdirectories
    for j = 1:numel(subSubDirs)
        % Get the current sub-subdirectory
        currentSubSubDir = fullfile(currentTopLevelDir, subSubDirs(j).name);
        
        % Get a list of JPG files in the current sub-subdirectory
        jpgFiles = dir(fullfile(currentSubSubDir, '*.jpg'));
        vidcsv = [];
        cd(currentSubSubDir);
        % Process the JPG files in the current sub-subdirectory (replace this with your processing code)
        for k = 1:numel(jpgFiles)
            currentFilePath = fullfile(currentSubSubDir, jpgFiles(k).name);
            % Replace the following line with your actual processing code

            fprintf('Processing file: %s\n', currentFilePath);

            img = imread(currentFilePath);
            
            % Create a cascade object detector for mouth detection
            mouthDetector = vision.CascadeObjectDetector('Mouth', 'MergeThreshold', 42);
            
            % Detect mouths in the image
            bbox = step(mouthDetector, img);
            
            % Check if at least one mouth is detected
            if ~isempty(bbox)
                % Choose the first detected mouth (you can modify this logic if needed)
                mouthBox = bbox(1, :);
            
                % Crop the image to the detected mouth box
                croppedImg = imcrop(img, mouthBox);
            
                % Convert the cropped image to grayscale
                grayCroppedImg = rgb2gray(croppedImg);
                
                %save the cropped greyscale image
                
                
                % Resize the cropped image and binary mask to the desired size
                targetSize = [256, 256];  % Adjust as needed
                resizedCroppedImg = imresize(grayCroppedImg, targetSize);

                imwrite(resizedCroppedImg, ['cropped_gray_' jpgFiles(k).name])
                % Apply a threshold to create a binary mask
                threshold = graythresh(resizedCroppedImg);
                binaryMask = imbinarize(resizedCroppedImg, threshold);
            
                % Perform morphological operations to clean up the binary mask
                % se = strel('disk', 5);  % Adjust the disk size as needed
                % binaryMask = imopen(binaryMask, se);
                % binaryMask = imclose(binaryMask, se);
            
                
                %resizedBinaryMask = imresize(binaryMask, targetSize);
            
                %get the stats of the image with region props
                stats = regionprops(binaryMask, 'Area', 'BoundingBox','Eccentricity', 'Centroid');
                
            % Find the index of the region with the maximum area
                [~, maxAreaIndex] = max([stats.Area]);
                
                % Extract information only from the most significant region
                if ~isempty(stats)
                    % Extract features for the most significant region
                    maxAreaStats = stats(maxAreaIndex);
                    area = maxAreaStats.Area;
                    width = maxAreaStats.BoundingBox(3);
                    height = maxAreaStats.BoundingBox(4);
                    eccentricity = maxAreaStats.Eccentricity;


                    [feat] = [area, width, height, eccentricity];
                    vidcsv = [vidcsv; {jpgFiles(k).name}, num2cell(feat)];
                    % Display extracted features
                    % fprintf('Area: %.2f\n', maxAreaStats.Area);
                    % fprintf('Bounding Box: %.2f\n',boundingBox);
                    % fprintf('Centroid: (%.2f, %.2f)\n', centroid(1), centroid(2));
                else
                    fprintf('No regions detected in the binary mask.\n');
                end
            
                % 
                % % Display the original cropped image
                % figure;
                % subplot(1, 3, 1);
                % imshow(resizedCroppedImg);
                % title('Cropped Image with Detected Mouth');
                % 
                % % Display the resized binary mask after morphological operations
                % subplot(1, 3, 2);
                % imshow(binaryMask);
                % title('Binary Mask');
                
            else
                fprintf('No mouths detected in the image.\n');
            end
            
        end
        writetable(cell2table(vidcsv), [subSubDirs(j).name '_binary_features.csv'], 'WriteVariableNames', false);
    end
end














































































































currentDir = pwd();
parentDir = fullfile(currentDir,'/recordings');
% cd(parentDir);
% recordingsDir = pwd();
% subDirs = dir(recordingsDir);
% subDirs = ([subDirs.isdir]);
filelist = dir(fullfile(parentDir, '**\*.*'));
filelist = filelist(~[filelist.isdir]); 

for nameDirIndex = 1:numel(subDirs)
    % nameVidDir = dir(fullfile(subDirs(nameDirIndex).name));
    nameVidDir = dir(subDirs(nameDirIndex).name);

    % Skip parent directory and current/parent directory indicators
    if strcmp(nameVidDir, '.') || strcmp(nameVidDir, '..')
        continue;
    end

    for vidDirIndex = 1:numel(nameVidDir)
        % vidNumDir = dir(fullfile(nameVidDir(vidDirIndex).folder, nameVidDir(vidDirIndex).name));
        vidNumDir = dir(nameVidDir(vidDirIndex).name);

         % Skip parent directory and current/parent directory indicators
        if strcmp(vidNumDir, '.') || strcmp(vidNumDir, '..')
            continue;
        end

        sourceDir = fullfile(parentDir,nameVidDir.name,vidNumDir.name);
        cd(sourceDir);
        jpgFiles = dir('output*.jpg');
        % jpgFiles = dir(fullfile(sourceDir,'*.jpg'));

        vidcsv = [];

        for i= 1:numel(jpgFiles)
            
            filName = jpgFiles(i).name;

            filePath = fullfile(sourceDir,fileName);

               % Load an image
            img = imread(filePath);
            
            % Create a cascade object detector for mouth detection
            mouthDetector = vision.CascadeObjectDetector('Mouth', 'MergeThreshold', 5);
            
            % Detect mouths in the image
            bbox = step(mouthDetector, img);
            
            % Check if at least one mouth is detected
            if ~isempty(bbox)
                % Choose the first detected mouth (you can modify this logic if needed)
                mouthBox = bbox(1, :);
            
                % Crop the image to the detected mouth box
                croppedImg = imcrop(img, mouthBox);
            
                % Convert the cropped image to grayscale
                grayCroppedImg = rgb2gray(croppedImg);
                
                %save the cropped greyscale image
                
                
                % Resize the cropped image and binary mask to the desired size
                targetSize = [256, 256];  % Adjust as needed
                resizedCroppedImg = imresize(grayCroppedImg, targetSize);

                imwrite(resizedCroppedImg, 'cropped_gray_'+fileName)
                % Apply a threshold to create a binary mask
                threshold = graythresh(resizedCroppedImg);
                binaryMask = imbinarize(resizedCroppedImg, threshold);
            
                % Perform morphological operations to clean up the binary mask
                % se = strel('disk', 5);  % Adjust the disk size as needed
                % binaryMask = imopen(binaryMask, se);
                % binaryMask = imclose(binaryMask, se);
            
                
                %resizedBinaryMask = imresize(binaryMask, targetSize);
            
                %get the stats of the image with region props
                stats = regionprops(binaryMask, 'Area', 'BoundingBox','Eccentricity', 'Centroid');
                
            % Find the index of the region with the maximum area
                [~, maxAreaIndex] = max([stats.Area]);
                
                % Extract information only from the most significant region
                if ~isempty(stats)
                    % Extract features for the most significant region
                    maxAreaStats = stats(maxAreaIndex);
                    area = maxAreaStats.Area;
                    width = maxAreaStats.BoundingBox(3);
                    height = maxAreaStats.BoundingBox(4);
                    eccentricity = maxAreaStats.Eccentricity;


                    [feat] = [area, width, height, eccentricity];
                    vidcsv = [vidcsv; {fileName}, num2cell(feat)];
                    % Display extracted features
                    % fprintf('Area: %.2f\n', maxAreaStats.Area);
                    % fprintf('Bounding Box: %.2f\n',boundingBox);
                    % fprintf('Centroid: (%.2f, %.2f)\n', centroid(1), centroid(2));
                else
                    fprintf('No regions detected in the binary mask.\n');
                end
            
                % 
                % % Display the original cropped image
                % figure;
                % subplot(1, 3, 1);
                % imshow(resizedCroppedImg);
                % title('Cropped Image with Detected Mouth');
                % 
                % % Display the resized binary mask after morphological operations
                % subplot(1, 3, 2);
                % imshow(binaryMask);
                % title('Binary Mask');
                
            else
                fprintf('No mouths detected in the image.\n');
            end
        
        writetable(cell2table(vidcsv), vidNumDir.name+'_binary_features.csv', 'WriteVariableNames', false);
            
        end


    end


end 