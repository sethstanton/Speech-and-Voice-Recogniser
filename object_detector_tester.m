pathName = 'recordings/Ben/Ben_50/output_Ben_video_50_frame_90.jpg';

img = imread(pathName);
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
    % imwrite(grayCroppedImg, ['cropped_gray_' fileName]);
    
    % Resize the cropped image and binary mask to the desired size
    targetSize = [256, 256];  % Adjust as needed
    resizedCroppedImg = imresize(grayCroppedImg, targetSize);
    figure;
    imshow(resizedCroppedImg);
    % imwrite(resizedCroppedImg, ['cropped_gray_' fileName]);
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
        % vidcsv = [vidcsv; {fileName}, num2cell(feat)];
        % vidcsv = [vidcsv,fileName, feat];
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