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
        CroppedGray = dir(fullfile(currentSubSubDir, 'cropped_gray_*'));
        vidcsv = [];
        cd(currentSubSubDir);
        % Process the JPG files in the current sub-subdirectory (replace this with your processing code)
        for k = 1:numel(CroppedGray)
            currentFilePath = fullfile(currentSubSubDir, CroppedGray(k).name);
            % Replace the following line with your actual processing code
            fprintf('Processing file: %s\n', currentFilePath);
            fileName = CroppedGray(i).name;

               % Load an image
            img = imread(currentFilePath);

            tmp = upper_left_tri(img,length(img));
            flatTmp = reshape(tmp.',1,[]);
            zeroTmp = nonzeros(flatTmp');
            zeroTmp = reshape(zeroTmp.',1,[]);
            if~isempty(tmp)
                vidcsv = [vidcsv;{fileName},zeroTmp];
            else
                fprintf('tmp is empty');
            end


        end
    writetable(cell2table(vidcsv), [subSubDirs(j).name '_dct_features.csv'], 'WriteVariableNames', false);

    end
end












































% 
% currentDir = pwd();
% parentDir = fullfile(currentDir,'/recordings');
% subDirs = dir(parentDir);
% subDirs = ([subDirs.isdir]);
% 
% for nameDirIndex = 1:numel(subDirs)
%     nameVidDir = dir(fullfile(subDirs(nameDirIndex).name));
% 
%     % Skip parent directory and current/parent directory indicators
%     if strcmp(nameVidDir, '.') || strcmp(nameVidDir, '..')
%         continue;
%     end
% 
%     for vidDirIndex = 1:numel(nameVidDir)
%         % vidNumDir = dir(nameVidDir(vidDirIndex));
%         vidNumDir = dir(fullfile(nameVidDir(vidDirIndex).folder, nameVidDir(vidDirIndex).name));
%          % Skip parent directory and current/parent directory indicators
%         if strcmp(vidNumDir, '.') || strcmp(vidNumDir, '..')
%             continue;
%         end
% 
%         sourceDir = fullfile(parentDir,nameVidDir.name,vidNumDir.name);
%         cd(sourceDir);
%         % croppedGrayFiles = dir(fullfile(sourceDir,'cropped_gray_*'));
%         croppedGrayFiles = dir('cropped_gray_*');
% 
%         vidcsv = [];
% 
%         for i= 1:numel(croppedGrayFiles)
% 
%             fileName = croppedGrayFiles(i).name;
% 
%             filePath = fullfile(sourceDir,fileName);
% 
%                % Load an image
%             img = imread(filePath);
% 
%             tmp = upper_left_tri(img,length(img));
%             flatTmp = reshape(tmp.',1,[]);
%             zeroTmp = nonzeros(flatTmp');
%             zeroTmp = reshape(zeroTmp.',1,[]);
%             if~isempty(tmp)
%                 vidcsv = [vidcsv;{fileName},zeroTmp];
%             else
%                 fprintf('tmp is empty');
%             end
% 
%         end
%     writetable(cell2table(vidcsv), fullfile(sourceDir, [vidNumDir.name, '_dct_features.csv']), 'WriteVariableNames', false);
%     % writematrix(vidcsv,vidNumDir.name+'_dct_features.csv')
%     end
% 
% 
% end 

function triangle = upper_left_tri(img,triangle_size)
    img_dct = dct2(img);
    img_zeros = (double(zeros(size(img))));
    [rows,cols] = size(img);
    for i=1:rows
        for j=1:cols
            if j<=((triangle_size/32)-i)
                img_zeros(i,j)=1;
            end
        end
    end

    triangle = img_dct.*img_zeros;

    % figure; 
    % imshow(log(abs(triangle)),[],'colormap',jet(64));
    % triangle = idct2(triangle);
end