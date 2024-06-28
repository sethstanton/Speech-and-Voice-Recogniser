import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd 
from pathlib import Path
import os
import glob
import numpy as np
import pandas as pd 
from pathlib import Path
import os
import glob

def visual_feature_interp(visual_feat, audio_feat):
    """
    Return visual features matching the number of frames of the supplied audio
    feature. The time dimension must be the first dim of each feature
    matrix; uses the cubic spline interpolation method - adapted from Matlab.

    Args:
        visual_feat: the input visual features size: (visual_num_frames, visual_feature_len)
        audio_feat: the input audio features size: (audio_num_frames, audio_feature_len)

    Returns:
        visual_feat_interp: visual features that match the number of frames of the audio feature
    """

    audio_timesteps = audio_feat#.shape[0]

    # Initialize an array to store interpolated visual features
    visual_feat_interp = np.zeros((audio_timesteps, visual_feat.shape[1]))

    for feature_dim in range(visual_feat.shape[1]):
        cubicSpline = CubicSpline(np.arange(visual_feat.shape[0]), visual_feat[:, feature_dim])
        visual_feat_interp[:, feature_dim] = cubicSpline(np.linspace(0, visual_feat.shape[0] - 1, audio_timesteps))

    return visual_feat_interp

# df = pd.read_csv('recordings/Ben/Ben_01/Ben_01_binary_features.csv', header=None)
# print(df.shape[0])
# binary_feat = np.array(df.iloc[:, 1:].values)
# print(binary_feat.shape[0])




max_rows = 0

for csv_file in sorted(glob.glob('recordings/*/*/*.csv')):
    df = pd.read_csv(csv_file)
    num_rows = df.shape[0]
    if num_rows > max_rows:
        max_rows = num_rows


DIR_PATH = Path('recordings/')
for child in DIR_PATH.glob('*'):
    # print(child)
    for subDir in child.glob('*'):
        # print(subDir)
        binary_file = None
        for subsubDir in subDir.glob('*'):
                # print(subsubDir)
                subsubDir = str(subsubDir)
                if 'binary' in subsubDir.split('_'):
                    binary_file = subsubDir

        if binary_file != None:
            # print(binary_file)
            # print(mfcc_file)
            df = pd.read_csv(binary_file, header=None)
            binary_feat = np.array(df.iloc[:, 1:].values)
            interp_features = visual_feature_interp(binary_feat,max_rows)
            filename = str(subDir).split('\\')[2]+'_binaryarray_features.npy'
            filename = os.path.join(subDir,filename)
            print(filename)
            np.save(filename,interp_features)



for child in DIR_PATH.glob('*'):
    # print(child)
    for subDir in child.glob('*'):
        # print(subDir)
        dct_file = None

        for subsubDir in subDir.glob('*'):
                # print(subsubDir)
                subsubDir = str(subsubDir)
                if 'dct' in subsubDir.split('_'):
                    dct_file = subsubDir

        if dct_file != None:
            # print(binary_file)
            # print(mfcc_file)
            df = pd.read_csv(dct_file, header=None)
            dct_feat = np.array(df.iloc[:, 1:].values)
            interp_features = visual_feature_interp(dct_feat,max_rows)
            filename = str(subDir).split('\\')[2] +'_dctarray_features.npy'
            filename = os.path.join(subDir,filename)
            print(filename)
            np.save(filename,interp_features)

# Used for deleting bad data
# for img in sorted(glob.glob('recordings/*/*/cropped_gray_*')):
#   print(img)
#   os.remove(img)