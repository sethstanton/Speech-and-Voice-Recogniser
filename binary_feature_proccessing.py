import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd 
from pathlib import Path
import os

"""
mfcc_features = np.load('recordings/Ben/Ben_02/Ben_audio_02_mfcc_features.npy')
print(mfcc_features.shape)
shape = (429,16)
"""
# file_path = 'recordings/Ben/Ben_01/Ben_01_binary_features.csv'

# # Load CSV into pandas DataFrame
# df = pd.read_csv(file_path, header=None)

# # Split the first element into the first column and the rest into a list
# # df[1] = df.apply(lambda row: [int(x) for x in row[1:].tolist()], axis=1)

# # Display the resulting DataFrame
# # print(df)
# binaryFeaturesArray = np.array(df.iloc[:, 1:].values)

# Display the resulting NumPy array
# print(binaryFeaturesArray)

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

    audio_timesteps = audio_feat.shape[0]

    # Initialize an array to store interpolated visual features
    visual_feat_interp = np.zeros((audio_timesteps, visual_feat.shape[1]))

    for feature_dim in range(visual_feat.shape[1]):
        cubicSpline = CubicSpline(np.arange(visual_feat.shape[0]), visual_feat[:, feature_dim])
        visual_feat_interp[:, feature_dim] = cubicSpline(np.linspace(0, visual_feat.shape[0] - 1, audio_timesteps))

    return visual_feat_interp


DIR_PATH = Path('recordings/')
for child in DIR_PATH.glob('*'):
    # print(child)
    for subDir in child.glob('*'):
        # print(subDir)
        binary_file = None
        mfcc_file = None
        for subsubDir in subDir.glob('*'):
                # print(subsubDir)
                subsubDir = str(subsubDir)
                if 'binary' in subsubDir.split('_'):
                    binary_file = subsubDir
                elif 'mfcc' in subsubDir.split('_'):
                     mfcc_file = subsubDir
        
        if (binary_file != None) and (mfcc_file != None):
            # print(binary_file)
            # print(mfcc_file)
            df = pd.read_csv(binary_file, header=None)
            visual_feat = np.array(df.iloc[:, 1:].values)
            audio_feat = np.load(mfcc_file)
            interp_features = visual_feature_interp(visual_feat,audio_feat)
            filename = str(subDir).split('\\')[2]+'_ib_features.npy'
            filename = os.path.join(subDir,filename)
            print(filename)
            np.save(filename,interp_features)