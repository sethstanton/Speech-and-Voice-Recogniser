import cv2
import numpy as np
import os
import tensorflow as tf
from typing import List
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d



# mfcc_features = np.random.rand(298, 16)  # Replace with your actual data
# video_data = np.random.rand(119, 4)  # Replace with your actual data

# # Create an interpolation function for each feature dimension
# interp_funcs = [interp1d(np.linspace(0, 1, len(video_data)), video_data[:, i], kind='linear') for i in range(video_data.shape[1])]

# # Interpolate video frames to match the number of MFCC features
# interpolated_video_frames = np.array([func(np.linspace(0, 1, len(mfcc_features))) for func in interp_funcs]).T

# print('before normalising')
# print(interpolated_video_frames)

# print(interpolated_video_frames.shape)

# interpolated_video_frames = ((interpolated_video_frames - np.min(interpolated_video_frames)) / (np.max(interpolated_video_frames) - np.min(interpolated_video_frames)) * 255).astype(np.uint8)
# print('---------------------------------------------------------------------------------------')
# print('after normalising')
# print(interpolated_video_frames)

# print(interpolated_video_frames.shape)


























































# def read_video(video_path):

#     cap = cv2.VideoCapture(video_path)

#     frames=[]
#     while True:
#         ret,frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#     cap.release()
#     return frames

# def interpolate_frames(frames,target_length):
#     current_length = len(frames)
#     x = np.linspace(0,1,current_length)
#     new_x = np.linspace(0,1,target_length)
#     interpolated_frames = []

#     for i in range(frames.shape[1]):
#         f = interp1d(x, frames[:,1], kind='linear', fill_value='extrapolate')
#         interpolated_frames.append(f(new_x))

#     return np.array(interpolated_frames).T

# def extract_lips(frame,lower_colour, upper_colour):
    
#     hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv_frame,lower_colour,upper_colour)
#     result = cv2.bitwise_and(frame, frame, mask=mask)
#     return result

# def on_change_lower(value):
#     lower_colour[0]=value
# def on_change_upper(value):
#     upper_colour[0]=value

# lower_colour = np.array([0,0,0])
# upper_colour = np.array([255,255,255])
# path = Path('recordings/')
# image_path_list = list(path.glob('*.png'))
# def main():
#     video_path = 'enter path here'
#     audio_mfcc = 'put np.load(mfcc_file path.npy) here'

#     frames = read_video(video_path)

#     target_length = audio_mfcc.shape[0]
#     interpolated_frames = interpolate_frames(np.array(frames),target_length)

#     # cv2.namedWindow('Lip Extraction')

#     # # create track bars for colour adjustment
#     # cv2.createTrackbar('Hue','Lip Extraction', lower_colour[0],255,on_change_lower)
#     # cv2.createTrackbar('Saturation','Lip Extraction', lower_colour[1],255,on_change_lower)
#     # cv2.createTrackbar('Value','Lip Extraction', lower_colour[2],255,on_change_lower)

#     # cv2.createTrackbar('Hue upper','Lip Extraction', upper_colour[0],255,on_change_upper)
#     # cv2.createTrackbar('Saturation upper','Lip Extraction', upper_colour[1],255,on_change_upper)
#     # cv2.createTrackbar('Value upper','Lip Extraction', upper_colour[2],255,on_change_upper)   

#     while True:
#         for i in range(len(interpolated_frames)):
#             frame = interpolated_frames






















































































































































































