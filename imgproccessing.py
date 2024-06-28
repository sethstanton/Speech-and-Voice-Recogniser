import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
notes

- open cv image shapes are (H,W,C)
    * read in with img.shape

"""

'''

video = cv2.VideoCapture('recordings/output_Yubo_video_21.mp4')

# fps = video.get(cv2.CAP_PROP_FPS)
# total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
# print(f'frames per second = {fps}')
# print(f'total frames = {total_frames}')


# video.set(cv2.CAP_PROP_POS_FRAMES,0)

ret, frame = video.read()
# print(frame)
# cv2.imshow('frame',frame)
# cv2.waitKey(0)
# cv2.imwrite('test_video_frame.png',frame)

i = 0
while ret:
    cv2.imwrite('test_video_frame_'+str(i)+'.png',frame)
    cv2.waitKey(0)
    i+=1
    ret, frame = video.read()
'''

# def frame_extractor
#     video =cv2.VideoCapture
#


# to iterate through a folder called recordings where 20 names for each name
# I have audio wav files, just visual mp4 files, and a combined audio visual mp4
# files. I need a folder for each person and i need a folder for each sample for
# that person. I have 50 samples for each person.

# video = cv2.VideoCapture('recordings/output_Yubo_video_21.mp4')

# # Initialize frame counter
# i = 0

# while True:
#     ret, frame = video.read()
#     if not ret:
#         break

#     # Save the frame
#     cv2.imwrite('test_video_frame_'+str(i)+'.png', frame)

#     # If it's the first frame, select the ROI
#     if i == 0:
#         r = cv2.selectROI(frame)
#         cv2.destroyAllWindows()

#     # Crop the frame to get the ROI
#     roi = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

#     # Calculate the mean color of each channel
#     mean_color = cv2.mean(roi)[:3]

#     # Calculate the Euclidean distance
#     euclidean_distance = np.sqrt(np.sum([np.square(mc - np.mean(mean_color)) for mc in mean_color]))

#     # Apply threshold (replace 'threshold_value' with your threshold)
#     _, thresholded = cv2.threshold(frame, euclidean_distance, 255, cv2.THRESH_BINARY)

#     # Save the thresholded frame
#     cv2.imwrite('thresholded_frame_'+str(i)+'.png', thresholded)

#     i += 1

# video.release()
# cv2.destroyAllWindows()

def extract_frames_from_subdirs(parent_dir):
    # Loop through all subdirectories in the parent directory
    for subdir in os.listdir(parent_dir):
        subdir_path = os.path.join(parent_dir, subdir)

        # Check if the path is indeed a directory
        if os.path.isdir(subdir_path):
            # Find the video file inside the subdirectory
            for file in os.listdir(subdir_path):
                if file.__contains__("output"):
                    if file.endswith(".mp4"):
                        video_path = os.path.join(subdir_path, file)
                        process_video(video_path)

def process_video(video_path):
    # Create a VideoCapture object
    video = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    # Extract the directory and base name from the video path
    base_dir, video_file = os.path.split(video_path)
    base_name = os.path.splitext(video_file)[0]

    # Initialize frame count
    frame_count = 0

    # Read until video is completed
    while True:
        # Capture frame-by-frame
        ret, frame = video.read()

        # Break the loop if there are no more frames
        if not ret:
            break

        # Save frame as JPEG file
        output_filename = os.path.join(base_dir, f"{base_name}_frame_{frame_count}.jpg")
        cv2.imwrite(output_filename, frame)
        frame_count += 1

    # When everything is done, release the video capture object
    video.release()
    print(f"Extraction complete for {video_path}. {frame_count} frames were saved.")

# Example usage - replace 'recordings/Yubo' with the correct parent directory path
extract_frames_from_subdirs('test_recordings/Yubo')

















""" 
define area by mouse
"""
# img = cv.imread('test_video_frame_0.png')
# cv.namedWindow('ROI') 

# r=cv.selectROI('ROI', img,False,False)
# imROI = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
# cv.destroyWindow('ROI')
# cv.imshow("ROI", imROI)
# cv.waitKey(0)
# cv.destroyAllWindows()
# print(imROI)

"""
colour converter

"""

# img = cv2.imread('test_video_frame_20.png')

# hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# cv2.imshow('HSV',hsv)
# cv2.waitKey(0)

# def select_lips(image):
    
#     cv2.namedWindow('Select Lips')
#     cv2.imshow('Select Lips', image)

#     selected_colour = None
#     def mouse_callback(event,x,y,flags, param):
#         nonlocal selected_colour

#         if event == cv2.EVENT_LBUTTONDOWN:
#             selected_colour = image[y,x]
#             print('selected colour (bgr):',selected_colour)

#     cv2.setMouseCallback('Select lips', mouse_callback)

#     while True:
#         key = cv2.waitKey(0)
#         if key ==27:
#             break
#     cv2.destroyAllWindows()

#     return selected_colour

# selected_colour = select_lips(img)
























