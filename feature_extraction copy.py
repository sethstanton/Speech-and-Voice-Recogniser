import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt
from scipy.fftpack import dct


def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def framing(signal, sample_rate, frame_size=0.020, frame_stride=0.01):
    frame_samples = int(frame_size * sample_rate)
    step_samples = int(frame_stride * sample_rate)
    frames = []
    start = 0
    while start < len(signal) - frame_samples:
        frames.append(signal[start:start + frame_samples])
        start += step_samples
    return frames

def windowing(frames, frame_length):
    hamming_window = np.hamming(frame_length)
    return [frame * hamming_window for frame in frames]

def mel_filterbank(num_filters, NFFT, sample_rate, low_freq=0, high_freq=None):
    if not high_freq:
        high_freq = sample_rate / 2
    low_mel = 2595 * np.log10(1 + low_freq / 700.)
    high_mel = 2595 * np.log10(1 + high_freq / 700.)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595.0) - 1)
    bin_points = (NFFT + 1) * hz_points / sample_rate
    bin_points = bin_points.astype(int)

    filters = np.zeros((num_filters, int(NFFT / 2 + 1)))
    for i in range(1, num_filters + 1):
        filters[i - 1, bin_points[i - 1]:bin_points[i]] = (np.arange(bin_points[i - 1], bin_points[i]) - bin_points[i - 1]) / (bin_points[i] - bin_points[i - 1])
        filters[i - 1, bin_points[i]:bin_points[i + 1]] = 1 - (np.arange(bin_points[i], bin_points[i + 1]) - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
    return filters

def dct_II(x):
    return dct(x, type=2, norm='ortho')


def extract_mfcc(file_path, num_ceps=13, num_filters=26, NFFT=512):
    signal, sample_rate = sf.read(file_path, dtype='float32')
    emphasized = pre_emphasis(signal)
    frames_list = framing(emphasized, sample_rate)
    windowed_frames = windowing(frames_list, len(frames_list[0]))
    magnitude_spectrum = [np.abs(np.fft.fft(frame, n=NFFT))[:NFFT//2 + 1] for frame in windowed_frames]
    mel_filters = mel_filterbank(num_filters, NFFT, sample_rate)
    filtered_energy = np.dot(magnitude_spectrum, mel_filters.T)
    filtered_energy = np.where(filtered_energy == 0, np.finfo(float).eps, filtered_energy)
    log_values = np.log(filtered_energy)

    # Calculating the energy for each frame and storing in a container
    energy = np.array([np.sum(np.square(frame)) for frame in windowed_frames])
    energy = 0.99 * energy / max(abs(energy))
    # Calculating the velocity of the energy from frame to frame
    velocity = np.diff(energy, prepend=energy[0])
    velocity = 0.99 * velocity / max(abs(velocity))

    # Calculating the acceleration of the velocity from frame to frame
    acceleration = np.diff(velocity, prepend=velocity[0])
    acceleration = 0.99 * acceleration / max(abs(acceleration))

    # Apply DCT to each frame
    mfcc_values = np.array([dct_II(frame)[1:num_ceps + 1] for frame in log_values])

    features = np.column_stack((mfcc_values,energy,velocity,acceleration))

    # Creating file paths for mfcc plots
    # file_path_split = file_path.split("\\")
    # file_name_split = file_path_split[-1].split(".")
    # output_location = file_name_split[0]
    # output_directory_temp = output_location.split("_")
    # output_directory = output_directory_temp[0]
    # file_number_temp = file_name_split[0].split("_")
    # file_number = file_number_temp[1]
    # # print("file_number  = "+file_number)
    # # print("output_directory = "+output_directory)
    #
    #
    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory)
    #
    # # Generating and saving plots to directory
    # if file_number == "0":
    #     plt.figure(figsize=(15, 4))
    #
    #     # Plot the time domain signal
    #     plt.subplot(1, 3, 1)
    #     plt.title("Time Domain Signal")
    #     plt.plot(signal)
    #     plt.xlabel('Time')
    #     plt.ylabel('Amplitude')
    #
    #     # Plots the magnitude spectrum
    #     plt.subplot(1, 3, 2)
    #     plt.title("Magnitude Spectrum")
    #     plt.imshow(np.log(np.array(magnitude_spectrum).T), cmap='viridis', origin='lower', aspect='auto')
    #     plt.xlabel('Frequency Bins')
    #     plt.ylabel('Magnitude Log Scaled')
    #
    #     # Plots the Mel filterbank
    #     plt.subplot(1, 3, 3)
    #     plt.title("Mel Filterbank")
    #     plt.imshow(log_values, cmap='viridis', origin='lower', aspect='auto')
    #     plt.xlabel('Mel Filterbanks')
    #     plt.ylabel('Log Energy')
    #
    #     plt.savefig(os.path.join(output_directory, f"combined_plot_{output_directory}.png"))
    #     plt.close()

    return features


# def extract_mfcc_predictor(audio,name, num_ceps=13, num_filters=26, NFFT=512):
#     sample_rate = 16000.0
#     emphasized = pre_emphasis(audio)
#     frames_list = framing(emphasized, sample_rate)
#     windowed_frames = windowing(frames_list, len(frames_list[0]))
#     magnitude_spectrum = [np.abs(np.fft.fft(frame, n=NFFT))[:NFFT//2 + 1] for frame in windowed_frames]
#     mel_filters = mel_filterbank(num_filters, NFFT, sample_rate)
#     filtered_energy = np.dot(magnitude_spectrum, mel_filters.T)
#     filtered_energy = np.where(filtered_energy == 0, np.finfo(float).eps, filtered_energy)
#     log_values = np.log(filtered_energy)
#
#     # Calculating the energy for each frame and storing in a container
#     energy = np.array([np.sum(np.square(frame)) for frame in windowed_frames])
#     energy = 0.99*energy/max(abs(energy))
#     # Calculate the velocity of the energy from frame to frame
#     velocity = np.diff(energy, prepend=energy[0])
#     velocity = 0.99*velocity/max(abs(velocity))
#
#     # Calculating the acceleration of the velocity from frame to frame
#     acceleration = np.diff(velocity, prepend=velocity[0])
#     acceleration = 0.99*acceleration/max(abs(acceleration))
#
#     # Apply DCT to each frame
#     mfcc_values = np.array([dct_II(frame)[1:num_ceps + 1] for frame in log_values])
#
#     features = np.column_stack((mfcc_values,energy,velocity,acceleration))
#
#
#     # Generating and saving plots to directory
#
#     # Plot the time domain signal
#     # plt.figure(figsize=(10, 4))
#     # plt.subplot(1, 2, 1)
#     # plt.title("Time Domain Signal")
#     # plt.plot(audio)
#     # plt.xlabel('Time')
#     # plt.ylabel('Amplitude')
#     # plt.savefig(name+'_prediction_time_domain.png')
#     #
#     # # Plots the magnitude spectrum
#     # plt.subplot(1, 2, 2)
#     # plt.title("Magnitude Spectrum")
#     # plt.imshow(np.log(np.array(magnitude_spectrum).T), cmap='viridis', origin='lower', aspect='auto')
#     # plt.xlabel('Frequency Bins')
#     # plt.ylabel('Magnitude Log Scaled')
#     # plt.savefig(name+'_prediction_magnitude_spectrum.png')
#     #
#     # # Plots the Mel filterbank
#     # plt.subplot(1, 3, 3)
#     # plt.title("Mel Filterbank")
#     # plt.imshow(log_values, cmap='viridis', origin='lower', aspect='auto')
#     # plt.xlabel('Mel Filterbanks')
#     # plt.ylabel('Log Energy')
#     # plt.savefig(name+'_prediction_mel_filterbank.png')
#     #
#     # plt.close()
#
#     return features

# base_directory = "recordings"
# mfcc_list = []
# file_paths = []
#
# for person_folder in os.listdir(base_directory):
#     person_path = os.path.join(base_directory, person_folder)
#
#     if os.path.isdir(person_path):
#         for file in os.listdir(person_path):
#             if file.endswith(".wav"):
#                 file_path = os.path.join(person_path, file)
#                 mfcc_values = extract_mfcc(file_path)
#                 mfcc_array = np.array(mfcc_values)
#                 file_paths_array = np.array(file_path)
#                 file_name = file_path.split('.')[0]+"_mfcc_features.npy"
#                 path_name = file_path.split('.')[0]+"_file_npy_paths.npy"
#                 print(f' saving: {file_name}')
#                 np.save(file_name, mfcc_array)
#                 print(f' saving: {path_name}')
#                 np.save(path_name, file_paths_array)
#
#                 # mfcc_list.append(mfcc_values)
#                 # file_paths.append(file_path)




# use np.load to load them later
# def process_wav_files_in_subdirs(parent_dir):
#     for subdir in os.listdir(parent_dir):
#         subdir_path = os.path.join(parent_dir, subdir)
#
#         # Check if the path is indeed a directory
#         if os.path.isdir(subdir_path):
#             for file in os.listdir(subdir_path):
#                 if file.endswith(".wav"):
#                     process_wav_file(subdir_path, file)
#
# def process_wav_file(subdir_path, file):
#     file_path = os.path.join(subdir_path, file)
#     mfcc_values = extract_mfcc(file_path)
#     mfcc_array = np.array(mfcc_values)
#
#     # Save MFCC features as .npy file
#     file_name_mfcc = os.path.join(subdir_path, file.split('.')[0] + "_mfcc_features.npy")
#     np.save(file_name_mfcc, mfcc_array)
#     print(f'Saved MFCC features to: {file_name_mfcc}')
#
# # Example usage - replace 'recordings/Yubo' with the correct parent directory path
# process_wav_files_in_subdirs('recordings/Ben')

base_directory = "recordings/Ben"

mfcc_list = []
file_paths = []
##### NEW CODE #####
target_length = 442
####################
for person_folder in os.listdir(base_directory):
    person_path = os.path.join(base_directory, person_folder)
    if os.path.isdir(person_path):
        for file in os.listdir(person_path):
            if file.endswith(".wav"):
                file_path = os.path.join(person_path, file)
                mfcc_values = extract_mfcc(file_path)
                mfcc_array = np.array(mfcc_values)
                ############ NEW CODE ################
                padding_before = (target_length - mfcc_array.shape[0]) // 2
                padding_after = target_length - mfcc_array.shape[0] - padding_before

                mfcc_features_padded = np.pad(
                    mfcc_array,
                    ((padding_before, padding_after), (0, 0)),
                    mode='reflect'
                )
                ########################################
                file_paths_array = np.array(file_path)
                file_name = file_path.split('.')[0] + "_mfcc_features.npy"

                print(f' saving: {file_name}')
                np.save(file_name, mfcc_features_padded)