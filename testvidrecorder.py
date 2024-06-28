import cv2 as cv
import sys
import os
import threading
import queue
import sounddevice as sd
import soundfile as sf
import time

recordings_folder = 'recordings'
if not os.path.exists(recordings_folder):
    os.makedirs(recordings_folder)

def generate_unique_filename(base_name, dir_path, extension):
    index = 1
    while True:
        new_filename = f"{base_name}_{str(index).zfill(2)}"
        full_path = os.path.join(dir_path, f"{new_filename}.{extension}")
        if not os.path.exists(full_path):
            return new_filename
        index += 1


class VideoRecorder(object):
    def __init__(self, name):
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        assert self.cap.isOpened(), "Failed to open video capture device"

        base_video_filename = f'{name}_video'
        base_audio_filename = f'{name}_audio'

        video_filename = generate_unique_filename(base_video_filename, recordings_folder, 'mp4')
        self.video_file_name = os.path.join(recordings_folder, f"{video_filename}.mp4")
        unique_audio_filename = generate_unique_filename(base_audio_filename, recordings_folder, 'wav')

        self.frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        self.codec = cv.VideoWriter_fourcc('m', 'p', '4', 'v')

        self.frame_rate = 15
        self.output_video = cv.VideoWriter(self.video_file_name, self.codec, self.frame_rate, (self.frame_width, self.frame_height))
        assert self.output_video.isOpened(), "Failed to open video writer"

        audio_file_path = os.path.join(recordings_folder, f'{unique_audio_filename}.wav')
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

        self.audio_thread = AudioRecorder()
        self.audio_thread.start(unique_audio_filename, recordings_folder)

        start_time = time.time()
        while time.time() - start_time < 4:  # Record for 4 seconds
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            self.output_video.write(frame)

            # Display the frame in a window
            cv.imshow('Recording', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.audio_thread.stop()
        self.output_video.release()
        cv.destroyAllWindows()  # Close the display window

        outputfilename = os.path.join(recordings_folder, f"output_{video_filename}.mp4")
        cmd = f"ffmpeg -y -r {self.frame_rate} -i {audio_file_path} -r {self.frame_rate} -i {self.video_file_name} -c:v copy -c:a aac -strict experimental {outputfilename}"
        os.system(cmd)

class AudioRecorder():

    def __init__(self):
        self.open = True
        self.channels = 1
        self.q = queue.Queue()
        device_info = sd.query_devices(0, 'input')
        self.samplerate = int(device_info['default_samplerate'])

    def callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def record(self):
        with sf.SoundFile(self.file_name, mode='x', samplerate=self.samplerate,
                      channels=self.channels) as file:
            with sd.InputStream(samplerate=self.samplerate,
                                channels=self.channels, callback=self.callback):

                while(self.open == True):
                    file.write(self.q.get())

    def stop(self):
        self.open = False

    def start(self, file_name, file_dir):
        self.open = True
        self.file_name = os.path.join(file_dir, f'{file_name}.wav')  # Add extension here
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()


if __name__ == '__main__':
    names = ['Test']
    # names = ["PomatoPlant", "PlasticStraw", "Johnathan", "Nathan"]
    recorded_names = set()  # Set to keep track of names already recorded
    # names = ['Chiedozie', 'Seth', 'Marc', 'Ryan', 'Charlie', 'Francesca', 'Nischal', 'Carlos', 'Lindon', 'James',
    #               'Yubo', 'Jack', 'Ethan', 'Bonney', 'William', 'Yubo', 'El', 'Jake', 'Robin', 'Ben']

    for name in names:  # Loop through the list of names
        print(f"Recording : {name}")
        if name in recorded_names:
            print(f"Skipping duplicate name: {name}")
            continue  # Skip the recording for this duplicate name

        rec = VideoRecorder(name)
        recorded_names.add(name)  # Add the name to the set of recorded names
        time.sleep(1)  # 1-second pause between each recording

