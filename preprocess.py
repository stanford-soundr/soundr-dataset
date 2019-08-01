# %%
import scipy.io.wavfile
import os
import psutil
import numpy as np
import pickle
import scipy.signal
import math
from numpy import array
import quaternion

import seaborn as sns
import matplotlib.pyplot as plt

import webrtcvad

data_dir = "/home/soundr-share/Soundr-Data"
tracking_file_name = "tracking.pickle"  # 7 x 1
sound_file_name = "audio.wav"  # 8 x 1; last channel unused
# VR onboard mic data is 1x1

window_size = 0.3
vad = webrtcvad.Vad(1)


def audio_range_from_tracking(i, temp_window_size=window_size):
    """

    :param i: center index of audio of ith sample
    :param temp_window_size:
    :return: range of audio indices of window size
    """
    time = i / tracking_sample_rate
    start_time = time - temp_window_size / 2
    sample_count = int(temp_window_size * audio_sample_rate)
    audio_start_index = int(start_time * audio_sample_rate)
    audio_end_index = sample_count + audio_start_index
    return audio_start_index, audio_end_index


def validate_audio_range(audio_range):
    """
    Validates range is within the data
    :param audio_range:
    :return:
    """
    audio_start_index, audio_end_index = audio_range
    return audio_start_index >= 0 and audio_end_index < len(audio_data)


data_sub_dirs = os.listdir(data_dir)

input_array = []
output_array = []

segment_length = 0.1
energy_segment_length = 0.3

def get_audio_segment(audio_data, sample_rate, segment_id, override_length=None):
    """
    if you ovverride length, segment will still be segmented as original length
    :param audio_data:
    :param sample_rate:
    :param segment_id:
    :param override_length:
    :return:
    """
    start_idx = int(sample_rate * segment_id * segment_length)
    end_idx = int(sample_rate * (segment_id + 1) * segment_length)

    if override_length is not None:
        new_start_idx = int((start_idx + end_idx - sample_rate * override_length) / 2)
        new_end_idx = int((start_idx + end_idx + sample_rate * override_length) / 2)
        start_idx = new_start_idx
        end_idx = new_end_idx

    if start_idx >= 0 and end_idx < len(audio_data):
        return audio_data[start_idx: end_idx]
    else:
        return None

def get_tracking_sample(tracking_data, sample_rate, segment_id):
    try:
        idx = sample_rate * (segment_id + 0.5) * segment_length
        prev_idx = int(math.floor(idx))
        next_idx = int(prev_idx + 1)
        return tracking_data[prev_idx]

    except IndexError:
        return None


combined_segments_data = [[], []]


def vad_segment_valid(sample_frame: np.array, sample_rate=48000, skip=3):
    """
    Detect if audio contains speech
    :param sample_frame: frame of audio, sample rate 16000, int32 format
    :param sample_rate: only in 16000, 32000, or 48000 Hz
    :param skip: sample one in how many frames
    """
    vad_valid = False
    vad_length = int(sample_rate * 0.03)
    for start in range(0, len(sample_frame), vad_length * skip):
        if start + 480 > len(sample_frame):
            start = len(sample_frame) - vad_length
        end = start + vad_length
        sub_frame = np.int16(sample_frame[start:end] / 65536 / 8)
        sub_frame_valid = vad.is_speech(sub_frame.tobytes(), sample_rate)
        vad_valid = vad_valid or sub_frame_valid
    return vad_valid


for data_sub_dir in data_sub_dirs:#choose how many dirs [0:3]
    if data_sub_dir.startswith(".git"):
        continue

    if data_sub_dir.startswith("_"):
        continue
    data_path = os.path.join(data_dir, data_sub_dir)

    tracking_file_path = os.path.join(data_path, tracking_file_name)
    sound_file_path = os.path.join(data_path, sound_file_name)

    audio_sample_rate, audio_data = scipy.io.wavfile.read(sound_file_path)
    audio_data = np.delete(audio_data, [7, 15], 1)  # these two channels are empty
    with open(tracking_file_path, "rb") as tracking_file:
        tracking_sample_rate = 90
        tracking_data = np.array(pickle.load(tracking_file))

    xs = []
    zs = []
    for data in tracking_data:  # vector with length 7- x,y,z, a,b,c,d
        if data is not None:
            xs += [data[0]]
            zs += [data[2]]
    sns.scatterplot(xs, zs)
    plt.title(data_sub_dir)
    plt.show()

    audio_mono_data = audio_data.sum(axis=1)

    #%%
    audio_energy_threshold = 38

    def get_segments(offset = 0):
        segments = []
        last_valid = False

        for i in range(int(len(audio_mono_data) / (segment_length * audio_sample_rate))):
            energy_segment = get_audio_segment(audio_mono_data, audio_sample_rate, i + offset, override_length=0.5)
            if energy_segment is None:
                audio_valid = False
            else:
                audio_valid = vad_segment_valid(energy_segment[::3], 16000)
            audio_segment = get_audio_segment(audio_data, audio_sample_rate, i + offset)
            tracking_sample = get_tracking_sample(tracking_data, tracking_sample_rate, i + offset)
            valid = audio_valid and audio_segment is not None and tracking_sample is not None
            if valid:
                if not last_valid:
                    segments += [[]]
                segments[len(segments) - 1] += [(tracking_sample, audio_segment)]
            last_valid = valid
        return segments

    # offsets = [0, 0.033333, 0.066667]
    offsets = [0]

    segments = [segment for offset in offsets for segment in get_segments(offset)]

    segment_long_threshold = 20

    def process_segment(segment):
        if len(segment) > 20:
            new_len = len(segment) / math.ceil(len(segment) / 20)
            new_segments = []
            # for start in range(0, len(segment), int(new_len / 2)):
            for start in range(0, len(segment), int(new_len)):
                end = start + int(new_len)
                if end > len(segment):
                    break
                new_segments += [segment[start:end]]
            return new_segments
        else:
            return [segment]


    segment_length_stat = [len(segment) for segment in segments]

    sns.distplot(segment_length_stat)
    plt.title(data_sub_dir)
    plt.show()

    # all old segments less than 20 and then all new segments that are old (>20) segments chopped up
    processed_segments = [new_segment for segment in segments for new_segment in process_segment(segment)]

    new_segments_length_stat = [len(segment) for segment in processed_segments]

    combined_segments = [[], []]

    for segment in processed_segments:
        input_array = []
        output_array = []
        for data in segment:
            input_array += [np.transpose(data[1])]
            output_array += [data[0]]
        combined_segments[0] += [np.array(input_array)]
        combined_segments[1] += [np.array(output_array)]

    combined_segments_data[0] += combined_segments[0]
    combined_segments_data[1] += combined_segments[1]

    print(f"loop {psutil.virtual_memory()}")

#%%
csdInput = array(combined_segments_data[0])
csdOutput = array(combined_segments_data[1])
#convert to np array
print(f"pre-save {psutil.virtual_memory()}")
# np.save("/home/soundr-share/train_set4_example_old.npy", combined_segments_data)

np.save("/home/soundr-share/train_set4_input_small_batch.npy", csdInput)
np.save("/home/soundr-share/train_set4_output_small_batch.npy", csdOutput)
print(f"post-save {psutil.virtual_memory()}")
#convert & save separately
