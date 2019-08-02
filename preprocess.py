# %%
import os
import psutil
import numpy as np
import math
from numpy import array

import seaborn as sns
import matplotlib.pyplot as plt

import webrtcvad

data_dir = "/home/soundr-share/Soundr-Data-2"
tracking_file_name = "vr_tracking_data.npy"  # 7 x 1
sound_file_name = "mic_data.npy"  # 8 x 1; last channel unused
vad_file_name = "vr_audio_data.npy"  # VR onboard mic data is 1x1; use specifically for VAD
offset_file_name = "offset.npy"

window_size = 0.3  # window of each pieced audio segment
vad = webrtcvad.Vad(1)  # voice activity detection to understand when during the audio someone is actually speaking

data_sub_dirs = os.listdir(data_dir)

input_array = []
output_array = []

segment_length = 0.1
energy_segment_length = 0.3


def get_audio_segment(audio_datas, sample_rate, segment_id, offset_rate, override_length=None):
    """
    if you ovverride length, segment will still be segmented as original length
    :rtype: object
    :param offset_rate:
    :param audio_datas:
    :param sample_rate:
    :param segment_id:
    :param override_length:
    :return:
    """
    start_idx = int(sample_rate * segment_id * segment_length * offset_rate)
    end_idx = start_idx + int(sample_rate * segment_length)

    if override_length is not None:
        new_start_idx = int((start_idx + end_idx - sample_rate * override_length) / 2)
        new_end_idx = int((start_idx + end_idx + sample_rate * override_length) / 2)
        start_idx = new_start_idx
        end_idx = new_end_idx

    if start_idx >= 0 and end_idx < len(audio_datas):
        return audio_datas[start_idx: end_idx]
    else:
        return None


def get_tracking_sample(tracking_datas, sample_rate, segment_id, offset_rate):
    try:
        idx = sample_rate * (segment_id + 0.5) * segment_length * offset_rate
        prev_idx = int(math.floor(idx))
        # next_idx = int(prev_idx + 1)
        return tracking_datas[prev_idx]

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
        sub_frame = np.int16(sample_frame[start:end] * (2 ** 15))
        sub_frame_valid = vad.is_speech(sub_frame.tobytes(), sample_rate)
        vad_valid = vad_valid or sub_frame_valid
    return vad_valid


for data_sub_dir in data_sub_dirs:  # choose how many dirs [0:3]
    if data_sub_dir.startswith(".git"):  # omit any git folders
        continue

    if data_sub_dir.startswith("_"):  # omit selected data folders
        continue

    data_path = os.path.join(data_dir, data_sub_dir)  # data path is one specific set of collected data

    # all lead to npy files
    tracking_file_path = os.path.join(data_path, tracking_file_name)
    sound_file_path = os.path.join(data_path, sound_file_name)
    vad_file_path = os.path.join(data_path, vad_file_name)
    offset_file_path = os.path.join(data_path, offset_file_name)

    # audio_sample_rate, audio_data = scipy.io.wavfile.read(sound_file_path)
    AUDIO_SAMPLE_RATE = 48000
    TRACKING_SAMPLE_RATE = 72
    audio_data = np.load(sound_file_path)
    audio_data = np.delete(audio_data, [7], 1)  # this channel is empty
    tracking_data = np.load(tracking_file_path)
    vad_data = np.load(vad_file_path)
    offsets = np.load(offset_file_path)

    data_length = np.array([vad_data.shape[0], audio_data.shape[0], tracking_data.shape[0]])

    offset_rates = data_length / (offsets.astype(float) + data_length)

    xs = []
    zs = []
    for data in tracking_data:  # vector with length 7- x,y,z, a,b,c,d
        if data is not None:
            xs += [data[0]]
            zs += [data[2]]
    sns.scatterplot(xs, zs)
    plt.title(data_sub_dir)
    plt.show()

    audio_mono_data = vad_data  # maybe misaligned; using onboard vr mic data for thresholding instead of normal mic

    # %%
    AUDIO_ENERGY_THRESHOLD = 38


    def get_segments(offset, offset_rates_param):
        segments_internal = []
        last_valid = False

        for i in range(int(len(audio_mono_data) / (segment_length * AUDIO_SAMPLE_RATE))):
            energy_segment = get_audio_segment(
                audio_mono_data,
                AUDIO_SAMPLE_RATE,
                i + offset,
                offset_rates_param[0],
                override_length=0.5
            )
            if energy_segment is None:
                audio_valid = False
            else:
                audio_valid = vad_segment_valid(energy_segment[::3], 16000)
            audio_segment = get_audio_segment(audio_data, AUDIO_SAMPLE_RATE, i + offset, offset_rates_param[1])
            tracking_sample = \
                get_tracking_sample(tracking_data, TRACKING_SAMPLE_RATE, i + offset, offset_rates_param[2])
            valid = audio_valid and audio_segment is not None and tracking_sample is not None
            if valid:
                if not last_valid:
                    segments_internal += [[]]
                segments_internal[len(segments_internal) - 1] += [(tracking_sample, audio_segment)]
            last_valid = valid
        return segments_internal


    # offsets = [0, 0.033333, 0.066667]
    start_offsets = [0]

    segments = [segment for start_offset in start_offsets for segment in get_segments(start_offset, offset_rates)]

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

# %%
csdInput = array(combined_segments_data[0])
csdOutput = array(combined_segments_data[1])
# convert to np array
print(f"pre-save {psutil.virtual_memory()}")
# np.save("/home/soundr-share/train_set4_example_old.npy", combined_segments_data)

np.save("/home/soundr-share/train_set10_input.npy", csdInput)
np.save("/home/soundr-share/train_set10_output.npy", csdOutput)
print(f"post-save {psutil.virtual_memory()}")
# convert & save separately
