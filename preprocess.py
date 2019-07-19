# %%
import scipy.io.wavfile
import os
import numpy as np
import pickle
import scipy.signal
import math
import quaternion

import seaborn as sns
import matplotlib.pyplot as plt

import webrtcvad

data_dir = "/home/soundr-share/Soundr-Data"
tracking_file_name = "tracking.pickle"
sound_file_name = "audio.wav"

window_size = 0.3
vad = webrtcvad.Vad(1)

def audio_range_from_tracking(i, temp_window_size=window_size):
    time = i / tracking_sample_rate
    start_time = time - temp_window_size / 2
    sample_count = int(temp_window_size * audio_sample_rate)
    audio_start_index = int(start_time * audio_sample_rate)
    audio_end_index = sample_count + audio_start_index
    return audio_start_index, audio_end_index


def validate_audio_range(audio_range):
    audio_start_index, audio_end_index = audio_range
    return audio_start_index >= 0 and audio_end_index < len(audio_data)


data_sub_dirs = os.listdir(data_dir)

input_array = []
output_array = []

segment_length = 0.1
energy_segment_length = 0.3

def get_audio_segment(audio_data, sample_rate, segment_id, override_length=None):
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
        # if tracking_data[prev_idx] is None:
        #     prev_idx -= 1
        #
        # if tracking_data[next_idx] is None:
        #     next_idx += 1
        #
        # prev_data = tracking_data[prev_idx]
        # next_data = tracking_data[next_idx]
        # if prev_data is None or next_data is None:
        #     return None
        #
        # progress = (idx - prev_idx) / (next_idx - prev_idx)
        # prev_pos = prev_data[0:3]
        # prev_quat = quaternion.as_quat_array(prev_data[3:7])
        # next_pos = next_data[0:3]
        # next_quat = quaternion.as_quat_array(next_data[3:7])
        # print(progress)
        # idx_pos = progress * next_pos + (1 - progress) * prev_pos
        # idx_quat = quaternion.slerp(prev_quat, next_quat, 0, 1, 0.5)
        # return np.concatenate([idx_pos, quaternion.as_float_array(idx_quat)])
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


for data_sub_dir in data_sub_dirs:
    if data_sub_dir.startswith(".git"):
        continue

    if data_sub_dir.startswith("_"):
        continue
    data_path = os.path.join(data_dir, data_sub_dir)

    tracking_file_path = os.path.join(data_path, tracking_file_name)
    sound_file_path = os.path.join(data_path, sound_file_name)

    audio_sample_rate, audio_data = scipy.io.wavfile.read(sound_file_path)
    audio_data = np.delete(audio_data, [7, 15], 1)
    with open(tracking_file_path, "rb") as tracking_file:
        tracking_sample_rate = 90
        tracking_data = np.array(pickle.load(tracking_file))

    xs = []
    zs = []
    for data in tracking_data:
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

#%%

np.save("/home/soundr-share/train_set3.npy", combined_segments_data)


# #%%
#
#
# data_frame_raw = []
#
#
#
# for i in range(0, len(audio_valid), 500):
#     data_frame_raw += [['valid', i, audio_valid[i] * 1e10]]
#     data_frame_raw += [['data', i, audio_mono_data[i]]]
#
# import pandas as pd
#
# data_frame = pd.DataFrame(data_frame_raw, columns=["type", "x", "value"])
#
# plt.clf()
# plt.gcf().set_size_inches(50, 10)
# sns.lineplot(x="x", y="value", hue="type", data=data_frame)
# plt.savefig("/tmp/audio.png")
# #%%
#
#
#
# valid_tracking_data = []
# valid_audio_slices = []
# valid_audio_energy = []
#
# # for i in range(len(tracking_data)):
# for i in range(len(tracking_data)):
#     audio_range = audio_range_from_tracking(i)
#     if not validate_audio_range(audio_range):
#         print(f"{i} is not validated!")
#     if validate_audio_range(audio_range) and tracking_data[i] is not None:
#         valid_tracking_data += [tracking_data[i]]
#         data_audio_range = audio_range_from_tracking(i, 0.16)
#         data_audio_start_index, data_audio_end_index = data_audio_range
#         audio_slice = audio_data[data_audio_start_index: data_audio_end_index]
#         valid_audio_slices += [audio_slice[0::3]]
#         audio_start_index, audio_end_index = audio_range
#         audio_energy_slice = audio_energy[audio_start_index: audio_end_index]
#         valid_audio_energy += [math.log(np.average(audio_energy_slice))]
#         # print(f"{i}: {audio_range}, {data_audio_range}")
#
# # plt.clf()
# # valid_tracking_data_np = np.array(valid_tracking_data)
# # sns.scatterplot(valid_tracking_data_np[:, 0], valid_tracking_data_np[:, 2], linewidth=0)
# # plt.show()
#
# speaking_tracking_data = []
# speaking_audio_slices = []
# speaking_audio_energy = []
#
# for i in range(len(valid_tracking_data)):
#     if valid_audio_energy[i] > 39:
#         speaking_tracking_data += [valid_tracking_data[i]]
#         speaking_audio_slices += [valid_audio_slices[i]]
#         speaking_audio_energy += [valid_audio_energy[i]]
#
# input_array += speaking_audio_slices
# output_array += speaking_tracking_data
#
# input_data = np.array(input_array)
# output_data = np.array(output_array)
#
# data = input_data, output_data
#
# with open("train_set2.pickle", "wb") as train_set:
#     pickle.dump(data, train_set, protocol=4)

#
#
# def rolling_avg(data, window):
#    return np.convolve(audio_energy, scipy.signal.gaussian(window, window), 'same') / window
#
# # # %%
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# plt.interactive(True)
#
# plt.cla()
# plt.clf()
# plt.figure(figsize=(100,10))
# plt.plot(valid_audio_energy)
# plt.show()
# plt.close()
#
# # %%
# import math
#
# threshold = math.e ** 39.5
#
# audio_speaking = np.greater(avg_audio_energy, threshold).astype(float)
# avg_audio_speaking = np.convolve(audio_speaking, np.ones(540), 'same') / 540
#
# # %%
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.interactive(True)
#
# plt.cla()
# plt.clf()
# plt.figure(figsize=(100,10))
# plt.plot(avg_audio_speaking.astype(int))
# plt.show()
# plt.close()
#
#
# #%%
#
# plt.cla()
# plt.clf()
# plt.figure(figsize=(100,10))
# plt.plot(np.log(avg_audio_energy))
# plt.show()
# plt.close()
#
# #%%
# plt.cla()
# plt.clf()
# plt.figure(figsize=(100,10))
# plt.plot(scipy.signal.gaussian(30000, 15000))
# plt.show()
# plt.close()
#
# # %%
#
# plt.clf()
# sns.distplot(speaking_audio_energy)
# plt.show()
