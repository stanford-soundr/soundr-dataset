# %%
import scipy.io.wavfile
import os
import numpy as np
import pickle
import scipy.signal
import math

import seaborn as sns
import matplotlib.pyplot as plt

data_dir = "/home/soundr-share/Soundr-Data"
tracking_file_name = "tracking.pickle"
sound_file_name = "audio.wav"

window_size = 0.3


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

    audio_mono_data = audio_data.sum(axis=1)
    audio_energy = audio_mono_data ** 2

    valid_tracking_data = []
    valid_audio_slices = []
    valid_audio_energy = []

    for i in range(len(tracking_data)):
        audio_range = audio_range_from_tracking(i)
        if not validate_audio_range(audio_range):
            print(f"{i} is not validated!")
        if validate_audio_range(audio_range) and tracking_data[i] is not None:
            valid_tracking_data += [tracking_data[i]]
            data_audio_range = audio_range_from_tracking(i, 0.16)
            data_audio_start_index, data_audio_end_index = data_audio_range
            audio_slice = audio_data[data_audio_start_index: data_audio_end_index]
            valid_audio_slices += [audio_slice[0::3]]
            audio_start_index, audio_end_index = audio_range
            audio_energy_slice = audio_energy[audio_start_index: audio_end_index]
            valid_audio_energy += [math.log(np.average(audio_energy_slice))]
            # print(f"{i}: {audio_range}, {data_audio_range}")

    # plt.clf()
    # valid_tracking_data_np = np.array(valid_tracking_data)
    # sns.scatterplot(valid_tracking_data_np[:, 0], valid_tracking_data_np[:, 2], linewidth=0)
    # plt.show()

    speaking_tracking_data = []
    speaking_audio_slices = []
    speaking_audio_energy = []

    for i in range(len(valid_tracking_data)):
        if valid_audio_energy[i] > 39:
            speaking_tracking_data += [valid_tracking_data[i]]
            speaking_audio_slices += [valid_audio_slices[i]]
            speaking_audio_energy += [valid_audio_energy[i]]

    input_array += speaking_audio_slices
    output_array += speaking_tracking_data

input_data = np.array(input_array)
output_data = np.array(output_array)

data = input_data, output_data

with open("train_set2.pickle", "wb") as train_set:
    pickle.dump(data, train_set, protocol=4)

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
