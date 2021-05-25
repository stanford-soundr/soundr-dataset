# %%
import argparse
import os
import psutil
import numpy as np
import numpy.lib.format
import math
import json

import seaborn as sns
import matplotlib.pyplot as plt

import webrtcvad
from typing import List, Dict, Optional, Tuple

parser = argparse.ArgumentParser(description='Process Soundr training data')
parser.add_argument('data_dir', type=str, help='path to data dir')
parser.add_argument('--output', type=str, help='output path')
args = parser.parse_args()
data_dir = args.data_dir
tracking_file_name = "vr_tracking_data.npy"  # 8 x 1 need to delete the first channel (timestamp)
sound_file_name = "mic_data.npy"  # 16 x 1
vad_file_name = "vr_audio_data.npy"  # VR on-board mic data is 1x1; use specifically for VAD
offset_file_name = "offset.npy"
config_file_name = "config.json"

vad = webrtcvad.Vad(3)  # voice activity detection to understand when during the audio someone is actually speaking

data_sub_dirs = os.listdir(data_dir)

SEGMENT_LENGTH = 0.1
ENERGY_SEGMENT_LENGTH = 0.3

AUDIO_CHANNELS = 16
TRACKING_CHANNELS = 7
AUDIO_SAMPLE_RATE = 48000
TRACKING_SAMPLE_RATE = 72

SEGMENT_LONG_THRESHOLD = 20


def get_audio_segment(audio_data, sample_rate, segment_id, offset_rate, override_length=None):
    """
    if you ovverride length, segment will still be segmented as original length
    :rtype: object
    :param offset_rate:
    :param audio_data:
    :param sample_rate:
    :param segment_id:
    :param override_length:
    :return:
    """
    start_idx = get_audio_segment_idx(audio_data, sample_rate, segment_id, offset_rate, override_length)
    if start_idx is None:
        return None
    end_idx = start_idx + int(sample_rate * override_length)
    return audio_data[start_idx: end_idx]


def get_audio_segment_idx(audio_data, sample_rate, segment_id, offset_rate, override_length=None) -> Optional[int]:
    """
    if you ovverride length, segment will still be segmented as original length
    :rtype: object
    :param offset_rate:
    :param audio_data:
    :param sample_rate:
    :param segment_id:
    :param override_length:
    :return:
    """
    start_idx = int(sample_rate * segment_id * SEGMENT_LENGTH * offset_rate)
    end_idx = start_idx + int(sample_rate * SEGMENT_LENGTH)

    if override_length is not None:
        new_start_idx = int((start_idx + end_idx - sample_rate * override_length) / 2)
        new_end_idx = int((start_idx + end_idx + sample_rate * override_length) / 2)
        start_idx = new_start_idx
        end_idx = new_end_idx

    if start_idx >= 0 and end_idx < len(audio_data):
        return start_idx
    else:
        return None


def get_tracking_sample(sample_rate, segment_id, offset_rate, skipped):
    try:
        idx = sample_rate * (segment_id - skipped + 0.5) * SEGMENT_LENGTH * offset_rate
        prev_idx = int(math.floor(idx))
        return prev_idx

    except IndexError:
        return None


combined_segments_data: List[List[Dict]] = [[], []]


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


def load_audio_data(audio_data_path):
    audio_data = np.load(audio_data_path, mmap_mode='r')
    assert (audio_data.shape[1] == AUDIO_CHANNELS)
    return audio_data


def load_tracking_data(tracking_data_path):
    tracking_data = np.load(tracking_data_path, mmap_mode='r')
    track_timestamp = tracking_data[:, 0:1]
    tracking_data = np.delete(tracking_data, [0], 1)

    # realign columns of tracking data
    track_xyz = tracking_data[:, 0:3]
    track_q_xyz = tracking_data[:, 3:6]
    track_q_w = tracking_data[:, 6:7]

    import quaternion
    track_q = np.concatenate((track_q_w, track_q_xyz), axis=1)
    q_track_q = quaternion.as_quat_array(track_q)
    forward = [0, 0, 1]
    directions = quaternion.rotate_vectors(q_track_q, forward)

    track_q_no_z_list = []
    for direction in directions:
        dot_product = np.dot(forward, direction)
        if dot_product >= 1 - 1e-8:
            track_q_no_z_list += [[1, 0, 0, 0]]
        elif dot_product <= -1 + 1e-8:
            track_q_no_z_list += [[0, 0, 1, 0]]
        else:
            s = np.sqrt((1 + dot_product) * 2)
            inv_s = 1 / s

            axis = np.cross(forward, direction)
            track_q_no_z_list += [[s / 2, axis[0] * inv_s, axis[1] * inv_s, axis[2] * inv_s]]

    track_q_no_z = np.array(track_q_no_z_list)
    q_track_q_no_z = np.normalized(quaternion.as_quat_array(track_q_no_z))
    directions_no_z = quaternion.rotate_vectors(q_track_q_no_z, [0, 0, 1])

    # assert np.allclose(directions, directions_no_z, rtol=1.e-5, atol=1.e-5)
    assert len(directions_no_z) == len(directions)
    for i in range(len(directions_no_z)):
        if not np.allclose(directions[i], directions_no_z[i], rtol=1.e-4, atol=1.e-4):
            print(f"Warning: direction: ({directions[i]}, {directions_no_z[i]}) are not close")

    track_q_wxyz = quaternion.as_float_array(q_track_q_no_z)

    tracking_data = np.concatenate((track_xyz, track_q_wxyz, track_timestamp), axis=1)
    assert (tracking_data.shape[1] == TRACKING_CHANNELS + 1)  # one more for timestamp
    return tracking_data


def process_tracking_data(tracking_data: np.array) -> List[Tuple[int, int]]:
    tracking_timestamp = tracking_data[:, 7]
    missing_tracking = []
    for i, (cur_timestamp, next_timestamp) in enumerate(zip(tracking_timestamp, tracking_timestamp[1:])):
        if next_timestamp - cur_timestamp > 10.0 / TRACKING_SAMPLE_RATE:
            missed_frames = (next_timestamp - cur_timestamp) * TRACKING_SAMPLE_RATE
            missed_frames_int = int(round(missed_frames))
            missing_tracking += [(i, missed_frames_int)]
            error = missed_frames - missed_frames_int
            print(f"Missing tracking frame detected at {i}, length {missed_frames_int}, error {error}!")
    return missing_tracking


def get_offset_segments(audio_mono_data, audio_data, tracking_length, offset_rates, missing_tracking):
    offset = 0
    segments_internal = [[]]
    skipped_tracking = 0
    max_i = int(len(audio_mono_data) / (SEGMENT_LENGTH * AUDIO_SAMPLE_RATE))

    i = 0
    while i < max_i:
        energy_segment = get_audio_segment(
            audio_mono_data,
            AUDIO_SAMPLE_RATE,
            i + offset,
            offset_rates[1],
            override_length=ENERGY_SEGMENT_LENGTH
        )

        if energy_segment is None:
            audio_valid = False
        else:
            audio_valid = vad_segment_valid(energy_segment[::3], 16000)
            # audio_valid = False

        audio_segment_start_idx = get_audio_segment_idx(
            audio_data, AUDIO_SAMPLE_RATE, i + offset, offset_rates[1]
        )
        tracking_sample_idx = get_tracking_sample(
            TRACKING_SAMPLE_RATE, i + offset, offset_rates[2], skipped_tracking
        )
        if tracking_sample_idx > tracking_length:
            break
        if len(missing_tracking) > 0:
            if tracking_sample_idx > missing_tracking[0][0]:  # passed the first missing tracking frame
                # skip missing tracking segments
                skipped_audio_frames = \
                    (missing_tracking[0][1] / TRACKING_SAMPLE_RATE / offset_rates[2]) / SEGMENT_LENGTH
                skipped_audio_frames_int = int(round(skipped_audio_frames))
                error = skipped_audio_frames - skipped_audio_frames_int
                print(f"Skipping {skipped_audio_frames_int} frames, error {error}")
                i += skipped_audio_frames_int
                # create a new place to store segments
                segments_internal += [[]]
                # remove the first missing_tracking place
                missing_tracking = missing_tracking[1:]
                continue

        valid = audio_segment_start_idx is not None and tracking_sample_idx is not None
        if not valid:
            print(f"End at {i}")
            break
        segments_internal[-1] += [(tracking_sample_idx, audio_segment_start_idx, audio_valid)]
        i += 1
    return segments_internal


def main():
    for data_sub_dir in data_sub_dirs:  # choose how many dirs [0:3]
        print(f"Working on {data_sub_dir}")
        if data_sub_dir.startswith("."):  # omit any hidden file/folders
            continue

        if data_sub_dir.startswith("_"):  # omit selected data folders
            continue

        data_path = os.path.join(data_dir, data_sub_dir)  # data path is one specific set of collected data

        # all lead to npy files
        tracking_file_path = os.path.join(data_path, tracking_file_name)
        sound_file_path = os.path.join(data_path, sound_file_name)
        vad_file_path = os.path.join(data_path, vad_file_name)
        offset_file_path = os.path.join(data_path, offset_file_name)
        config_file_path = os.path.join(data_path, config_file_name)

        audio_data = load_audio_data(sound_file_path)
        tracking_data = load_tracking_data(tracking_file_path)
        missing_tracking = process_tracking_data(tracking_data)
        total_missing_frames = sum([missing_frames for i, missing_frames in missing_tracking])

        vad_data = np.load(vad_file_path, mmap_mode='r')
        try:
            offsets = np.load(offset_file_path)
            offsets[0] = 0  # TODO: fix offset in soundr-data-collector-2
        except FileNotFoundError:
            print(f"File not exist, replacing with default: {offset_file_path}")
            vad_offset = 0
            audio_offset = -9860
            tracking_offset = tracking_data.shape[0] * 0.0025
            offsets = np.array([vad_offset, audio_offset, tracking_offset])

        offsets[2] -= total_missing_frames  # add back the frames that we already found
        try:
            with open(config_file_path, "r") as config_file:
                config_json = json.load(config_file)
        except FileNotFoundError:
            config_json = {"user_id": 0, "trial_id": -1}

        if "room_id" not in config_json:
            config_json["room_id"] = 0

        print(config_json)

        data_length = np.array([vad_data.shape[0], audio_data.shape[0], tracking_data.shape[0]])
        offset_rates = data_length / (offsets.astype(float) + data_length)
        print(f"{offset_rates}, {offsets}: path \"{offset_file_path}\"")

        xs = []
        zs = []
        for data in tracking_data:  # vector with length 7- x,y,z, a,b,c,d
            if data is not None:
                xs += [data[0]]
                zs += [data[2]]
        sns.scatterplot(xs, zs)
        plt.title(data_sub_dir)
        plt.show()

        audio_mono_data = vad_data

        segments_array = \
            get_offset_segments(audio_mono_data, audio_data, len(tracking_data), offset_rates, missing_tracking)

        for segments in segments_array:
            # create a dictionary so that each set of indices is matched to a specific data file
            combined_segments_data[0] += [{"filename": sound_file_path, "segments": segments, "config": config_json}]
            combined_segments_data[1] += [{"filename": tracking_file_path, "segments": segments}]

        print(f"loop {psutil.virtual_memory()}")

    # find the total size of the array we need to allocate
    sound_segment_lengths = [len(data_dic["segments"]) for data_dic in combined_segments_data[0]]

    tracking_segment_lengths = [len(data_dic["segments"]) for data_dic in combined_segments_data[1]]

    sound_segments_total_length = sum(sound_segment_lengths)
    tracking_segments_total_length = sum(tracking_segment_lengths)

    data_directory = args.output
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # allocate fixed arrays to fill with data using indices that are stored
    segment_samples = int(AUDIO_SAMPLE_RATE * SEGMENT_LENGTH)
    audio_new_data = np.lib.format.open_memmap(
        data_directory + "input.npy",
        dtype=np.float32,
        mode='w+',
        shape=(sound_segments_total_length, AUDIO_CHANNELS, segment_samples)
    )
    tracking_new_data = np.lib.format.open_memmap(
        data_directory + "output.npy",
        dtype=np.float32,
        mode='w+',
        shape=(tracking_segments_total_length, TRACKING_CHANNELS + 1)  # one more for voice activity
    )

    audio_current_idx = 0
    tracking_current_idx = 0

    recording_starts = []

    for data_dic in combined_segments_data[0]:
        recording_start = {"start": audio_current_idx, "end": None, "config": data_dic["config"]}
        sound_file_path = data_dic["filename"]
        sound_segments = data_dic["segments"]
        sound_data = load_audio_data(sound_file_path)

        for sound_segment_tuple in sound_segments:
            sound_segment_index = sound_segment_tuple[1]
            audio_new_data[audio_current_idx] = \
                sound_data[sound_segment_index: sound_segment_index + segment_samples, :].T
            audio_current_idx += 1
        recording_start["end"] = audio_current_idx
        recording_starts += [recording_start]

    for data_dic in combined_segments_data[1]:
        tracking_file_path = data_dic["filename"]
        tracking_segments = data_dic["segments"]
        tracking_data = load_tracking_data(tracking_file_path)

        for tracking_segment_tuple in tracking_segments:
            tracking_segment_index = tracking_segment_tuple[0]
            tracking_new_data[tracking_current_idx, 0:7] = tracking_data[tracking_segment_index, 0:7]
            tracking_new_data[tracking_current_idx, 7] = 1.0 if tracking_segment_tuple[2] else 0.0
            tracking_current_idx += 1

    assert(tracking_current_idx == audio_current_idx)
    np.save(data_directory + "starts.npy", recording_starts)

    print("Pre-processing finished")


if __name__ == '__main__':
    main()
