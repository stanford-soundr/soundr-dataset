Dataset of "Soundr: Head Position and Orientation Prediction Using a Microphone Array"
==================

### Dataset download

The original dataset can be found at [Stanford Library](https://searchworks.stanford.edu/view/jn901sr3775).

A script is provided for your convenience to download the data from Stanford Library: `./download.sh`.
It will download and extract all the data to the current folder.
Make sure you have at least 125GB of free space on your current drive before running the script.

A preprocessed version that can be used for training ML models can be found at [Google Drive link](https://drive.google.com/drive/folders/1OGNJWVLocsMuxaJVn5IGD0wNAERxPs5W?usp=sharing).

### Dataset content

The original dataset is divided by each collection session (folders).

Every folder contains:
* Session configuration: `config.json`

  {'user_id': User ID, 'trial_id': Trial ID, 'room_id': Room ID}
* Multi-channel microphone data: `mic_data.npy`

  A `np.array` with shape (number of samples, the audio channel (always 16)).
  The audio sampling rate is 48KHz.
* VR Headset Tracking data: `vr_tracking_data.npy`

  A `np.array` with shape (number of samples, data channels (UNIX timestamp, x, y, z, q_x, q_y, q_z, q_w)).
  The data sampling rate is around 72Hz.
* Headset Audio data: `vr_audio_data.npy`

  A `np.array` with shape (number of samples).
  The audio sampling rate is 48KHz.
* Offset data: `offset.npy`

  A `np.array` with a length of three (VR Headset audio, Multi-channel microphone, VR Headset tracking).
  Used to correct for missing samples from multi-channel microphone data and headset tracking and audio data.
  The three numbers are the number of samples missing or added comparing with the expected sample size (expected_samples - actual_samples).


The processed dataset only contains the multi-channel microphone data and headset tracking data.

It contains three files:
* Input multi-channel audio data `input.npy`:

  A `np.array` with shape (number of segments, the audio channel (always 16), segment length).
* Output headset tracking data `output.npy`:

  A `np.array` with shape (number of segments, tracking data (x, y, z, q_x, q_y, q_z, q_w, vad)).

  Each segment in the output headset tracking data corresponds to the input audio data of the same index.
  The `vad` is voice activity detection result from the headset audio: 1 (voice activity present); 0 (voice activity not present).
  When training a machine learning model, only segments with voice activity should be used.

  Note that the head rotation in processed data only kept the pitch and the yaw of the user's head on the Y-axis, as the roll can't be inferred from audio data alone.
* Data sessions config `starts.npy`:

  [{'start': start index, 'end': end index, 'config': {'user_id': User ID, 'trial_id': Trial ID, 'room_id': Room ID}}]

### Run the preprocessing script

You can download the preprocessed data from the link above. If you would like to preprocess the data yourself, here is the command:

```bash
python3 -m pip install -r requirements.txt
python3 ./preprocess.py {data folder} --output {output folder}
```
