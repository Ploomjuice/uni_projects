import librosa
import librosa.feature
import math
import io
import base64
import scipy as sp
import numpy as np


def load_audio(upload):
    # extract contents from the uploaded file
    content_type, content_string = upload.split(',')

    # decode the file content
    decoded = io.BytesIO(base64.b64decode(content_string))

    # get signal data and sample rate from decoded content
    signal, sr = librosa.load(decoded, sr=None)

    t = librosa.frames_to_time(range(signal.size), sr=48000, hop_length=256)
    return signal, sr, t


def band_energy_ratio(signal, div_freq):
    """Calculates the band energy ratio at each timeframe in the signal
    :param signal: input signal (array)
    :param div_freq: threshold frequency for comparison (int/float)
    :return: Band energy ratio (below/above)
    """

    # get spectrogram data
    spectrogram_data = librosa.stft(signal, n_fft=512, hop_length=256)

    # Get split frequencies
    # shape of the spectrogram = (# of frequency bins, time)
    freq_gap = 24000 / spectrogram_data.shape[0]
    divider_bin = int(np.floor(div_freq / freq_gap))

    # Calculate BER
    power_spectrogram = np.abs(spectrogram_data) ** 2
    # transpose to get (time, frequency bins) to iterate through the list of frequency bins at each time frame
    power_spectrogram = power_spectrogram.T
    ber = np.array([np.sum(frame[:divider_bin]) / np.sum(frame[divider_bin:]) for frame in power_spectrogram])

    # clean infinity values
    ber[ber == np.inf] = np.nan
    ber = np.nan_to_num(ber)

    # frames
    t_ber = librosa.frames_to_time(range(ber.size), sr=48000, hop_length=256)
    return ber, t_ber


def signal_energy(signal):
    # energy analysis data
    ae = np.array([max(signal[i:i + 512]) for i in range(0, len(signal), 256)])
    rms = librosa.feature.rms(y=signal, frame_length=512, hop_length=256)

    # set x_arrays to use for x-axes of graphs
    t_ae = librosa.frames_to_time(range(ae.size), sr= 48000, hop_length=256)
    t_rms = librosa.frames_to_time(range(rms.size), sr=48000, hop_length=256)

    return ae, t_ae, rms, t_rms


def attack_release(signal, amp_env):
    atk_time_idx = np.argmax(amp_env)
    attack_time = librosa.samples_to_time(atk_time_idx, sr=48000)

    release_time_index = np.argmax(amp_env[::-1])
    release_time = librosa.samples_to_time(len(signal) - release_time_index, sr=48000)

    return attack_time, -release_time


def mfcc_features(signal, sr):
    mfccs = [np.mean(i) for i in librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)]
    delta_mfccs = [np.mean(i) for i in librosa.feature.delta(mfccs, mode='nearest')]
    delta2_mfccs = [np.mean(i) for i in librosa.feature.delta(delta_mfccs, mode='nearest')]
    return [mfccs, delta_mfccs, delta2_mfccs]


def levels(signal, sr):
    """
    bass: 0-150hz
    mids: 150-1200hz
    treble: 1200-8000hz
    air: 8000+  hz
    """
    fourier = sp.fft.fft(signal)
    mag = np.absolute(fourier)

    freq_range = np.linspace(0, sr, len(mag))
    bass = sum(mag[0:150])
    mids = sum(mag[150:1200])
    treble = sum(mag[1200:8000])
    air = sum(mag[8000:24000])

    return bass, mids, treble, air

def brightness(bass, mids, treble, air):
    brightness_score = (air + treble) / (bass + mids + treble)
    return brightness_score

