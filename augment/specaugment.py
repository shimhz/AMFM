import librosa
import numpy as np
import random
import torch


def spec_augment(
    mel_spectrogram,
    time_warping_para=80,
    frequency_masking_para=27,
    time_masking_para=100,
    frequency_mask_num=1,
    time_mask_num=1,
):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    # print (mel_spectrogram.size()) #[24, 128, 251]
    """
    shape = mel_spectrogram.shape
    mel_spectrogram = torch.reshape(mel_spectrogram, (-1, shape[2], shape[1]))
    print (mel_spectrogram.size())
    """

    v = mel_spectrogram.shape[1]  # freq
    tau = mel_spectrogram.shape[2]  # time

    # Step 1 : Time warping
    # warped_mel_spectrogram = time_warp(mel_spectrogram, W=time_warping_para)

    # Step 2 : Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, v - f)
        # warped_mel_spectrogram[:, f0:f0+f, :] = 0
        mel_spectrogram[:, f0 : f0 + f, :] = 0

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau - t)
        # warped_mel_spectrogram[:, :, t0:t0+t] = 0
        mel_spectrogram[:, :, t0 : t0 + t] = 0

    # return warped_mel_spectrogram
    return mel_spectrogram
