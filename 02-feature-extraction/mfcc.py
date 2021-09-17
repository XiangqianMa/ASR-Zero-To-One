import librosa
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from scipy.fftpack import dct


def plot_spectrogram(spec, note,file_name):
    """Draw the spectrogram picture
        :param spec: a feature_dim by num_frames array(real)
        :param note: title of the picture
        :param file_name: name of the file
    """
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.savefig(file_name)


def pre_emphasis(signal, coefficient=0.97):
    """
    对输入信号执行预加重操作（高通滤波）
    :param signal:输入信号
    :param coefficient:预加重系数
    :return: 预加重后的信号
    """
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1])


def enframe(signal, frame_len=400, frame_shift=160, win=np.hamming(400)):
    """
    对输入信号执行加窗操作
    :param signal:输入信号
    :param frame_len: 帧长
    :param frame_shift: 帧移
    :param win: 窗函数
    :return: 加窗后的信号
    """
    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift) + 1

    frames = np.zeros((int(num_frames), frame_len))

    # 分帧与加窗
    for i in range(int(num_frames)):
        frames[i, :] = signal[i * frame_shift: i * frame_shift + frame_len]
        frames[i, :] = win * frames[i, :]

    return frames


def get_spectrum(frames, fft_len=512):
    """
    使用离散傅立叶变换计算得到频谱图
    :param frames:
    :param fft_len:
    :return:
    """
    c_fft = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2.0) + 1
    spectrum = np.abs(c_fft[:, 0: valid_len])

    return spectrum


def filter_bank(spectrum, fs, fft_len, num_filter=23):
    low_freq_mel = 0
    high_freq_mel = 2596 * np.log10(1 + (fs / 2) / 700)
    # 计算各个中心点的梅尔频率
    k_bm = np.linspace(low_freq_mel, high_freq_mel, num_filter + 2)
    # 将梅尔频率转换为线性频率
    k_bm = 700 * (10 ** (k_bm / 2595) - 1)

    feats = np.zeros((num_filter, int(fft_len / 2 + 1)))
    # 将滤波器的频率与频谱中各个频率相对应
    filter_bins = (k_bm / (fs / 2)) * (fft_len / 2)
    for i in range(1, num_filter + 1):
        k_bm_center = int(filter_bins[i])
        k_bm_left = int(filter_bins[i - 1])
        k_bm_right = int(filter_bins[i + 1])
        for j in range(k_bm_left, k_bm_center):
            feats[i - 1, j + 1] = (j + 1 - filter_bins[i - 1]) / (filter_bins[i] - filter_bins[i - 1])
        for j in range(k_bm_center, k_bm_right):
            feats[i - 1, j + 1] = (filter_bins[i + 1] - (j + 1)) / (filter_bins[i + 1] - filter_bins[i])

    filter_banks = np.dot(spectrum, feats.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    return filter_banks


def mfcc(fbank, num_mfcc=12):
    feats = dct(fbank, type=2, axis=1, norm='ortho')[:, 1: (num_mfcc + 1)]
    (n_frames, n_coeff) = feats.shape
    n = np.arange(n_coeff)
    lift = 1 + (num_mfcc / 2) * np.sin(np.pi * n / num_mfcc)
    feats *= lift
    return feats


def write_file(feats, file_name):
    """
    Write the feature to file
    :param feats: a num_frames by feature_dim array(real)
    :param file_name: name of the file
    """
    f = open(file_name, 'w')
    (row, col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i, j]) + ' ')
        f.write(']\n')
    f.close()


if __name__ == '__main__':
    wav_file = "./test.wav"
    alpha = 0.97

    frame_len = 400
    frame_shift = 160
    fft_len = 512

    num_filter = 23
    num_mfcc = 12

    wav, fs = librosa.load(wav_file, sr=None)
    signal = pre_emphasis(wav, coefficient=alpha)
    frames = enframe(signal, frame_len, frame_shift)
    spectrum = get_spectrum(frames, fft_len=fft_len)
    plot_spectrogram(spectrum.T, 'Spectrum', 'spectrum.png')
    fbank_feats = filter_bank(spectrum, fs, fft_len, num_filter=num_filter)
    plot_spectrogram(fbank_feats.T, 'Filter Bank', 'fbank.png')
    write_file(fbank_feats, 'test.fbank')
    mfcc_feats = mfcc(fbank_feats, num_mfcc=num_mfcc)
    plot_spectrogram(mfcc_feats.T, 'MFCC', 'mfcc.png')
    write_file(mfcc_feats, 'test.mfcc')