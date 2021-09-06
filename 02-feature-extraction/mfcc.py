import librosa
import numpy as np

wav_file = "./test.wav"

alpha = 0.97

frame_len = 400
frame_shift = 160
fft_len = 512

num_filter = 23
num_mfcc = 12

wav, fs = librosa.load(wav_file, sr=None)


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


def filter_bank(spectrum, num_filter=23):
    feats = np.zeros(spectrum.shape[0], num_filter)

    return feats


def mfcc(fbank, num_mfcc=12):
    feats = np.zeros((fbank.shape[0], num_mfcc))

    return feats


def write_file(feats, file_name):
    """
    Write the feature to file
    :param feats: a num_frames by feature_dim array(real)
    :param file_name: name of the file
    """
    f=open(file_name,'w')
    (row,col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i,j])+' ')
        f.write(']\n')
    f.close()


if __name__ == '__main__':
    pass
