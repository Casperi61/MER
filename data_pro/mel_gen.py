import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

mel_metrics = []
mel = {}
mel_tensor = np.array([])
folder_path = 'MEMD_audio/'
files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp3')]
files.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or -1))

# 设置参数
n_fft = 2048  # FFT窗口大小
n_mels = 96  # 梅尔频带数量
# 加载音频文件
for filename in files:
    if filename.lower().endswith('.mp3'):
        print(filename + "\n")
        file_path = os.path.join(folder_path, filename)
        y, sr = librosa.load(file_path, sr=None)
        print(sr)
        hop_length = int(sr / 88)

        # 计算梅尔频谱（功率）
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

        # 转换为对数分贝单位
        S_dB = librosa.power_to_db(S, ref=np.max)


        S_dB = np.array(S_dB)

        S_dB = S_dB[:, 15 * 88:45 * 88]

        mel[filename[:-4]] = S_dB

        mel_metrics.append(S_dB)


# 打印矩阵（行：频率，列：时间）
print("梅尔频谱图矩阵（单位：dB，对数分贝）")
print(mel_metrics[0].shape)

print(mel)

# 可视化mel
plt.figure(figsize=(10, 5))
librosa.display.specshow(mel_metrics[0], sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.show()

import pickle

# 保存数据
data = {'mel': mel}

with open('mel_data.pkl', 'wb') as f:
    pickle.dump(data, f)