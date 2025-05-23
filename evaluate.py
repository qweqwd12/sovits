import os
import torch
import torchaudio
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn

# 定义使用torchaudio获取梅尔谱的函数
def get_mel_spectrogram(waveform, sample_rate=22050, n_fft=1024, n_mels=80, hop_length=256, win_length=1024):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=1,
        center=True,
        normalized=False,
        norm=None
    )
    mel = mel_transform(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel

# 定义相似度计算函数
def compute_similarity(mel_spec1, mel_spec2):
    mel_spec1 = mel_spec1.cpu().numpy()
    mel_spec2 = mel_spec2.cpu().numpy()
    # 对梅尔频谱进行零填充以保证维度一致
    max_len = max(mel_spec1.shape[1], mel_spec2.shape[1])
    mel_spec1_padded = np.pad(mel_spec1, ((0, 0), (0, max_len - mel_spec1.shape[1])), mode='constant')
    mel_spec2_padded = np.pad(mel_spec2, ((0, 0), (0, max_len - mel_spec2.shape[1])), mode='constant')

    mel_spec1_padded = mel_spec1_padded.flatten().reshape(1, -1)
    mel_spec2_padded = mel_spec2_padded.flatten().reshape(1, -1)
    similarity = cosine_similarity(mel_spec1_padded, mel_spec2_padded)[0][0]
    return similarity

# 定义计算梅尔谱损失的函数
def compute_mel_loss(mel_spec1, mel_spec2):
    # 对梅尔谱进行零填充以保证维度一致
    max_len = max(mel_spec1.size(2), mel_spec2.size(2))
    mel_spec1_padded = torch.nn.functional.pad(mel_spec1, (0, max_len - mel_spec1.size(2)))
    mel_spec2_padded = torch.nn.functional.pad(mel_spec2, (0, max_len - mel_spec2.size(2)))

    criterion = nn.MSELoss()
    loss = criterion(mel_spec1_padded, mel_spec2_padded)
    return loss.item()

# 定义绘制梅尔频谱图函数
def plot_mel_spectrograms(mel_spec1, mel_spec2, title1, title2, save_path):
    mel_spec1 = mel_spec1.cpu().squeeze().numpy()
    mel_spec2 = mel_spec2.cpu().squeeze().numpy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    im1 = librosa.display.specshow(mel_spec1, sr=22050, x_axis='time', y_axis='mel', ax=axes[0])
    axes[0].set_title(title1)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Mel Frequency')
    plt.colorbar(im1, ax=axes[0], format='%+2.0f dB')

    im2 = librosa.display.specshow(mel_spec2, sr=22050, x_axis='time', y_axis='mel', ax=axes[1])
    axes[1].set_title(title2)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Mel Frequency')
    plt.colorbar(im2, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 定义评估函数
def evaluate_speaker_similarity(original_dir, generated_dir):
    original_files = [f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f)) and f.endswith('.wav')]
    generated_files = [f for f in os.listdir(generated_dir) if os.path.isfile(os.path.join(generated_dir, f)) and f.endswith('.wav')]

    best_similarity = -1
    best_original_file = None
    best_generated_file = None
    best_original_mel_spec = None
    best_generated_mel_spec = None

    total_loss = 0
    valid_pair_count = 0

    for original_file in original_files:
        original_path = os.path.join(original_dir, original_file)
        generated_file = [f for f in generated_files if f.startswith(original_file.split('.')[0])]
        if generated_file:
            generated_path = os.path.join(generated_dir, generated_file[0])

            # 加载音频文件
            original_waveform, _ = torchaudio.load(original_path)
            generated_waveform, _ = torchaudio.load(generated_path)

            # 计算梅尔频谱
            original_mel_spec = get_mel_spectrogram(original_waveform)
            generated_mel_spec = get_mel_spectrogram(generated_waveform)

            # 计算相似度
            similarity = compute_similarity(original_mel_spec, generated_mel_spec)

            if similarity > best_similarity:
                best_similarity = similarity
                best_original_file = original_file
                best_generated_file = generated_file[0]
                best_original_mel_spec = original_mel_spec
                best_generated_mel_spec = generated_mel_spec

            # 计算梅尔谱损失
            loss = compute_mel_loss(original_mel_spec, generated_mel_spec)
            total_loss += loss
            valid_pair_count += 1

    if valid_pair_count > 0:
        average_loss = total_loss / valid_pair_count
        print(f"平均梅尔谱损失: {average_loss}")
    else:
        print("未找到有效的音频对，无法计算平均梅尔谱损失。")

    print(f"最优的说话人相似度: {best_similarity}")
    save_path = 'best_similarity_comparison.png'
    plot_mel_spectrograms(best_original_mel_spec, best_generated_mel_spec,
                          f"Original - {best_original_file}",
                          f"Generated - {best_generated_file}",
                          save_path)

# 调用评估函数
original_dir = '/root/autodl-tmp/so-vits-svc-5.0/dataset_eval/audio'
generated_dir = '/root/autodl-tmp/so-vits-svc-5.0/_svc_out'
evaluate_speaker_similarity(original_dir, generated_dir)
    