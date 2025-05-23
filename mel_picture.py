import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_mel_spectrograms(wav_path_1, wav_path_2, output_path, target_sr=24000):
    """
    绘制并比较两个音频文件的梅尔频谱图
    参数:
        wav_path_1: 第一个音频文件路径
        wav_path_2: 第二个音频文件路径
        output_path: 输出图像路径
        target_sr: 目标采样率(默认24000)
    """
    # 加载音频文件
    y1, sr1 = librosa.load(wav_path_1, sr=None)
    y2, sr2 = librosa.load(wav_path_2, sr=None)
    
    # 采样率统一处理
    if sr1 != sr2:
        print(f"警告: 采样率不一致(音频1:{sr1}Hz, 音频2:{sr2}Hz)，将统一重采样到{target_sr}Hz")
        
        # 重采样到目标采样率
        if sr1 != target_sr:
            y1 = librosa.resample(y1, orig_sr=sr1, target_sr=target_sr)
            sr1 = target_sr
            print(f"音频1已重采样到{target_sr}Hz")
            
        if sr2 != target_sr:
            y2 = librosa.resample(y2, orig_sr=sr2, target_sr=target_sr)
            sr2 = target_sr
            print(f"音频2已重采样到{target_sr}Hz")
    
    # 最终采样率确认
    assert sr1 == sr2, f"采样率处理失败，最终采样率仍不一致(音频1:{sr1}Hz, 音频2:{sr2}Hz)"
    sr = sr1
    
    # 计算梅尔频谱
    S1 = librosa.feature.melspectrogram(y=y1, sr=sr, n_mels=100, fmax=8000)
    S2 = librosa.feature.melspectrogram(y=y2, sr=sr, n_mels=100, fmax=8000)

    # 转换为分贝单位
    S1_db = librosa.power_to_db(S1, ref=np.max)
    S2_db = librosa.power_to_db(S2, ref=np.max)

    # 绘制两张图
    plt.figure(figsize=(12, 8))  # 增大图像尺寸

    # 原始音频频谱
    plt.subplot(2, 1, 1)
    librosa.display.specshow(S1_db, sr=sr, cmap='viridis', 
                           y_axis='mel', x_axis='time', fmax=8000)
    plt.colorbar(format="%+2.0f dB")
    plt.title(f'Original Audio (SR: {sr}Hz)')

    # 生成音频频谱
    plt.subplot(2, 1, 2)
    librosa.display.specshow(S2_db, sr=sr, cmap='viridis',
                           y_axis='mel', x_axis='time', fmax=8000)
    plt.colorbar(format="%+2.0f dB")
    plt.title(f'Generated Audio (SR: {sr}Hz)')

    plt.tight_layout()

    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 提高DPI到300
    plt.close()
    print(f"梅尔频谱对比图已保存至: {output_path}")
    print(f"最终使用的采样率: {sr}Hz")


if __name__ == "__main__":
    # 示例文件路径
    ori_wav = "/root/autodl-tmp/so-vits-svc-5.0/test_result_radio/fdt.wav"
    gen_wav = "/root/autodl-tmp/so-vits-svc-5.0/test_result_radio/sovits_manori.wav"
    output_img = "/root/autodl-tmp/so-vits-svc-5.0/test_result_radio/sovits_对比图3.png"

    # 调用函数并处理可能的异常
    try:
        plot_mel_spectrograms(ori_wav, gen_wav, output_img)
    except Exception as e:
        print(f"发生错误: {str(e)}")