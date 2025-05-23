import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import torch

from whisper.model import Whisper, ModelDimensions
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram


def load_model(path, device) -> Whisper:
    checkpoint = torch.load(path, map_location="cpu")
    dims = ModelDimensions(**checkpoint["dims"])
    # print(dims)
    model = Whisper(dims)
    del model.decoder
    cut = len(model.encoder.blocks) // 4
    cut = -1 * cut
    del model.encoder.blocks[cut:]
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    if not (device == "cpu"):
        model.half()
    model.to(device)
    # torch.save({
    #     'dims': checkpoint["dims"],
    #     'model_state_dict': model.state_dict(),
    # }, "large-v2.pt")
    return model


def pred_ppg(whisper: Whisper, wavPath, ppgPath, device):
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    ppg_a = []
    idx_s = 0
    while (idx_s + 15 * 16000 < audln):
        short = audio[idx_s:idx_s + 15 * 16000]
        idx_s = idx_s + 15 * 16000
        ppgln = 15 * 16000 // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(device)
        if not (device == "cpu"):
            mel = mel.half()
        with torch.no_grad():
            mel = mel + torch.randn_like(mel) * 0.1
            ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            ppg = ppg[:ppgln,]  # [length, dim=1024]
            ppg_a.extend(ppg)
    if (idx_s < audln):
        short = audio[idx_s:audln]
        ppgln = (audln - idx_s) // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(device)
        if not (device == "cpu"):
            mel = mel.half()
        with torch.no_grad():
            mel = mel + torch.randn_like(mel) * 0.1
            ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            ppg = ppg[:ppgln,]  # [length, dim=1024]
            ppg_a.extend(ppg)
    np.save(ppgPath, ppg_a, allow_pickle=False)

def text_to_ppg(whisper: Whisper, text, device):
    # 这里需要实现将文本转换为PPG的逻辑
    # 由于Whisper主要用于音频处理，没有直接的文本转PPG方法，
    # 你可能需要借助其他模型（如TTS模型的文本编码器）来实现
    # 这里简单返回一个零向量作为示例
    dummy_ppg = np.zeros((100, 1024))  # 假设PPG的长度为100，维度为1024
    return dummy_ppg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-p", "--ppg", help="ppg", dest="ppg", required=True)
    parser.add_argument("--use-text-input", action="store_true", help="Use text input instead of audio features")
    args = parser.parse_args()
    print(args.wav)
    print(args.ppg)

    wavPath = args.wav
    ppgPath = args.ppg

    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper = load_model(os.path.join("whisper_pretrain", "large-v2.pt"), device)
    
    if args.use_text_input:
        text = input("请输入要合成的文本: ")
        ppg = text_to_ppg(whisper, text, device)
        np.save(ppgPath, ppg, allow_pickle=False)
    else:
        pred_ppg(whisper, wavPath, ppgPath, device)
