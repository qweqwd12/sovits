import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tqdm
import torch
import argparse

from whisper.inference import load_model, pred_ppg

import importlib.util

# How to use
# python svc_inference_batch.py --config configs/base.yaml --model vits_pretrain/sovits5.0.pth --wave test_waves/ --spk configs/singers/singer0047.npy

out_path = "./_svc_out"
os.makedirs(out_path, exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="/root/autodl-tmp/so-vits-svc-5.0/configs/base.yaml",
                        help="yaml file for config.")#required=True,
    parser.add_argument('--model', type=str, default="/root/autodl-tmp/so-vits-svc-5.0/chkpt/sovits5.0/sovits5.0_0500.pt",
                        help="path of model for evaluation")#, required=True
    parser.add_argument('--wave', type=str, default="/root/autodl-tmp/so-vits-svc-5.0/dataset_eval/audio" ,
                        help="Path of raw audio.")#required=True,
    parser.add_argument('--spk', type=str, default="/root/autodl-tmp/so-vits-svc-5.0/data_svc/singer/losl.spk.npy",
                    help="Path of speaker.")#required=True,
    parser.add_argument('--shift', type=int, default=0,
                    help="Pitch shift key.")
    args = parser.parse_args()
    wave_path = args.wave
    assert os.path.isdir(wave_path), f"{wave_path} is not folder"
    waves = [file for file in os.listdir(wave_path) if file.endswith(".wav")]
    for file in waves:
        print(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    whisper = load_model(os.path.join("whisper_pretrain", "large-v2.pt"), device=device)
    for file in tqdm.tqdm(waves, desc="whisper"):
        pred_ppg(whisper, f"{wave_path}/{file}", f"{out_path}/{file}.ppg.npy", device=device)
    del whisper

    for file in tqdm.tqdm(waves, desc="svc"):
        os.system(
            f"python svc_inference.py --config {args.config} --model {args.model} "
            f"--wave {wave_path}/{file} --ppg {out_path}/{file}.ppg.npy "
            f"--spk {args.spk} --shift {args.shift}")
        os.system(f"mv svc_out.wav {out_path}/{file}")
        os.system(f"rm {out_path}/{file}.ppg.npy")
