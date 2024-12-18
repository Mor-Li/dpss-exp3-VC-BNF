import os
import numpy as np
import argparse
from models.wenet.bin.recognize import AsrReco
from models import BLSTMConversionModel
from config import Hparams
from utils import (
    load_wav,
    _preemphasize,
    melspectrogram,
    inv_mel_spectrogram,
    inv_preemphasize,
    save_wav,
    F0Extractor,
    reform_input_audio,
)
import matplotlib.pyplot as plt
import torch


def main():
    hps = Hparams
    parser = argparse.ArgumentParser("VC inference")
    parser.add_argument("--src_wav", type=str, help="source wav file path")
    parser.add_argument("--ckpt", type=str, help="model ckpt path")
    parser.add_argument("--save_dir", type=str, help="synthesized wav save directory")
    args = parser.parse_args()

    src_wav_arr = load_wav(args.src_wav)
    src_wav_arr[src_wav_arr > 1] = 1.0
    pre_emphasized_wav = _preemphasize(src_wav_arr)

    # 1. extract bnfs
    print("Set up BNFs extraction network")
    # Set up network
    bnf_config = "./config/asr_config.yaml"
    asr_checkpoint_path = "./pretrained_model/asr_model/final.pt"
    print("Loading BNFs extractor from {}".format(bnf_config))
    bnf_extractor = AsrReco(bnf_config, asr_checkpoint_path, False)

    fid = args.src_wav.split("/")[-1].split(".wav")[0]
    reform_input_audio(args.src_wav, fid + "-temp.wav")
    BNFs, feat_lengths, PPGs = bnf_extractor.recognize(fid + "-temp.wav")

    # 2. extract normed_f0, mel-spectrogram
    pitch_ext = F0Extractor("praat", sample_rate=16000)
    f0 = pitch_ext.extract_f0_by_frame(src_wav_arr, True)
    # mel-spectrogram is extracted for comparison
    mel_spec = melspectrogram(pre_emphasized_wav).astype(np.float32).T

    # 3. prepare inputs
    min_len = min(f0.shape[0], BNFs.shape[0])
    vc_inputs = np.concatenate([BNFs[:min_len, :], f0[:min_len, :]], axis=1)
    vc_inputs = np.expand_dims(vc_inputs, axis=1)  # [time, batch, dim]

    # 4. setup vc model and do the inference
    model = BLSTMConversionModel(
        in_channels=hps.Audio.bn_dim + 2,
        out_channels=hps.Audio.num_mels,
        lstm_hidden=hps.BLSTMConversionModel.lstm_hidden,
    )
    device = torch.device("cpu")
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()
    predicted_mels = model(torch.tensor(vc_inputs).to(torch.float32))
    predicted_mels = np.squeeze(predicted_mels.detach().numpy(), axis=1)

    # 5. synthesize wav
    synthesized_wav = inv_preemphasize(inv_mel_spectrogram(predicted_mels.T))
    resynthesized_wav = inv_preemphasize(inv_mel_spectrogram(mel_spec.T))
    ckpt_name = args.ckpt.split("/")[-1].split(".")[0]
    wav_name = args.src_wav.split("/")[-1].split(".")[0]
    save_wav(
        synthesized_wav,
        os.path.join(args.save_dir, "{}-{}-converted.wav".format(wav_name, ckpt_name)),
    )
    save_wav(
        resynthesized_wav,
        os.path.join(args.save_dir, "{}-{}-src-resyn.wav".format(wav_name, ckpt_name)),
    )

    # Step 6: Save Mel-spectrogram images
    def save_mel_plot(mel, save_path, title):
        plt.figure(figsize=(10, 4))
        plt.imshow(mel.T, aspect="auto", origin="lower", interpolation="none")
        plt.colorbar(format="%+2.0f dB")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # Save predicted Mel-spectrogram
    predicted_mel_plot_path = os.path.join(
        args.save_dir, f"{wav_name}-predicted-mel.png"
    )
    save_mel_plot(
        predicted_mels, predicted_mel_plot_path, title="Predicted Mel-Spectrogram"
    )

    # Save ground truth Mel-spectrogram
    groundtruth_mel_plot_path = os.path.join(
        args.save_dir, f"{wav_name}-groundtruth-mel.png"
    )
    save_mel_plot(
        mel_spec, groundtruth_mel_plot_path, title="Ground Truth Mel-Spectrogram"
    )

    # Step 7: Calculate MSE between Mel-spectrograms
    mse = np.mean((mel_spec[: predicted_mels.shape[0], :] - predicted_mels) ** 2)
    print(f"MSE between ground truth and predicted Mel-spectrogram: {mse}")

    # Save MSE to a text file
    mse_save_path = os.path.join(args.save_dir, f"{wav_name}-mse.txt")
    with open(mse_save_path, "w") as mse_file:
        mse_file.write(f"MSE: {mse}\n")

    return


if __name__ == "__main__":
    main()
