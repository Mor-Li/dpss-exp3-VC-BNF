import os
import numpy as np
import argparse
from models.wenet.bin.recognize import AsrReco
from models import BLSTMToManyConversionModel
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
import torch

def main():
    hps = Hparams
    parser = argparse.ArgumentParser("VC inference")
    parser.add_argument("--src_wav", type=str, help="source wav file path")
    parser.add_argument("--ckpt", type=str, help="model ckpt path")
    parser.add_argument(
        "--tgt_spk",
        type=str,
        help="target speaker name(s), separated by '/'",
        # 移除 choices 参数，方便传递多个目标说话人
        # choices=hps.SPEAKERS.spk_to_inds,
    )
    parser.add_argument("--save_dir", type=str, help="synthesized wav save directory")
    args = parser.parse_args()

    # 将多个目标说话人名称分割成列表
    tgt_spk_names = args.tgt_spk.split('/')

    # 检查目标说话人名称是否合法
    valid_speakers = hps.SPEAKERS.spk_to_inds
    for spk_name in tgt_spk_names:
        if spk_name not in valid_speakers:
            raise ValueError(f"Invalid target speaker name: {spk_name}. "
                             f"Valid choices are: {valid_speakers}")

    # 0. 加载源语音
    src_wav_arr = load_wav(args.src_wav)
    src_wav_arr[src_wav_arr > 1] = 1.0
    pre_emphasized_wav = _preemphasize(src_wav_arr)

    # 1. 提取 BNFs 特征
    bnf_config = "./config/asr_config.yaml"
    asr_checkpoint_path = "./pretrained_model/asr_model/final.pt"
    print("Loading BNFs extractor from {}".format(bnf_config))
    bnf_extractor = AsrReco(bnf_config, asr_checkpoint_path, False)
    fid = args.src_wav.split("/")[-1].split(".wav")[0]
    reform_input_audio(args.src_wav, fid + "-temp.wav")
    BNFs, feat_lengths, PPGs = bnf_extractor.recognize(fid + "-temp.wav")

    # 2. 提取 F0 特征和梅尔谱用于对比
    pitch_ext = F0Extractor("praat", sample_rate=16000)
    f0 = pitch_ext.extract_f0_by_frame(src_wav_arr, True)
    mel_spec = melspectrogram(pre_emphasized_wav).astype(np.float32).T

    # 3. 准备输入特征
    min_len = min(f0.shape[0], BNFs.shape[0])
    vc_inputs = np.concatenate([BNFs[:min_len, :], f0[:min_len, :]], axis=1)
    vc_inputs = np.expand_dims(vc_inputs, axis=1)  # [time, batch=1, dim]

    # 4. 加载 VC 模型
    model = BLSTMToManyConversionModel(
        in_channels=hps.Audio.bn_dim + 2,
        out_channels=hps.Audio.num_mels,
        num_spk=hps.SPEAKERS.num_spk,
        embd_dim=hps.BLSTMToManyConversionModel.spk_embd_dim,
        lstm_hidden=hps.BLSTMToManyConversionModel.lstm_hidden,
    )
    device = torch.device("cpu")
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    # 5. 对每个目标说话人进行转换
    for tgt_spk in tgt_spk_names:
        # 获取目标说话人索引
        tgt_spk_ind = torch.LongTensor([hps.SPEAKERS.spk_to_inds.index(tgt_spk)])

        # 进行语音转换
        predicted_mels = model(torch.tensor(vc_inputs).to(torch.float), tgt_spk_ind)
        predicted_mels = np.squeeze(predicted_mels.detach().numpy(), axis=1)

        # 6. 合成语音
        synthesized_wav = inv_preemphasize(inv_mel_spectrogram(predicted_mels.T))
        ckpt_name = args.ckpt.split("/")[-1].split(".")[0].split("-")[-1]
        wav_name = args.src_wav.split("/")[-1].split(".")[0]

        # 创建保存目录（如果不存在）
        os.makedirs(args.save_dir, exist_ok=True)

        # 保存转换后的语音
        output_wav_path = os.path.join(
            args.save_dir,
            f"{wav_name}-to-{tgt_spk}-converted-{ckpt_name}.wav"
        )
        save_wav(synthesized_wav, output_wav_path)
        print(f"Saved converted wav for target speaker '{tgt_spk}' at: {output_wav_path}")

    # 7. 保存原始语音的重合成版本（可选）
    resynthesized_wav = inv_preemphasize(inv_mel_spectrogram(mel_spec.T))
    resyn_wav_path = os.path.join(args.save_dir, f"{wav_name}-src-cp-syn-{ckpt_name}.wav")
    save_wav(resynthesized_wav, resyn_wav_path)
    print(f"Saved resynthesized source wav at: {resyn_wav_path}")

if __name__ == "__main__":
    main()
