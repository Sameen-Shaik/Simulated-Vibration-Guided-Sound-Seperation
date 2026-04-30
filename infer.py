"""
infer.py
========
Inference script: separate overlapping birds from a new audio file.

Usage:
  python infer.py \
    --input mixed_birds.wav \
    --checkpoint experiments/checkpoints/vibration/best.pt \
    --output_dir separated/

The script:
  1. Loads the mixed audio
  2. Generates simulated vibration signals from the mixture
     (inference-time: derived from mixture itself, per thesis Section 4.3)
  3. Runs the separator
  4. Saves separated tracks as WAV files
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

import numpy as np
import torch
import soundfile as sf
from pathlib import Path

from data.dataset import VibrationSimulator
from models.separator import build_model


# Default config matching training
DEFAULT_CFG = {
    "model": {
        "num_speakers": 2,
        "audio_encoder_channels": 256,
        "audio_encoder_kernel_size": 16,
        "audio_encoder_stride": 8,
        "vib_encoder_channels": 128,
        "vib_encoder_layers": 4,
        "tcn_channels": 256,
        "tcn_kernel_size": 3,
        "tcn_layers": 8,
        "tcn_stacks": 3,
        "film_hidden_dim": 256,
    }
}


def load_wav(path: str, target_sr: int = 22050) -> np.ndarray:
    import librosa
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    # Normalise
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        audio = audio / max_val
    return audio.astype(np.float32)


def infer(
    input_path: str,
    checkpoint_path: str,
    output_dir: str,
    num_speakers: int = 2,
    sample_rate: int = 22050,
    device: str = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Load model
    cfg = DEFAULT_CFG.copy()
    cfg["model"]["num_speakers"] = num_speakers
    model = build_model(cfg, "vibration")

    ckpt = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(dev)
    print(f"  Model loaded from: {checkpoint_path}")

    # Load audio
    audio = load_wav(input_path, sample_rate)
    print(f"  Input: {input_path} ({len(audio)/sample_rate:.2f}s)")

    # Simulate vibrations from mixture (inference-time approach per thesis §4.3)
    # Since we don't have clean per-bird audio at inference, we derive N vibration
    # signals from the mixture using different frequency bands / energy patterns.
    vib_sim = VibrationSimulator(sample_rate=sample_rate, noise_std=0.02)
    per_bird_vibs = []
    for bird_idx in range(num_speakers):
        vib = vib_sim.simulate(audio, bird_id=bird_idx)
        per_bird_vibs.append(vib)

    # Prepare tensors
    mixture_t = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(dev)  # (1, 1, T)
    vibrations_t = torch.stack(
        [torch.from_numpy(v) for v in per_bird_vibs], dim=0
    ).unsqueeze(0).to(dev)  # (1, N, T)

    # Run separation
    with torch.no_grad():
        separated, _ = model(mixture_t, vibrations_t)

    separated_np = separated.squeeze(0).cpu().numpy()   # (N, T)

    # Save outputs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_stem = Path(input_path).stem

    for n in range(num_speakers):
        out_path = out_dir / f"{input_stem}_separated_bird{n+1}.wav"
        sf.write(str(out_path), separated_np[n], sample_rate)
        print(f"  Saved: {out_path}")

    print(f"  Done. {num_speakers} tracks saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Separate birds from mixed audio")
    parser.add_argument("--input", required=True, help="Mixed WAV file")
    parser.add_argument("--checkpoint",
                        default="experiments/checkpoints/vibration/best.pt")
    parser.add_argument("--output_dir", default="separated")
    parser.add_argument("--num_speakers", type=int, default=2)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    infer(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_speakers=args.num_speakers,
        sample_rate=args.sample_rate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
