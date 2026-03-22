# =============================================================================
# SoundSense — Preprocessing Pipeline
# Converts all .wav files to mel-spectrograms saved as .npy arrays
# Input  : data/mimii/valve/valve/id_XX/normal & abnormal
# Output : data/processed/id_XX_normal.npy & id_XX_abnormal.npy
# =============================================================================

import os
import numpy as np
import librosa

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_PATH    = r"C:\soundsense\data\mimii\valve\valve"
OUTPUT_PATH  = r"C:\soundsense\data\processed"
MACHINE_IDS  = ["id_00", "id_02", "id_04", "id_06"]
N_MELS       = 128       # number of mel bands
SR           = 16000     # sample rate
DURATION     = 10        # seconds per clip

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── Helper function ────────────────────────────────────────────────────────────
def wav_to_melspec(filepath):
    """Load a .wav file and return its mel-spectrogram in dB."""
    audio, sr = librosa.load(filepath, sr=SR, duration=DURATION)
    mel        = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_db     = librosa.power_to_db(mel, ref=np.max)
    return mel_db

# ── Main pipeline ──────────────────────────────────────────────────────────────
for machine_id in MACHINE_IDS:
    for label in ["normal", "abnormal"]:

        folder = os.path.join(BASE_PATH, machine_id, label)
        files  = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")]

        print(f"Processing {machine_id} / {label} — {len(files)} files...")

        spectrograms = []
        for f in files:
            mel = wav_to_melspec(f)
            spectrograms.append(mel)

        # Save as numpy array
        out_array = np.array(spectrograms)
        out_file  = os.path.join(OUTPUT_PATH, f"{machine_id}_{label}.npy")
        np.save(out_file, out_array)

        print(f"   Saved {out_array.shape} → {out_file}")

print("\n Preprocessing complete!")