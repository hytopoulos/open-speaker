import os
import torch, torchaudio

from torch.utils.data import Dataset
from torchaudio.functional import preemphasis
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, AddNoise
from torchvision.transforms import Compose
from tqdm import tqdm

class AudioDataset(Dataset):
    def __init__(self, config, cache_dir=None):
        self.cache_dir = os.path.join(cache_dir, f"{config.root.replace('/', '_')}.pt") if cache_dir is not None else None
        self.transform = Compose([
            preemphasis,
            MelSpectrogram(window_fn=torch.hamming_window, n_fft=600),
            AmplitudeToDB(top_db=80)
        ])
        self.num_samples = 10 * 16000
        self.samples = self.load_data(config.root)

    def load_data(self, data_root):
        if self.cache_dir is not None and os.path.exists(self.cache_dir):
            return torch.load(self.cache_dir)

        assert len(os.listdir(data_root)) % 3 == 0, "Expected 3 files per sample"
        num_samples = len(os.listdir(data_root)) // 3

        wav_files = sorted([f for f in os.listdir(data_root) if f.endswith(".wav")])
        spkid_files = sorted([f for f in os.listdir(data_root) if f.endswith(".spkid.txt")])

        samples = []
        for i in tqdm(range(num_samples), desc="Loading data"):
            wav_path = os.path.join(data_root, wav_files[i])
            spkid_path = os.path.join(data_root, spkid_files[i])

            wav_data, _ = torchaudio.load(wav_path)

            with open(spkid_path, 'r') as f:
                spkid = torch.tensor(int(f.read().strip()))

            wav_data = self.pad_wav(wav_data)
            wav_data = self.transform(wav_data)
            samples.append({
                "wav": wav_data,
                "label": spkid
            })

        if self.cache_dir is not None:
            torch.save(samples, self.cache_dir)
        return samples

    def pad_wav(self, wav):
        if wav.shape[1] < self.num_samples:
            pad_size = self.num_samples - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad_size))
        return wav

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
