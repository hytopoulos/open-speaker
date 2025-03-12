import os, re, random
import torch, torchaudio

from torch.utils.data import Dataset, DataLoader
from torchaudio.functional import preemphasis
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, AddNoise, FrequencyMasking
from torchvision.transforms import Compose
from tqdm import tqdm

class EpisodicDataloader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super(EpisodicDataloader, self).__init__(dataset, **kwargs)

    def __iter__(self):
        return iter(self.dataset)

class AudioDataset(Dataset):
    def __init__(self, config, cache_dir=None):
        self.cache_dir = os.path.join(cache_dir, f"{config.root.replace('/', '_')}.pt") if cache_dir is not None else None
        self.transform = Compose([
            preemphasis,
            MelSpectrogram(window_fn=torch.hamming_window, n_fft=600),
            AmplitudeToDB(top_db=80)
        ])

        # self.augmentations = Compose([
        #     FrequencyMasking(freq_mask_param=30, iid_masks=True)
        # ])
        self.num_samples = 10 * 16000
        self.samples = self.load_data(config.root)
        self.make_episodes()

    def load_data(self, data_root):
        if self.cache_dir is not None and os.path.exists(self.cache_dir):
            return torch.load(self.cache_dir)

        assert len(os.listdir(data_root)) % 3 == 0, "Expected 3 files per sample"
        num_samples = len(os.listdir(data_root)) // 3

        wav_files = sorted([f for f in os.listdir(data_root) if f.endswith(".wav")])
        spkid_files = sorted([f for f in os.listdir(data_root) if f.endswith(".spkid.txt")])

        samples = {}
        for i in tqdm(range(num_samples), desc="Loading data"):
            wav_path = os.path.join(data_root, wav_files[i])
            spkid_path = os.path.join(data_root, spkid_files[i])

            wav_data, _ = torchaudio.load(wav_path)

            with open(spkid_path, 'r') as f:
                spkid = torch.tensor(int(f.read().strip()))

            wav_data = self.pad_wav(wav_data)
            wav_data = self.transform(wav_data)
            # if data_root.find("train") != -1:
            #     wav_data = self.augmentations(wav_data)

            if samples.get(spkid.item()) is None:
                samples[spkid.item()] = []
            samples[spkid.item()].append({
                "wav": wav_data,
                "label": spkid
            })
        if self.cache_dir is not None:
            torch.save(samples, self.cache_dir)
        return samples

    def make_episodes(self, num_episodes=100, num_cls=8, num_shots=4):
        episodes = []
        available_speakers = list(self.samples.keys())

        for _ in range(num_episodes):
            classes = random.sample(available_speakers, num_cls)
            episode = { "wav": [], "label": [] }
            for c in classes:
                if len(self.samples[c]) < num_shots:
                    continue
                shots = torch.tensor(random.sample(range(len(self.samples[c])), num_shots))
                for s in shots:
                    episode["wav"].append(self.samples[c][s]["wav"])
                    episode["label"].append(self.samples[c][s]["label"])
            episode["wav"] = torch.cat(episode["wav"], dim=0)
            episode["label"] = torch.stack(episode["label"])
            episodes.append(episode)
        self.episodes = episodes


    def pad_wav(self, wav):
        if wav.shape[1] < self.num_samples:
            pad_size = self.num_samples - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad_size))
        return wav

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]

class TestAudioDataset(Dataset):
    def __init__(self, config):
        self.transform = Compose([
            preemphasis,
            MelSpectrogram(window_fn=torch.hamming_window, n_fft=600),
            AmplitudeToDB(top_db=80)
        ])
        self.num_samples = 10 * 16000
        self.samples = self.load_data(f"{config.input_dir}")

    def load_data(self, input_dir):
        wav_paths = []
        with open(f"{input_dir}/task1.script.txt", 'r') as f:
            for line in f:
                wav_paths.append(line.split())

        samples = []
        for w1, w2 in wav_paths:
            wav1, _ = torchaudio.load(f"{input_dir}/{w1}")
            wav2, _ = torchaudio.load(f"{input_dir}/{w2}")

            wav1 = self.pad_wav(wav1)
            wav2 = self.pad_wav(wav2)

            wav1 = self.transform(wav1)
            wav2 = self.transform(wav2)

            samples.append({
                "wav1": wav1.unsqueeze(0),
                "wav2": wav2.unsqueeze(0)
            })
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
