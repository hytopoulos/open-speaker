import os, random
import torch, torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchaudio.functional import preemphasis
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import Compose, Lambda
from torchvision.transforms.functional import crop
from tqdm import tqdm

class EpisodicDataloader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super(EpisodicDataloader, self).__init__(dataset, **kwargs)

    def __iter__(self):
        return iter(self.dataset)

class BaseDataset(Dataset):
    def __init__(self, config):
        self._transform = Compose([
            preemphasis,
            MelSpectrogram(window_fn=torch.hamming_window, n_fft=600, n_mels=512),
            AmplitudeToDB(top_db=80),
            Lambda(lambda x: crop(x, 534-512, 0, 512, 512))
        ])
        self.num_samples = 10 * 16000
        self.samples = self.load_data(f"{config.root}")
        self.config = config

    def transform(self, wav):
        # pad wav to num_samples
        pad_size = self.num_samples - wav.shape[1]
        wav = F.pad(wav, (pad_size, 0))

        # create mask
        mask = torch.zeros((wav.shape[1]), dtype=torch.float32)
        mask[-pad_size:] = 1

        # apply transformations
        wav = self._transform(wav)

        # resample mask
        mask = F.interpolate(mask[None, None, :], size=(wav.shape[2]))[0, 0, :]

        return wav, mask

    def load_data(self, data_root):
        raise NotImplementedError()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class AudioDataset(BaseDataset):
    def __init__(self, config, cache_dir=None):
        self.cache_dir = os.path.join(cache_dir, f"{config.root.replace('/', '_')}.pt") if cache_dir is not None else None
        super(AudioDataset, self).__init__(config)
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

            wav, _ = torchaudio.load(wav_path)

            with open(spkid_path, 'r') as f:
                spkid = torch.tensor(int(f.read().strip()))

            wav, mask = self.transform(wav)

            if samples.get(spkid.item()) is None:
                samples[spkid.item()] = []
            samples[spkid.item()].append({
                "wav": wav,
                "pad_mask": mask,
                "label": spkid
            })
        if self.cache_dir is not None:
            torch.save(samples, self.cache_dir)
        return samples

    def make_episodes(self):
        episodes = []
        available_speakers = list(self.samples.keys())

        for _ in range(self.config.num_episodes):
            classes = random.sample(available_speakers, self.config.num_ways)
            episode = { "wav": [], "label": [], "pad_mask": [] }
            for c in classes:
                if len(self.samples[c]) < self.config.num_shots:
                    continue
                shots = torch.tensor(random.sample(range(len(self.samples[c])), self.config.num_shots))
                for s in shots:
                    episode["wav"].append(self.samples[c][s]["wav"])
                    episode["label"].append(self.samples[c][s]["label"])
                    episode["pad_mask"].append(self.samples[c][s]["pad_mask"])
            episode["wav"] = torch.cat(episode["wav"], dim=0)
            episode["label"] = torch.stack(episode["label"])
            episode["pad_mask"] = torch.stack(episode["pad_mask"])
            episodes.append(episode)
        self.episodes = episodes

class TestAudioDataset(BaseDataset):
    def load_data(self, data_root):
        wav_paths = []
        with open(f"{data_root}/task1.script.txt", 'r') as f:
            for line in f:
                wav_paths.append(line.split())

        samples = []
        for w1, w2 in wav_paths:
            wav1, _ = torchaudio.load(f"{data_root}/{w1}")
            wav2, _ = torchaudio.load(f"{data_root}/{w2}")

            wav1, mask1 = self.transform(wav1)
            wav2, mask2 = self.transform(wav2)

            samples.append({
                "wav1": wav1.unsqueeze(0),
                "pad_mask1": mask1.unsqueeze(0),
                "wav2": wav2.unsqueeze(0),
                "pad_mask2": mask2.unsqueeze(0),
            })
        return samples
