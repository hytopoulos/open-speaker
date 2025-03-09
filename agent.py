import os, sys
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision.models import resnet18
from dataset import AudioDataset

pbar = 0

class SpeakerNetModel2(nn.Module):
    def __init__(self, config):
        super(SpeakerNetModel2, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=config.params.init_k)
        self.resnet = resnet18(num_classes=config.params.resnet_dim)
        self.dropout = nn.Dropout(config.params.dropout)
        self.fc = nn.Linear(config.params.resnet_dim, config.params.fc_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class SpeakerNetModel(nn.Module):
    def __init__(self, config):
        super(SpeakerNetModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=config.params.init_k)
        self.resnet = resnet18(num_classes=config.params.resnet_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet(x)
        return x

class SpeakerNetAgent():
    def __init__(self, config):

        if not os.path.exists(config.cache_dir):
            os.makedirs(config.cache_dir)

        self.device = torch.device("cuda")
        self.config = config
        self.train_dataset = AudioDataset(config.dataset.train, cache_dir=config.cache_dir)
        # self.train_dataset.samples = [s for s in self.train_dataset.samples if s['label'] < 20]
        self.dev_dataset = AudioDataset(config.dataset.dev, cache_dir=config.cache_dir)
        # self.dev_dataset.samples = [s for s in self.dev_dataset.samples if s['label'] < 100]
        self.train_dataloader = DataLoader(self.train_dataset, shuffle=True, **config.loader)
        self.dev_dataloader = DataLoader(self.dev_dataset, shuffle=False, **config.loader)
        self.model = globals()[config.model.name](config.model).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **config.optimizer)
        self.loss_fn = nn.TripletMarginLoss()

    def run(self):
        self.train()

    def train(self):
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self.train_one_epoch()
            
            if epoch % self.config.val_interval == 0:
                self.validate()

            if epoch % self.config.checkpoint_freq == 0:
                self.save_model()

    def save_model(self):
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
        torch.save(self.model.state_dict(), os.path.join(self.config.checkpoint_dir, f"model_epoch_{self.epoch}.pt"))

    def validate(self):
        self.model.eval()

        all_confs = []
        num_correct = 0
        total = 0

        for batch in self.dev_dataloader:
            wavs = batch["wav"]
            labels = batch["label"]
            wavs = wavs.to(self.device)
            labels = labels.to(self.device)

            embs = self.model(wavs)
            a, p, n = self.calc_triplets(embs, labels)
            loss = self.loss_fn(embs[a], embs[p], embs[n])
            dists = torch.cdist(embs, embs)
            confs = 1 / np.exp(dists[a, p].detach().cpu().numpy())
            all_confs.extend(confs)
            num_correct += torch.sum(dists[a, n] > dists[a, p]).item()
            total += len(labels)

        print(f"Val {self.epoch} | Loss: {loss.item():.3f} | Acc: {num_correct/total:.3f} | Conf Avg: {np.mean(all_confs):.3f} | Conf Std: {np.std(all_confs):.3f}", file=sys.stderr)

    def train_one_epoch(self):
        self.model.train()

        all_confs = []
        num_correct = 0
        total = 0

        if pbar: batch_tqdm = tqdm(self.train_dataloader, total=len(self.train_dataloader), desc=f"[train {self.epoch}]")
        for batch in self.train_dataloader:
            wavs = batch["wav"]
            labels = batch["label"]
            wavs = wavs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            embs = self.model(wavs)
            a, p, n = self.calc_triplets(embs, labels)
            loss = self.loss_fn(embs[a], embs[p], embs[n])
            loss.backward()

            dists = torch.cdist(embs, embs)

            dists = torch.cdist(embs, embs)
            confs = 1 / np.exp(dists[a, p].detach().cpu().numpy())
            all_confs.extend(confs)
            num_correct += torch.sum(dists[a, n] > dists[a, p]).item()
            total += len(labels)

            self.optimizer.step()

        print(f"Epoch {self.epoch} | Loss: {loss.item():.3f} | Acc: {num_correct}/{total} | Conf Avg: {np.mean(all_confs):.3f} | Conf Std: {np.std(all_confs):.3f}", file=sys.stderr)

    @torch.no_grad()
    def calc_triplets(self, embs, labels):
        dists = torch.cdist(embs, embs)

        anchors = torch.arange(len(labels))
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        negative_mask = labels.unsqueeze(0) != labels.unsqueeze(1)

        # hard positives for each anchor
        pos = torch.max(torch.masked_fill(dists, negative_mask, float('-inf')), dim=1).indices
        # hard negatives for each anchor
        neg = torch.min(torch.masked_fill(dists, positive_mask, float('inf')), dim=1).indices
        return anchors, pos, neg
