import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision.models import resnet34
from dataset_episodic import AudioDataset, TestAudioDataset

pbar = 0

class SpeakerNetModel2(nn.Module):
    def __init__(self, config):
        super(SpeakerNetModel2, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=config.params.init_k)
        self.resnet = resnet34(num_classes=config.params.resnet_dim)
        self.dropout = nn.Dropout(config.params.dropout)
        self.fc = nn.Linear(config.params.resnet_dim, config.params.fc_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class SpeakerNetModel(nn.Module):
    '''
        ResNet18 model for speaker verification

        Inspired by : Deep CNNs With Self-Attention for Speaker Identification (An et al. 2019)
    '''

    def __init__(self, config):
        super(SpeakerNetModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=config.params.init_k)
        self.resnet = resnet34(num_classes=config.params.resnet_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet(x)
        x = self.activation(x)
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
        # self.loss_fn = nn.TripletMarginLoss(margin=config.loss.margin)
        self.loss_fn = self.compute_loss if config.loss.name == "triplet" else self.compute_loss_cosine

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

        self.dev_dataset.make_episodes()
        for episode in self.dev_dataset.episodes:
            wavs = episode["wav"].unsqueeze(1)
            labels = episode["label"]
            wavs = wavs.to(self.device)
            labels = labels.to(self.device)

            embs = self.model(wavs)
            loss, acc, score = self.loss_fn(embs, labels, val=True)
            if loss is None:
                continue

        print(f"Val   {self.epoch} | Loss: {loss.item() if loss else 0:.3f} | Acc: {acc:.3f} | Score: {score:.3f}", file=sys.stderr)

    def train_one_epoch(self):
        self.model.train()

        all_confs = []
        num_correct = 0
        total = 0

        self.train_dataset.make_episodes()
        for episode in self.train_dataset.episodes:
            wavs = episode["wav"].unsqueeze(1)
            labels = episode["label"]
            wavs = wavs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            embs = self.model(wavs)

            loss, acc, score = self.loss_fn(embs, labels)
            if loss is None:
                continue
            loss.backward()

            self.optimizer.step()

        print(f"Epoch {self.epoch} | Loss: {loss.item() if loss else 0:.3f} | Acc: {acc:.3f} | Score: {score:.3f}", file=sys.stderr)

    def compute_loss(self, embs, labels, val=False):
        '''
            Implements triplet mining for hard/semihard/easy pos/neg supporting points

            Inspired by : Improved Embeddings with Easy Positive Triplet Mining (Xuan et al. 2018)
        '''

        # dists: (batch_size, batch_size)
        dists = torch.cdist(embs, embs)

        anchors, pos, neg = self.mine_triplets(embs, labels, maximize=False)
        if anchors is None:
            print("Warning: No valid triplets found", file=sys.stdout)
            return None, 0, 0

        loss = F.triplet_margin_loss(a, p, n, margin=self.config.loss.margin)

        a = embs[anchors]
        p = embs[pos]
        n = embs[neg]

        # with torch.no_grad():
        acc = torch.mean((dists[anchors, pos] < dists[anchors, neg]).float()).detach().cpu().numpy()
        preds = torch.cat([dists[anchors, pos], dists[anchors, neg]])
        targets = torch.cat([torch.ones_like(dists[anchors, pos]), torch.zeros_like(dists[anchors, neg])])
        score = F.binary_cross_entropy_with_logits(preds, targets).detach().cpu().numpy()
        return loss, acc, score

    def compute_loss_cosine(self, embs, labels, val=False):
        # map onto unit sphere
        embs = F.normalize(embs, p=2, dim=1)
        # normalize similarity to [0, 1]
        sims = (F.cosine_similarity(embs.unsqueeze(0), embs.unsqueeze(1), dim=2) + 1) / 2

        anchors, pos, neg = self.mine_triplets(embs, sims, labels, maximize=True, val=val)

        preds = torch.cat([sims[anchors, pos], sims[anchors, neg]])
        # labels for a x p = 1, a x n = 0
        targets = torch.cat([torch.ones_like(sims[anchors, pos]), torch.zeros_like(sims[anchors, neg])])
        loss = F.binary_cross_entropy_with_logits(preds, targets)
        acc = torch.mean((sims[anchors, pos] > sims[anchors, neg]).float()).detach().cpu().numpy()
        score = F.binary_cross_entropy_with_logits(preds, targets).detach().cpu().numpy()
        return loss, acc, score

    def mine_triplets(self, embs, scores, labels, maximize, val):
        # anchors: (batch_size)
        anchors = torch.arange(len(labels))
        # positive_mask: (batch_size, batch_size)
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask.fill_diagonal_(True)
        # negative_mask: (batch_size, batch_size)
        negative_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
        negative_mask.fill_diagonal_(True)

        if maximize:
            scores *= -1

        if self.config.loss.positive == "random" or val:
            # create array of random values of same shape, mask out positives and take max
            pos = torch.masked_fill(torch.rand_like(scores), negative_mask, float('-inf'))
            posv, pos = torch.max(pos, dim=1)
        elif self.config.loss.positive == "hard":
            # pos: (batch_size)
            pos = torch.masked_fill(scores, negative_mask, float('-inf'))
            posv, pos = torch.max(pos, dim=1)
        elif self.config.loss.positive == "easy":
            # pos: (batch_size)
            pos = torch.masked_fill(scores, negative_mask, float('inf'))
            posv, pos = torch.min(pos, dim=1)

        if self.config.loss.negative == "random" or val:
            # create array of random values of same shape, mask out positives and take max
            neg = torch.masked_fill(torch.rand_like(scores), positive_mask, float('-inf'))
            negv, neg = torch.max(neg, dim=1)
        elif self.config.loss.negative == "hard":
            # neg: (batch_size)
            neg = torch.masked_fill(scores, positive_mask, float('inf'))
            negv, neg = torch.min(neg, dim=1)
        elif self.config.loss.negative == "easy":
            # neg: (batch_size)
            neg = torch.masked_fill(scores, positive_mask, float('-inf'))
            negv, neg = torch.max(neg, dim=1)
        elif self.config.loss.negative == "semihard":
            # neg: (batch_size)
            neg = torch.masked_fill(scores, positive_mask, float('inf'))
            mask = (scores > scores[anchors, pos])
            neg = torch.masked_fill(neg, mask, float('inf'))
            negv, neg = torch.min(neg, dim=1)

        if torch.isinf(posv).any() or torch.isinf(negv).any():
            print("Warning: Inf values in pos/neg distances", file=sys.stdout)
            return None, None, None
        return anchors, pos, neg

    def eval_test(self):
        self.model.eval()

        preds = []
        test_dataset = TestAudioDataset(self.config.dataset.test)
        for batch in test_dataset:
            wav1 = batch["wav1"]
            wav2 = batch["wav2"]
            wav1 = wav1.to(self.device)
            wav2 = wav2.to(self.device)

            embs1 = self.model(wav1)
            embs2 = self.model(wav2)
            # map onto unit sphere
            embs1 = F.normalize(embs1, p=2, dim=1)
            embs2 = F.normalize(embs2, p=2, dim=1)
            dist = torch.dist(embs1, embs2)
            confidence = 1 / np.exp(dist.item())
            print(f"Confidence: {confidence}")
            preds.append(confidence)

        preds = np.array(preds, dtype=np.float32)
        np.save(f"{self.config.dataset.test.output_dir}/task1_predictions.npy", preds)
        print(preds)

    def predict_batch(self, batch):
        self.model.eval()
        wavs = batch["wav"]
        wavs = wavs.to(self.device)
        embs = self.model(wavs)
        return embs.detach().cpu().numpy()
