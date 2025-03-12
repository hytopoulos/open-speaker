import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import TransformerEncoder
from torchvision.models import resnet34
from dataset_episodic import AudioDataset, TestAudioDataset

pbar = 0

class SpeakerNetTransformer(nn.Module):
    def __init__(self, config):
        super(SpeakerNetTransformer, self).__init__()
        self.transformer = TransformerEncoder(
            nn.TransformerEncoderLayer(128, 8, batch_first=True),
            num_layers=6,
        )

        self.dropout = nn.Dropout(config.params.dropout)
        self.fc = nn.Linear(128, 128)

    def forward(self, x, masks):
        x = x.transpose(1, 2) # swap axes to (batch_size, time_dim, feature_dim)
        x = self.transformer(x, src_key_padding_mask=masks)
        # pool over time dimension
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class SpeakerNetModel(nn.Module):
    def __init__(self, config):
        super(SpeakerNetModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=config.params.init_k)
        self.resnet = resnet34(num_classes=config.params.resnet_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.resnet(x)
        x = self.activation(x)
        x = x.squeeze(1)
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
        if config.loss.type=="triplet":
            self.loss_fn = self.triplet_loss
        elif config.loss.type=="contrastive":
            self.loss_fn = self.contrastive_loss
        elif config.loss.type=="cosine":
            self.loss_fn = self.cosine_embedding_loss
        else:
            raise ValueError("Invalid loss type")

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

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        acc_store = []
        score_store = []
        loss_store = []

        self.dev_dataset.make_episodes()
        for episode in self.dev_dataset.episodes:
            wavs = episode["wav"]
            masks = episode["pad_mask"]
            labels = episode["label"]
            wavs = wavs.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)

            embs = self.model(wavs, masks)
            loss, acc, score = self.loss_fn(embs, labels, val=True)
            if loss is None:
                continue
            acc_store.append(acc)
            score_store.append(score)
            loss_store.append(loss.item())

        print(f"Val   {self.epoch} | Loss: {np.mean(loss_store):.3f} | Acc: {np.mean(acc_store):.3f} | Score: {np.mean(score_store):.3f}", file=sys.stderr)

    def train_one_epoch(self):
        self.model.train()

        acc_store = []
        score_store = []
        loss_store = []

        self.train_dataset.make_episodes()
        for episode in self.train_dataset.episodes:
            wavs = episode["wav"]
            masks = episode["pad_mask"]
            labels = episode["label"]
            wavs = wavs.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            
            embs = self.model(wavs, masks)
            # map onto unit sphere
            embs = F.normalize(embs, p=2, dim=1)

            dists = 1 - F.cosine_similarity(embs.unsqueeze(0), embs.unsqueeze(1), dim=2)
            loss = self.loss_fn(embs, dists, labels)
            acc, score = self.compute_accuracy_score(embs, labels)
            if loss is None:
                continue

            acc_store.append(acc)
            score_store.append(score)
            loss_store.append(loss.item())

            loss.backward()
            self.optimizer.step()

        print(f"Epoch {self.epoch} | Loss: {np.mean(loss_store):.3f} | Acc: {np.mean(acc_store):.3f} | Score: {np.mean(score_store):.3f}", file=sys.stderr)

    @torch.no_grad()
    def compute_accuracy_score(self, dists, labels):
        anchors, pos, neg = self.mine_triplets(dists, labels, val=True)
        if anchors is None:
            return 0
        # define accuracy as triplets where a->p < a->n
        acc = torch.mean((dists[anchors, pos] < dists[anchors, neg]).float()).detach().cpu().numpy()
        preds = torch.cat([dists[anchors, pos], dists[anchors, neg]])
        targets = torch.cat([torch.ones_like(dists[anchors, pos]), torch.zeros_like(dists[anchors, neg])])
        score = F.binary_cross_entropy_with_logits(preds, targets).detach().cpu().numpy()
        return acc, score

    def cosine_embedding_loss(self, embs, dists, labels):
        anchors, pos, neg = self.mine_triplets(embs, dists, labels)
        if anchors is None:
            return None

        anchors2 = torch.cat([anchors, anchors])
        posneg = torch.cat([pos, neg])
        targets = torch.cat([torch.ones_like(dists[anchors, pos]), -torch.ones_like(dists[anchors, neg])])
        return torch.cosine_embedding_loss(embs[anchors2], embs[posneg], targets)

    def contrastive_loss(self, embs, dists, labels):
        anchors, pos, neg = self.mine_triplets(embs, dists, labels)
        if anchors is None:
            return None

        preds = torch.cat([dists[anchors, pos], dists[anchors, neg]])
        # labels for a x p = 1, a x n = 0
        targets = torch.cat([torch.ones_like(dists[anchors, pos]), torch.zeros_like(dists[anchors, neg])])

        return F.binary_cross_entropy_with_logits(preds, targets)

    def triplet_loss(self, embs, dists, labels, val=False):
        anchors, pos, neg = self.mine_triplets(dists, labels, val=val)
        if anchors is None:
            return None

        return F.triplet_margin_loss(embs[anchors], embs[pos], embs[neg], margin=self.config.loss.margin)

    def mine_triplets(self, scores, labels, val):
        # anchors: (batch_size)
        anchors = torch.arange(len(labels))
        # positive_mask: (batch_size, batch_size)
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask.fill_diagonal_(True)
        # negative_mask: (batch_size, batch_size)
        negative_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
        negative_mask.fill_diagonal_(True)

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
            mask = (scores < self.config.loss.margin)
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
            pad_mask1 = batch["pad_mask1"]
            wav2 = batch["wav2"]
            pad_mask2 = batch["pad_mask2"]
            wav1 = wav1.to(self.device)
            pad_mask1 = pad_mask1.to(self.device)
            wav2 = wav2.to(self.device)
            pad_mask2 = pad_mask2.to(self.device)

            embs1 = self.model(wav1, pad_mask1)
            embs2 = self.model(wav2, pad_mask2)
            # map onto unit sphere
            embs1 = F.normalize(embs1, p=2, dim=1)
            embs2 = F.normalize(embs2, p=2, dim=1)
            dist = torch.pairwise_distance(embs1, embs2)
            confidence = 1 / np.exp(dist.item())
            print(f"Confidence: {confidence}")
            preds.append(confidence)

        preds = np.array(preds, dtype=np.float32)
        np.save(f"{self.config.dataset.test.output_dir}/task1_predictions.npy", preds)
        print(preds)

    def predict_batch(self, batch):
        self.model.eval()
        wavs = batch["wav"]
        pad_masks = batch["pad_mask"]
        wavs = wavs.to(self.device)
        pad_masks = pad_masks.to(self.device)
        embs = self.model(wavs, pad_masks)
        return embs.detach().cpu().numpy()
