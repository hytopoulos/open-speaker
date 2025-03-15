import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_episodic import AudioDataset, TestAudioDataset

from model import SpeakerResNet34, SpeakerResNet50, SpeakerResNet101

class SpeakerNetAgent():
    def __init__(self, config):
        if not os.path.exists(config.cache_dir):
            os.makedirs(config.cache_dir)

        self.device = torch.device("cuda")
        self.config = config

        self.train_dataset = AudioDataset(config.dataset.train, cache_dir=config.cache_dir)
        self.dev_dataset = AudioDataset(config.dataset.dev, cache_dir=config.cache_dir)
        self.train_dataloader = DataLoader(self.train_dataset, shuffle=True, **config.loader)
        self.dev_dataloader = DataLoader(self.dev_dataset, shuffle=False, **config.loader)

        self.model = globals()[config.model.name](config.model).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **config.optimizer)

        self.dev_scores = []

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

                if self.dev_scores[-1] != None and min(self.dev_scores) == self.dev_scores[-1]:
                    print("Saving best model")
                    self.save_model("model_best.pt")

            if epoch % self.config.checkpoint_freq == 0:
                self.save_model(f"model_epoch_{epoch}.pt")


    def load_model(self, name):
        if os.path.exists(self.config.checkpoint_dir):
            self.model.load_state_dict(torch.load(os.path.join(self.config.checkpoint_dir, name)))
            print(f"Model loaded from {self.config.checkpoint_dir}")

    def save_model(self, name):
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
        torch.save(self.model.state_dict(), os.path.join(self.config.checkpoint_dir, name))

    def train_one_epoch(self):
        self.model.train()

        acc_store = []
        score_store = []
        loss_store = []

        self.train_dataset.make_episodes()
        for episode in self.train_dataset.episodes:

            self.optimizer.zero_grad()
            loss, acc, score = self.forward(episode)
            if loss is None:
                continue

            loss.backward()
            self.optimizer.step()

            acc_store.append(acc)
            score_store.append(score)
            loss_store.append(loss.item())

        print(f"Epoch {self.epoch} | Loss: {np.mean(loss_store):.3f} | Acc: {np.mean(acc_store):.3f} | Score: {np.mean(score_store):.3f}", file=sys.stderr)

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        acc_store = []
        score_store = []
        loss_store = []

        self.dev_dataset.make_episodes()
        for episode in self.dev_dataset.episodes:
            
            loss, acc, score = self.forward(episode)
            if loss is None:
                loss = torch.zeros(1) # if no valid triplets for loss computation, thats fine..

            acc_store.append(acc)
            score_store.append(score)
            loss_store.append(loss.item())

        self.dev_scores.append(np.mean(score_store))
        print(f"Val   {self.epoch} | Loss: {np.mean(loss_store):.3f} | Acc: {np.mean(acc_store):.3f} | Score: {np.mean(score_store):.3f}", file=sys.stderr)

    def forward(self, batch):
        wavs = batch["wav"].to(self.device)
        masks = batch["pad_mask"].to(self.device)
        labels = batch["label"].to(self.device)

        embs = self.model(wavs, masks)

        # map onto unit sphere, create distance matrix
        embs = F.normalize(embs, p=2, dim=1)
        dists = torch.cdist(embs, embs, p=2)

        loss = self.loss_fn(embs, dists, labels)
        acc, score = self.compute_accuracy_score(dists, labels)

        return loss, acc, score

    @torch.no_grad()
    def compute_accuracy_score(self, dists, labels):
        positive_mask, negative_mask = self.compute_masks(labels)

        positive_mask.fill_diagonal_(False)
        negative_mask.fill_diagonal_(False)

        inclass_dists = dists[positive_mask].view(-1)
        outclass_dists = dists[negative_mask].view(-1)

        dists = torch.cat([inclass_dists, outclass_dists], dim=0)
        dists_norm = torch.clamp(dists, min=0, max=self.config.loss.margin) / self.config.loss.margin
        targets = torch.cat([torch.zeros_like(inclass_dists), torch.ones_like(outclass_dists)], dim=0)
        acc = ((dists > self.config.loss.margin) == targets).float().mean().detach().cpu().numpy()
        score = F.binary_cross_entropy_with_logits(dists_norm, targets, reduction='none')
        score[len(inclass_dists):] /= self.config.dataset.train.num_shots
        score = score.mean().detach().cpu().numpy()
        return acc, score

    def cosine_embedding_loss(self, embs, dists, labels):
        anchors, pos, neg = self.mine_triplets(dists, labels)
        if anchors is None:
            return None

        anchors2 = torch.cat([anchors, anchors])
        posneg = torch.cat([pos, neg])
        targets = torch.cat([torch.ones_like(pos), -torch.ones_like(neg)])
        return F.cosine_embedding_loss(embs[anchors2], embs[posneg], targets, margin=self.config.loss.margin)

    def contrastive_loss(self, embs, dists, labels):
        anchors, pos, neg = self.mine_triplets(dists, labels)
        if anchors is None:
            return None

        logits = 1 / torch.exp(torch.cat([dists[anchors, pos], dists[anchors, neg]]))
        targets = torch.cat([torch.zeros_like(pos), torch.ones_like(neg)]).float()
        return F.binary_cross_entropy_with_logits(logits, targets)

    def triplet_loss(self, embs, dists, labels):
        anchors, pos, neg = self.mine_triplets(dists, labels)
        if anchors is None:
            return None

        return F.triplet_margin_loss(embs[anchors], embs[pos], embs[neg], margin=self.config.loss.margin)


    def mine_triplets(self, dists, labels, val=False):
        ''' take current batch (online), find optimal triplets (mining)
            
            `dists`: distance matrix

            `labels`: speaker IDs
            
            `val`: if True, use random positive/negative samples
        '''
        anchors = torch.arange(len(labels))
        positive_mask, negative_mask = self.compute_masks(labels)

        if self.config.loss.positive == "random" or val:
            pos = torch.masked_fill(torch.rand_like(dists), negative_mask, float('-inf'))
            posv, pos = torch.max(pos, dim=1)
        elif self.config.loss.positive == "hard":
            pos = torch.masked_fill(dists, negative_mask, float('-inf'))
            posv, pos = torch.max(pos, dim=1)
        elif self.config.loss.positive == "easy":
            pos = torch.masked_fill(dists, negative_mask, float('inf'))
            posv, pos = torch.min(pos, dim=1)

        if self.config.loss.negative == "random" or val:
            neg = torch.masked_fill(torch.rand_like(dists), positive_mask, float('-inf'))
            negv, neg = torch.max(neg, dim=1)
        elif self.config.loss.negative == "hard":
            neg = torch.masked_fill(dists, positive_mask, float('inf'))
            negv, neg = torch.min(neg, dim=1)
        elif self.config.loss.negative == "easy":
            neg = torch.masked_fill(dists, positive_mask, float('-inf'))
            negv, neg = torch.max(neg, dim=1)
        elif self.config.loss.negative == "semihard":
            neg = torch.masked_fill(dists, positive_mask, float('inf'))
            mask = (dists < self.config.loss.margin)
            neg = torch.masked_fill(neg, mask, float('inf'))
            negv, neg = torch.min(neg, dim=1)

        if torch.isinf(posv).any() or torch.isinf(negv).any():
            print(f"Inf distance, val={val}, p:{self.config.loss.positive}, n:{self.config.loss.negative}", file=sys.stdout)
            return None, None, None
        return anchors, pos, neg

    def eval_test(self):
        ''' Evaluate the model on the test set, containing pairs of clips
        '''
        self.model.eval()

        preds = []
        test_dataset = TestAudioDataset(self.config.dataset.test)
        for batch in test_dataset:
            wav1 = batch["wav1"].to(self.device)
            pad_mask1 = batch["pad_mask1"].to(self.device)
            wav2 = batch["wav2"].to(self.device)
            pad_mask2 = batch["pad_mask2"].to(self.device)

            embs1 = self.model(wav1, pad_mask1)
            embs2 = self.model(wav2, pad_mask2)
            
            embs1 = F.normalize(embs1, p=2, dim=1)
            embs2 = F.normalize(embs2, p=2, dim=1)
            dist = torch.cdist(embs1, embs2, p=2)

            clamped_dist = torch.clamp(dist, min=0, max=1)
            preds.append(-clamped_dist)

        preds = np.array(preds, dtype=np.float32)
        np.save(f"{self.config.dataset.test.output_dir}/task1_predictions.npy", preds)
        print(preds)

    def compute_masks(self, labels):
        # for masking out same speakers
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask.fill_diagonal_(True)
        # for masking out different speakers
        negative_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
        negative_mask.fill_diagonal_(True)

        return positive_mask, negative_mask

    def predict_batch(self, batch):
        self.model.eval()
        wavs = batch["wav"].to(self.device)
        pad_masks = batch["pad_mask"].to(self.device)
        embs = self.model(wavs, pad_masks)
        return embs.detach().cpu().numpy()
