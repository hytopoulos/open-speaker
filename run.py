#!/usr/bin/env python3

import sys, json
from agent import SpeakerNetAgent
from dotmap import DotMap
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

np.set_printoptions(precision=3)

def main(args):
    config_path = args[1]
    config = load_config(config_path, args)

    agent = SpeakerNetAgent(config)
    agent.run()

    if config.pca_on_finish:
        make_pca(agent, dev=True)
        make_pca(agent, dev=False)
    if config.tsne_on_finish:
        make_tsne(agent, dev=True)
        make_tsne(agent, dev=False)

    if config.eval_test:
        agent.eval_test()

def make_pca(agent, dev=True):
    embs = []
    labels = []
    loader = agent.dev_dataloader if dev else agent.train_dataloader
    for batch in loader:
        labels.extend(batch["label"].numpy())
        embs.extend(agent.predict_batch(batch))
    embs = np.array(embs)
    pca = PCA(n_components=2)
    out = pca.fit_transform(embs)
    figure = plt.figure()
    plt.scatter(out[:, 0], out[:, 1], c=labels)
    plt.title("PCA")
    plt.savefig(f"{'dev' if dev else 'train'}_pca.png")
    plt.close(figure)

def make_tsne(agent, dev=True):
    labels = []
    embs = []
    loader = agent.dev_dataloader if dev else agent.train_dataloader
    for batch in loader:
        labels.extend(batch["label"].numpy())
        embs.extend(agent.predict_batch(batch))
    embs = np.array(embs)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    out = tsne.fit_transform(embs)
    figure = plt.figure()
    plt.scatter(out[:, 0], out[:, 1], c=labels)
    plt.title("TSNE")
    plt.savefig(f"{'dev' if dev else 'train'}_tsne.png")
    plt.close(figure)

def load_config(config_path, args):
    with open(config_path, 'r') as f:
        config = DotMap(json.load(f))
        # override the configuration
        for arg in args[2:]:
            key, value = arg.split("=")
            attrs = key.split(".")
            obj = config
            for attr in attrs[:-1]:
                obj = getattr(obj, attr)
            setattr(obj, attrs[-1], type(getattr(obj, attrs[-1]))(value))
    return config

if __name__ == "__main__":
    main(sys.argv)
