# Open-Set Speaker Verification

## Methods

We train a model to produce an embedded representation that can uniquely identify the speaker and compare several architecture decisions, which will be referred to as the study.

### Feature Extraction

Waveforms are first left-padded to a length of 10 seconds. Waveform feature extraction began with preemphasis. Mel spectrograms (`n_mels=128`) were then extracted from the waveforms and converted from amplitude to decibel scale. These transformations were found to be common in audio processing.

### Models

A transformer encoder [Vaswani et al., 2017] and ResNet34 [He et al., 2016] were tested as candidate architectures for the study. The transformer was configured with 6 self-attention heads and masked attention for spectrograms with padding. The time dimension was pooled using the mean of the last hidden state before a linear transformation. The output dimension of the transformer is 128 logits due to the spectrogram settings.

The ResNet model contains a 2D-convolution layer to map the spectrogram to 3 channels, which is then passed to the PyTorch implementation of ResNet34. The output dimension of the ResNet was configured to 512 logits.

### Training

Training is accomplished using contrastive learning. Contrastive learning trained a model `f` to minimize the distance metric `d` between points with the same label, and maximize the distance between separate labels. The distance chosen for this study is the `l_2` norm of the embedding. Triplet loss [Wang et al., 2019] and pairwise loss were both examined for optimizing the model.

#### Triplet Loss

Triplet loss is defined with the following equation:

```math
L = \max(0, d_{f(a),f(p)} - d_{f(a),f(n)} + \alpha)
```

The terms are defined as `a` being the anchor sample, `p` being the same label as the anchor, and `d` being a different label as the anchor. Separation between labels is enforced by the `α` hyperparameter. Sets of triplets must be picked strategically, since the number of combinations scales cubically with the size of the dataset. Online triplet mining is a technique where optimal triplets are chosen from a batch during training. One type of triplet mining involves finding the hardest examples, i.e., the negative closest to the anchor and the positive furthest from the anchor. This is compared with random selection in the results section. The margin was fixed to `α=1`, which is the standard for Euclidean distance and gave the best results.

#### Pairwise Loss

Pairwise loss simply minimizes the distance between points in the same class and maximizes distance between points in different classes. The implementation in the study is the inverse exponential of the distance with separate terms for minimizing positive label distance and maximizing negative label distance.

#### Episodic Training

Models were trained over batches of episodes, where an episode contains a random subset of `K` classes each with `N` samples. All models were trained with `K=8` and `N=4` for 500 epochs, with each epoch containing 100 episodes.

## Results

After selecting the best model from each trial, we find that training the model on harder samples yielded better accuracy than training randomly selected samples (see Figure below). While the transformer encoder was able to fit to the training samples, it performed poorly on validation. Accuracy is defined as the number of seeded random positive and negative samples where `p_{d,p} < p_{d,n}`. The learning rate for the following experiments was `lr=1e-4` with weight decay `w=1e-5`.

### Model and Triplet Study

| Model      | Loss Type | Triplet Type       | Dev Acc. | Dev Loss |
|------------|----------|--------------------|----------|----------|
| ResNet34   | Triplet  | max(P), min(N)     | 0.955    | 0.166    |
| ResNet34   | Triplet  | rand(P), rand(N)   | 0.908    | 0.164    |
| ResNet34   | Pairwise | max(P), min(N)     | 0.933    | 0.164    |
| ResNet34   | Pairwise | rand(P), rand(N)   | 0.776    | 0.162    |
| Transformer | Pairwise | rand(P), rand(N)  | 0.529    | 0.693    |
| Transformer | Triplet  | rand(P), rand(N)  | 0.529    | 0.693    |
| ResNet101  | Triplet  | rand(P), rand(N)   | 0.775    | 0.161    |

From the study, ResNet101 was determined to be the optimal model. Random triplets were chosen to balance accuracy (clustering) and the loss. The full configuration is located in **config/main.json**.

## Conclusions

Computer vision techniques are suitable for processing audio spectrograms. Metric learning and self-supervised tasks benefit from large amounts of data to model the feature space. It is unclear why the transformer performed poorly, but future work would include constructing embeddings of individual frames as is done in the literature.

