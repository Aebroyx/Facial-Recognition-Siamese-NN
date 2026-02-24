# One-Shot Face Recognition with a Siamese Network

A Keras-based face recognition system built with a Siamese network. Siamese networks enable **one-shot learning**: recognizing faces from very few examples instead of large labeled datasets.

---

## Getting Started

This section covers environment setup, how to train the model, and how to run the face recognition app, plus a short overview of one-shot learning and Siamese networks.

### Prerequisites

- **Python 3.x**
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Data & model storage (local only)

Training data and the trained model are **not** in this repo (they are ignored because of size). You need to **store them locally** on your machine:

- **Training data** — After running `datafetch.py`, keep the downloaded datasets in your chosen directory (e.g. under `src` or a separate `data/` folder).
- **Saved model** — After training, the checkpoint is saved as `saved_best` (and/or `saved_best.h5`). Keep these files locally; they are required to run the app with `-m`.

Do not commit these folders/files to the repo; they are listed in `.gitignore`.

### Training the Network

1. **Download datasets** (from the `src` directory). Training and evaluation data are fetched in parallel; this can take around an hour.

```bash
python datafetch.py
```

2. **Train the model.** The best checkpoint is saved as `saved_best`.

```bash
python train.py
```

### Running the App

Run the app with `main.py` using:

- `-db` — path to the face database
- `-m` — path to the saved model
- `-i` — paths to one or more images to run recognition on

**Example:**

```bash
python main.py -db ../database -m ../saved_best -i ../myself.jpg image2.jpg
```

On the evaluation set, the app typically achieves **75–85% accuracy**, depending on the samples used.

---

## One-Shot Learning

Deep neural networks excel at learning from high-dimensional data (images, speech, etc.) when given large amounts of labeled data. Humans, by contrast, can often learn from a single example—e.g., after seeing one picture of a spatula, they can reliably tell spatulas apart from other kitchen tools.

Several recent works have tackled one-shot learning with neural networks. This project uses one such approach: the **Siamese network**.

---

## Siamese Network

The idea is to train a network that takes **two images** and predicts whether they belong to the **same class**. At test time, the network compares the query image to each image in a support set and assigns the label of the most similar one. So we need an architecture that takes two images and outputs a probability that they share the same class.

Simply concatenating the two images and feeding them into one network would use different weights for each input, which breaks symmetry. A better design is to use **two identical subnetworks with shared weights** (the “twins”), process both images through them, and feed their **absolute difference** into a linear classifier. That shared-weight, twin architecture is the Siamese network—two identical branches joined at the head.

![Siamese Network](https://github.com/aebroyx/Facial-Recognition-Siamese-NN/blob/master/screenshots/siamese.png)

The final output is passed through a **sigmoid** so it lies in [0, 1]. Targets are \(t = 1\) when the two images share a class and \(t = 0\) otherwise. Training uses **binary cross-entropy** between predictions and targets, plus an **L2 weight decay** term to favor smaller, less noisy weights and better generalization.

For a one-shot task, the Siamese net compares the test image to every image in the support set and assigns the class of the one it judges most similar.

---

## Author

- **Sangalabror Pujianto** — [Portfolio](https://sangalabror.aebroyx.dev)

## License

This project is licensed under the MIT License — see [LICENSE.md](https://github.com/aebroyx/Facial-Recognition-Siamese-NN/blob/master/LICENSE) for details.

## Acknowledgements

- [One-shot learning with Siamese networks](https://sorenbouma.github.io/blog/oneshot/)
