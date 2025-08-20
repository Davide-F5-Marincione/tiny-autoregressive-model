import random
from typing import List
import tqdm

from binary_mamba import *


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    MAX_LEN = 32
    # Create a vocabulary mapping for lowercase letters, space, and period

    VOCAB = {
        l: i for i, l in enumerate("\nabcdefghijklmnopqrstuvwxyz")
    }
    INVERSE_VOCAB = {v: k for k, v in VOCAB.items()}
    VOCAB_SIZE = len(VOCAB)

    model = BinaryMamba(vocab_size=VOCAB_SIZE, d_model=16)
    model.load('binary_mamba16.tiny')

    samples = []
    # with open('text.txt', 'r') as f:
    #     sample = ""
    #     for line in f:
    #         line = line.strip()
    #         if not line:
    #             if sample:
    #                 samples.append(sample)
    #                 sample = ""
    #         else:
    #             sample += ' ' + line
    with open('human_names.txt', 'r') as f:
        samples = f.read().splitlines(keepends=True)

    # Calculate cross-entropy on full dataset
    total_loss = 0.0
    total_samples = 0
    for sample in tqdm.tqdm(samples):
        sample = sample.strip().lower()
        state = None
        for i in range(len(sample)-1):
            total_samples += 1
            idx = VOCAB[sample[i]]
            next_idx = VOCAB[sample[i+1]]
            output, state = model.step([idx], state)
            output = np.array(output.data)
            preds = np.exp(output-np.max(output))  # Softmax normalization
            preds = preds / np.sum(preds, axis=-1, keepdims=True)  # Softmax normalization
            total_loss += -np.log(preds[next_idx] + 1e-10) # Cross-entropy loss

    print(f"Total loss: {total_loss / total_samples:.4f}")