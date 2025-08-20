from binary_mamba import *


if __name__ == "__main__":
    # Beam search parameters
    beam_width = 5
    max_length = 20
    temp = 1  # Temperature for softmax scaling
    end_bias = lambda x: len(x) * 0.1  # Bias for end token based on sequence length
    repetition_penalty = 1.2  # Penalty for repeated tokens
    prompt = "m"

    VOCAB = {
        l: i for i, l in enumerate("\nabcdefghijklmnopqrstuvwxyz")
    }
    INVERSE_VOCAB = {v: k for k, v in VOCAB.items()}
    VOCAB_SIZE = len(VOCAB)

    model = BinaryMamba(vocab_size=VOCAB_SIZE, d_model=16)
    model.load('binary_mamba16_10epochs.tiny')

    # Greedy search initialization
    state = None
    result = prompt
    while result[-1] != '\n' and len(result) < max_length:
        last_index = VOCAB[result[-1]]
        output, state = model.step([last_index], state)
        output = np.array(output.data)
        output = output / temp
        output = np.exp(output - np.max(output))
        output[VOCAB['\n']] += end_bias(result)
        output[VOCAB[result[-1]]] /= repetition_penalty
        output = output / np.sum(output)

        next_index = np.random.choice(
            range(VOCAB_SIZE), p=output
        )
        result += INVERSE_VOCAB[next_index]

    result = result.strip()
    print(f"Greedy search result: {result}")

    # Beam search initialization
    beams = [(prompt, 0.0, None)]  # (sequence, score, state)
    for _ in range(max_length - len(prompt)):
        new_beams = []
        for seq, score, state in beams:
            last_index = VOCAB[seq[-1]]
            output, state = model.step([last_index], state)
            output = np.array(output.data)
            output = output / temp  # Apply temperature scaling
            output = np.exp(output - np.max(output))
            output[VOCAB['\n']] += end_bias(seq)  # Bias for end token
            output[VOCAB[seq[-1]]] /= repetition_penalty  # Apply repetition penalty
            output = output / np.sum(output)

            # Select top-k candidates
            top_indices = np.argsort(output)[-beam_width:][::-1]
            for idx in top_indices:
                new_seq = seq + INVERSE_VOCAB[idx]
                new_score = score - np.log(output[idx] + 1e-10)  # Negative log likelihood
                new_beams.append((new_seq, new_score, state))

        # Keep only the best beams
        beams = sorted(new_beams, key=lambda x: x[1])[:beam_width]
        if beams[0][0].endswith('\n'):
            break

    # Print the best beam
    best_seq, best_score, _ = beams[0]
    best_seq = best_seq.strip()
    print(f"Beam search result: {best_seq}")
