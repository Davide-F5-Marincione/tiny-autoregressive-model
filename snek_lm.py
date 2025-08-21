import array
from typing import Tuple
import pickle
import numpy as np

def popcount(x: int) -> int:
    """Count the number of set bits in a byte using SWAR algorithm."""
    x = x - ((x >> 1) & 0x55) # 0x55 = 01010101
    x = (x & 0x33) + ((x >> 2) & 0x33) # 0x33 = 00110011
    return ((x + (x >> 4)) & 0x0F)  # 0x0F = 00001111

def binarize_vector(vec: array.array, data: array.array) -> array.array:
    for i in range(8):
        data[0] = ((vec[i] < 0) << 7) & 0x80 | (data[0] >> 1) & 0x7F # Last & is needed only because Python.
        data[1] = ((vec[i + 8] < 0) << 7) & 0x80 | (data[1] >> 1) & 0x7F
    return data

def add_vectors(vec1: array.array, vec2: array.array) -> array.array:
    for i in range(vec1.__len__()):
        vec1[i] += vec2[i]
    return vec1

def matmul(mat1: array.array, mat2T: array.array, result: array.array) -> array.array:
    for c in range(0, mat2T.__len__(), 2): # 16
        xor_mul = mat1[0] ^ mat2T[c]
        row_sum = popcount(xor_mul)
        xor_mul = mat1[1] ^ mat2T[c + 1]
        row_sum += popcount(xor_mul)
        
        result[c >> 1] = 16 - (row_sum << 1)
    return result

def matmul_T(mat1: array.array, mat2T: array.array, result: array.array) -> array.array:
    w = 0
    for c in range(16):
        B = c >= 8
        k = c % 8
        for r in range(B, 16, 2):
            w = (w >> 1) & 0x7F | (mat2T[r] << (7 - k)) & 0x80
        xor_mul = mat1[0] ^ w
        row_sum = popcount(xor_mul)
        for r in range(B + 16, 32, 2):
            w = (w >> 1) & 0x7F | (mat2T[r] << (7 - k)) & 0x80
        xor_mul = mat1[1] ^ w
        row_sum += popcount(xor_mul)
        
        result[c] = 16 - (row_sum << 1)
    return result

        
def select_embedding(embeddings: array.array, index: int, data: array.array) -> array.array:
    index = index << 1
    data[0] = embeddings[index]
    data[1] = embeddings[index + 1]
    return data

class Snek:
    def __init__(self, vocab_size=256):
        """
        An SSM-based language model, with binary matrix operations and embeddings.
        """
        super().__init__()

        self.vocab_size = vocab_size

        # Binary input projection → u, delta
        self.u_projs = array.array('B', (0 for _ in range(vocab_size << 1)))
        self.deltas = array.array('B', (0 for _ in range(vocab_size << 1)))

        # Binary versions of A, B, C, D
        self.At = array.array('B', (0 for _ in range(32)))
        self.Bt = array.array('B', (0 for _ in range(32)))
        self.Dt = array.array('B', (0 for _ in range(32)))
    
    def __call__(self, index, state: array.array, inference_space: array.array) -> Tuple[array.array, array.array]:        
        data1 = array.array('B',inference_space[:2].tobytes())
        data2 = array.array('B',inference_space[2:4].tobytes())
        data3 = array.array('b',inference_space[4:20].tobytes())
        data4 = array.array('b',inference_space[20:36].tobytes())
        
        u_proj = select_embedding(self.u_projs, index, data1)
        delta = select_embedding(self.deltas, index, data2)
        
        decay_logit = matmul(delta, self.At, data3)
        state_proposal = matmul(u_proj, self.Bt, data4)
        
        for i in range(16):
            state[i] = (((decay_logit[i] + 16) * (state[i] - state_proposal[i])) >> 5) + state_proposal[i]
        
        y_t = binarize_vector(add_vectors(matmul_T(binarize_vector(state, data2), self.Bt, data3), matmul(u_proj, self.Dt, data4)), data1)

        return matmul(y_t, self.u_projs, inference_space[:self.vocab_size]), state
    
    def create_inference_space(self):
        """Create an inference space for the model."""
        return array.array('b', (0 for _ in range(max(36, self.vocab_size))))
    
    def bytes_count(self):
        return (self.u_projs.__len__() + 
                self.deltas.__len__() + 
                self.At.__len__() + 
                self.Bt.__len__() + 
                self.Dt.__len__())
    
    def load(self, path):
        """Load the model from a file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.u_projs = data['u_projs'].data
            self.deltas = data['deltas'].data
            self.At = data['At'].data
            self.Bt = data['Bt'].data
            self.Dt = data['Dt'].data

        self.vocab_size = len(self.u_projs) // 2
        return self
    
if __name__ == "__main__":
    
    MAX_LEN = 256
    VOCAB_SIZE = 27
    
    model = Snek().load("binary_mamba16_10epochs.tiny")
    print(f"Snek model initialized with {model.bytes_count()} bytes.")
    inference_space = model.create_inference_space()
    state = array.array('b', (0 for _ in range(16)))  # Initial state for the SSM

    print("Tot bytes used:", model.bytes_count() + inference_space.__len__() + state.__len__())

    # Beam search parameters
    beam_width = 5
    max_length = 20
    temp = 1  # Temperature for softmax scaling
    end_bias = lambda x: len(x) * 1e-2  # Bias for end token based on sequence length
    repetition_penalty = 1.2  # Penalty for repeated tokens
    prompt = "k"

    VOCAB = {
        l: i for i, l in enumerate("\nabcdefghijklmnopqrstuvwxyz")
    }
    INVERSE_VOCAB = {v: k for k, v in VOCAB.items()}
    VOCAB_SIZE = len(VOCAB)

    # Greedy search initialization
    result = prompt
    while result[-1] != '\n' and len(result) < max_length:
        last_index = VOCAB[result[-1]]
        output, state = model(last_index, state, inference_space)
        output = np.array(output)
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
    beams = [(prompt, 0.0, array.array('b', (0 for _ in range(16))))]  # (sequence, score, state)
    for _ in range(max_length - len(prompt)):
        new_beams = []
        for seq, score, state in beams:
            last_index = VOCAB[seq[-1]]
            output, state = model(last_index, array.array('b', (v for v in state)), inference_space)
            output = np.array(output)
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