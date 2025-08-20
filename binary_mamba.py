import random
from typing import List
import tqdm
import pickle

from binarylinalg import *

class BinaryEmbeddings:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = BinMatrix(num_embeddings, embedding_dim).randomize()
        
    def select(self, indices: List[int]) -> BinMatrix:
        selected = BinMatrix(len(indices), self.embedding_dim)
        actual_cols = (self.embedding_dim >> 3) + ((self.embedding_dim & 8) > 0)
        for i, idx in enumerate(indices):
            for j in range((self.embedding_dim >> 3) + ((self.embedding_dim & 8) > 0)):
                selected.data[i * actual_cols + j] = self.embeddings.data[idx * actual_cols + j]
        return selected
    

class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01, weight_clip=2):
        self.params = params  # A list or dict of your model's parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.weight_clip = weight_clip  # Limit for parameter updates
        
        # Initialize moment vectors and timestep counter
        self.m = {key: np.zeros_like(p) for key, p in params.items()}
        self.v = {key: np.zeros_like(p) for key, p in params.items()}
        self.t = 0

    def step(self, grads):
        """Performs a single optimization step."""
        self.t += 1
        
        for key, param in self.params.items():
            if key in grads:
                g = grads[key]
                
                # 1. Update first and second moment vectors
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g * g)
                
                # 2. Bias correction
                m_hat = self.m[key] / (1 - self.beta1**self.t)
                v_hat = self.v[key] / (1 - self.beta2**self.t)
                
                # 3. Update parameters
                update_value = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                param[...] =np.clip(param - update_value - param * self.weight_decay, -self.weight_clip, self.weight_clip) # This updates the parameter in place

class BinaryMamba:
    def __init__(self, vocab_size=256, d_model=16):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Binary input projection → u, delta
        self.u_projs = BinaryEmbeddings(vocab_size, d_model)
        self.deltas = BinaryEmbeddings(vocab_size, d_model)

        # Binary versions of A, B, C, D
        self.At = BinMatrix(d_model, d_model)
        self.Bt = BinMatrix(d_model, d_model)
        # self.C = BinMatrix(d_model, d_model).randomize()
        self.Dt = BinMatrix(d_model, d_model)

        self.u_projs_float = np.clip(np.random.randn(self.u_projs.num_embeddings, self.d_model), -2, 2)
        self.deltas_float = np.clip(np.random.randn(self.u_projs.num_embeddings, self.d_model), -2, 2)
        self.At_float = np.clip(np.random.randn(self.d_model, self.d_model), -2, 2)
        self.Bt_float = np.clip(np.random.randn(self.d_model, self.d_model), -2, 2)
        # self.C_float = bin2float(self.C)
        self.Dt_float = np.clip(np.random.randn(self.d_model, self.d_model), -2, 2)

        self.binarize_weights()
    
    def step(self, index, state:ByteMatrix = None):
        
        u_proj = self.u_projs.select(index)
        delta = self.deltas.select(index)
        
        if state is None:
            state = ByteMatrix(len(index), self.d_model)

        decay_logit = delta.matmul_other_T_no_bin(self.At)
        state_proposal = u_proj.matmul_other_T_no_bin(self.Bt)
        
        for i in range(state.data.__len__()):
            state.data[i] = max(min((((decay_logit.data[i] + self.d_model) * (state.data[i] - state_proposal.data[i])) // (2 * self.d_model)) + state_proposal.data[i], self.d_model), -self.d_model)

        y_t = state.binarize().matmul_other_T_no_bin(self.Bt.transpose()).add(u_proj.matmul_other_T_no_bin(self.Dt)).binarize()

        return y_t.matmul_other_T_no_bin(self.u_projs.embeddings), state
    
    def train(self, batch):
        grads = {
            'u_projs': np.zeros((self.u_projs.num_embeddings, self.d_model), dtype=np.float32),
            'deltas': np.zeros((self.deltas.num_embeddings, self.d_model), dtype=np.float32),
            'At': np.zeros((self.d_model, self.d_model), dtype=np.float32),
            'Bt': np.zeros((self.d_model, self.d_model), dtype=np.float32),
            # 'C': np.zeros((self.d_model, self.d_model), dtype=np.float32),
            'Dt': np.zeros((self.d_model, self.d_model), dtype=np.float32),
        }

        loss = 0.0

        batch_size = len(batch[0])
        N = len(batch)

        mask = np.ones((batch_size, N), dtype=np.float32)

        # Sentence is a list of indices in the embedding space
        # Each target at a step is the next index in the sentence
        # loss is cross-entropy loss, thus gradients are y - max(pred)
        states = [ByteMatrix(batch_size, self.d_model)]
        inputs = []
        outputs = []
        targets = []
        y_ts = []
        u_projs = []
        deltas = []
        decay_logits = []
        proposals = []

        for i in range(N - 1):
            indexes = list(batch[i])
            next_indexes = list(batch[i + 1])

            for j in range(batch_size):
                if indexes[j] is None or next_indexes[j] is None:
                    mask[j, i] = 0.0
                    indexes[j] = 0
                    next_indexes[j] = 0
            
            u_proj = self.u_projs.select(indexes)
            delta = self.deltas.select(indexes)

            decay_logit = delta.matmul_other_T_no_bin(self.At)
            state_proposal = u_proj.matmul_other_T_no_bin(self.Bt)

            state = ByteMatrix(batch_size, self.d_model)
            
            for j in range(state.data.__len__()):
                state.data[j] = max(min((((decay_logit.data[j] + self.d_model) * (states[-1].data[j] - state_proposal.data[j])) // (2 * self.d_model)) + state_proposal.data[j], self.d_model), -self.d_model)

            y_t = state.binarize().matmul_other_T_no_bin(self.Bt.transpose()).add(u_proj.matmul_other_T_no_bin(self.Dt))

            output = y_t.binarize().matmul_other_T_no_bin(self.u_projs.embeddings)

            inputs.append(indexes)
            outputs.append(output)
            states.append(state)
            targets.append(next_indexes)
            y_ts.append(y_t)
            u_projs.append(u_proj)
            deltas.append(delta)
            decay_logits.append(decay_logit)
            proposals.append(state_proposal)

        # Backprop
        tot = np.sum(mask)
        incoming_grad = np.zeros((batch_size, self.d_model), dtype=np.float32)
        for i in range(len(outputs)):
            i = N - i - 1 
            inpt = inputs.pop()
            output = byte2float(outputs.pop())
            target = targets.pop()
            state = byte2float(states.pop())
            y_t = byte2float(y_ts.pop())
            u_proj = bin2float(u_projs.pop())
            delta = bin2float(deltas.pop())
            decay_logit = (byte2float(decay_logits.pop()) + self.d_model)/ 2 / self.d_model
            proposal = byte2float(proposals.pop())

            preds = np.exp(output-np.max(output))  # Softmax normalization
            preds = preds / np.sum(preds, axis=-1, keepdims=True)  # Softmax normalization
            loss += (-np.log(preds[range(batch_size), target] + 1e-10) / tot * mask[:, i]).sum() # Cross-entropy loss

            out_grad = preds
            out_grad[range(batch_size), target] -= 1.0
            out_grad *= 1 / tot * mask[:, i:i+1]# Normalize by batch and sentence length
            grads['u_projs'] += (y_t.T @ out_grad).T
            y_t_grad = out_grad @ self.u_projs_float
            grads['Dt'] += (u_proj.T @ y_t_grad).T
            grads['Bt'] += state.T @ y_t_grad
            state_grad = y_t_grad @ self.Bt_float.T + incoming_grad
            incoming_grad = state_grad * decay_logit
            proposal_grad = state_grad * (1 - decay_logit)
            decay_logit_grad = state_grad * (byte2float(states[-1]) - proposal) / 2 / self.d_model
            grads['Bt'] += (u_proj.T @ proposal_grad).T
            grads['At'] += (delta.T @ decay_logit_grad).T
            u_proj_grad = y_t_grad @ self.Dt_float + proposal_grad @ self.Bt_float
            delta_grad = decay_logit_grad @ self.At_float
            grads['u_projs'][inpt] += u_proj_grad
            grads['deltas'][inpt] += delta_grad

        return loss, grads
    
    def binarize_weights(self):
        self.u_projs.embeddings = float2bin(self.u_projs_float)
        self.deltas.embeddings = float2bin(self.deltas_float)
        self.At = float2bin(self.At_float)
        self.Bt = float2bin(self.Bt_float)
        # self.C = float2bin(self.C_float)
        self.Dt = float2bin(self.Dt_float)
    
    def save(self, path):
        """Save the model to a file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'u_projs': self.u_projs.embeddings,
                'deltas': self.deltas.embeddings,
                'At': self.At,
                'Bt': self.Bt,
                'Dt': self.Dt,
            }, f)
            
    def load(self, path):
        """Load the model from a file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.u_projs.embeddings = data['u_projs']
            self.deltas.embeddings = data['deltas']
            self.At = data['At']
            self.Bt = data['Bt']
            self.Dt = data['Dt']
        
        # Convert back to binary
        self.u_projs_float = bin2float(self.u_projs.embeddings)
        self.deltas_float = bin2float(self.deltas.embeddings)
        self.At_float = bin2float(self.At)
        self.Bt_float = bin2float(self.Bt)
        self.Dt_float = bin2float(self.Dt)

        self.vocab_size = self.u_projs.num_embeddings
        self.d_model = self.u_projs.embedding_dim


class DataGenerator:
    def __init__(self, samples: List[str], vocab: dict, max_len: int, batch_size: int = 32, shuffle: bool = True):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.samples)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.samples):
            raise StopIteration
        batch = []
        while self.i < len(self.samples):
            sample = self.samples[self.i]
            self.i += 1
            sample = sample.strip().lower()
            if len(sample) > self.max_len:
                sample = sample[:self.max_len]
            indices = [self.vocab.get(char, 0) for char in sample]
            batch.append(indices)
            if len(batch) == self.batch_size:
                break
        max_len = max(len(s) for s in batch)
        batch = [s + [None] * (max_len - len(s)) for s in batch]
        batch = [s for s in zip(*batch) if any(s)] # Reset batch
        return batch


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
    params_to_optimize = {
        'u_projs': model.u_projs_float,
        'deltas': model.deltas_float,
        'At': model.At_float,
        'Bt': model.Bt_float,
        'Dt': model.Dt_float,
    }
    optimizer = AdamOptimizer(params_to_optimize, lr=4e-3, weight_decay=0.0, weight_clip=2)

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
    data_gen = DataGenerator(samples, VOCAB, max_len=MAX_LEN, batch_size=32, shuffle=True)


    lr_schedule = lambda epoch: 1e-2 * (0.9 ** (epoch // 2))  # Decrease learning rate every 2 epochs
    for i in range(10):
        print(f"Epoch {i+1}/10")
        optimizer.lr = lr_schedule(i)
        for batch in (pbar:=tqdm.tqdm(data_gen, total=len(samples)//32 + 1, desc="Training")):
            loss, grads = model.train(batch)
            optimizer.step(grads)
            model.binarize_weights()
            pbar.set_postfix(loss=loss)

    # Save
    model.save('binary_mamba16_10epochs.tiny')