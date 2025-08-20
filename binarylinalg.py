import array
import random
import numpy as np

def popcount(x: int) -> int:
    """Count the number of set bits in a byte using SWAR algorithm."""
    x = x - ((x >> 1) & 0x55) # 0x55 = 01010101
    x = (x & 0x33) + ((x >> 2) & 0x33) # 0x33 = 00110011
    return ((x + (x >> 4)) & 0x0F)  # 0x0F = 00001111

class DataHolder:
    def __init__(self, size: int, dtype: str = 'B'):
        assert size > 0, "Size must be a positive integer."
        self.data = array.array(dtype, (0 for _ in range(size)))
        
    def randomize(self):
        if self.data.typecode == 'b':
            a,b = -128, 127
        elif self.data.typecode == 'B':
            a, b = 0, 255
        for i in range(len(self.data)):
            self.data[i] = random.randint(a, b)
        return self

class ByteMatrix(DataHolder):
    def __init__(self, rows: int, cols: int):
        super().__init__(rows * cols, 'b')
        self.rows = rows
        self.cols = cols        
        
    def binarize(self):
        result = BinMatrix(self.rows, self.data.__len__() // self.rows)
        for i in range(self.data.__len__()):
            result.data[i >> 3] |= (self.data[i] < 0) << (i % 8)
        return result
    
    def transpose(self):
        result = ByteMatrix(self.cols, self.rows)
        for r in range(self.rows):
            for c in range(self.cols):
                result.data[c * self.rows + r] = self.data[r * self.cols + c]
        return result
    
    def add(self, other: 'ByteMatrix'):
        for i in range(self.data.__len__()):
            self.data[i] += other.data[i]
        return self
    
    def add_new(self, other: 'ByteMatrix'):
        new_vector = ByteMatrix(self.rows, self.cols)
        for i in range(self.data.__len__()):
            new_vector.data[i] = self.data[i] + other.data[i]
        return new_vector
    
    def __repr__(self):
        # Generate a string representation of the binary matrix
        matrix_str = []
        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                row_str.append(str(self.data[r * self.cols + c]))
            matrix_str.append(' '.join(row_str))
        return '\n'.join(matrix_str)

class BinMatrix(DataHolder):
    def __init__(self, rows: int, cols: int, grad: bool = False):
        super().__init__(rows * ((cols >> 3) + ((cols % 8) > 0)), 'B')
        self.rows = rows
        self.cols = cols
        
    def transpose(self):
        transposed = BinMatrix(self.cols, self.rows)
        actual_cols = (self.cols >> 3) + ((self.cols % 8) > 0)
        actual_rows = (self.rows >> 3) + ((self.rows % 8) > 0)
        for r in range(self.rows):
            for c in range(self.cols):
                bit = (self.data[r * actual_cols + (c >> 3)] >> (c % 8)) & 0x1
                transposed.data[c * actual_rows + (r >> 3)] |= bit << (r % 8)
        return transposed
    
    # def matmul(self, other: 'BinMatrix') -> 'BinMatrix':
    #     return self.matmul_other_T(other.transpose())
    
    # def matmul_other_T(self, other: 'BinMatrix') -> 'BinMatrix':
    #     assert self.cols == other.cols, "Incompatible dimensions for matrix multiplication."
    #     result = BinMatrix(self.rows, other.rows)
    #     actual_cols = (self.cols >> 3) + ((self.cols % 8) > 0)
    #     actual_res_cols = (other.rows >> 3) + ((other.rows % 8) > 0)
    #     for r in range(self.rows):
    #         for c in range(other.rows):
    #             row_sum = 0
    #             for k in range(actual_cols):
    #                 xor_mul = self.data[r * actual_cols + k] ^ other.data[c * actual_cols + k]
    #                 if k == actual_cols - 1 and self.cols % 8 > 0:
    #                     xor_mul &= (1 << (self.cols % 8)) - 1
    #                 row_sum += popcount(xor_mul)
    #             result.data[r * actual_res_cols + (c >> 3)] |= (row_sum >= (self.cols >> 1)) << (c % 8)
    #     return result

    def matmul_no_bin(self, B: 'BinMatrix') -> ByteMatrix:
        return self.matmul_other_T_no_bin(B.transpose())

    def matmul_other_T_no_bin(self, B: 'BinMatrix') -> ByteMatrix:
        assert self.cols == B.cols, "Incompatible dimensions for matrix multiplication."
        result = ByteMatrix(self.rows, B.rows)
        actual_cols = (self.cols >> 3) + ((self.cols % 8) > 0)
        for r in range(self.rows):
            for c in range(B.rows):
                row_sum = 0
                for k in range(actual_cols):
                    xor_mul = self.data[r * actual_cols + k] ^ B.data[c * actual_cols + k]
                    row_sum += popcount(xor_mul)
                result.data[r * B.rows + c] = self.cols - (row_sum << 1)

        return result

    def __repr__(self):
        # Generate a string representation of the binary matrix
        matrix_str = []
        actual_cols = (self.cols >> 3) + ((self.cols % 8) > 0)
        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                bit = (self.data[r * actual_cols + (c >> 3)] >> (c % 8)) & 0x1
                row_str.append(str(bit))
            matrix_str.append(' '.join(row_str))
        return '\n'.join(matrix_str)
    

def bin2float(binary_matrix: BinMatrix) -> np.ndarray:
    """Convert a binary matrix to a float matrix."""
    rows, cols = binary_matrix.rows, binary_matrix.cols
    float_matrix = np.zeros((rows, cols), dtype=np.float32)
    actual_cols = (cols >> 3) + ((cols % 8) > 0)
    
    for r in range(rows):
        for c in range(cols):
            bit = (binary_matrix.data[r * actual_cols + (c >> 3)] >> (c % 8)) & 0x1
            float_matrix[r, c] = -1 if bit else 1
    
    return float_matrix

def byte2float(byte_matrix: ByteMatrix) -> np.ndarray:
    """Convert a byte matrix to a float matrix."""
    rows, cols = byte_matrix.rows, byte_matrix.cols
    float_matrix = np.zeros((rows, cols), dtype=np.float32)
    
    for r in range(rows):
        for c in range(cols):
            float_matrix[r, c] = byte_matrix.data[r * cols + c]
    
    return float_matrix

def float2bin(float_matrix: np.ndarray) -> BinMatrix:
    """Convert a float matrix to a binary matrix."""
    rows, cols = float_matrix.shape
    binary_matrix = BinMatrix(rows, cols)
    actual_cols = (cols >> 3) + ((cols % 8) > 0)
    
    for r in range(rows):
        for c in range(cols):
            binary_matrix.data[r * actual_cols + (c >> 3)] |= (float_matrix[r, c] < 0) << (c % 8)
    
    return binary_matrix
    
if __name__ == "__main__":

    # Matrix multiplication: y = x @ A
    # grad_A = x^T @ grad_y
    # grad_x = grad_y @ A^T
    # 

    x = BinMatrix(2, 16).randomize()
    A = BinMatrix(16, 16).randomize()
    # print("Matrix x:")
    # print(x)
    print("Matrix A:")
    print(A)
    y = x.matmul_no_bin(A)
    # print("Result of A * B:")
    # print(y)

    grad_y = 100 * np.random.randn(2, 16).astype(np.float32)

    grad_A = bin2float(x).T @ grad_y
    
    A = grad_update(A, grad_A, learning_rate=0.01)
    print("Updated Matrix A:")
    print(A)