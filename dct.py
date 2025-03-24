import numpy as np

def dct2(chunks : np.ndarray) -> np.ndarray:
    chunk_size = chunks.shape[0]
    
    loop_indices = np.arange(0, chunk_size, step=1)
    dct_k, dct_n = np.meshgrid(loop_indices, loop_indices, indexing="ij")
    dct_matrix = np.cos(np.pi / chunk_size * (dct_n + 0.5) * dct_k)

    dct_coeffs = np.matmul(dct_matrix, chunks)

    return dct_coeffs

# DCT 3 scaled
def dct2_inv(chunks : np.ndarray) -> np.ndarray:
    chunk_size = chunks.shape[0]

    dct_n = np.arange(1, chunk_size, step=1)
    dct_k = np.arange(0, chunk_size, step=1)

    dct_k, dct_n = np.meshgrid(dct_k, dct_n, indexing="ij")

    dct_matrix = np.cos(np.pi / chunk_size * (dct_k + 0.5) * dct_n)

    bias = chunks[:1, :]
    coeffs = chunks[1:, :]

    values = np.matmul(dct_matrix, coeffs) + 0.5 * bias
    values = values * 2 / chunk_size

    return values

if __name__ == "__main__":
    sequence = np.random.random((13112, 20, 7))
    #sequence = np.reshape(sequence, (1, sequence.shape[1], 1))

    reconstructed = dct2_inv(dct2(sequence))

    err = sequence - reconstructed
    print(sequence)
    print(reconstructed)
    print(err)

    print(f"Err is {np.sum(err)}")