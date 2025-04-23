import numpy as np

import torch
import math

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

def hash_dct(chunk : np.ndarray, xor_primes : np.ndarray | torch.Tensor, quant_bins : int, num_entries : int):
    if type(xor_primes) is torch.Tensor:
        xor_primes = xor_primes.detach().cpu().numpy()

    dct_chunk = dct2(chunk)

    qdct_chunk = (dct_chunk * quant_bins // 2)

    qdct_chunk = qdct_chunk.astype(np.int32).clip(-quant_bins // 2, quant_bins // 2).flatten()

    hash = np.bitwise_xor.reduce(qdct_chunk * xor_primes, axis=0).item() % num_entries

    return hash

"""
Does NOT use xor hashing!
"""
def hash_dct_torch_batch(chunks : torch.Tensor, weights : torch.Tensor, num_entries : int):
    chunk_size = chunks.shape[1]

    loop_indices = torch.arange(0, chunk_size, step=1)
    dct_k, dct_n = torch.meshgrid(loop_indices, loop_indices, indexing="ij")
    dct_matrix = torch.cos(torch.pi / chunk_size * (dct_n + 0.5) * dct_k).unsqueeze(0).to("cuda")

    # (1, 20, 20) x (N, 20, 7) = (N, 20, 7)
    dct_coeffs = torch.matmul(dct_matrix, chunks)

    weights = weights.unsqueeze(0).unsqueeze(2)

    # (20) x (N, 20, 7) = (N, 20, 7)
    modulated = (weights * dct_coeffs).flatten(1)

    hash = torch.remainder(modulated.sum(1).int(), num_entries)

    return hash.unsqueeze(1)

prime_sets = torch.tensor(
    [
        [268817, 739099, 244553, 718433, 569603, 698171, 578467, 737351, 542093, 640583, 212587, 341743, 292561, 821383, 504289, 808837, 991511, 649921, 960989, 654821, 370813, 732181, 424117, 74161, 272269, 504001, 509203, 610879, 653801, 526397, 807281, 804901, 358747, 583267, 617237, 256163, 96013, 123887, 652063, 221497, 128339, 457829, 987739, 864541, 203789, 428149, 500231, 955193, 483127, 733751, 951997, 941599, 572939, 97919, 875209, 142837, 605609, 87877, 863921, 254209, 982697, 116167, 243781, 631751, 656597, 238163, 720997, 70921, 625621, 718759, 938843, 76123, 173431, 410119, 715171, 941429, 385397, 131251, 162889, 201167, 842813, 520241, 586919, 534637, 741193, 721577, 267373, 377789, 618439, 443533, 334289, 958679, 159193, 252481, 891379, 459037, 958043, 630701, 230849, 143719, 711427, 465119, 145193, 206447, 117499, 225353, 315109, 431449, 324179, 421517, 169111, 706507, 709957, 110291, 211727, 214729, 282493, 687541, 304739, 638467, 347143, 682439, 493333, 741809, 151523, 904193, 928559, 741991, 465701, 665527, 584377, 528811, 864307, 961151, 74611, 662953, 345643, 909683, 988417, 928471],
        [588877, 316423, 63361, 831643, 99439, 649073, 725381, 106867, 336521, 311749, 513899, 578021, 251257, 86017, 809779, 763039, 455573, 709921, 411337, 77137, 418391, 910781, 515227, 627433, 460393, 760847, 756011, 813199, 213253, 585643, 126461, 117877, 958963, 588121, 450479, 729907, 142237, 69457, 591791, 117701, 399181, 444557, 267217, 300851, 77617, 116981, 700537, 688679, 114689, 526573, 789377, 378509, 952507, 751997, 328381, 274403, 502133, 897473, 935339, 123923, 497851, 423307, 626929, 455393, 414949, 756919, 366167, 177473, 756271, 847967, 384017, 536867, 821851, 318349, 974147, 755861, 875717, 404099, 412073, 535387, 550757, 524893, 592939, 771019, 559211, 737537, 825883, 520291, 518113, 396527, 352333, 870161, 689093, 349303, 269791, 873139, 534671, 458639, 374483, 222791, 418189, 973591, 162059, 994337, 703471, 84457, 353783, 245963, 690407, 344821, 664691, 804847, 760843, 372709, 243889, 266411, 291209, 406969, 827857, 345089, 94771, 393901, 710051, 582601, 408631, 76597, 331739, 221317, 175411, 360439, 969407, 302663, 986191, 737053, 701179, 413629, 754711, 939413, 296819, 86539]
    ]
)

fast_hash_weights = torch.Tensor(
    [5 * math.exp(-2 * i) for i in range(20)]
).to("cuda")

if __name__ == "__main__":
    sequence = np.random.random((13112, 20, 7))
    #sequence = np.reshape(sequence, (1, sequence.shape[1], 1))

    reconstructed = dct2_inv(dct2(sequence))

    err = sequence - reconstructed
    print(sequence)
    print(reconstructed)
    print(err)

    print(f"Err is {np.sum(err)}")