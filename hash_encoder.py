import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dct import *
from testing_framework import TrajectoryEncoder
from data_stuff import ActionChunkDataset

import tqdm



# hash map is immutable, so we insert all chunks upon creation
class HashEncoder:
    def __init__(self, num_entries : int, quant_bins : int, xor_hash_primes : torch.Tensor, chunks : torch.Tensor):
        self.num_entries = num_entries
        self.xor_hash_primes = xor_hash_primes.detach().cpu().numpy()
        self.quant_bins = quant_bins

        self.mappings = [ [] for _ in range(self.num_entries) ]

        num_chunks = chunks.shape[0]
        np_chunks = chunks.detach().cpu().numpy()
        for i in tqdm.tqdm(range(num_chunks)):
            self.mappings[self.hash_chunk(np_chunks[i])].append(i)

    def hash_chunk(self, chunk : np.ndarray) -> int:
        dct_chunk = dct2(chunk)

        qdct_chunk = (dct_chunk * self.quant_bins // 2)

        qdct_chunk = qdct_chunk.astype(np.int32).clip(-self.quant_bins // 2, self.quant_bins // 2).flatten()

        used_primes = self.xor_hash_primes[:qdct_chunk.size]

        hash = np.bitwise_xor.reduce(qdct_chunk * used_primes, axis=0).item() % self.num_entries

        return hash

    """
    Returns indices of which action chunks may map to this chunk
    """
    def fetch(self, chunk : torch.Tensor) -> list[int]:
        np_chunk = chunk.detach().cpu().numpy()

        return self.mappings[self.hash_chunk(np_chunk)]


class DoubleHashEncoder(TrajectoryEncoder):
    def __init__(self, num_entries : int, quant_bins : int):
        self.num_entries = num_entries
        self.quant_bins = quant_bins

    def fit(self, acds : ActionChunkDataset, **kwargs):
        prime_sets = torch.tensor(
            [
                [268817, 739099, 244553, 718433, 569603, 698171, 578467, 737351, 542093, 640583, 212587, 341743, 292561, 821383, 504289, 808837, 991511, 649921, 960989, 654821, 370813, 732181, 424117, 74161, 272269, 504001, 509203, 610879, 653801, 526397, 807281, 804901, 358747, 583267, 617237, 256163, 96013, 123887, 652063, 221497, 128339, 457829, 987739, 864541, 203789, 428149, 500231, 955193, 483127, 733751, 951997, 941599, 572939, 97919, 875209, 142837, 605609, 87877, 863921, 254209, 982697, 116167, 243781, 631751, 656597, 238163, 720997, 70921, 625621, 718759, 938843, 76123, 173431, 410119, 715171, 941429, 385397, 131251, 162889, 201167, 842813, 520241, 586919, 534637, 741193, 721577, 267373, 377789, 618439, 443533, 334289, 958679, 159193, 252481, 891379, 459037, 958043, 630701, 230849, 143719, 711427, 465119, 145193, 206447, 117499, 225353, 315109, 431449, 324179, 421517, 169111, 706507, 709957, 110291, 211727, 214729, 282493, 687541, 304739, 638467, 347143, 682439, 493333, 741809, 151523, 904193, 928559, 741991, 465701, 665527, 584377, 528811, 864307, 961151, 74611, 662953, 345643, 909683, 988417, 928471],
                [588877, 316423, 63361, 831643, 99439, 649073, 725381, 106867, 336521, 311749, 513899, 578021, 251257, 86017, 809779, 763039, 455573, 709921, 411337, 77137, 418391, 910781, 515227, 627433, 460393, 760847, 756011, 813199, 213253, 585643, 126461, 117877, 958963, 588121, 450479, 729907, 142237, 69457, 591791, 117701, 399181, 444557, 267217, 300851, 77617, 116981, 700537, 688679, 114689, 526573, 789377, 378509, 952507, 751997, 328381, 274403, 502133, 897473, 935339, 123923, 497851, 423307, 626929, 455393, 414949, 756919, 366167, 177473, 756271, 847967, 384017, 536867, 821851, 318349, 974147, 755861, 875717, 404099, 412073, 535387, 550757, 524893, 592939, 771019, 559211, 737537, 825883, 520291, 518113, 396527, 352333, 870161, 689093, 349303, 269791, 873139, 534671, 458639, 374483, 222791, 418189, 973591, 162059, 994337, 703471, 84457, 353783, 245963, 690407, 344821, 664691, 804847, 760843, 372709, 243889, 266411, 291209, 406969, 827857, 345089, 94771, 393901, 710051, 582601, 408631, 76597, 331739, 221317, 175411, 360439, 969407, 302663, 986191, 737053, 701179, 413629, 754711, 939413, 296819, 86539]
            ]
        )

        self.chunks = acds.train_split()

        self.henc = [
            HashEncoder(num_entries=self.num_entries, quant_bins=self.quant_bins, xor_hash_primes=prime_sets[0], chunks=self.chunks),
            HashEncoder(num_entries=self.num_entries, quant_bins=self.quant_bins, xor_hash_primes=prime_sets[1], chunks=self.chunks),
        ]

    def encode(self, trajectories : torch.Tensor) -> int:
        indices = []
        for i in tqdm.tqdm(range(trajectories.shape[0])):
            traj = trajectories[i]

            candidates = sorted(self.henc[0].fetch(traj) + self.henc[1].fetch(traj))

            selected_index = -1
            for i in range(1, len(candidates)):
                if candidates[i] == candidates[i - 1]:
                    selected_index = candidates[i]
                    break

            if selected_index == -1 and len(candidates) > 0:
                selected_index = candidates[0]

            indices.append(selected_index)

        return indices

    def decode(self, indices : list[int]) -> torch.Tensor:
        return torch.stack(
            [
                self.chunks[idx] if idx != -1 else torch.zeros_like(self.chunks[0]) 
                for idx in indices
            ]
        )