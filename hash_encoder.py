import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dct import *
from data_stuff import ActionChunkDataset

import tqdm

hash_primes = torch.tensor(
    [
        [268817, 739099, 244553, 718433, 569603, 698171, 578467, 737351, 542093, 640583, 212587, 341743, 292561, 821383, 504289, 808837, 991511, 649921, 960989, 654821, 370813, 732181, 424117, 74161, 272269, 504001, 509203, 610879, 653801, 526397, 807281, 804901, 358747, 583267, 617237, 256163, 96013, 123887, 652063, 221497, 128339, 457829, 987739, 864541, 203789, 428149, 500231, 955193, 483127, 733751, 951997, 941599, 572939, 97919, 875209, 142837, 605609, 87877, 863921, 254209, 982697, 116167, 243781, 631751, 656597, 238163, 720997, 70921, 625621, 718759, 938843, 76123, 173431, 410119, 715171, 941429, 385397, 131251, 162889, 201167, 842813, 520241, 586919, 534637, 741193, 721577, 267373, 377789, 618439, 443533, 334289, 958679, 159193, 252481, 891379, 459037, 958043, 630701, 230849, 143719, 711427, 465119, 145193, 206447, 117499, 225353, 315109, 431449, 324179, 421517, 169111, 706507, 709957, 110291, 211727, 214729, 282493, 687541, 304739, 638467, 347143, 682439, 493333, 741809, 151523, 904193, 928559, 741991, 465701, 665527, 584377, 528811, 864307, 961151, 74611, 662953, 345643, 909683, 988417, 928471],
        [588877, 316423, 63361, 831643, 99439, 649073, 725381, 106867, 336521, 311749, 513899, 578021, 251257, 86017, 809779, 763039, 455573, 709921, 411337, 77137, 418391, 910781, 515227, 627433, 460393, 760847, 756011, 813199, 213253, 585643, 126461, 117877, 958963, 588121, 450479, 729907, 142237, 69457, 591791, 117701, 399181, 444557, 267217, 300851, 77617, 116981, 700537, 688679, 114689, 526573, 789377, 378509, 952507, 751997, 328381, 274403, 502133, 897473, 935339, 123923, 497851, 423307, 626929, 455393, 414949, 756919, 366167, 177473, 756271, 847967, 384017, 536867, 821851, 318349, 974147, 755861, 875717, 404099, 412073, 535387, 550757, 524893, 592939, 771019, 559211, 737537, 825883, 520291, 518113, 396527, 352333, 870161, 689093, 349303, 269791, 873139, 534671, 458639, 374483, 222791, 418189, 973591, 162059, 994337, 703471, 84457, 353783, 245963, 690407, 344821, 664691, 804847, 760843, 372709, 243889, 266411, 291209, 406969, 827857, 345089, 94771, 393901, 710051, 582601, 408631, 76597, 331739, 221317, 175411, 360439, 969407, 302663, 986191, 737053, 701179, 413629, 754711, 939413, 296819, 86539]
    ]
)

# hash map is immutable, so we insert all chunks upon creation
class HashEncoder:
    def __init__(self, num_entries : int, qbins : int, primes : torch.Tensor, chunks : torch.Tensor):
        self.num_entries = num_entries
        self.primes = primes.detach().cpu().numpy()
        self.qbins = qbins

        self.mappings = [ [] for _ in range(self.num_entries) ]

        num_chunks = chunks.shape[0]
        np_chunks = chunks.detach().cpu().numpy()
        for i in tqdm.tqdm(range(num_chunks)):
            self.mappings[self.hash_chunk(np_chunks[i])].append(i)

    def hash_chunk(self, chunk : np.ndarray) -> int:
        # utilize FSQ-like quanitzation
        dct_chunk = dct2(chunk)
        dct_chunk = np.tanh(dct_chunk)
        qdct_chunk = (dct_chunk * self.qbins // 2)

        # round and clip
        qdct_chunk = (qdct_chunk + 0.5).astype(np.int32)
        qdct_chunk = qdct_chunk.clip(-self.qbins // 2, self.qbins // 2)

        # use lower frequency components only 
        qdct_chunk = qdct_chunk[:3]

        hash = np.bitwise_xor.reduce(qdct_chunk.flatten() * self.primes[:qdct_chunk.size], axis=0).item() % self.num_entries

        return hash

    """
    Returns indices of which action chunks may map to this chunk
    """
    def fetch(self, chunk : torch.Tensor) -> list[int]:
        np_chunk = chunk.detach().cpu().numpy()

        return self.mappings[self.hash_chunk(np_chunk)]


# hash map is immutable, so we insert all chunks upon creation
class MultiFrequencyHashEncoder:
    def __init__(
            self, 
            time_horizon : int,
            # low and high frequency num entries
            lfne : int, 
            hfne : int,
            # low and high requency quantization bins
            lfqb : int,
            hfqb : int, 
            # channel divider
            chnl_div : int,
            primes : torch.Tensor, 
            chunks : torch.Tensor
        ):
        
        self.chnl_div = chnl_div
        self.time_horizon = time_horizon
        self.sep_dct = True

        self.lfne = lfne
        self.hfne = hfne

        self.lfqb = lfqb
        self.hfqb = hfqb
        self.qbins = np.concatenate(
            (
                np.full((self.chnl_div, 1), fill_value=self.lfqb),
                np.full((self.time_horizon - self.chnl_div, 1), fill_value=self.hfqb),
            ),
            axis=0
        )

        self.rqbins = np.right_shift(self.qbins, 1)

        self.lfm = [ None for _ in range(self.lfne)]
        self.hfm = [ None for _ in range(self.hfne)]


        self.primes = primes.detach().cpu().numpy()

        num_chunks = chunks.shape[0]
        np_chunks = chunks.detach().cpu().numpy()

        for i in tqdm.tqdm(range(num_chunks)):
            lfi, hfi, lfdct, hfdct = self.hash_chunk(np_chunks[i])

            self.lfm[lfi] = lfdct
            self.hfm[hfi] = hfdct

    def hash_chunk(self, chunk : np.ndarray) -> int:
        # utilize FSQ-like quanitzation
        if self.sep_dct:
            coeffs = dct2(chunk[1:, :])
            dct_chunk = np.concatenate((chunk[:1, :], coeffs), axis=0)

            qdct_in = dct_chunk
        else:
            dct_chunk = dct2(chunk)

            qdct_in = np.tanh(dct_chunk)
        
        qdct_chunk = (qdct_in * self.rqbins)

        # round and clip
        qdct_chunk = (qdct_chunk + np.where(qdct_chunk < 0, -0.5, 0.5)).astype(np.int32)
        qdct_chunk = qdct_chunk.clip(-self.rqbins, self.rqbins)

        # use lower frequency components only 
        lfdct = qdct_chunk[:self.chnl_div]
        hfdct = qdct_chunk[self.chnl_div:]

        lfi = np.bitwise_xor.reduce(lfdct.flatten() * self.primes[:lfdct.size], axis=0).item() % self.lfne
        hfi = np.bitwise_xor.reduce(hfdct.flatten() * self.primes[lfdct.size:], axis=0).item() % self.hfne

        # we save the non-quanitized DCT chunks in our mappings
        return lfi, hfi, dct_chunk[:self.chnl_div], dct_chunk[self.chnl_div:]

    """
    Returns indices of which action chunks may map to this chunk
    """
    def fetch(self, chunk : torch.Tensor) -> list[int]:
        np_chunk = chunk.detach().cpu().numpy()

        lfi, hfi, a, b = self.hash_chunk(np_chunk)

        lfc = self.lfm[lfi] if True else a
        hfc = self.hfm[hfi] if False else b

        if self.sep_dct:
            resid = dct2_inv(hfc)

            traj = np.concatenate((lfc, resid), axis=0)

            traj = np.cumsum(traj, axis=0)
        else:
            #dct = np.concatenate((self.lfm[lfi], self.hfm[hfi]), axis=0)
            dct = np.concatenate((lfc, np.zeros_like(hfc)), axis=0)

            traj = dct2_inv(dct)

        return torch.from_numpy(traj)



acds = ActionChunkDataset()

resid = acds.residuals()

# disable sci mode and print 6 digits
np.set_printoptions(precision=6, suppress=True)

train_data = acds.train_split()
test_data = acds.test_split()

# comment to change between single hashing and multi hashing
if True:
    print(f"Using multifreq hash encoding")

    mfhe = MultiFrequencyHashEncoder(
        time_horizon=20,
        lfne=4096,
        hfne=8192,
        lfqb=5,
        hfqb=32,
        chnl_div=1,
        primes=hash_primes[0],
        chunks=resid
    )

    reconstructed = torch.stack([
        mfhe.fetch(resid[i]) for i in tqdm.tqdm(range(resid.shape[0]))
    ])


    """
    print(f"Using single hash encoding")
    he = HashEncoder(num_entries=4096, qbins=16, primes=hash_primes[0], chunks=acds.all_chunks)

    reconstructed = torch.stack([
        acds.all_chunks[he.fetch(acds.all_chunks[i])[0]] for i in range(acds.all_chunks.shape[0])
    ])
    """
else:
    print(f"Using multi hash encoding")

    mhe0 = HashEncoder(num_entries=4096, qbins=16, primes=hash_primes[0], chunks=train_data)
    mhe1 = HashEncoder(num_entries=4096, qbins=16, primes=hash_primes[1], chunks=train_data)

    def multi_fetch(chunk : torch.Tensor) -> torch.Tensor:
        candidates = sorted(mhe0.fetch(chunk) + mhe1.fetch(chunk))

        for i in range(1, len(candidates)):
            if candidates[i] == candidates[i - 1]:
                return train_data[candidates[i]]
            
        print("UNEXPECTED: NO SUITABLE CANDIDATE FOUND!")

        # at least return something from the hashmap
        # if there's nothing then return zero chunk
        return train_data[candidates[0]] if len(candidates) > 0 else torch.zeros_like(chunk)


    reconstructed = torch.stack([
        multi_fetch(test_data[i]) for i in tqdm.tqdm(range(test_data.shape[0]))
    ])
    
print(f"Reconstructed MSE loss was {F.mse_loss(reconstructed, test_data)}")