import dct
import h5py

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
chunk_size = 20
quantization_bins = 16
hashmap_size = 512
files = ["low_dim_v141(4).hdf5", "low_dim_v141(3).hdf5", "low_dim_v141(2).hdf5", "low_dim_v141(1).hdf5", "low_dim_v141.hdf5"]

hash_primes0 = np.array([
    268817, 739099, 244553, 718433, 569603, 698171, 578467, 737351, 542093, 640583, 212587, 341743, 292561, 821383, 504289, 808837, 991511, 649921, 960989, 654821, 370813, 732181, 424117, 74161, 272269, 504001, 509203, 610879, 653801, 526397, 807281, 804901, 358747, 583267, 617237, 256163, 96013, 123887, 652063, 221497, 128339, 457829, 987739, 864541, 203789, 428149, 500231, 955193, 483127, 733751, 951997, 941599, 572939, 97919, 875209, 142837, 605609, 87877, 863921, 254209, 982697, 116167, 243781, 631751, 656597, 238163, 720997, 70921, 625621, 718759, 938843, 76123, 173431, 410119, 715171, 941429, 385397, 131251, 162889, 201167, 842813, 520241, 586919, 534637, 741193, 721577, 267373, 377789, 618439, 443533, 334289, 958679, 159193, 252481, 891379, 459037, 958043, 630701, 230849, 143719, 711427, 465119, 145193, 206447, 117499, 225353, 315109, 431449, 324179, 421517, 169111, 706507, 709957, 110291, 211727, 214729, 282493, 687541, 304739, 638467, 347143, 682439, 493333, 741809, 151523, 904193, 928559, 741991, 465701, 665527, 584377, 528811, 864307, 961151, 74611, 662953, 345643, 909683, 988417, 928471
])

hash_primes1 = np.array([
    588877, 316423, 63361, 831643, 99439, 649073, 725381, 106867, 336521, 311749, 513899, 578021, 251257, 86017, 809779, 763039, 455573, 709921, 411337, 77137, 418391, 910781, 515227, 627433, 460393, 760847, 756011, 813199, 213253, 585643, 126461, 117877, 958963, 588121, 450479, 729907, 142237, 69457, 591791, 117701, 399181, 444557, 267217, 300851, 77617, 116981, 700537, 688679, 114689, 526573, 789377, 378509, 952507, 751997, 328381, 274403, 502133, 897473, 935339, 123923, 497851, 423307, 626929, 455393, 414949, 756919, 366167, 177473, 756271, 847967, 384017, 536867, 821851, 318349, 974147, 755861, 875717, 404099, 412073, 535387, 550757, 524893, 592939, 771019, 559211, 737537, 825883, 520291, 518113, 396527, 352333, 870161, 689093, 349303, 269791, 873139, 534671, 458639, 374483, 222791, 418189, 973591, 162059, 994337, 703471, 84457, 353783, 245963, 690407, 344821, 664691, 804847, 760843, 372709, 243889, 266411, 291209, 406969, 827857, 345089, 94771, 393901, 710051, 582601, 408631, 76597, 331739, 221317, 175411, 360439, 969407, 302663, 986191, 737053, 701179, 413629, 754711, 939413, 296819, 86539
])


num_demos = 0

all_demos = []

class MotionLibraryHashmap:
    def __init__(self, primes):
        self.primes = primes
        self.hashmap = [ [] for i in range(hashmap_size) ]

    # returns hash indices
    def insert(self, eef_dct_quant):
        eef_dct_hash = self.calc_hash(eef_dct_quant)

        for i in range(eef_dct_quant.shape[0]):
            self.hashmap[eef_dct_hash[i]].append(i)

        return eef_dct_hash
    
    def fetch(self, eef_dct_hash):
        entries = [self.hashmap[eef_dct_hash[i]] for i in range(eef_dct_hash.shape[0])]
        return entries

    def calc_hash(self, eef_dct_quant):
        eef_dct_hash = np.bitwise_xor.reduce(eef_dct_quant * self.primes, axis=1) % hashmap_size
        eef_dct_hash = np.reshape(eef_dct_hash, (-1))
        return eef_dct_hash

mlhm0 = MotionLibraryHashmap(hash_primes0)
mlhm1 = MotionLibraryHashmap(hash_primes1)

for real_robot_path in files:
    with h5py.File(real_robot_path, 'r') as f:

        """
        Example of how to dump stuff we can print:

            print(f"demo: {f['data'].keys()}")
            print(f"demo keys {f['data/demo_1'].keys()}")
            print(f"demo keys {f['data/demo_1/obs'].keys()}")

        Robomimic stores stuff in 7-d format
        """


        for demo in f['data']:
            num_demos += 1

            demo_pos = f['data'][demo]['obs']['robot0_eef_pos'][:]
            demo_rot = f['data'][demo]['obs']['robot0_eef_quat'][:]

            demo_eef_actions = np.concatenate((demo_pos, demo_rot), axis=1)

            total_num_chunks = ((demo_eef_actions.shape[0] - 1) // chunk_size + 1)
            round_up_size = chunk_size * total_num_chunks - demo_eef_actions.shape[0]

            all_demos.append(demo_eef_actions)
            all_demos.append(np.zeros((round_up_size, demo_eef_actions.shape[1])))

eef_raw_actions = np.concatenate(all_demos, axis=0)

# FAST normalization
sorted_coeffs = np.sort(eef_raw_actions, axis=1)
lower = sorted_coeffs[int(sorted_coeffs.shape[0] * 0.01)]
upper = sorted_coeffs[int(sorted_coeffs.shape[0] * 0.99)]
quantile_range = upper - lower
eef_actions = (eef_raw_actions - lower) / quantile_range

# apply DCT2 transform
num_chunks = eef_actions.shape[0] // chunk_size
eef_actions = np.reshape(eef_actions, (num_chunks, chunk_size, 7))
print(f"Input EEF is {eef_actions.shape}")
eef_dct = dct.dct2(eef_actions)
eef_dct_flattened = np.reshape(eef_dct, (num_chunks, -1))

# quantize and hash
eef_dct_quant = (quantization_bins * eef_dct_flattened + 0.5).astype(np.int32)
hash_entries0 = mlhm0.fetch(mlhm0.insert(eef_dct_quant))
hash_entries1 = mlhm1.fetch(mlhm1.insert(eef_dct_quant))

combined_hash_entries = [hash_entries0[i] + hash_entries1[i] for i in range(eef_actions.shape[0])]


trunc_hash_entries = [stats.mode(comb_hash_en)[0] for comb_hash_en in combined_hash_entries]
#trunc_hash_entries = [comb_hash_en[0] for comb_hash_en in combined_hash_entries]

# fetch from hashmap and reconstruct
fetched_chunks = eef_dct[trunc_hash_entries]
fetched_chunks = np.stack(fetched_chunks)
reconstructed_chunks = dct.dct2_inv(fetched_chunks)

# calculate error
err = reconstructed_chunks - eef_actions
mse = np.mean(err ** 2) / err.size
print(f"MSE was {mse}")
