import h5py
import matplotlib.pyplot as plt
import numpy as np

chunk_size = 20

files = ["low_dim_v141(4).hdf5", "low_dim_v141(3).hdf5", "low_dim_v141(2).hdf5", "low_dim_v141(1).hdf5", "low_dim_v141.hdf5"]

real_robot_path = "low_dim_v141(4).hdf5"

num_demos = 0

all_demos = []

for real_robot_path in files:
    with h5py.File(real_robot_path, 'r') as f:
        print(f"demo: {f['data'].keys()}")
        print(f"demo keys {f['data/demo_1'].keys()}")
        print(f"demo keys {f['data/demo_1/obs'].keys()}")

        for demo in f['data']:
            num_demos += 1

            demo_pos = f['data'][demo]['obs']['robot0_eef_pos'][:]
            demo_rot = f['data'][demo]['obs']['robot0_eef_quat'][:]

            demo_eef_actions = np.concatenate((demo_pos, demo_rot), axis=1)


            round_up_size = chunk_size * ((demo_eef_actions.shape[0] - 1) // chunk_size + 1) - demo_eef_actions.shape[0]

            all_demos.append(demo_eef_actions)
            all_demos.append(np.zeros((round_up_size, demo_eef_actions.shape[1])))

eef_actions = np.concatenate(all_demos, axis=0)

# linearly remap [1st percentile, 99th percentile] to [-1.0, 1.0] as described in the paper
# this is important for the scaling and rounding step
def fast_normalization(array):
    sorted_array = np.sort(array, axis=1)

    idx_1st = int(sorted_array.shape[0] * 0.01)
    idx_99th = int(sorted_array.shape[0] * 0.99)

    lower = sorted_array[idx_1st]
    upper = sorted_array[idx_99th]

    remapped = (array - lower) / (upper - lower)

    return remapped

# divide input into chunk_size continous chunks
# for the last chunk we zero pad because
# 1) it's easier to do it this way
# 2) our number of output actions must still be a multiple of the chunk size, so we cannot have unevenly sized chunks
def dct_all_chunks(array):
    if (array.shape[0] % chunk_size != 0):
        raise RuntimeError("Bruh")

    num_chunks = array.shape[0] // chunk_size
    chunks = np.reshape(array, (num_chunks, chunk_size, array.shape[1]))
    
    dct_int = np.arange(0, chunk_size, step=1)
    dct_x, dct_y = np.meshgrid(dct_int, dct_int)
    dct_matrix = np.cos(np.pi / chunk_size * (dct_x + 0.5) * dct_y)

    dct_coeffs = np.matmul(dct_matrix, chunks)

    return dct_coeffs

print(f"Num demos is {num_demos}")

"""
We have n actions in each DCT input. We run a loseless DCT transform, so we have n coefficients for each dimension. 
Eventually, our DCT coefficients are a n x 7 matrix. If we believe that our data is sparse, then we can reduce the number
of rows. Thus, we must perform PCA on the transpose matrix, 7 x n, and reduce to to 7 x m. Chaojian suggested n=20 and m=10.

Some random rambling to make sure I understand the correct way to do this:
It's quite unusual that we are "mixing" features like this (normally you would keep them independent in seperate columns).
If we believe there is sparsity, then that would mean if we plot each 7-d vector in a grid, we would find that they tend to cluster
around a few points. Thus, we should ideally be doing PCA on the 7-d vector. But 7-d is small already. The issue is, we have many
different action chunk coefficients and we want to encourage sparsity. 

If we look back at the original inspiration for the FAST paper, we want each action chunk to be as different as possible from the others
in order to force the VLA to create as new information as possible with each token. What if we flatten the coefficients matrix for each
action chunk and then do PCA? That seems like something that would make a lot more sense. 
"""


eef_actions = fast_normalization(eef_actions)
dct = dct_all_chunks(eef_actions)

# hmmm, it's tempting to do a DCT of a DCT here
dct = np.reshape(dct, (dct.shape[0], -1))
dct = dct[:, :28]

print(f"Flattened DCT shape is {dct.shape}")

PCA_coeff, s, Vt = np.linalg.svd(dct, full_matrices=False)
s = np.diag(s)

pca_dim = 10

dct_reconstruct = np.matmul(PCA_coeff[:, :pca_dim], np.matmul(s[:pca_dim, :pca_dim], Vt[:pca_dim, :]))

dct_err = dct_reconstruct - dct
dct_err_fro = np.linalg.norm(dct_err, ord="fro")
print(f"Fro norm of reconstruction: {dct_err_fro}")

channel_wise_err = np.matmul(dct_err.T, dct_err) / dct.shape[0]
channel_wise_err = np.diag(channel_wise_err)
print(f"Channelwise average L2 error  {channel_wise_err}")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

if False:
    axis_size = 999.0
    ax.set_xlim([-axis_size, axis_size])
    ax.set_ylim([-axis_size, axis_size])
    ax.set_zlim([-axis_size, axis_size])
ax.scatter(xs=PCA_coeff[:, 0], ys=PCA_coeff[:, 1], zs=PCA_coeff[:, 2], s=0.1, marker='o')

plt.show()