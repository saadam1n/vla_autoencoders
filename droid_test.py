import tqdm
import torch
from torch.utils.data import DataLoader
import tensorflow as tf
from typing import Any, Dict
from robomimic.utils.rlds_utils import droid_dataset_transform, robomimic_transform, TorchRLDSDataset

from octo.data.dataset import make_dataset_from_rlds, make_interleaved_dataset
from octo.data.utils.data_utils import combine_dataset_statistics
from octo.utils.spec import ModuleSpec

tf.config.set_visible_devices([], "GPU")

# ------------------------------ Get Dataset Information ------------------------------
DATA_PATH = "/data/droid/d100/"              # UPDATE WITH PATH TO RLDS DATASETS
DATASET_NAMES = ["droid_100"]   # You can add additional co-training datasets here
sample_weights = [1]        # Add to this if you add additional co-training datasets

# ------------------------------ Construct Dataset ------------------------------
BASE_DATASET_KWARGS = {
    "data_dir": DATA_PATH,
    "image_obs_keys": {"primary": "exterior_image_1_left", "secondary": "exterior_image_2_left"},
    "state_obs_keys": ["cartesian_position", "gripper_position"],
    "language_key": "language_instruction",
    "norm_skip_keys":  ["proprio"],
    "action_proprio_normalization_type": "bounds",
    "absolute_action_mask": [False] * 7,                    # we don't want abs actions
    "action_normalization_mask": [False] * 6 + [False],      # don't normalize anything
    "standardize_fn": None,
}

# By default, only use success trajectories in DROID
filter_functions = [
    [
        ModuleSpec.create(
            "robomimic.utils.rlds_utils:filter_success"
        )
    ] if d_name == "droid" else [] for d_name in DATASET_NAMES
]

dataset_kwargs_list = [
    {
        "name": d_name,
        "filter_functions": f_functions,
        **BASE_DATASET_KWARGS
    }
    for d_name, f_functions in zip(DATASET_NAMES, filter_functions)
]

# Compute combined normalization stats. Note: can also set this to None to normalize each dataset separately
combined_dataset_statistics = combine_dataset_statistics(
    [make_dataset_from_rlds(**dataset_kwargs, train=True)[1] for dataset_kwargs in dataset_kwargs_list]
)

dataset = make_interleaved_dataset(
    dataset_kwargs_list,
    sample_weights,
    train=True,
    shuffle_buffer_size=100000,         # adjust this based on your system RAM
    batch_size=None,                    # batching will be handled in PyTorch Dataloader object
    balance_weights=False,
    dataset_statistics=combined_dataset_statistics,
    traj_transform_kwargs=dict(
        window_size=1,
        future_action_window_size=19,
        subsample_length=100,
        skip_unlabeled=True,            # skip all trajectories without language annotation
    ),
    frame_transform_kwargs=dict(
        image_augment_kwargs=dict(
        ),
        resize_size=dict(
            primary=[128, 128],
            secondary=[128, 128],
        ),
        num_parallel_calls=128,
    ),
    traj_transform_threads=128,
    traj_read_threads=128,
)


def traj_only_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "actions": trajectory["action"],
    }

dataset = dataset.map(traj_only_transform, num_parallel_calls=128)


# ------------------------------ Create Dataloader ------------------------------

pytorch_dataset = TorchRLDSDataset(dataset)
train_loader = DataLoader(
    pytorch_dataset,
    batch_size=128,
    num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
)


boundry = "=" * 32
print(f"\n{boundry} END OF DATASET INIT {boundry}\n")

next_traj = 0

all_traj = None

for i, sample in tqdm.tqdm(enumerate(train_loader)):
    print(f"Processing batch {i}")

    traj = sample["actions"]
    
    all_traj = traj if all_traj is None else torch.cat((all_traj, traj), dim=0)
    print(f"All traj size is now {all_traj.shape} and is taking up {all_traj.numel()} elements (dtype is {all_traj.dtype})")

    if i == 1000: 
        break

print(f"Saving all traj...")
torch.save(all_traj, "data/traj_128k.pt")