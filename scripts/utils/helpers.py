## { MODULE

##
## === DEPENDENCIES
##

import argparse
from pathlib import Path

##
## === FUNCTIONS
##


def get_axs_grid(axs, num_rows, num_cols):
    ## normalise axis for 2D indexing [row, col]
    if (num_rows == 1) and (num_cols == 1):
        axs_grid = [[axs]]
    elif num_rows == 1:
        axs_grid = [list(axs)]
    elif num_cols == 1:
        axs_grid = [[ax] for ax in axs]
    else:
        axs_grid = axs
    return axs_grid


def get_user_input():
    parser = argparse.ArgumentParser(description="Test Quokka simulation(s).")
    parser.add_argument(
        "--dir",
        "-d",
        type=lambda path: Path(path).expanduser().resolve(),
        default=None,
        help="Optional path to a Quokka simulation or dataset directory.",
    )
    parser.add_argument(
        "--components",
        "-c",
        nargs="+",
        default=None,
        help="Optional list of components to plot.",
    )
    parser.add_argument(
        "--axes",
        "-a",
        nargs="+",
        default=None,
        help="Optional list of axes to plot.",
    )
    parser.add_argument(
        "--fields",
        "-f",
        nargs="+",
        default=None,
        help="Optional list of axes to plot.",
    )
    args = parser.parse_args()
    return args


def get_latest_dataset_dirs(
    sim_dir: Path,
) -> list[Path]:
    dataset_dirs = [
        sub_dir for sub_dir in sim_dir.iterdir()
        if sub_dir.is_dir() and ("plt" in sub_dir.name) and ("old" not in sub_dir.name)
    ]
    dataset_dirs.sort(key=lambda dataset_dir: int(dataset_dir.name.split("plt")[1]))  # sort by plt-index
    return dataset_dirs


def resolve_dataset_dirs(
    input_dir: Path,
) -> list[Path]:
    if "plt" in input_dir.name:
        dataset_dirs = [input_dir]
        return dataset_dirs
    dataset_dirs = get_latest_dataset_dirs(sim_dir=input_dir)
    assert len(dataset_dirs) != 0
    return dataset_dirs


def subsample_dirs(
    dataset_dirs,
    target_max: int = 10,
):
    num_dirs = len(dataset_dirs)
    if num_dirs <= target_max:
        return dataset_dirs
    stride_ratio = (num_dirs - 1) // (target_max - 1)
    indices_to_keep = [round(_index * stride_ratio) for _index in range(target_max)]
    return [dataset_dirs[dir_index] for dir_index in indices_to_keep]


## } MODULE
