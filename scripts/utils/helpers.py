## { MODULE

##
## === DEPENDENCIES
##

import argparse
from pathlib import Path

##
## === FUNCTIONS
##


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
    args = parser.parse_args()
    return args


def get_latest_dataset_dirs(
    sim_dir: Path,
) -> list[Path]:
    dataset_dirs = [
        sub_dir for sub_dir in sim_dir.iterdir()
        if sub_dir.is_dir() and ("plt" in sub_dir.name) and ("old" not in sub_dir.name)
    ]
    dataset_dirs.sort(key=lambda dataset_dir: int(dataset_dir.name.split("plt")[1]))  # sort by index
    return dataset_dirs


## } MODULE
