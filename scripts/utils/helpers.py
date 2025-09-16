## { MODULE

##
## === DEPENDENCIES ===
##

import argparse
from pathlib import Path

##
## === FUNCTIONS ===
##


def get_user_input():
    parser = argparse.ArgumentParser(description="Test Quokka simulation(s).")
    parser.add_argument(
        "--sim-dir",
        "-sd",
        type=lambda path: Path(path).expanduser().resolve(),
        default=None,
        help="Optional path to a Quokka sim-directory.",
    )
    args = parser.parse_args()
    return args.sim_dir


def get_latest_plt_dirs(
    sim_dir: Path,
) -> list[Path]:
    plt_dirs = [
        sub_dir for sub_dir in sim_dir.iterdir()
        if sub_dir.is_dir() and ("plt" in sub_dir.name) and ("old" not in sub_dir.name)
    ]
    plt_dirs.sort(key=lambda plt_dir: int(plt_dir.name.split("plt")[1]))  # sort by index
    return plt_dirs


## } MODULE
