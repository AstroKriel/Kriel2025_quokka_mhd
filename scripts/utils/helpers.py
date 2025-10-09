## { MODULE

##
## === DEPENDENCIES
##

import argparse
from pathlib import Path
from jormi.ww_plots import plot_manager

##
## === FUNCTIONS
##


def create_axes_grid(
    num_rows: int,
    num_cols: int,
    add_cbar_space: bool = False,
):
    fig, axs = plot_manager.create_figure(
        num_rows=num_rows,
        num_cols=num_cols,
        share_x=False,
        y_spacing=0.25,
        x_spacing=0.75 if add_cbar_space else 0.25,
    )
    axs_grid = plot_manager.get_axs_grid(
        axs=axs,
        num_rows=num_rows,
        num_cols=num_cols,
    )
    return fig, axs_grid


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


## } MODULE
