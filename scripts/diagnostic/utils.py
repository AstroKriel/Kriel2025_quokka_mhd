## { MODULE

##
## === DEPENDENCIES
##

import argparse
from pathlib import Path
from jormi.ww_plots import plot_manager

##
## === QUOKKA FIELD MAPPINGS
##

QUOKKA_FIELD_LOOKUP = {
    "rho": {
        "loader": "load_density_sfield",
        "cmap": "Greys",
        "color": "black",
    },
    "vel": {
        "loader": "load_velocity_vfield",
        "cmap": "Blues",
        "color": "royalblue",
    },
    "mag": {
        "loader": "load_magnetic_vfield",
        "cmap": "Oranges",
        "color": "orangered",
    },
    "Etot": {
        "loader": "load_total_energy_sfield",
        "cmap": "cividis",
        "color": "black",
    },
    "Ekin": {
        "loader": "load_kinetic_energy_sfield",
        "cmap": "magma",
        "color": "royalblue",
    },
    "Ekin_div": {
        "loader": "load_div_kinetic_energy_sfield",
        "cmap": "magma",
        "color": "lightskyblue",
    },
    "Ekin_sol": {
        "loader": "load_sol_kinetic_energy_sfield",
        "cmap": "magma",
        "color": "cornflowerblue",
    },
    "Ekin_bulk": {
        "loader": "load_bulk_kinetic_energy_sfield",
        "cmap": "magma",
        "color": "dodgerblue",
    },
    "Emag": {
        "loader": "load_magnetic_energy_sfield",
        "cmap": "plasma",
        "color": "darkorchid",
    },
    "Eint": {
        "loader": "load_internal_energy_sfield",
        "cmap": "magma",
        "color": "violet",
    },
    "pressure": {
        "loader": "load_pressure_sfield",
        "cmap": "Purples",
        "color": "orchid",
    },
    "divb": {
        "loader": "load_divb_sfield",
        "cmap": "bwr",
        "color": "sandybrown",
    },
}

##
## === FUNCTIONS
##


def get_user_args():
    parser = argparse.ArgumentParser(description="Test Quokka simulation(s).")
    parser.add_argument(
        "--dir",
        "-d",
        type=lambda path: Path(path).expanduser().resolve(),
        default=None,
        help="Optional path to a Quokka simulation or dataset directory.",
    )
    parser.add_argument(
        "--comps",
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
    parser.add_argument(
        "--animate-only",
        action="store_true",
        default=False,
        help="Skip to animation if provided (default: False).",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        default=False,
        help="Fit to some data (default: False).",
    )
    args = parser.parse_args()
    return args


def create_figure(
    num_rows: int,
    num_cols: int,
    add_cbar_space: bool = False,
):
    fig, axs_grid = plot_manager.create_figure(
        num_rows=num_rows,
        num_cols=num_cols,
        share_x=False,
        y_spacing=0.25,
        x_spacing=0.75 if add_cbar_space else 0.25,
    )
    return fig, axs_grid


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


def get_dataset_index_str(
    dataset_dir: Path,
) -> str:
    dataset_name = dataset_dir.name
    name_parts = dataset_name.split("plt")
    if len(name_parts) < 2:
        raise ValueError(f"Unexpected dataset name format: {dataset_name}")
    digits_str = name_parts[1].split(".")[0]
    if not digits_str.isdigit():
        raise ValueError(f"Expected digits after 'plt' in {dataset_name}")
    return digits_str


def get_max_index_width(
    dataset_dirs: list[Path],
) -> int:
    if not dataset_dirs: return 1
    index_widths: list[int] = []
    for dataset_dir in dataset_dirs:
        dataset_index_str = get_dataset_index_str(dataset_dir)
        index_widths.append(len(dataset_index_str))
    return max(index_widths) if index_widths else 1


## } MODULE
