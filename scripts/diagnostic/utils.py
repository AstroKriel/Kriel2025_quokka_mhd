## { MODULE

##
## === DEPENDENCIES
##

import numpy
import argparse

from pathlib import Path

from jormi.utils import list_utils
from jormi.ww_types import type_checks
from jormi.ww_io import log_manager
from jormi.ww_plots import plot_manager

##
## === QUOKKA FIELDS
##

QUOKKA_FIELD_LOOKUP = {
    "rho": {
        "loader": "load_3d_density_sfield",
        "cmap": "Greys",
        "color": "black",
    },
    "vel": {
        "loader": "load_3d_velocity_vfield",
        "cmap": "Blues",
        "color": "royalblue",
    },
    "mag": {
        "loader": "load_3d_magnetic_vfield",
        "cmap": "Oranges",
        "color": "orangered",
    },
    "Etot": {
        "loader": "load_3d_total_energy_sfield",
        "cmap": "cividis",
        "color": "black",
    },
    "Ekin": {
        "loader": "load_3d_kinetic_energy_sfield",
        "cmap": "magma",
        "color": "royalblue",
    },
    "Ekin_div": {
        "loader": "load_3d_div_kinetic_energy_sfield",
        "cmap": "magma",
        "color": "lightskyblue",
    },
    "Ekin_sol": {
        "loader": "load_3d_sol_kinetic_energy_sfield",
        "cmap": "magma",
        "color": "cornflowerblue",
    },
    "Ekin_bulk": {
        "loader": "load_3d_bulk_kinetic_energy_sfield",
        "cmap": "magma",
        "color": "dodgerblue",
    },
    "Emag": {
        "loader": "load_3d_magnetic_energy_sfield",
        "cmap": "plasma",
        "color": "darkorchid",
    },
    "Eint": {
        "loader": "load_3d_internal_energy_sfield",
        "cmap": "magma",
        "color": "violet",
    },
    "pressure": {
        "loader": "load_3d_pressure_sfield",
        "cmap": "Purples",
        "color": "orchid",
    },
    "divb": {
        "loader": "load_3d_divb_sfield",
        "cmap": "bwr",
        "color": "sandybrown",
    },
    "cur": {
        "loader": "load_current_density_sfield",
        "cmap": "cubehelix",
        "color": "black",
    },
}

##
## === HELPER FUNCTIONS
##


def get_user_args():
    parser = argparse.ArgumentParser(
        description="Diagnostic plots to make of quantities in a Quokka (BoxLib) data-directory.",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=lambda path: Path(path).expanduser().resolve(),
        default=None,
        help="Optional path to a Quokka simulation or dataset directory.",
    )
    parser.add_argument(
        "--tag",
        "-t",
        default="plt",
        help=
        "Dataset tag used to identify output directories (e.g., `plt` -> plt00010, plt00020). Default: `plt`.",
    )
    field_list = list_utils.as_string(elems=sorted(QUOKKA_FIELD_LOOKUP.keys()))
    parser.add_argument(
        "--fields",
        "-f",
        nargs="+",
        default=None,
        help=f"List of (vector and/or scalar) fields to plot. Options: {field_list}",
    )
    parser.add_argument(
        "--comps",
        "-c",
        nargs="+",
        default=None,
        help="Optional list of vector field components to show.",
    )
    parser.add_argument(
        "--axes",
        "-a",
        nargs="+",
        default=None,
        help="Optional list of axes to slice.",
    )
    parser.add_argument(
        "--animate-only",
        action="store_true",
        default=False,
        help="Skip straight to animation (default: False).",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        default=False,
        help="Perform the relevant fitting routine (default: False).",
    )
    user_args = parser.parse_args()
    return user_args


def create_figure(
    num_rows: int,
    num_cols: int,
    add_cbar_space: bool = False,
):
    if (num_rows == 1) and (num_cols == 1):
        fig, ax = plot_manager.create_figure(
            share_x=False,
            share_y=False,
        )
        if add_cbar_space:
            fig.subplots_adjust(right=0.82)
        axs_grid = numpy.asarray([[ax]], dtype=object)
        return fig, axs_grid
    fig, axs_grid = plot_manager.create_figure(
        num_rows=num_rows,
        num_cols=num_cols,
        share_x=False,
        share_y=False,
        y_spacing=0.25,
        x_spacing=0.75 if add_cbar_space else 0.25,
    )
    return fig, axs_grid


def looks_like_boxlib_dir(
    dataset_dir: Path,
) -> bool:
    type_checks.ensure_type(param=dataset_dir, valid_types=Path)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return False
    has_header = (dataset_dir / "Header").is_file()
    has_level0 = (dataset_dir / "Level_0").is_dir()
    return has_header and has_level0


def ensure_looks_like_boxlib_dir(
    dataset_dir: Path,
) -> None:
    if not looks_like_boxlib_dir(dataset_dir=dataset_dir):
        log_manager.log_error(
            "Provided dataset does not appear to be a valid BoxLib-like plotfile.",
            notes={
                "Path": str(dataset_dir),
                "Expected": "both a `Header` file and `Level_0` directory",
            },
        )
        raise ValueError(f"Directory is not valid: {dataset_dir}")


def get_latest_dataset_dirs(
    sim_dir: Path,
    dataset_tag: str,
) -> list[Path]:
    dataset_dirs = [
        sub_dir for sub_dir in sim_dir.iterdir()
        if sub_dir.is_dir() and (dataset_tag in sub_dir.name) and ("old" not in sub_dir.name)
    ]
    dataset_dirs.sort(
        key=lambda dataset_dir: int(get_dataset_index_string(dataset_dir, dataset_tag)),
    )
    return dataset_dirs


def resolve_dataset_dirs(
    input_dir: Path,
    dataset_tag: str,
    max_elems: int | None = None,
) -> list[Path]:
    if (dataset_tag in input_dir.name) or looks_like_boxlib_dir(input_dir):
        return [input_dir]
    dataset_dirs = get_latest_dataset_dirs(
        sim_dir=input_dir,
        dataset_tag=dataset_tag,
    )
    if not dataset_dirs:
        raise ValueError(f"No dataset directories found using tag `{dataset_tag}` under: {input_dir}")
    if max_elems is not None:
        dataset_dirs = list_utils.sample_list(
            elems=dataset_dirs,
            max_elems=100,
        )
    return dataset_dirs


def get_dataset_index_string(
    dataset_dir: Path,
    dataset_tag: str,
) -> str:
    dataset_name = dataset_dir.name
    if dataset_tag not in dataset_name:
        raise ValueError(f"Dataset tag `{dataset_tag}` was not found in `{dataset_name}`.")
    name_parts = dataset_name.split(dataset_tag)
    if len(name_parts) < 2:
        raise ValueError(f"Unexpected dataset name format: {dataset_name}")
    digits_string = name_parts[1].split(".")[0]
    if not digits_string.isdigit():
        raise ValueError(f"Expected digits after `{dataset_tag}` in {dataset_name}")
    return digits_string


def get_max_index_width(
    dataset_dirs: list[Path],
    dataset_tag: str,
) -> int:
    if not dataset_dirs: return 1
    index_widths: list[int] = []
    for dataset_dir in dataset_dirs:
        dataset_index_string = get_dataset_index_string(
            dataset_dir=dataset_dir,
            dataset_tag=dataset_tag,
        )
        index_widths.append(len(dataset_index_string))
    return max(index_widths) if len(index_widths) > 0 else 1


## } MODULE
