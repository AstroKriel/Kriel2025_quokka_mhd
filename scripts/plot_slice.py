## { SCRIPT

##
## === DEPENDENCIES
##

import numpy
from typing import Literal
from pathlib import Path
from dataclasses import dataclass
from jormi.ww_io import io_manager
from jormi.utils import parallel_utils
from jormi.ww_plots import plot_manager, plot_data, annotate_axis
from jormi.ww_fields import field_types
from ww_quokka_sims.sim_io import load_dataset
from utils import helpers

##
## === DATA TYPES
##

Axis = Literal["x", "y", "z"]

LOOKUP_AXIS_INDEX: dict[Axis, int] = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class SliceArgs:
    data_2d: numpy.ndarray
    label: str
    min_value: float
    max_value: float


@dataclass(frozen=True)
class FieldArgs:
    field_name: str
    field_loader: str
    cmap_name: str


@dataclass(frozen=True)
class DataItem:
    data_3d: numpy.ndarray
    label: str


@dataclass(frozen=True)
class SnapshotArgs:
    fig_dir: Path
    dataset_dir: Path
    field_args: FieldArgs
    components_to_plot: list[Axis]  # empty for scalars
    axes_to_slice: list[Axis]
    verbose: bool


##
## === HELPERS
##


def _get_slice_bounds(
    domain_details: field_types.UniformDomain,
    axis_to_slice: Axis,
) -> tuple[float, float, float, float]:
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = domain_details.domain_bounds
    if axis_to_slice == "z": return (x_min, x_max, y_min, y_max)
    if axis_to_slice == "y": return (x_min, x_max, z_min, z_max)
    if axis_to_slice == "x": return (y_min, y_max, z_min, z_max)
    raise ValueError("axis_to_slice must be one of: x, y, z")


def _get_slice_labels(
    axis_to_slice: Axis,
) -> tuple[str, str]:
    if axis_to_slice == "z": return ("x", "y")
    if axis_to_slice == "y": return ("x", "z")
    if axis_to_slice == "x": return ("y", "z")
    raise ValueError("axis_to_slice must be one of: x, y, z")


def slice_field(
    data_3d: numpy.ndarray,
    axis_to_slice: Axis,
) -> SliceArgs:
    num_cells_x, num_cells_y, num_cells_z = data_3d.shape
    slice_index_x = num_cells_x // 2
    slice_index_y = num_cells_y // 2
    slice_index_z = num_cells_z // 2
    if axis_to_slice == "z":
        data_2d = data_3d[:, :, slice_index_z]
        label = r"$(x, y, z=L_z/2)$"
    elif axis_to_slice == "y":
        data_2d = data_3d[:, slice_index_y, :]
        label = r"$(x, y=L_y/2, z)$"
    elif axis_to_slice == "x":
        data_2d = data_3d[slice_index_x, :, :]
        label = r"$(x=L_x/2, y, z)$"
    else:
        raise ValueError("axis_to_slice must be one of: x, y, z")
    min_value = float(numpy.min(data_2d))
    max_value = float(numpy.max(data_2d))
    return SliceArgs(
        data_2d=data_2d,
        label=label,
        min_value=min_value,
        max_value=max_value,
    )


##
## === PLOTTING
##


def _plot_slice(
    ax,
    sim_time: float,
    field_slice: SliceArgs,
    domain_details: field_types.UniformDomain,
    axis_to_slice: Axis,
    label: str,
    cmap_name: str,
) -> None:
    plot_data.plot_sfield_slice(
        ax=ax,
        field_slice=field_slice.data_2d,
        axis_bounds=_get_slice_bounds(domain_details, axis_to_slice),
        cmap_name=cmap_name,
        add_colorbar=True,
        cbar_label=label,
        cbar_side="right",
        cbar_bounds=(field_slice.min_value, field_slice.max_value),
    )
    annotate_axis.add_text(
        ax=ax,
        x_pos=0.5,
        y_pos=0.95,
        x_alignment="center",
        y_alignment="top",
        label=f"({field_slice.min_value:.2e}, {field_slice.max_value:.2e})",
        fontsize=16,
        box_alpha=0.5,
        add_box=True,
    )
    annotate_axis.add_text(
        ax=ax,
        x_pos=0.5,
        y_pos=0.5,
        x_alignment="center",
        y_alignment="center",
        label=rf"$t = {sim_time:.2f}$",
        fontsize=16,
        box_alpha=0.5,
        add_box=True,
    )
    annotate_axis.add_text(
        ax=ax,
        x_pos=0.5,
        y_pos=0.05,
        x_alignment="center",
        y_alignment="bottom",
        label=field_slice.label,
        fontsize=16,
        box_alpha=0.5,
        add_box=True,
    )


def _plot_snapshot(
    snapshot: SnapshotArgs,
) -> None:
    with load_dataset.QuokkaDataset(dataset_dir=snapshot.dataset_dir, verbose=snapshot.verbose) as ds:
        domain_details = ds.load_domain_details()
        loader = getattr(ds, snapshot.field_args.field_loader)
        field = loader()  # ScalarField or VectorField
    sim_time = float(field.sim_time)
    plt_index = snapshot.dataset_dir.name.split("plt")[1]
    data_items: list[DataItem] = []
    if isinstance(field, field_types.VectorField):
        if not snapshot.components_to_plot:
            raise ValueError(
                f"Vector field '{snapshot.field_args.field_name}' requires at least one component via -c",
            )
        for comp in sorted(snapshot.components_to_plot):
            data_items.append(
                DataItem(
                    data_3d=field.data[LOOKUP_AXIS_INDEX[comp]],
                    label=field.labels[LOOKUP_AXIS_INDEX[comp]],
                ),
            )
    elif isinstance(field, field_types.ScalarField):
        data_items = [
            DataItem(
                data_3d=field.data,
                label=field.label,
            ),
        ]
    else:
        raise ValueError(f"{snapshot.field_args.field_name} is an unrecognised field type.")
    fig, axs_grid = helpers.create_axes_grid(
        num_rows=len(data_items),
        num_cols=len(snapshot.axes_to_slice),
        add_cbar_space=True,
    )
    for row_index, data_item in enumerate(data_items):
        for col_index, axis_to_slice in enumerate(snapshot.axes_to_slice):
            ax = axs_grid[row_index][col_index]
            field_slice = slice_field(
                data_3d=data_item.data_3d,
                axis_to_slice=axis_to_slice,
            )
            _plot_slice(
                ax=ax,
                sim_time=sim_time,
                field_slice=field_slice,
                domain_details=domain_details,
                axis_to_slice=axis_to_slice,
                label=data_item.label,
                cmap_name=snapshot.field_args.cmap_name,
            )
    num_rows = len(axs_grid)
    for row_index in range(num_rows):
        for col_index, axis_to_slice in enumerate(snapshot.axes_to_slice):
            ax = axs_grid[row_index][col_index]
            xlabel, ylabel = _get_slice_labels(axis_to_slice)
            if (num_rows == 1) or (row_index == num_rows - 1):
                ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
    fig_name = f"{snapshot.field_args.field_name}_slice_{plt_index}.png"
    fig_path = snapshot.fig_dir / fig_name
    plot_manager.save_figure(fig=fig, fig_path=fig_path, verbose=snapshot.verbose)


##
## === OPERATOR CLASS
##


class Plotter:

    VALID_FIELDS = {
        "rho": {
            "loader": "load_density_sfield",
            "cmap": "Greys",
        },
        "vel": {
            "loader": "load_velocity_vfield",
            "cmap": "Oranges",
        },
        "mag": {
            "loader": "load_magnetic_vfield",
            "cmap": "Blues",
        },
        "Etot": {
            "loader": "load_total_energy_sfield",
            "cmap": "cividis",
        },
        "Ekin": {
            "loader": "load_kinetic_energy_sfield",
            "cmap": "magma",
        },
        "Emag": {
            "loader": "load_magnetic_energy_density_sfield",
            "cmap": "plasma",
        },
        "Eint": {
            "loader": "load_internal_energy_sfield",
            "cmap": "magma",
        },
        "pressure": {
            "loader": "load_pressure_sfield",
            "cmap": "Purples",
        },
        "divb": {
            "loader": "load_div_b_sfield",
            "cmap": "bwr",
        },
    }

    def __init__(
        self,
        *,
        input_dir: Path,
        fields_to_plot: list[str],
        components_to_plot: list[Axis],
        axes_to_slice: list[Axis],
        use_parallel: bool = True,
        animate_only: bool = False,
    ):
        valid_fields = set(self.VALID_FIELDS.keys())
        if not fields_to_plot or not set(fields_to_plot).issubset(valid_fields):
            raise ValueError(f"Provide one or more field to plot (via -f) from: {sorted(valid_fields)}")
        valid_axes: set[Axis] = {"x", "y", "z"}
        if not components_to_plot: components_to_plot = ["x", "y", "z"]
        elif not set(components_to_plot).issubset(valid_axes):
            raise ValueError("Provide one or more components (via -c) from: x, y, z")
        if not axes_to_slice: axes_to_slice = ["x", "y", "z"]
        elif not set(axes_to_slice).issubset(valid_axes):
            raise ValueError("Provide one or more axes (via -a) from: x, y, z")
        self.input_dir = Path(input_dir)
        self.fields_to_plot = fields_to_plot
        self.components_to_plot = components_to_plot
        self.axes_to_slice = axes_to_slice
        self.use_parallel = bool(use_parallel)
        self.animate_only = bool(animate_only)

    def run(self) -> None:
        dataset_dirs = helpers.resolve_dataset_dirs(self.input_dir)
        if not dataset_dirs: return
        fig_dir = dataset_dirs[0].parent
        if not self.animate_only:
            snapshots = self._prepare_snapshots(
                dataset_dirs=dataset_dirs,
                fig_dir=fig_dir,
            )
            if not snapshots: return
            if self.use_parallel and len(snapshots) > 5:
                parallel_utils.run_in_parallel(
                    func=_plot_snapshot,
                    grouped_args=snapshots,
                    timeout_seconds=120,
                    show_progress=True,
                    enable_plotting=True,
                )
            else:
                [_plot_snapshot(snapshot) for snapshot in snapshots]
        for field_name in self.fields_to_plot:
            fig_paths = io_manager.ItemFilter(
                prefix=f"{field_name}_slice_",
                suffix=".png",
                include_folders=False,
                include_files=True,
            ).filter(directory=fig_dir)
            if len(fig_paths) < 3: continue
            mp4_path = Path(fig_dir) / f"{field_name}_slices.mp4"
            plot_manager.animate_pngs_to_mp4(
                frames_dir=fig_dir,
                mp4_path=mp4_path,
                pattern=f"{field_name}_slice_*.png",
                fps=30,
                timeout_seconds=120,
            )

    def _prepare_snapshots(
        self,
        dataset_dirs: list[Path],
        fig_dir: Path,
    ) -> list[SnapshotArgs]:
        snapshots: list[SnapshotArgs] = []
        for field_name in self.fields_to_plot:
            field_meta = self.VALID_FIELDS[field_name]
            field_args = FieldArgs(
                field_name=field_name,
                field_loader=field_meta["loader"],
                cmap_name=field_meta["cmap"],
            )
            for dataset_dir in dataset_dirs:
                snapshots.append(
                    SnapshotArgs(
                        fig_dir=Path(fig_dir),
                        dataset_dir=Path(dataset_dir),
                        field_args=field_args,
                        components_to_plot=self.components_to_plot,
                        axes_to_slice=self.axes_to_slice,
                        verbose=False,
                    ),
                )
        return snapshots


##
## === MAIN PROGRAM
##


def main():
    args = helpers.get_user_input()
    plotter = Plotter(
        input_dir=args.dir,
        fields_to_plot=args.fields,
        components_to_plot=args.components,
        axes_to_slice=args.axes,
        use_parallel=True,
        animate_only=args.animate_only,
    )
    plotter.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
