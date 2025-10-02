## { SCRIPT

##
## === DEPENDENCIES
##

import numpy
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


@dataclass(frozen=True)
class PlotArgs:
    fig_dir: Path
    dataset_dir: Path
    field_name: str
    components_to_plot: list[str]
    axes_to_slice: list[str]
    loader_name: str
    cmap_name: str
    verbose: bool


##
## === HELPERS
##


def _get_slice_bounds(
    domain: field_types.UniformDomain,
    axis_to_slice: str,
):
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = domain.domain_bounds
    if axis_to_slice == "z": return (x_min, x_max, y_min, y_max)
    if axis_to_slice == "y": return (x_min, x_max, z_min, z_max)
    if axis_to_slice == "x": return (y_min, y_max, z_min, z_max)
    raise ValueError("axis_to_slice must be one of: x, y, z")


def _get_slice_labels(
    axis_to_slice: str,
):
    if axis_to_slice == "z": return ("x", "y")
    if axis_to_slice == "y": return ("x", "z")
    if axis_to_slice == "x": return ("y", "z")
    raise ValueError("axis_to_slice must be one of: x, y, z")


def _slice_sfield(
    field_data: numpy.ndarray,
    axis_to_slice: str,
):
    num_cells_x, num_cells_y, num_cells_z = field_data.shape
    slice_index_x = num_cells_x // 2
    slice_index_y = num_cells_y // 2
    slice_index_z = num_cells_z // 2
    if axis_to_slice == "z": return field_data[:, :, slice_index_z], r"$(x, y, z=L_z/2)$"
    if axis_to_slice == "y": return field_data[:, slice_index_y, :], r"$(x, y=L_y/2, z)$"
    if axis_to_slice == "x": return field_data[slice_index_x, :, :], r"$(x=L_x/2, y, z)$"
    raise ValueError("axis_to_slice must be one of: x, y, z")


##
## === PLOTTING
##


def _plot_slice(
    ax,
    sim_time,
    field_data,
    domain,
    axis_to_slice: str,
    label: str,
    cmap_name: str,
):
    field_slice, slice_label = _slice_sfield(field_data, axis_to_slice)
    min_value = numpy.min(field_slice)
    max_value = numpy.max(field_slice)
    plot_data.plot_sfield_slice(
        ax=ax,
        field_slice=field_slice,
        axis_bounds=_get_slice_bounds(domain, axis_to_slice),
        cmap_name=cmap_name,
        add_colorbar=True,
        cbar_label=label,
        cbar_side="right",
        cbar_bounds=(min_value, max_value),
    )
    annotate_axis.add_text(
        ax=ax,
        x_pos=0.5,
        y_pos=0.95,
        x_alignment="center",
        y_alignment="top",
        label=f"({min_value:.2e}, {max_value:.2e})",
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
        label=slice_label,
        fontsize=16,
        box_alpha=0.5,
        add_box=True,
    )


def _plot_snapshot(
    plot_args: PlotArgs,
):
    with load_dataset.QuokkaDataset(dataset_dir=plot_args.dataset_dir, verbose=plot_args.verbose) as ds:
        domain = ds.load_domain()
        loader = getattr(ds, plot_args.loader_name)
        field = loader()  # ScalarField or VectorField
    sim_time = field.sim_time
    lookup_comp_index = {"x": 0, "y": 1, "z": 2}
    data_args: list[tuple[str, numpy.ndarray]] = []
    if isinstance(field, field_types.VectorField):
        if len(plot_args.components_to_plot) == 0:
            raise ValueError(f"Vector field '{plot_args.field_name}' requires at least one component via -c")
        for comp_name in sorted(plot_args.components_to_plot):
            comp_index = lookup_comp_index[comp_name]
            data_args.append((field.labels[comp_index], field.data[comp_index]))
    elif isinstance(field, field_types.ScalarField):
        data_args = [(field.label, field.data)]
    else:
        raise ValueError(f"{plot_args.field_name} is an unrecognised field type.")
    fig, axs_grid = helpers.create_axes_grid(
        num_rows=len(data_args),
        num_cols=len(plot_args.axes_to_slice),
        add_cbar_space=True
    )
    for row_index, (data_label, data_array) in enumerate(data_args):
        for col_index, axis_to_slice in enumerate(plot_args.axes_to_slice):
            ax = axs_grid[row_index][col_index]
            _plot_slice(
                ax=ax,
                sim_time=sim_time,
                field_data=data_array,
                domain=domain,
                axis_to_slice=axis_to_slice,
                label=data_label,
                cmap_name=plot_args.cmap_name,
            )
    num_rows = len(axs_grid)
    for row_index in range(num_rows):
        for col_index, axis_to_slice in enumerate(plot_args.axes_to_slice):
            ax = axs_grid[row_index][col_index]
            xlabel, ylabel = _get_slice_labels(axis_to_slice)
            if num_rows == 1:
                ax.set_xlabel(xlabel)
            elif num_rows > 1 and row_index == num_rows-1:
                ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
    index_label = plot_args.dataset_dir.name.split("plt")[1]
    fig_name = f"{plot_args.field_name}_slice_{index_label}.png"
    fig_path = plot_args.fig_dir / fig_name
    plot_manager.save_figure(
        fig=fig,
        fig_path=fig_path,
        verbose=plot_args.verbose,
    )


##
## === OPERATOR CLASS
##


class Plotter:

    VALID_FIELDS = {
        "divb": {
            "loader": "load_div_b_sfield",
            "cmap": "bwr",
        },
        "mag": {
            "loader": "load_magnetic_vfield",
            "cmap": "Blues",
        },
        "vel": {
            "loader": "load_velocity_vfield",
            "cmap": "Oranges",
        },
        "rho": {
            "loader": "load_density_sfield",
            "cmap": "Greys",
        },
        "Etot": {
            "loader": "load_total_energy_sfield",
            "cmap": "cividis",
        },
        "Emag": {
            "loader": "load_magnetic_energy_sfield",
            "cmap": "plasma",
        },
    }

    def __init__(
        self,
        *,
        input_dir: Path,
        fields_to_plot: list[str],
        components_to_plot: list[str],
        axes_to_slice: list[str],
        use_parallel: bool = True,
    ):
        valid_fields = set(self.VALID_FIELDS.keys())
        if not fields_to_plot or not set(fields_to_plot).issubset(valid_fields):
            raise ValueError(f"Provide one or more field to plot (via -f) from: {sorted(valid_fields)}")
        valid_axes = {"x", "y", "z"}
        ## default to all components/axes (if not provided)
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

    def run(self) -> None:
        dataset_dirs = helpers.resolve_dataset_dirs(self.input_dir)
        fig_dir = dataset_dirs[0].parent
        grouped_args: list[PlotArgs] = []
        for field_name in self.fields_to_plot:
            field_meta = self.VALID_FIELDS[field_name]
            loader_name = field_meta["loader"]
            cmap_name = field_meta["cmap"]
            for dataset_dir in dataset_dirs:
                grouped_args.append(
                    PlotArgs(
                        fig_dir=Path(fig_dir),
                        dataset_dir=Path(dataset_dir),
                        field_name=field_name,
                        components_to_plot=self.components_to_plot,
                        axes_to_slice=self.axes_to_slice,
                        loader_name=loader_name,
                        cmap_name=cmap_name,
                        verbose=False,
                    ),
                )
        if not grouped_args: return
        if self.use_parallel and len(grouped_args) > 5:
            parallel_utils.run_in_parallel(
                func=_plot_snapshot,
                grouped_args=grouped_args,
                timeout_seconds=30,
                show_progress=True,
                enable_plotting=True,
            )
        else:
            [_plot_snapshot(plot_args) for plot_args in grouped_args]
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
    )
    plotter.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
