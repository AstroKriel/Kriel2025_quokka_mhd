## { SCRIPT

##
## === DEPENDENCIES
##

import numpy
from typing import Literal
from pathlib import Path
from dataclasses import dataclass
from jormi.ww_plots import plot_manager, add_color
from jormi.ww_fields import field_types
from ww_quokka_sims.sim_io import load_dataset
from utils import helpers

##
## === DATA TYPES
##

Axis = Literal["x", "y", "z"]

LOOKUP_AXIS_INDEX: dict[Axis, int] = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class PlotArgs:
    fig_dir: Path
    dataset_dirs: list[Path]
    field_name: str
    components_to_plot: list[Axis]
    axes_to_slice: list[Axis]
    field_loader: str
    cmap_name: str
    verbose: bool = False


@dataclass(frozen=True)
class ProfileData:
    sim_time: float
    x_positions: numpy.ndarray
    ## for scalars: y_profile.shape == (num_axes,)
    ## for vectors: y_profile.shape == (num_axes, num_comps)
    y_profile: numpy.ndarray
    axes_labels: list[Axis]
    comp_labels: list[str]

    def get(
        self,
        axis_index: int,
        comp_index: int | None = None,
    ) -> numpy.ndarray:
        if self.num_comps == 1:
            return self.y_profile[axis_index]
        if comp_index is None:
            raise ValueError("comp_index is required for vector fields")
        return self.y_profile[axis_index, comp_index]

    @property
    def num_axes(
        self,
    ) -> int:
        return len(self.axes_labels)

    @property
    def num_comps(
        self,
    ) -> int:
        return len(self.comp_labels)


##
## === HELPERS
##


def _compute_centers(
    uniform_domain,
    axis: Axis,
) -> numpy.ndarray:
    (x_min, _), (y_min, _), (z_min, _) = uniform_domain.domain_bounds
    num_cells_x, num_cells_y, num_cells_z = uniform_domain.resolution
    cell_width_x, cell_width_y, cell_width_z = uniform_domain.cell_widths
    if axis == "x": return x_min + (numpy.arange(num_cells_x) + 0.5) * cell_width_x
    if axis == "y": return y_min + (numpy.arange(num_cells_y) + 0.5) * cell_width_y
    if axis == "z": return z_min + (numpy.arange(num_cells_z) + 0.5) * cell_width_z
    raise ValueError("axis must be one of: x, y, z")


def _extract_profile(
    sfield: numpy.ndarray,
    axis: Axis,
) -> numpy.ndarray:
    num_cells_x, num_cells_y, num_cells_z = sfield.shape
    slice_index_x = num_cells_x // 2
    slice_index_y = num_cells_y // 2
    slice_index_z = num_cells_z // 2
    if axis == "x": return sfield[:, slice_index_y, slice_index_z]
    if axis == "y": return sfield[slice_index_x, :, slice_index_z]
    if axis == "z": return sfield[slice_index_x, slice_index_y, :]
    raise ValueError("axis must be one of: x, y, z")


def _style_axes(
    axs_grid,
    comp_labels: list[str],
    axes: list[Axis],
) -> None:
    for comp_index in range(len(axs_grid)):
        for axis_index, axis in enumerate(axes):
            ax = axs_grid[comp_index][axis_index]
            if axis_index == 0:
                ax.set_ylabel(comp_labels[comp_index])
            ax.set_xlabel(axis)


##
## === LOAD DATASETS
##


def load_field_profiles(
    plot_args: PlotArgs,
) -> list[ProfileData]:
    field_profiles: list[ProfileData] = []
    for dataset_dir in plot_args.dataset_dirs:
        with load_dataset.QuokkaDataset(dataset_dir=dataset_dir, verbose=False) as ds:
            uniform_domain = ds.load_domain_details()
            field_loader = getattr(ds, plot_args.field_loader)
            field = field_loader()
        sim_time = field.sim_time
        axes_names = sorted(plot_args.axes_to_slice)
        x_positions = numpy.empty((len(axes_names), ), dtype=object)
        for axis_index, axis_name in enumerate(axes_names):
            x_positions[axis_index] = _compute_centers(uniform_domain, axis_name)
        if isinstance(field, field_types.VectorField):
            if len(plot_args.components_to_plot) == 0:
                raise ValueError(
                    f"Vector field '{plot_args.field_name}' requires at least one component via -c",
                )
            comp_names = sorted(plot_args.components_to_plot)
            comp_labels = [rf"$({plot_args.field_name})_{{{comp_name}}}$" for comp_name in comp_names]
            profile_data = numpy.empty((len(axes_names), len(comp_names)), dtype=object)
            for axis_index, axis_name in enumerate(axes_names):
                for comp_index, comp_name in enumerate(comp_names):
                    field_data = field.data[LOOKUP_AXIS_INDEX[comp_name]]
                    profile_data[axis_index, comp_index] = _extract_profile(field_data, axis_name)
            field_profiles.append(
                ProfileData(
                    sim_time=sim_time,
                    x_positions=x_positions,
                    y_profile=profile_data,
                    axes_labels=axes_names,
                    comp_labels=comp_labels,
                ),
            )
        elif isinstance(field, field_types.ScalarField):
            profile_data = numpy.empty((len(axes_names), ), dtype=object)
            for axis_index, axis_name in enumerate(axes_names):
                profile_data[axis_index] = _extract_profile(field.data, axis_name)
            field_profiles.append(
                ProfileData(
                    sim_time=sim_time,
                    x_positions=x_positions,
                    y_profile=profile_data,
                    axes_labels=axes_names,
                    comp_labels=[field.field_label],
                ),
            )
        else:
            raise ValueError(f"{plot_args.field_name} is an unrecognised field type.")
    field_profiles.sort(key=lambda field_profile: field_profile.sim_time)
    return field_profiles


##
## === PLOTTING
##


def _plot_snapshot(
    *,
    axs_grid,
    field_profile: ProfileData,
    color,
) -> None:
    ## rows = components (or a single row for scalars)
    ## cols = axes along which profiles were extracted
    for comp_index in range(field_profile.num_comps):
        for axis_index in range(field_profile.num_axes):
            ax = axs_grid[comp_index][axis_index]
            x = field_profile.x_positions[axis_index]
            y = field_profile.get(axis_index=axis_index, comp_index=comp_index)
            ax.plot(x, y, lw=2.0, color=color)


def _plot_series(
    *,
    axs_grid,
    field_profiles: list[ProfileData],
    cmap_name: str,
) -> None:
    cmap, norm = add_color.create_cmap(
        cmap_name=cmap_name,
        cmin=0.25,
        vmin=0,
        vmax=len(field_profiles) - 1,
    )
    for profile_index, field_profile in enumerate(field_profiles):
        color = cmap(norm(profile_index))
        _plot_snapshot(
            axs_grid=axs_grid,
            field_profile=field_profile,
            color=color,
        )
    add_color.add_cbar_from_cmap(
        ax=axs_grid[-1][-1],
        label=r"dump index",
        cmap=cmap,
        norm=norm,
        side="right",
        ax_percentage=0.05,
    )


def _plot_field(
    plot_args: PlotArgs,
) -> None:
    field_profiles = load_field_profiles(plot_args)
    if not field_profiles: return
    fig, axs_grid = helpers.create_figure(
        num_rows=field_profiles[0].num_comps,
        num_cols=field_profiles[0].num_axes,
    )
    if len(field_profiles) == 1:
        _plot_snapshot(
            axs_grid=axs_grid,
            field_profile=field_profiles[0],
            color="black",
        )
    else:
        _plot_series(
            axs_grid=axs_grid,
            field_profiles=field_profiles,
            cmap_name=plot_args.cmap_name,
        )
    _style_axes(
        axs_grid=axs_grid,
        comp_labels=field_profiles[0].comp_labels,
        axes=plot_args.axes_to_slice,
    )
    fig_path = plot_args.fig_dir / f"{plot_args.field_name}_profiles.png"
    plot_manager.save_figure(
        fig=fig,
        fig_path=fig_path,
        verbose=plot_args.verbose,
    )


##
## === OPERATOR
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
    }

    def __init__(
        self,
        *,
        input_dir: Path,
        fields_to_plot: list[str],
        components_to_plot: list[Axis],
        axes_to_slice: list[Axis],
        verbose: bool = True,
    ):
        valid_fields = set(self.VALID_FIELDS.keys())
        if not fields_to_plot or not set(fields_to_plot).issubset(valid_fields):
            raise ValueError(f"Provide fields via -f from: {sorted(valid_fields)}")
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
        self.components_to_plot = components_to_plot  # ignored for scalars
        self.axes_to_slice = axes_to_slice
        self.verbose = bool(verbose)

    def run(
        self,
    ) -> None:
        dataset_dirs = helpers.resolve_dataset_dirs(self.input_dir)
        if not dataset_dirs:
            return
        fig_dir = dataset_dirs[0].parent
        for field_name in self.fields_to_plot:
            field_meta = self.VALID_FIELDS[field_name]
            plot_args = PlotArgs(
                fig_dir=Path(fig_dir),
                field_name=field_name,
                dataset_dirs=dataset_dirs,
                components_to_plot=self.components_to_plot,
                axes_to_slice=self.axes_to_slice,
                field_loader=field_meta["loader"],
                cmap_name=field_meta["cmap"],
                verbose=self.verbose,
            )
            _plot_field(plot_args)


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
        verbose=True,
    )
    plotter.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
