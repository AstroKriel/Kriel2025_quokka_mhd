## { SCRIPT

##
## === DEPENDENCIES
##

import numpy
from typing import Literal
from pathlib import Path
from dataclasses import dataclass
from jormi.utils import type_utils
from jormi.ww_plots import plot_manager, add_color
from jormi.ww_fields import field_types
from ww_quokka_sims.sim_io import load_dataset
import utils

##
## === DATA TYPES
##

Axis = Literal["x", "y", "z"]

LOOKUP_AXIS_INDEX: dict[Axis, int] = {"x": 0, "y": 1, "z": 2}

##
## === DATA CLASSES
##


@dataclass(frozen=True)
class CompProfile:
    sim_time: float
    comp_label: str
    axis_labels: list[Axis]
    x_array_by_axis: list[numpy.ndarray]
    y_array_by_axis: list[numpy.ndarray]

    @property
    def num_axes(
        self,
    ) -> int:
        return len(self.axis_labels)

    def get(
        self,
        *,
        axis_index: int,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        return (
            self.x_array_by_axis[axis_index],
            self.y_array_by_axis[axis_index],
        )


##
## === OPERATOR CLASSES
##


class ComputeCompProfiles:

    def __init__(
        self,
        *,
        dataset_dirs: list[Path],
        field_name: str,
        field_loader: str,
        comps_to_plot: tuple[Axis, ...],
        axes_to_slice: tuple[Axis, ...],
    ):
        self.dataset_dirs = dataset_dirs
        self.field_name = field_name
        self.field_loader = field_loader
        self.comps_to_plot = comps_to_plot
        self.axes_to_slice = axes_to_slice

    @staticmethod
    def _compute_cell_centers(
        *,
        uniform_domain: field_types.UniformDomain,
        axis_to_slice: Axis,
    ) -> numpy.ndarray:
        (x_min, _), (y_min, _), (z_min, _) = uniform_domain.domain_bounds
        num_cells_x, num_cells_y, num_cells_z = uniform_domain.resolution
        cell_width_x, cell_width_y, cell_width_z = uniform_domain.cell_widths
        if axis_to_slice == "x": return x_min + (numpy.arange(num_cells_x) + 0.5) * cell_width_x
        if axis_to_slice == "y": return y_min + (numpy.arange(num_cells_y) + 0.5) * cell_width_y
        if axis_to_slice == "z": return z_min + (numpy.arange(num_cells_z) + 0.5) * cell_width_z
        raise ValueError("axis must be one of: x, y, z")

    @staticmethod
    def _extract_1d_midplane_profile(
        *,
        data_3d: numpy.ndarray,
        axis_to_slice: Axis,
    ) -> numpy.ndarray:
        num_cells_x, num_cells_y, num_cells_z = data_3d.shape
        slice_index_x = num_cells_x // 2
        slice_index_y = num_cells_y // 2
        slice_index_z = num_cells_z // 2
        if axis_to_slice == "x": return data_3d[:, slice_index_y, slice_index_z]
        if axis_to_slice == "y": return data_3d[slice_index_x, :, slice_index_z]
        if axis_to_slice == "z": return data_3d[slice_index_x, slice_index_y, :]
        raise ValueError("axis must be one of: x, y, z")

    @staticmethod
    def _get_sim_time(
        *,
        field: field_types.ScalarField | field_types.VectorField,
    ) -> float:
        sim_time = field.sim_time
        type_utils.ensure_finite_float(
            var_obj=sim_time,
            var_name="sim_time",
            allow_none=False,
        )
        assert sim_time is not None
        return float(sim_time)

    def _compute_scalar_profiles(
        self,
        *,
        field: field_types.ScalarField,
        uniform_domain: field_types.UniformDomain,
    ) -> list[CompProfile]:
        field_types.ensure_sfield(field)
        sim_time = self._get_sim_time(field=field)
        axis_labels = list(self.axes_to_slice)
        x_array_by_axis: list[numpy.ndarray] = []
        y_array_by_axis: list[numpy.ndarray] = []
        for axis_to_slice in axis_labels:
            x_positions = ComputeCompProfiles._compute_cell_centers(
                uniform_domain=uniform_domain,
                axis_to_slice=axis_to_slice,
            )
            field_profile = ComputeCompProfiles._extract_1d_midplane_profile(
                data_3d=field.data,
                axis_to_slice=axis_to_slice,
            )
            x_array_by_axis.append(x_positions)
            y_array_by_axis.append(field_profile)
        return [
            CompProfile(
                sim_time=sim_time,
                axis_labels=axis_labels,
                comp_label=field.field_label,
                x_array_by_axis=x_array_by_axis,
                y_array_by_axis=y_array_by_axis,
            ),
        ]

    def _compute_vector_profiles(
        self,
        *,
        field: field_types.VectorField,
        uniform_domain: field_types.UniformDomain,
    ) -> list[CompProfile]:
        if len(self.comps_to_plot) == 0:
            raise ValueError(f"Vector field '{self.field_name}' requires at least one component via -c")
        field_types.ensure_vfield(field)
        sim_time = self._get_sim_time(field=field)
        comp_names = sorted(self.comps_to_plot)
        axis_labels = list(self.axes_to_slice)
        comp_profiles: list[CompProfile] = []
        for comp_name in comp_names:
            comp_label = rf"$({self.field_name})_{{{comp_name}}}$"
            x_array_by_axis: list[numpy.ndarray] = []
            y_array_by_axis: list[numpy.ndarray] = []
            for axis_to_slice in axis_labels:
                x_positions = ComputeCompProfiles._compute_cell_centers(
                    uniform_domain=uniform_domain,
                    axis_to_slice=axis_to_slice,
                )
                comp_data_3d = field.data[LOOKUP_AXIS_INDEX[comp_name]]
                comp_profile = ComputeCompProfiles._extract_1d_midplane_profile(
                    data_3d=comp_data_3d,
                    axis_to_slice=axis_to_slice,
                )
                x_array_by_axis.append(x_positions)
                y_array_by_axis.append(comp_profile)
            comp_profiles.append(
                CompProfile(
                    sim_time=sim_time,
                    axis_labels=axis_labels,
                    comp_label=comp_label,
                    x_array_by_axis=x_array_by_axis,
                    y_array_by_axis=y_array_by_axis,
                ),
            )
        return comp_profiles

    def run(
        self,
    ) -> dict[str, list[CompProfile]]:
        comp_profiles_lookup: dict[str, list[CompProfile]] = {}
        for dataset_dir in self.dataset_dirs:
            with load_dataset.QuokkaDataset(dataset_dir=dataset_dir, verbose=False) as ds:
                uniform_domain = ds.load_uniform_domain()
                loader_fn = getattr(ds, self.field_loader)
                field = loader_fn()  # ScalarField or VectorField
            if isinstance(field, field_types.ScalarField):
                comp_profiles = self._compute_scalar_profiles(
                    field=field,
                    uniform_domain=uniform_domain,
                )
            elif isinstance(field, field_types.VectorField):
                comp_profiles = self._compute_vector_profiles(
                    field=field,
                    uniform_domain=uniform_domain,
                )
            else:
                raise ValueError(f"{self.field_name} is an unrecognised field type.")
            for comp_profile in comp_profiles:
                comp_label = comp_profile.comp_label
                if comp_label not in comp_profiles_lookup:
                    comp_profiles_lookup[comp_label] = []
                comp_profiles_lookup[comp_label].append(comp_profile)
        for comp_label in comp_profiles_lookup:
            comp_profiles_lookup[comp_label].sort(key=lambda item: item.sim_time)
        return comp_profiles_lookup


class RenderCompProfiles:

    def __init__(
        self,
        *,
        dataset_dirs: list[Path],
        field_name: str,
        comps_to_plot: tuple[Axis, ...],
        axes_to_slice: tuple[Axis, ...],
        field_loader: str,
        cmap_name: str,
        fig_dir: Path,
        verbose: bool = False,
    ):
        self.dataset_dirs = dataset_dirs
        self.fig_dir = Path(fig_dir)
        self.field_name = field_name
        self.comps_to_plot = comps_to_plot
        self.axes_to_slice = axes_to_slice
        self.field_loader = field_loader
        self.cmap_name = cmap_name
        self.verbose = bool(verbose)

    @staticmethod
    def _style_axs(
        *,
        axs_grid,
        comp_labels: list[str],
        axis_labels: list[Axis],
    ) -> None:
        for row_index, comp_label in enumerate(comp_labels):
            for col_index, axis_label in enumerate(axis_labels):
                ax = axs_grid[row_index][col_index]
                if col_index == 0:
                    ax.set_ylabel(comp_label)
                ax.set_xlabel(axis_label)

    @staticmethod
    def _plot_comp_profile(
        *,
        axs_row,
        comp_profile: CompProfile,
        color,
    ) -> None:
        for axis_index in range(comp_profile.num_axes):
            ax = axs_row[axis_index]
            x, y = comp_profile.get(axis_index=axis_index)
            ax.plot(x, y, lw=2.0, color=color)

    def _plot_series_row(
        self,
        *,
        axs_row,
        comp_profiles: list[CompProfile],
    ) -> None:
        cmap, norm = add_color.create_cmap(
            cmap_name=self.cmap_name,
            cmin=0.25,
            vmin=0,
            vmax=max(
                0,
                len(comp_profiles) - 1,
            ),
        )
        for time_index, comp_profile in enumerate(comp_profiles):
            color = cmap(norm(time_index))
            RenderCompProfiles._plot_comp_profile(
                axs_row=axs_row,
                comp_profile=comp_profile,
                color=color,
            )
        add_color.add_cbar_from_cmap(
            ax=axs_row[-1],
            label=r"snapshot index",
            cmap=cmap,
            norm=norm,
            side="right",
            ax_percentage=0.05,
        )

    def run(
        self,
    ) -> None:
        compute_comp_profiles = ComputeCompProfiles(
            dataset_dirs=self.dataset_dirs,
            field_name=self.field_name,
            field_loader=self.field_loader,
            comps_to_plot=self.comps_to_plot,
            axes_to_slice=self.axes_to_slice,
        )
        comp_profiles_lookup = compute_comp_profiles.run()
        if not comp_profiles_lookup:
            return
        comp_labels = list(comp_profiles_lookup.keys())
        axis_labels = comp_profiles_lookup[comp_labels[0]][0].axis_labels
        num_rows = len(comp_labels)
        num_cols = len(axis_labels)
        fig, axs_grid = utils.create_figure(
            num_rows=num_rows,
            num_cols=num_cols,
        )
        for row_index, comp_label in enumerate(comp_labels):
            comp_profiles = comp_profiles_lookup[comp_label]
            if len(comp_profiles) == 1:
                RenderCompProfiles._plot_comp_profile(
                    axs_row=axs_grid[row_index],
                    comp_profile=comp_profiles[0],
                    color="black",
                )
            else:
                self._plot_series_row(
                    axs_row=axs_grid[row_index],
                    comp_profiles=comp_profiles,
                )
        RenderCompProfiles._style_axs(
            axs_grid=axs_grid,
            comp_labels=comp_labels,
            axis_labels=axis_labels,
        )
        fig_path = self.fig_dir / f"{self.field_name}_profiles.png"
        plot_manager.save_figure(
            fig=fig,
            fig_path=fig_path,
            verbose=self.verbose,
        )


class ScriptInterface:

    def __init__(
        self,
        *,
        input_dir: Path,
        fields_to_plot: list[str],
        comps_to_plot: tuple[Axis, ...] | list[Axis] | None,
        axes_to_slice: tuple[Axis, ...] | list[Axis] | None,
        verbose: bool = True,
    ):
        valid_fields = set(utils.QUOKKA_FIELDS_LOOKUP.keys())
        if not fields_to_plot or not set(fields_to_plot).issubset(valid_fields):
            raise ValueError(f"Provide fields via -f from: {sorted(valid_fields)}")
        valid_axes: set[Axis] = {"x", "y", "z"}
        if comps_to_plot is None:
            comps_to_plot = ("x", "y", "z")
        elif not set(comps_to_plot).issubset(valid_axes):
            raise ValueError("Provide one or more components (via -c) from: x, y, z")
        if axes_to_slice is None:
            axes_to_slice = ("x", "y", "z")
        elif not set(axes_to_slice).issubset(valid_axes):
            raise ValueError("Provide one or more axes (via -a) from: x, y, z")
        self.input_dir = Path(input_dir)
        self.fields_to_plot = type_utils.as_tuple(seq_obj=fields_to_plot)
        self.comps_to_plot = type_utils.as_tuple(seq_obj=comps_to_plot)
        self.axes_to_slice = type_utils.as_tuple(seq_obj=axes_to_slice)
        self.verbose = bool(verbose)

    def run(
        self,
    ) -> None:
        dataset_dirs = utils.resolve_dataset_dirs(self.input_dir)
        if not dataset_dirs:
            return
        fig_dir = dataset_dirs[0].parent
        for field_name in self.fields_to_plot:
            field_meta = utils.QUOKKA_FIELDS_LOOKUP[field_name]
            render_comp_profiles = RenderCompProfiles(
                dataset_dirs=dataset_dirs,
                fig_dir=fig_dir,
                field_name=field_name,
                comps_to_plot=self.comps_to_plot,
                axes_to_slice=self.axes_to_slice,
                field_loader=field_meta["loader"],
                cmap_name=field_meta["cmap"],
                verbose=self.verbose,
            )
            render_comp_profiles.run()


##
## === PROGRAM MAIN
##


def main():
    args = utils.get_user_args()
    script_interface = ScriptInterface(
        input_dir=args.dir,
        fields_to_plot=args.fields,
        comps_to_plot=args.comps,
        axes_to_slice=args.axes,
        verbose=True,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
