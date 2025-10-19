## { SCRIPT

##
## === DEPENDENCIES
##

import numpy
from typing import Literal, NamedTuple
from pathlib import Path
from dataclasses import dataclass
from jormi.ww_io import io_manager, log_manager
from jormi.utils import parallel_utils, type_utils
from jormi.ww_plots import plot_manager, plot_data, annotate_axis
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
class FieldArgs:
    field_name: str
    field_loader: str
    cmap_name: str


class WorkerArgs(NamedTuple):
    dataset_dir: str
    field_name: str
    field_loader: str
    comps_to_plot: tuple[Axis, ...]
    axes_to_slice: tuple[Axis, ...]
    cmap_name: str
    fig_dir: str
    index_width: int
    verbose: bool


@dataclass(frozen=True)
class Dataset:
    uniform_domain: field_types.UniformDomain
    field: field_types.ScalarField | field_types.VectorField

    @property
    def sim_time(self) -> float:
        sim_time = self.field.sim_time
        type_utils.ensure_finite_float(
            var_obj=sim_time,
            var_name="sim_time",
            allow_none=False,
        )
        assert sim_time is not None
        return float(sim_time)


@dataclass(frozen=True)
class FieldComp:
    data_3d: numpy.ndarray
    label: str


@dataclass(frozen=True)
class SlicedField:
    data_2d: numpy.ndarray
    label: str
    min_value: float
    max_value: float
    axis_bounds: tuple[float, float, float, float]


##
## === HELPERS
##


def get_slice_bounds(
    *,
    uniform_domain: field_types.UniformDomain,
    axis_to_slice: Axis,
) -> tuple[float, float, float, float]:
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = uniform_domain.domain_bounds
    if axis_to_slice == "z": return (x_min, x_max, y_min, y_max)
    if axis_to_slice == "y": return (x_min, x_max, z_min, z_max)
    if axis_to_slice == "x": return (y_min, y_max, z_min, z_max)
    raise ValueError("axis_to_slice must be one of: x, y, z")


def get_slice_labels(
    axis_to_slice: Axis,
) -> tuple[str, str]:
    if axis_to_slice == "z": return ("x", "y")
    if axis_to_slice == "y": return ("x", "z")
    if axis_to_slice == "x": return ("y", "z")
    raise ValueError("axis_to_slice must be one of: x, y, z")


def slice_field(
    *,
    data_3d: numpy.ndarray,
    axis_to_slice: Axis,
    uniform_domain: field_types.UniformDomain,
) -> SlicedField:
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
    axis_bounds = get_slice_bounds(
        uniform_domain=uniform_domain,
        axis_to_slice=axis_to_slice,
    )
    min_value = float(numpy.min(data_2d))
    max_value = float(numpy.max(data_2d))
    return SlicedField(
        data_2d=data_2d,
        label=label,
        min_value=min_value,
        max_value=max_value,
        axis_bounds=axis_bounds,
    )


##
## === OPERATOR CLASSES
##


@dataclass(frozen=True)
class FieldPlotter:
    field_args: FieldArgs
    comps_to_plot: tuple[Axis, ...]
    axes_to_slice: tuple[Axis, ...]

    @staticmethod
    def plot_slice(
        *,
        ax,
        sim_time: float,
        field_slice: SlicedField,
        label: str,
        cmap_name: str,
    ) -> None:
        plot_data.plot_sfield_slice(
            ax=ax,
            field_slice=field_slice.data_2d,
            axis_bounds=field_slice.axis_bounds,
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

    def _load_dataset(
        self,
        *,
        dataset_dir: Path,
        verbose: bool,
    ) -> Dataset:
        with load_dataset.QuokkaDataset(dataset_dir=dataset_dir, verbose=verbose) as ds:
            uniform_domain = ds.load_uniform_domain()
            loader_fn = getattr(ds, self.field_args.field_loader)
            field = loader_fn()  # ScalarField or VectorField
        return Dataset(
            uniform_domain=uniform_domain,
            field=field,
        )

    def _get_field_comps(
        self,
        *,
        field: field_types.ScalarField | field_types.VectorField,
    ) -> list[FieldComp]:
        field_name = self.field_args.field_name
        if isinstance(field, field_types.ScalarField):
            return [
                FieldComp(
                    data_3d=field.data,
                    label=field_name,
                ),
            ]
        if isinstance(field, field_types.VectorField):
            if not self.comps_to_plot:
                raise ValueError(f"Vector field '{field_name}' requires at least one component via -c")
            return [
                FieldComp(
                    data_3d=field.data[LOOKUP_AXIS_INDEX[comp_name]],
                    label=rf"$({field_name})_{{{comp_name}}}$",
                ) for comp_name in sorted(self.comps_to_plot)
            ]
        raise ValueError(f"{field_name} is an unrecognised field type.")

    def _plot_field_comps(
        self,
        *,
        axs_grid,
        field_comps: list[FieldComp],
        uniform_domain: field_types.UniformDomain,
        sim_time: float,
    ) -> None:
        for row_index, field_comp in enumerate(field_comps):
            for col_index, axis_to_slice in enumerate(self.axes_to_slice):
                ax = axs_grid[row_index][col_index]
                field_slice = slice_field(
                    data_3d=field_comp.data_3d,
                    axis_to_slice=axis_to_slice,
                    uniform_domain=uniform_domain,
                )
                self.plot_slice(
                    ax=ax,
                    sim_time=sim_time,
                    field_slice=field_slice,
                    label=field_comp.label,
                    cmap_name=self.field_args.cmap_name,
                )

    def _label_axes(
        self,
        *,
        axs_grid,
    ) -> None:
        num_rows = len(axs_grid)
        for row_index in range(num_rows):
            for col_index, axis_to_slice in enumerate(self.axes_to_slice):
                ax = axs_grid[row_index][col_index]
                x_label_str, y_label_str = get_slice_labels(axis_to_slice)
                if (num_rows == 1) or (row_index == num_rows - 1):
                    ax.set_xlabel(x_label_str)
                ax.set_ylabel(y_label_str)

    def plot_dataset(
        self,
        *,
        fig_dir: Path,
        dataset_dir: Path,
        index_width: int,
        verbose: bool,
    ) -> None:
        dataset = self._load_dataset(dataset_dir=dataset_dir, verbose=verbose)
        dataset_index = int(utils.get_dataset_index(dataset_dir))
        field_comps = self._get_field_comps(field=dataset.field)
        fig, axs_grid = utils.create_figure(
            num_rows=len(field_comps),
            num_cols=len(self.axes_to_slice),
            add_cbar_space=True,
        )
        self._plot_field_comps(
            axs_grid=axs_grid,
            field_comps=field_comps,
            uniform_domain=dataset.uniform_domain,
            sim_time=dataset.sim_time,
        )
        self._label_axes(axs_grid=axs_grid)
        figure_name = f"{self.field_args.field_name}_slice_{dataset_index:0{index_width}d}.png"
        figure_path = fig_dir / figure_name
        plot_manager.save_figure(
            fig=fig,
            fig_path=figure_path,
            verbose=verbose,
        )


def render_fields_in_serial(
    *,
    fields_to_plot: tuple[str, ...],
    comps_to_plot: tuple[Axis, ...],
    axes_to_slice: tuple[Axis, ...],
    dataset_dirs: list[Path],
    fig_dir: Path,
    index_width: int,
) -> None:
    for field_name in fields_to_plot:
        field_meta = utils.QUOKKA_FIELDS_LOOKUP[field_name]
        field_args = FieldArgs(
            field_name=field_name,
            field_loader=field_meta["loader"],
            cmap_name=field_meta["cmap"],
        )
        field_plotter = FieldPlotter(
            field_args=field_args,
            comps_to_plot=comps_to_plot,
            axes_to_slice=axes_to_slice,
        )
        for dataset_dir in dataset_dirs:
            field_plotter.plot_dataset(
                fig_dir=fig_dir,
                dataset_dir=dataset_dir,
                index_width=index_width,
                verbose=False,
            )


def _plot_dataset_worker(
    *args,
) -> None:
    worker_args = WorkerArgs(*args)
    field_args = FieldArgs(
        field_name=worker_args.field_name,
        field_loader=worker_args.field_loader,
        cmap_name=worker_args.cmap_name,
    )
    field_plotter = FieldPlotter(
        field_args=field_args,
        comps_to_plot=worker_args.comps_to_plot,
        axes_to_slice=worker_args.axes_to_slice,
    )
    field_plotter.plot_dataset(
        fig_dir=Path(worker_args.fig_dir),
        dataset_dir=Path(worker_args.dataset_dir),
        index_width=int(worker_args.index_width),
        verbose=bool(worker_args.verbose),
    )


def render_fields_in_parallel(
    *,
    fields_to_plot: tuple[str, ...],
    comps_to_plot: tuple[Axis, ...],
    axes_to_slice: tuple[Axis, ...],
    dataset_dirs: list[Path],
    fig_dir: Path,
    index_width: int,
) -> None:
    grouped_worker_args: list[WorkerArgs] = []
    for field_name in fields_to_plot:
        field_meta = utils.QUOKKA_FIELDS_LOOKUP[field_name]
        for dataset_dir in dataset_dirs:
            grouped_worker_args.append(
                WorkerArgs(
                    field_name=field_name,
                    dataset_dir=str(dataset_dir),
                    field_loader=field_meta["loader"],
                    comps_to_plot=comps_to_plot,
                    axes_to_slice=axes_to_slice,
                    cmap_name=field_meta["cmap"],
                    fig_dir=str(fig_dir),
                    index_width=index_width,
                    verbose=False,
                ),
            )
    parallel_utils.run_in_parallel(
        worker_fn=_plot_dataset_worker,
        grouped_worker_args=grouped_worker_args,
        timeout_seconds=120,
        show_progress=True,
        enable_plotting=True,
    )


class ScriptInterface:

    def __init__(
        self,
        *,
        input_dir: Path,
        fields_to_plot: tuple[str, ...] | list[str] | None,
        comps_to_plot: tuple[Axis, ...] | list[Axis] | None,
        axes_to_slice: tuple[Axis, ...] | list[Axis] | None,
        use_parallel: bool = True,
        animate_only: bool = False,
    ):
        valid_fields = set(utils.QUOKKA_FIELDS_LOOKUP.keys())
        if not fields_to_plot or not set(fields_to_plot).issubset(valid_fields):
            raise ValueError(f"Provide one or more field to plot (via -f) from: {sorted(valid_fields)}")
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
        self.use_parallel = bool(use_parallel)
        self.animate_only = bool(animate_only)

    def _animate_fields(
        self,
        *,
        fig_dir: Path,
    ) -> None:
        for field_name in self.fields_to_plot:
            fig_paths = io_manager.ItemFilter(
                prefix=f"{field_name}_slice_",
                suffix=".png",
                include_folders=False,
                include_files=True,
            ).filter(directory=fig_dir)
            if len(fig_paths) < 3:
                log_manager.log_hint(
                    text=(
                        f"Skipping animation for '{field_name}': "
                        f"only found {len(fig_paths)} frame(s), but need at least 3."
                    ),
                )
                continue
            mp4_path = Path(fig_dir) / f"{field_name}_slices.mp4"
            plot_manager.animate_pngs_to_mp4(
                frames_dir=fig_dir,
                mp4_path=mp4_path,
                pattern=f"{field_name}_slice_*.png",
                fps=60,
                timeout_seconds=120,
            )

    def run(
        self,
    ) -> None:
        dataset_dirs = utils.resolve_dataset_dirs(self.input_dir)
        if not dataset_dirs:
            return
        fig_dir = dataset_dirs[0].parent
        index_width = utils.get_max_index_width(dataset_dirs)
        if not self.animate_only:
            if self.use_parallel and (len(dataset_dirs) > 5):
                render_fields_in_parallel(
                    fields_to_plot=self.fields_to_plot,
                    comps_to_plot=self.comps_to_plot,
                    axes_to_slice=self.axes_to_slice,
                    dataset_dirs=dataset_dirs,
                    fig_dir=fig_dir,
                    index_width=index_width,
                )
            else:
                render_fields_in_serial(
                    fields_to_plot=self.fields_to_plot,
                    comps_to_plot=self.comps_to_plot,
                    axes_to_slice=self.axes_to_slice,
                    dataset_dirs=dataset_dirs,
                    fig_dir=fig_dir,
                    index_width=index_width,
                )
        self._animate_fields(fig_dir=fig_dir)


##
## === PROGRAM MAIN
##


def main():
    user_args = utils.get_user_args()
    script_interface = ScriptInterface(
        input_dir=user_args.dir,
        fields_to_plot=user_args.fields,
        comps_to_plot=user_args.comps,
        axes_to_slice=user_args.axes,
        animate_only=user_args.animate_only,
        use_parallel=True,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
