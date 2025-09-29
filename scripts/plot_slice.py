## { SCRIPT

##
## === DEPENDENCIES
##

import numpy
from jormi.ww_plots import plot_manager, plot_data, annotate_axis
from ww_quokka_sims.sim_io import load_dataset
from utils import helpers

##
## === HELPERS
##


def _compute_slice_bounds(
    domain,
    axis_to_slice: str,
):
    (x0, x1), (y0, y1), (z0, z1) = domain.domain_bounds
    if axis_to_slice == "z": return (x0, x1, y0, y1)
    if axis_to_slice == "y": return (x0, x1, z0, z1)
    if axis_to_slice == "x": return (y0, y1, z0, z1)
    raise ValueError("axis_to_slice must be one of: x, y, z")


def _compute_slice_labels(
    axis_to_slice: str,
):
    if axis_to_slice == "z": return ("x", "y")
    if axis_to_slice == "y": return ("x", "z")
    if axis_to_slice == "x": return ("y", "z")
    raise ValueError("axis_to_slice must be one of: x, y, z")


def _slice_sfield(
    sfield,
    axis_to_slice: str,
):
    nx, ny, nz = sfield.shape
    slice_index_x = nx // 2
    slice_index_y = ny // 2
    slice_index_z = nz // 2
    if axis_to_slice == "z": return sfield[:, :, slice_index_z], r"$(x, y, z=L_z/2)$"
    if axis_to_slice == "y": return sfield[:, slice_index_y, :], r"$(x, y=L_y/2, z)$"
    if axis_to_slice == "x": return sfield[slice_index_x, :, :], r"$(x=L_x/2, y, z)$"
    raise ValueError("axis_to_slice must be one of: x, y, z")


def _plot_slice(
    ax,
    vfield,
    domain,
    component: str,
    axis_to_slice: str,
    comp_labels: dict[str, str],
    cmap_name: str,
):
    component_to_index = {"x": 0, "y": 1, "z": 2}
    sfield = vfield.data[component_to_index[component]]
    sfield_slice, slice_label = _slice_sfield(sfield, axis_to_slice)
    min_value = numpy.min(sfield_slice)
    max_value = numpy.max(sfield_slice)
    plot_data.plot_sfield_slice(
        ax=ax,
        field_slice=sfield_slice,
        axis_bounds=_compute_slice_bounds(domain, axis_to_slice),
        cmap_name=cmap_name,
        add_colorbar=True,
        cbar_label=comp_labels[component],
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
        y_pos=0.05,
        x_alignment="center",
        y_alignment="bottom",
        label=slice_label,
        fontsize=16,
        box_alpha=0.5,
        add_box=True,
    )


##
## === OPERATOR CLASS
##


class Plotter:

    VALID_FIELDS = {
        "mag": {
            "loader": "load_magnetic_vfield",
            "labels": {
                "x": r"$b_x$",
                "y": r"$b_y$",
                "z": r"$b_z$",
            },
            "cmap": "Blues",
            "fig_prefix": "b_slice_",
        },
        "vel": {
            "loader": "load_velocity_vfield",
            "labels": {
                "x": r"$v_x$",
                "y": r"$v_y$",
                "z": r"$v_z$",
            },
            "cmap": "Oranges",
            "fig_prefix": "v_slice_",
        },
    }

    def __init__(
        self,
        input_dir,
        fields: list[str],
        components_to_plot: list[str],
        axes_to_slice: list[str],
    ):
        valid_axes = {"x", "y", "z"}
        valid_fields = set(self.VALID_FIELDS.keys())
        if not fields or not set(fields).issubset(valid_fields):
            raise ValueError(f"Provide one or more fields (via -f) from: {sorted(valid_fields)}")
        if not axes_to_slice or not set(axes_to_slice).issubset(valid_axes):
            raise ValueError("Provide one or more axes (via -a) from: x, y, z")
        if not components_to_plot or not set(components_to_plot).issubset(valid_axes):
            raise ValueError("Provide one or more components (via -c) from: x, y, z")
        self.input_dir = input_dir
        self.fields = list(fields)
        self.axes_to_slice = list(axes_to_slice)
        self.components_to_plot = list(components_to_plot)

    def run(
        self,
    ) -> None:
        dataset_dirs = helpers.resolve_dataset_dirs(self.input_dir)
        for field_name in self.fields:
            if len(dataset_dirs) == 1:
                fig_dir = dataset_dirs[0].parent
                self._plot_snapshot(
                    field_name=field_name,
                    dataset_dir=dataset_dirs[0],
                    fig_dir=fig_dir,
                )
            else:
                fig_dir = self.input_dir
                for dataset_dir in dataset_dirs:
                    self._plot_snapshot(
                        field_name=field_name,
                        dataset_dir=dataset_dir,
                        fig_dir=fig_dir,
                    )

    def _load_vfield(
        self,
        ds,
        field_name: str,
    ):
        field_meta = self.VALID_FIELDS[field_name]
        loader_name = field_meta["loader"]
        return getattr(ds, loader_name)()

    def _plot_snapshot(
        self,
        field_name: str,
        dataset_dir,
        fig_dir,
    ) -> None:
        field_meta = self.VALID_FIELDS[field_name]
        comp_labels = field_meta["labels"]
        cmap_name = field_meta["cmap"]
        num_rows = len(self.components_to_plot)
        num_cols = len(self.axes_to_slice)
        fig, axs = plot_manager.create_figure(
            num_rows=num_rows,
            num_cols=num_cols,
            y_spacing=0.35,
            x_spacing=0.25,
        )
        axs_grid = helpers.get_axs_grid(axs, num_rows, num_cols)
        with load_dataset.QuokkaDataset(dataset_dir=dataset_dir) as dataset:
            vfield = self._load_vfield(dataset, field_name)
            domain = dataset.load_domain()
        for row_index, component in enumerate(self.components_to_plot):
            for col_index, axis_to_slice in enumerate(self.axes_to_slice):
                ax = axs_grid[row_index][col_index]
                _plot_slice(
                    ax=ax,
                    vfield=vfield,
                    domain=domain,
                    component=component,
                    axis_to_slice=axis_to_slice,
                    comp_labels=comp_labels,
                    cmap_name=cmap_name,
                )
        self._style_slice_axes(axs_grid)
        index_label = dataset_dir.name.split("plt")[1]
        fig_prefix = field_meta["fig_prefix"]
        fig_name = f"{fig_prefix}{index_label}.png"
        fig_path = fig_dir / fig_name
        plot_manager.save_figure(fig=fig, fig_path=fig_path)

    def _style_slice_axes(
        self,
        axs_grid,
    ) -> None:
        for row_index, _ in enumerate(self.components_to_plot):
            for col_index, axis_to_slice in enumerate(self.axes_to_slice):
                ax = axs_grid[row_index][col_index]
                xlabel, ylabel = _compute_slice_labels(axis_to_slice)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)


##
## === MAIN PROGRAM
##


def main():
    args = helpers.get_user_input()
    plotter = Plotter(
        input_dir=args.dir,
        fields=args.fields,
        components_to_plot=args.components,
        axes_to_slice=args.axes,
    )
    plotter.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
