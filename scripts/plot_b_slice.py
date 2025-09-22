## { SCRIPT

##
## === DEPENDENCIES
##

import numpy
from jormi.ww_plots import plot_manager, plot_data, annotate_axis
from utils import helpers, load_quokka_dataset

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
    ix = 0 # nx // 2
    iy = 0 # ny // 2
    iz = 0 # nz // 2
    if axis_to_slice == "z": return sfield[:, :, iz], r"$(x, y, z=0)$"
    if axis_to_slice == "y": return sfield[:, iy, :], r"$(x, y=0, z)$"
    if axis_to_slice == "x": return sfield[ix, :, :], r"$(x=0, y, z)$"
    raise ValueError("axis_to_slice must be one of: x, y, z")


def _plot_slice(
    ax,
    vfield,
    domain,
    component: str,
    axis_to_slice: str,
):
    component_to_index = {"x": 0, "y": 1, "z": 2}
    component_labels = {"x": r"$|b_x|$", "y": r"$|b_y|$", "z": r"$|b_z|$"}
    sfield = vfield.data[component_to_index[component]]
    sfield_slice, label = _slice_sfield(sfield, axis_to_slice)
    plot_data.plot_sfield_slice(
        ax=ax,
        field_slice=sfield_slice,
        axis_bounds=_compute_slice_bounds(domain, axis_to_slice),
        cmap_name="cmr.arctic",
        add_colorbar=True,
        cbar_label=component_labels[component],
        cbar_side="right",
    )
    annotate_axis.add_text(
        ax=ax,
        x_pos = 0.5,
        y_pos = 0.95,
        x_alignment="center",
        y_alignment="top",
        label=f"({numpy.min(sfield_slice):.2e}, {numpy.max(sfield_slice):.2e})",
        fontsize=16,
        box_alpha=0.5,
        add_box=True,
    )
    annotate_axis.add_text(
        ax=ax,
        x_pos = 0.5,
        y_pos = 0.05,
        x_alignment="center",
        y_alignment="bottom",
        label=label,
        fontsize=16,
        box_alpha=0.5,
        add_box=True,
    )


def _cast_axs_to_grid(
    axs,
    num_rows: int,
    num_cols: int,
):
    if (num_rows == 1) and (num_cols == 1):
        return [[axs]]
    if num_rows == 1:
        return [list(axs)]
    if num_cols == 1:
        return [[ax] for ax in axs]
    return axs


##
## === OPERATOR CLASS
##


class Plotter:

    def __init__(
        self,
        input_dir,
        components_to_plot: list[str],
        axes_to_slice: list[str],
    ):
        valid_axes = {"x", "y", "z"}
        if not axes_to_slice or not set(axes_to_slice).issubset(valid_axes):
            raise ValueError("Provide one or more axes (via -a) from: x, y, z")
        if not components_to_plot or not set(components_to_plot).issubset(valid_axes):
            raise ValueError("Provide one or more components (via -c) from: x, y, z")
        self.input_dir = input_dir
        self.axes_to_slice = list(axes_to_slice)
        self.components_to_plot = list(components_to_plot)
        self.component_labels = {"x": r"$|b_x|$", "y": r"$|b_y|$", "z": r"$|b_z|$"}

    def run(
        self,
    ) -> None:
        dataset_dirs, is_single_dataset = self._resolve_dataset_dirs(self.input_dir)
        if is_single_dataset:
            fig_dir = dataset_dirs[0].parent
            self._plot_snapshot(
                dataset_dir=dataset_dirs[0],
                fig_dir=fig_dir
            )
        else:
            fig_dir = self.input_dir
            for dataset_dir in dataset_dirs:
                self._plot_snapshot(
                    dataset_dir=dataset_dir,
                    fig_dir=fig_dir
                )

    def _plot_snapshot(
        self,
        dataset_dir,
        fig_dir,
    ) -> None:
        num_rows = len(self.components_to_plot)
        num_cols = len(self.axes_to_slice)
        fig, axs = plot_manager.create_figure(
            num_rows=num_rows,
            num_cols=num_cols,
            y_spacing=0.35,
            x_spacing=0.25,
        )
        axs_grid = _cast_axs_to_grid(axs, num_rows, num_cols)
        with load_quokka_dataset.QuokkaDataset(dataset_dir=dataset_dir) as dataset:
            vfield = dataset.load_magnetic_field()
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
                )
        self._style_slice_axes(axs_grid)
        label = dataset_dir.name.split("plt")[1]
        fig_name = f"b_slice_{label}.png"
        fig_path = fig_dir / fig_name
        plot_manager.save_figure(fig=fig, fig_path=fig_path)

    @staticmethod
    def _resolve_dataset_dirs(
        input_dir,
    ):
        if "plt" in input_dir.name:
            dataset_dirs = [input_dir]
            return dataset_dirs, True
        dataset_dirs = helpers.get_latest_dataset_dirs(sim_dir=input_dir)
        assert len(dataset_dirs) != 0
        return dataset_dirs, False

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
