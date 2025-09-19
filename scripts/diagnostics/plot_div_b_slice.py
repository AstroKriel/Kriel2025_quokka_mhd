## { SCRIPT

##
## === DEPENDENCIES
##

from jormi.ww_plots import plot_manager, plot_data
from jormi.ww_fields import field_operators
from utils import helpers, load_quokka_dataset

##
## === HELPERS
##


def _compute_plane_bounds(
    domain,
    axis: str,
):
    ## return axis bounds for the slice plane corresponding to axis
    (x0, x1), (y0, y1), (z0, z1) = domain.domain_bounds
    if axis == "z":  ## xy plane at mid-z
        return (x0, x1, y0, y1)
    if axis == "y":  ## xz plane at mid-y
        return (x0, x1, z0, z1)
    if axis == "x":  ## yz plane at mid-x
        return (y0, y1, z0, z1)
    raise ValueError("axis must be one of: x, y, z")


def _slice_through_midplane(
    sfield_div_b,
    axis: str,
):
    ## extract mid-plane slice orthogonal to axis
    n_cells_x, n_cells_y, n_cells_z = sfield_div_b.shape
    index_x = n_cells_x // 2
    index_y = n_cells_y // 2
    index_z = n_cells_z // 2
    if axis == "z":  # xy plane
        return sfield_div_b[:, :, index_x]
    if axis == "y":  # xz plane
        return sfield_div_b[:, index_y, :]
    if axis == "x":  # yz plane
        return sfield_div_b[index_z, :, :]
    raise ValueError("axis must be one of: x, y, z")


def _compute_div_b(
    vfield_b,
    domain,
):
    return field_operators.compute_vfield_divergence(
        vfield=vfield_b.data,
        domain_lengths=domain.domain_lengths,
    )


def _plot_div_b_mid_slice(
    ax,
    sfield_div_b,
    domain,
    axis: str,
):
    plot_data.plot_sfield_slice(
        ax=ax,
        field_slice=_slice_through_midplane(sfield_div_b, axis),
        axis_bounds=_compute_plane_bounds(domain, axis),
        cmap_name="cmr.arctic",
        add_colorbar=True,
        cbar_label=r"$\nabla\cdot\vec{b}$",
        cbar_side="right",
    )


def _cast_to_grid(
    axs,
    num_rows: int,
    num_cols: int,
):
    ## normalise matplotlib axes to 2-D indexing [row][col]
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
        axes: list[str],
    ):
        ## validate axes and normalise to x/y/z
        _valid = {"x", "y", "z"}
        if not axes or not set(axes).issubset(_valid):
            raise ValueError("Provide one or more axes with -a chosen from: x, y, z")
        self.input_dir = input_dir
        self.axes = list(axes)

    def run(
        self,
    ) -> None:
        ## main routine: resolve datasets, branch, style, save
        dataset_dirs, single_snapshot = self._resolve_dataset_dirs(self.input_dir)
        if single_snapshot:
            num_rows, num_cols = 1, len(self.axes)
            fig, axs = plot_manager.create_figure(num_rows=num_rows, num_cols=num_cols)
            axs_grid = _cast_to_grid(axs, num_rows, num_cols)
            self._plot_single(axs_grid, dataset_dirs[0])
        else:
            num_rows, num_cols = 2, len(self.axes)
            fig, axs = plot_manager.create_figure(num_rows=num_rows, num_cols=num_cols)
            axs_grid = _cast_to_grid(axs, num_rows, num_cols)
            self._plot_series(axs_grid, dataset_dirs)
        self._style_axes(axs_grid)
        self._save(fig, out_name="div_b_slice.png")

    def _plot_single(
        self,
        axs_grid,
        dataset_dir,
    ) -> None:
        ## single snapshot: one row with columns per requested axis
        with load_quokka_dataset.QuokkaDataset(dataset_dir=dataset_dir) as ds:
            vfield_b = ds.load_magnetic_field()
            domain = ds.load_domain()
        sfield_div_b = _compute_div_b(vfield_b, domain)
        for col_index, axis in enumerate(self.axes):
            ax = axs_grid[0][col_index]
            _plot_div_b_mid_slice(ax, sfield_div_b, domain, axis)

    def _plot_series(
        self,
        axs_grid,
        dataset_dirs,
    ) -> None:
        ## series: first and last snapshots as two rows; columns per axis
        datasets_to_plot = (dataset_dirs[0], dataset_dirs[-1])
        for row_index, dataset_dir in enumerate(datasets_to_plot):
            with load_quokka_dataset.QuokkaDataset(dataset_dir=dataset_dir) as ds:
                vfield_b = ds.load_magnetic_field()
                domain = ds.load_domain()
            sfield_div_b = _compute_div_b(vfield_b, domain)
            for col_index, axis in enumerate(self.axes):
                ax = axs_grid[row_index][col_index]
                _plot_div_b_mid_slice(ax, sfield_div_b, domain, axis)

    @staticmethod
    def _resolve_dataset_dirs(
        input_dir,
    ):
        ## determine whether input is a single snapshot or a series
        if "plt" in input_dir.name:
            dataset_dirs = [input_dir]
            return dataset_dirs, True
        dataset_dirs = helpers.get_latest_dataset_dirs(sim_dir=input_dir)
        assert len(dataset_dirs) != 0
        return dataset_dirs, (len(dataset_dirs) == 1)

    def _style_axes(
        self,
        axs_grid,
    ) -> None:
        ## apply titles by column (axis label) and hide interior tick labels
        num_rows = len(axs_grid)
        num_cols = len(axs_grid[0])
        for row_index in range(num_rows):
            for col_index in range(num_cols):
                ax = axs_grid[row_index][col_index]
                ## hide x tick labels except bottom row
                if row_index != num_rows - 1:
                    ax.set_xticklabels([])
                ## hide y tick labels except left column
                if col_index != 0:
                    ax.set_yticklabels([])

    def _save(
        self,
        fig,
        out_name: str,
    ) -> None:
        ## save figure to input directory
        fig_path = self.input_dir / out_name
        plot_manager.save_figure(fig=fig, fig_path=fig_path)


##
## === MAIN PROGRAM
##


def main():
    ## parse user inputs and run
    args = helpers.get_user_input()
    plotter = Plotter(
        input_dir=args.dir,
        axes=args.axes,
    )
    plotter.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
