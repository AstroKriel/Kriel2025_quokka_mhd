## { SCRIPT

##
## === DEPENDENCIES
##

import numpy
from pathlib import Path
from jormi.ww_plots import plot_manager, add_color
from utils import helpers, load_quokka_dataset

##
## === HELPERS
##


def _compute_centers(
    domain,
    axis: str,
):
    ## return 1-D cell-center coordinates along chosen domain-axis
    (x0, x1), (y0, y1), (z0, z1) = domain.domain_bounds
    nx, ny, nz = domain.resolution
    if axis == "x":
        dx = (x1 - x0) / nx
        return x0 + (numpy.arange(nx) + 0.5) * dx
    if axis == "y":
        dy = (y1 - y0) / ny
        return y0 + (numpy.arange(ny) + 0.5) * dy
    if axis == "z":
        dz = (z1 - z0) / nz
        return z0 + (numpy.arange(nz) + 0.5) * dz
    raise ValueError("axis must be one of: 'x', 'y', 'z'")


def _extract_midline(
    component_3d: numpy.ndarray,
    axis: str,
):
    ## extract 1-D profile along axis through the middle of the other two axes
    nx, ny, nz = component_3d.shape
    ix = nx // 2
    iy = ny // 2
    iz = nz // 2
    if axis == "x":
        return component_3d[:, iy, iz]
    if axis == "y":
        return component_3d[ix, :, iz]
    if axis == "z":
        return component_3d[ix, iy, :]
    raise ValueError("axis must be one of: 'x', 'y', 'z'")


def _plot_profile_along_axis(
    ax,
    component_3d: numpy.ndarray,
    domain,
    axis: str,
    color,
):
    ## compute centerline profile along axis and draw a single line
    x = _compute_centers(domain, axis)
    centerline = _extract_midline(component_3d, axis)
    ax.plot(x, centerline, lw=2.0, color=color)


def _maybe_subsample_dirs(
    dataset_dirs,
    target_max: int = 25,
):
    ## subsample many snapshots to a fixed count for readability
    n = len(dataset_dirs)
    if n <= target_max:
        return dataset_dirs
    idx = numpy.linspace(0, n - 1, num=target_max, dtype=int)
    return [dataset_dirs[i] for i in idx]


##
## === OPERATOR CLASS
##


class Plotter:

    def __init__(
        self,
        input_dir: Path,
        components: list[str],
        axes: list[str],
    ):
        ## validate inputs and normalise to x/y/z
        _valid = {"x", "y", "z"}
        if not components or not axes:
            raise ValueError("You must provide at least one component (-c) and one axis (-a).")
        if not set(components).issubset(_valid) or not set(axes).issubset(_valid):
            raise ValueError("Components and axes must be chosen from: x y z.")
        self.input_dir = input_dir
        self.components = list(components)
        self.axes = list(axes)
        self.comp_to_idx = {"x": 0, "y": 1, "z": 2}
        self.comp_labels = {"x": r"$b_x$", "y": r"$b_y$", "z": r"$b_z$"}

    def run(
        self,
    ) -> None:
        dataset_dirs, single_snapshot = self._resolve_dataset_dirs(self.input_dir)
        num_rows = len(self.components)
        num_cols = len(self.axes)
        fig, axs = plot_manager.create_figure(
            num_rows=num_rows,
            num_cols=num_cols,
            share_x=False,
            y_spacing=0.15,
            x_spacing=0.1,
        )
        ## normalise axis for 2D indexing [row, col]
        if (num_rows == 1) and (num_cols == 1):
            axs_grid = [[axs]]
        elif num_rows == 1:
            axs_grid = [list(axs)]
        elif num_cols == 1:
            axs_grid = [[ax] for ax in axs]
        else:
            axs_grid = axs
        if single_snapshot:
            self._plot_single(axs_grid, dataset_dirs[0])
            self._style_axes(axs_grid)
            self._save(fig, out_name="b_profiles.png")
        else:
            dataset_dirs = _maybe_subsample_dirs(dataset_dirs, target_max=25)
            self._plot_series(axs_grid, dataset_dirs)
            self._style_axes(axs_grid)
            self._add_series_colorbar(axs_grid, num_snapshots=len(dataset_dirs))
        self._save(fig, out_name="b_profiles.png")

    def _plot_single(
        self,
        axs_grid,
        dataset_dir: Path,
    ) -> None:
        ## single snapshot: black centerline per subplot
        with load_quokka_dataset.QuokkaDataset(dataset_dir=dataset_dir) as ds:
            vfield_b = ds.load_magnetic_field()
            domain = ds.load_domain()
        for row_index, comp in enumerate(self.components):
            comp_idx = self.comp_to_idx[comp]
            comp_data = vfield_b.data[comp_idx]  ## (nx, ny, nz)
            for col_index, axis in enumerate(self.axes):
                ax = axs_grid[row_index][col_index]
                _plot_profile_along_axis(ax, comp_data, domain, axis, color="black")

    def _plot_series(
        self,
        axs_grid,
        dataset_dirs: list[Path],
    ) -> None:
        ## multiple snapshots: time-coloured centerlines with shared colorbar
        cmap, norm = add_color.create_cmap(
            cmap_name="Blues",
            cmin=0.25,
            vmin=0,
            vmax=len(dataset_dirs) - 1,
        )
        for t_index, dataset_dir in enumerate(dataset_dirs):
            with load_quokka_dataset.QuokkaDataset(dataset_dir=dataset_dir) as ds:
                vfield_b = ds.load_magnetic_field()
                domain = ds.load_domain()
            color = cmap(norm(t_index))
            for row_index, comp in enumerate(self.components):
                comp_idx = self.comp_to_idx[comp]
                comp_data = vfield_b.data[comp_idx]
                for col_index, axis in enumerate(self.axes):
                    ax = axs_grid[row_index][col_index]
                    _plot_profile_along_axis(ax, comp_data, domain, axis, color=color)

    def _add_series_colorbar(
        self,
        axs_grid,
        num_snapshots: int,
    ) -> None:
        ## attach colorbar on the bottom-right axes for series plots
        cmap, norm = add_color.create_cmap(
            cmap_name="Blues",
            cmin=0.25,
            vmin=0,
            vmax=num_snapshots - 1,
        )
        ax_ref = axs_grid[-1][-1]
        add_color.add_cbar_from_cmap(
            ax=ax_ref,
            cmap=cmap,
            norm=norm,
            side="right",
            ax_percentage=0.05,
        )

    @staticmethod
    def _resolve_dataset_dirs(
        input_dir: Path,
    ) -> tuple[list[Path], bool]:
        ## determine whether input is a single snapshot or a series
        if "plt" in input_dir.name:
            dataset_dirs = [input_dir]
            return dataset_dirs, True
        dataset_dirs = helpers.get_latest_dataset_dirs(sim_dir=input_dir)
        assert len(dataset_dirs) != 0
        return dataset_dirs, len(dataset_dirs) == 1

    def _style_axes(
        self,
        axs_grid,
    ) -> None:
        ## apply titles, labels, and hide interior tick labels
        for row_index, comp in enumerate(self.components):
            for col_index, axis in enumerate(self.axes):
                ax = axs_grid[row_index][col_index]
                if col_index == 0:
                    ax.set_ylabel(self.comp_labels[comp])
                ## hide x tick labels for all but the bottom row
                if row_index != len(axs_grid) - 1:
                    ax.set_xticklabels([])
                ## hide y tick labels for all but the left column
                if col_index != 0:
                    ax.set_yticklabels([])
        for col_index, axis in enumerate(self.axes):
            ax = axs_grid[-1][col_index]
            ax.set_xlabel(axis)

    def _save(
        self,
        fig,
        out_name: str,
    ) -> None:
        fig_path = Path(self.input_dir) / out_name
        plot_manager.save_figure(fig=fig, fig_path=fig_path)


##
## === MAIN PROGRAM
##


def main():
    args = helpers.get_user_input()
    plotter = Plotter(
        input_dir=args.dir,
        components=args.components,
        axes=args.axes,
    )
    plotter.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
