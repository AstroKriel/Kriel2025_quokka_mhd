## { SCRIPT

##
## === DEPENDENCIES
##

import numpy
from pathlib import Path
from jormi.ww_plots import plot_manager, add_color
from ww_quokka_sims.sim_io import load_dataset
from utils import helpers

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


def _extract_profile(
    sfield: numpy.ndarray,
    axis: str,
):
    ## extract 1-D profile along axis through the middle of the other two axes
    nx, ny, nz = sfield.shape
    ix = nx // 2
    iy = ny // 2
    iz = nz // 2
    if axis == "x": return sfield[:, iy, iz]
    if axis == "y": return sfield[ix, :, iz]
    if axis == "z": return sfield[ix, iy, :]
    raise ValueError("axis must be one of: 'x', 'y', 'z'")


def _plot_profile_along_axis(
    ax,
    sfield: numpy.ndarray,
    domain,
    axis: str,
    color,
):
    x_centers = _compute_centers(domain, axis)
    y_profile = _extract_profile(sfield, axis)
    ax.plot(x_centers, y_profile, lw=2.0, color=color)


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
            "fig_name": "b_profiles.png",
        },
        "vel": {
            "loader": "load_velocity_vfield",
            "labels": {
                "x": r"$v_x$",
                "y": r"$v_y$",
                "z": r"$v_z$",
            },
            "cmap": "Oranges",
            "fig_name": "v_profiles.png",
        },
    }

    def __init__(
        self,
        input_dir: Path,
        fields: list[str],
        comp_names: list[str],
        axes: list[str],
    ):
        _valid_axes = {"x", "y", "z"}
        _valid_fields = set(self.VALID_FIELDS.keys())
        if not fields or not comp_names or not axes:
            raise ValueError(
                "You must provide at least one field (-f vel/mag), one component (-c), and one axis (-a).",
            )
        if not set(fields).issubset(_valid_fields):
            raise ValueError(f"Fields must be chosen from: {sorted(_valid_fields)}.")
        if not set(comp_names).issubset(_valid_axes) or not set(axes).issubset(_valid_axes):
            raise ValueError("Components and axes must be chosen from: x y z.")
        self.input_dir = input_dir
        self.fields = list(fields)
        self.comp_names = list(comp_names)
        self.axes = list(axes)
        self.comp_to_idx = {"x": 0, "y": 1, "z": 2}

    def run(
        self,
    ) -> None:
        dataset_dirs = helpers.resolve_dataset_dirs(self.input_dir)
        for field in self.fields:
            self._plot_field(field, dataset_dirs[:20])

    def _plot_field(
        self,
        field,
        dataset_dirs,
    ):
        specs = self.VALID_FIELDS[field]
        comp_labels = specs["labels"]
        cmap_name = specs["cmap"]
        fig_name = specs["fig_name"]
        num_rows = len(self.comp_names)
        num_cols = len(self.axes)
        fig, axs = plot_manager.create_figure(
            num_rows=num_rows,
            num_cols=num_cols,
            share_x=False,
            y_spacing=0.25,
            x_spacing=0.25,
        )
        axs_grid = helpers.get_axs_grid(axs, num_rows, num_cols)
        if len(dataset_dirs) == 1:
            self._plot_snapshot(axs_grid, dataset_dirs[0], field, color="black")
            self._style_axes(axs_grid, comp_labels)
            self._save(fig, fig_name=fig_name)
        else:
            dataset_dirs_sub = helpers.subsample_dirs(dataset_dirs, target_max=25)
            self._plot_series(axs_grid, dataset_dirs_sub, field, cmap_name=cmap_name)
            self._style_axes(axs_grid, comp_labels)
            self._add_series_colorbar(axs_grid, num_snapshots=len(dataset_dirs_sub), cmap_name=cmap_name)
            self._save(fig, fig_name=fig_name)

    def _load_vfield(
        self,
        ds,
        field: str,
    ):
        specs = self.VALID_FIELDS[field]
        loader_name = specs["loader"]
        loader_fn = getattr(ds, loader_name)
        return loader_fn()

    def _plot_snapshot(
        self,
        axs_grid,
        dataset_dir: Path,
        field: str,
        color: str | tuple[float, float, float, float] = "black",
    ) -> None:
        ## single snapshot: one colored profile per subplot
        with load_dataset.QuokkaDataset(dataset_dir=dataset_dir) as ds:
            vfield = self._load_vfield(ds, field)
            domain = ds.load_domain()
        for row_index, comp_name in enumerate(self.comp_names):
            comp_index = self.comp_to_idx[comp_name]
            comp_data = vfield.data[comp_index]
            for col_index, axis in enumerate(self.axes):
                ax = axs_grid[row_index][col_index]
                _plot_profile_along_axis(ax, comp_data, domain, axis, color=color)

    def _plot_series(
        self,
        axs_grid,
        dataset_dirs: list[Path],
        field: str,
        cmap_name: str,
    ) -> None:
        ## multiple snapshots: time-coloured centerlines with shared colorbar
        cmap, norm = add_color.create_cmap(
            cmap_name=cmap_name,
            cmin=0.25,
            vmin=0,
            vmax=len(dataset_dirs) - 1,
        )
        for t_index, dataset_dir in enumerate(dataset_dirs):
            color = cmap(norm(t_index))
            self._plot_snapshot(axs_grid=axs_grid, dataset_dir=dataset_dir, field=field, color=color)

    def _add_series_colorbar(
        self,
        axs_grid,
        num_snapshots: int,
        cmap_name: str,
    ) -> None:
        ## attach colorbar on the bottom-right axes for series plots
        cmap, norm = add_color.create_cmap(
            cmap_name=cmap_name,
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

    def _style_axes(
        self,
        axs_grid,
        comp_labels: dict[str, str],
    ) -> None:
        ## apply titles, labels, and hide interior tick labels
        for row_index, comp in enumerate(self.comp_names):
            for col_index, axis in enumerate(self.axes):
                ax = axs_grid[row_index][col_index]
                if col_index == 0:
                    ax.set_ylabel(comp_labels[comp])
        for col_index, axis in enumerate(self.axes):
            ax = axs_grid[-1][col_index]
            ax.set_xlabel(axis)

    def _save(
        self,
        fig,
        fig_name: str,
    ) -> None:
        fig_path = Path(self.input_dir) / fig_name
        plot_manager.save_figure(fig=fig, fig_path=fig_path)


##
## === MAIN PROGRAM
##


def main():
    args = helpers.get_user_input()
    plotter = Plotter(
        input_dir=args.dir,
        fields=args.fields,
        comp_names=args.components,
        axes=args.axes,
    )
    plotter.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
