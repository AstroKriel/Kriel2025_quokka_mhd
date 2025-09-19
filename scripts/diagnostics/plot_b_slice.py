## { SCRIPT

##
## === DEPENDENCIES
##

from jormi.ww_plots import plot_manager, plot_data
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


def _plane_axis_labels(
    axis: str,
):
    ## return (xlabel, ylabel) for the slice plane corresponding to axis
    if axis == "z":  ## xy plane
        return ("x", "y")
    if axis == "y":  ## xz plane
        return ("x", "z")
    if axis == "x":  ## yz plane
        return ("y", "z")
    raise ValueError("axis must be one of: x, y, z")


def _slice_midplane(
    field_3d,
    axis: str,
):
    ## extract mid-plane slice orthogonal to axis from a (nx, ny, nz) array
    nx, ny, nz = field_3d.shape
    ix = nx // 2
    iy = ny // 2
    iz = nz // 2
    if axis == "z":  ## xy plane
        return field_3d[:, :, iz]
    if axis == "y":  ## xz plane
        return field_3d[:, iy, :]
    if axis == "x":  ## yz plane
        return field_3d[ix, :, :]
    raise ValueError("axis must be one of: x, y, z")


def _row_vrange_for_component(
    vfield_b,
    component: str,
    axes: list[str],
):
    ## compute shared vmin/vmax across all requested slices for this component row
    comp_to_idx = {"x": 0, "y": 1, "z": 2}
    field_3d = abs(vfield_b.data[comp_to_idx[component]])
    vmin = None
    vmax = None
    for axis in axes:
        sl = _slice_midplane(field_3d, axis)
        smin = float(sl.min())
        smax = float(sl.max())
        vmin = smin if vmin is None else min(vmin, smin)
        vmax = smax if vmax is None else max(vmax, smax)
    ## avoid degenerate colormap range
    assert vmin is not None
    if vmax == vmin: vmax = vmin + 1e-12
    return vmin, vmax


def _plot_component_amp_mid_slice(
    ax,
    vfield_b,
    domain,
    component: str,
    axis: str,
    vmin: float,
    vmax: float,
    add_colorbar: bool,
):
    ## render a mid-plane slice of |b_component| with shared vmin/vmax; draw colorbar optionally
    comp_to_idx = {"x": 0, "y": 1, "z": 2}
    comp_labels = {"x": r"$|b_x|$", "y": r"$|b_y|$", "z": r"$|b_z|$"}
    field_3d = vfield_b.data[comp_to_idx[component]]
    plot_data.plot_sfield_slice(
        ax=ax,
        field_slice=_slice_midplane(field_3d, axis),
        axis_bounds=_compute_plane_bounds(domain, axis),
        cmap_name="cmr.arctic",
        add_colorbar=add_colorbar,
        cbar_label=comp_labels[component],
        cbar_side="right",
        cbar_bounds=(vmin, vmax),
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
    ## wrapper for plotting mid-plane slices of |b_i| for selected components × axes
    ## sim dir  -> per-time image saved in sim dir with plt-index appended
    ## data dir -> single image saved in that data dir
    ## for each row (component), share color scale across columns; only rightmost subplot draws colorbar

    def __init__(
        self,
        input_dir,
        components: list[str],
        axes: list[str],
    ):
        ## validate inputs and normalise to x/y/z
        _valid = {"x", "y", "z"}
        if not axes or not set(axes).issubset(_valid):
            raise ValueError("Provide one or more axes with -a chosen from: x, y, z")
        if not components or not set(components).issubset(_valid):
            raise ValueError("Provide one or more components with -c chosen from: x, y, z")
        self.input_dir = input_dir
        self.axes = list(axes)
        self.components = list(components)
        self.comp_labels = {"x": r"$|b_x|$", "y": r"$|b_y|$", "z": r"$|b_z|$"}

    def run(
        self,
    ) -> None:
        ## main routine: resolve datasets; if sim dir, plot each snapshot separately
        dataset_dirs, is_single_dataset = self._resolve_dataset_dirs(self.input_dir)
        if is_single_dataset:
            ## single dataset dir (plt...): one figure
            self._plot_and_save_single_dataset(dataset_dir=dataset_dirs[0])
        else:
            ## sim dir: loop over all datasets and save each as its own image
            for dataset_dir in dataset_dirs:
                self._plot_and_save_sim_snapshot(dataset_dir=dataset_dir)

    def _plot_and_save_single_dataset(
        self,
        dataset_dir,
    ) -> None:
        ## build grid rows=components, cols=axes; plot with shared row ranges; style; save in dataset dir
        num_rows, num_cols = len(self.components), len(self.axes)
        fig, axs = plot_manager.create_figure(num_rows=num_rows, num_cols=num_cols, y_spacing=0.25, x_spacing=0.25)
        axs_grid = _cast_to_grid(axs, num_rows, num_cols)
        with load_quokka_dataset.QuokkaDataset(dataset_dir=dataset_dir) as ds:
            vfield_b = ds.load_magnetic_field()
            domain = ds.load_domain()
        ## per-row shared vmin/vmax
        row_ranges = {comp: _row_vrange_for_component(vfield_b, comp, self.axes) for comp in self.components}
        for row_index, comp in enumerate(self.components):
            vmin, vmax = row_ranges[comp]
            for col_index, axis in enumerate(self.axes):
                ax = axs_grid[row_index][col_index]
                add_cbar = (col_index == num_cols - 1)
                _plot_component_amp_mid_slice(
                    ax, vfield_b, domain, comp, axis, vmin=vmin, vmax=vmax, add_colorbar=add_cbar
                )
        self._style_axes(axs_grid)
        fig_path = dataset_dir / "b_component_slice.png"
        plot_manager.save_figure(fig=fig, fig_path=fig_path)

    def _plot_and_save_sim_snapshot(
        self,
        dataset_dir,
    ) -> None:
        ## build grid rows=components, cols=axes; plot with shared row ranges; style; save in sim dir with plt-index
        num_rows, num_cols = len(self.components), len(self.axes)
        fig, axs = plot_manager.create_figure(num_rows=num_rows, num_cols=num_cols, y_spacing=0.25, x_spacing=0.25)
        axs_grid = _cast_to_grid(axs, num_rows, num_cols)
        with load_quokka_dataset.QuokkaDataset(dataset_dir=dataset_dir) as ds:
            vfield_b = ds.load_magnetic_field()
            domain = ds.load_domain()
        ## per-row shared vmin/vmax
        row_ranges = {comp: _row_vrange_for_component(vfield_b, comp, self.axes) for comp in self.components}
        for row_index, comp in enumerate(self.components):
            vmin, vmax = row_ranges[comp]
            for col_index, axis in enumerate(self.axes):
                ax = axs_grid[row_index][col_index]
                add_cbar = (col_index == num_cols - 1)
                _plot_component_amp_mid_slice(
                    ax, vfield_b, domain, comp, axis, vmin=vmin, vmax=vmax, add_colorbar=add_cbar
                )
        self._style_axes(axs_grid)
        out_name = f"b_component_slice_{dataset_dir.name}.png"  ## append plt-index
        fig_path = self.input_dir / out_name
        plot_manager.save_figure(fig=fig, fig_path=fig_path)

    @staticmethod
    def _resolve_dataset_dirs(
        input_dir,
    ):
        ## determine whether input is a single dataset dir (plt...) or a sim dir
        if "plt" in input_dir.name:
            dataset_dirs = [input_dir]
            return dataset_dirs, True
        dataset_dirs = helpers.get_latest_dataset_dirs(sim_dir=input_dir)
        assert len(dataset_dirs) != 0
        return dataset_dirs, False

    def _style_axes(
        self,
        axs_grid,
    ) -> None:
        for row_index, comp in enumerate(self.components):
            for col_index, axis in enumerate(self.axes):
                ax = axs_grid[row_index][col_index]
                xlabel, ylabel = _plane_axis_labels(axis)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)


##
## === MAIN PROGRAM
##


def main():
    ## parse user inputs and run
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
