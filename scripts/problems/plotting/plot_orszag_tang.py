## { SCRIPT

##
## === DEPENDENCIES
##

import numpy
from typing import Literal
from pathlib import Path
from jormi.utils import list_utils
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager, plot_data
from ww_quokka_sims.sim_io import load_dataset

##
## === HELPER FUNCTIONS
##

##
## === OPERATOR CLASS
##


class CreatePlot():

    def __init__(
        self,
        dataset_dir: Path,
    ) -> None:
        self.dataset_dir = dataset_dir
        with load_dataset.QuokkaDataset(dataset_dir=self.dataset_dir, verbose=False) as ds:
            uniform_domain = ds.load_uniform_domain()
        (x_min, x_max), (y_min, y_max), _ = uniform_domain.domain_bounds
        self.domain_bounds = (x_min, x_max, y_min, y_max)
        self.z_slice_index = uniform_domain.resolution[2] // 2

    def _plotter(
        self,
        ax,
        data_slice,
        cmap_name,
        cbar_label,
        cbar_side: str | None = None,
    ) -> None:
        plot_data.plot_sfield_slice(
            ax=ax,
            data_slice=data_slice,
            data_format="xy",
            axis_bounds=self.domain_bounds,
            add_cbar=cbar_side is not None,
            cmap_name=cmap_name,
            cbar_label=cbar_label,
            cbar_side=cbar_side,
            cbar_bounds=(
                numpy.nanpercentile(data_slice, 1.5),
                numpy.nanpercentile(data_slice, 98.5),
            ),
        )

    def _plot_pressure_sfield(
        self,
        ax,
    ) -> None:
        with load_dataset.QuokkaDataset(dataset_dir=self.dataset_dir, verbose=False) as ds:
            pressure_sfield = ds.load_pressure_sfield()
        data_slice = pressure_sfield.data[:, :, self.z_slice_index]
        num_rows, num_cols = data_slice.shape
        # top wedge
        top_mask = GridMask2D.make_wedge_mask(num_rows, num_cols, side_name="top", include_diagonals=False)
        masked_slice = GridMask2D.apply_mask(data_slice, top_mask)
        self._plotter(
            ax=ax,
            data_slice=masked_slice,
            cmap_name="bone",
            cbar_label=pressure_sfield.field_label,
            cbar_side=None,
        )
        # bottom wedge
        bottom_mask = GridMask2D.make_wedge_mask(
            num_rows, num_cols, side_name="bottom", include_diagonals=False
        )
        masked_slice = GridMask2D.apply_mask(data_slice, bottom_mask)
        self._plotter(
            ax=ax,
            data_slice=masked_slice,
            cmap_name="bone",
            cbar_label=pressure_sfield.field_label,
            cbar_side="bottom",
        )

    def _plot_kinetic_energy_sfield(
        self,
        ax,
    ) -> None:
        with load_dataset.QuokkaDataset(dataset_dir=self.dataset_dir, verbose=False) as ds:
            Ekin_sfield = ds.load_kinetic_energy_sfield()
        data_slice = Ekin_sfield.data[:, :, self.z_slice_index]
        num_rows, num_cols = data_slice.shape
        top_mask = GridMask2D.make_wedge_mask(num_rows, num_cols, side_name="top", include_diagonals=False)
        masked_slice = GridMask2D.apply_mask(data_slice, top_mask)
        self._plotter(
            ax=ax,
            data_slice=numpy.log10(masked_slice),
            cmap_name="managua_r",
            cbar_label=Ekin_sfield.field_label,
            cbar_side=None,
        )
        bottom_mask = GridMask2D.make_wedge_mask(
            num_rows, num_cols, side_name="bottom", include_diagonals=False
        )
        masked_slice = GridMask2D.apply_mask(data_slice, bottom_mask)
        self._plotter(
            ax=ax,
            data_slice=numpy.log10(masked_slice),
            cmap_name="managua_r",
            cbar_label=Ekin_sfield.field_label,
            cbar_side="top",
        )

    def _plot_current_density_sfield(
        self,
        ax,
    ) -> None:
        with load_dataset.QuokkaDataset(dataset_dir=self.dataset_dir, verbose=False) as ds:
            cur_sfield = ds.load_current_density_sfield()
        data_slice = cur_sfield.data[:, :, self.z_slice_index]
        num_rows, num_cols = data_slice.shape

        left_mask = GridMask2D.make_wedge_mask(num_rows, num_cols, side_name="left", include_diagonals=False)
        masked_slice = GridMask2D.apply_mask(data_slice, left_mask)
        self._plotter(
            ax=ax,
            data_slice=masked_slice,
            cmap_name="cubehelix",
            cbar_label=cur_sfield.field_label,
            cbar_side=None,
        )
        right_mask = GridMask2D.make_wedge_mask(
            num_rows, num_cols, side_name="right", include_diagonals=False
        )
        masked_slice = GridMask2D.apply_mask(data_slice, right_mask)
        self._plotter(
            ax=ax,
            data_slice=masked_slice,
            cmap_name="cubehelix",
            cbar_label=cur_sfield.field_label,
            cbar_side="right",
        )

    def _plot_magnetic_energy_sfield(
        self,
        ax,
    ) -> None:
        with load_dataset.QuokkaDataset(dataset_dir=self.dataset_dir, verbose=False) as ds:
            Emag_sfield = ds.load_magnetic_energy_sfield()
        data_slice = Emag_sfield.data[:, :, self.z_slice_index]
        num_rows, num_cols = data_slice.shape
        left_mask = GridMask2D.make_wedge_mask(num_rows, num_cols, side_name="left", include_diagonals=False)
        masked_slice = GridMask2D.apply_mask(data_slice, left_mask)
        self._plotter(
            ax=ax,
            data_slice=numpy.log10(masked_slice),
            cmap_name="berlin",
            cbar_label=Emag_sfield.field_label,
            cbar_side=None,
        )
        right_mask = GridMask2D.make_wedge_mask(
            num_rows, num_cols, side_name="right", include_diagonals=False
        )
        masked_slice = GridMask2D.apply_mask(data_slice, right_mask)
        self._plotter(
            ax=ax,
            data_slice=numpy.log10(masked_slice),
            cmap_name="berlin",
            cbar_label=Emag_sfield.field_label,
            cbar_side="right",
        )

    def run(
        self,
    ) -> None:
        fig, ax = plot_manager.create_ax()
        fig, axs_grid = plot_manager.create_axs_grid(
            num_rows=2,
            num_cols=1,
            x_spacing=0.15,
            share_x=True,
            share_y=True,
        )
        axs_col = axs_grid.get_col()
        self._plot_kinetic_energy_sfield(ax=axs_col[0])
        self._plot_magnetic_energy_sfield(ax=axs_col[0])
        self._plot_pressure_sfield(ax=axs_col[1])
        self._plot_current_density_sfield(ax=axs_col[1])
        for ax in list_utils.flatten_list(list(axs_grid)):
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig_name = "orzsag_tang.png"
        fig_dir = io_manager.get_caller_directory()
        fig_path = fig_dir / fig_name
        plot_manager.save_figure(
            fig=fig,
            fig_path=fig_path,
            verbose=True,
        )


##
## === PROGRAM MAIN
##


def main():
    # dataset_dir = Path("/Users/necoturb/Downloads/plt03800")
    dataset_dir = Path("/Users/necoturb/Downloads/plt27500")
    create_plot = CreatePlot(dataset_dir)
    create_plot.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
