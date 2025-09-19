## { SCRIPT

##
## === DEPENDENCIES
##

from __future__ import annotations
import numpy
from pathlib import Path

from jormi.ww_data import compute_stats
from jormi.ww_plots import plot_manager, add_color
from jormi.ww_fields import field_operators
from utils import helpers, load_quokka_dataset

##
## === HELPERS
##


def _estimate_div_b_pdf(
    vfield_b,
    domain_lengths: tuple[float, float, float],
    num_bins: int = 50,
):
    ## compute divergence of magnetic field
    sfield_div_b = field_operators.compute_vfield_divergence(
        vfield=vfield_b.data,
        domain_lengths=domain_lengths,
    )
    ## estimate PDF of log10(|div(B)|)
    return compute_stats.estimate_pdf(
        values=numpy.log10(numpy.absolute(sfield_div_b)),
        num_bins=num_bins,
    )


##
## === OPERATOR CLASS
##


class Plotter:

    def __init__(
        self,
        input_dir: Path,
    ):
        self.input_dir = input_dir

    def run(
        self,
    ) -> None:
        dataset_dirs, single_snapshot = self._resolve_dataset_dirs(self.input_dir)
        fig, ax = plot_manager.create_figure()
        if single_snapshot:
            self._plot_single(ax, dataset_dirs[0])
        else:
            self._plot_series(ax, dataset_dirs)
        self._style_axes(ax)
        self._save(fig)

    def _plot_single(
        self,
        ax,
        dataset_dir: Path,
    ) -> None:
        ## plot a single dataset
        pdf = self._load_pdf(dataset_dir)
        ax.step(pdf.bin_centers, pdf.density, where="mid", lw=2.0, color="black", zorder=1)

    def _plot_series(
        self,
        ax,
        dataset_dirs: list[Path],
    ) -> None:
        ## plot multiple datasets with colormap
        cmap, norm = add_color.create_cmap(
            cmap_name="Blues",
            cmin=0.25,
            vmin=0,
            vmax=len(dataset_dirs),
        )
        for dataset_index, dataset_dir in enumerate(dataset_dirs[1:]):
            pdf = self._load_pdf(dataset_dir)
            ax.step(
                pdf.bin_centers,
                pdf.density,
                where="mid",
                lw=2.0,
                alpha=0.5,
                color=cmap(norm(dataset_index)),
                zorder=1.0 / (dataset_index + 1),
            )
        ## attach colorbar
        add_color.add_cbar_from_cmap(
            ax=ax,
            cmap=cmap,
            norm=norm,
            side="right",
            ax_percentage=0.05,
        )

    def _load_pdf(
        self,
        dataset_dir: Path,
    ):
        ## load dataset, extract B-field and domain, compute PDF
        with load_quokka_dataset.QuokkaDataset(dataset_dir=dataset_dir) as ds:
            vfield_b = ds.load_magnetic_field()
            domain = ds.load_domain()
        return _estimate_div_b_pdf(
            vfield_b=vfield_b,
            domain_lengths=domain.domain_lengths,
        )

    @staticmethod
    def _resolve_dataset_dirs(
        input_dir: Path,
    ) -> tuple[list[Path], bool]:
        ## determine whether input is a single snapshot or a series
        if "plt" in input_dir.name:
            dataset_dirs = [input_dir]
            return dataset_dirs, True
        else:
            dataset_dirs = helpers.get_latest_dataset_dirs(sim_dir=input_dir)
            assert len(dataset_dirs) != 0
            return dataset_dirs, len(dataset_dirs) == 1

    def _style_axes(
        self,
        ax,
    ) -> None:
        ax.set_xlabel(r"$\log_{10}(|\nabla\cdot\vec{b}|)$")
        ax.set_ylabel(r"$p(\log_{10}(|\nabla\cdot\vec{b}|))$")

    def _save(
        self,
        fig,
    ) -> None:
        fig_path = Path(self.input_dir) / "div_b_pdf.png"
        plot_manager.save_figure(fig=fig, fig_path=fig_path)


##
## === MAIN PROGRAM
##


def main():
    args = helpers.get_user_input()
    plotter = Plotter(input_dir=args.dir)
    plotter.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
