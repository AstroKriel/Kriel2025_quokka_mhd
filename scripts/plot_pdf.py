## { SCRIPT

##
## === DEPENDENCIES
##

import numpy
from typing import Literal
from pathlib import Path
from dataclasses import dataclass
from jormi.ww_plots import plot_manager, add_color
from jormi.ww_fields import field_types
from jormi.ww_data import compute_stats
from jormi.utils import type_utils, array_utils
from ww_quokka_sims.sim_io import load_dataset
import utils

##
## === DATA TYPES
##

Axis = Literal["x", "y", "z"]
LOOKUP_AXIS_INDEX: dict[Axis, int] = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class PDFData:
    sim_time: float
    grouped_bin_centers: list[numpy.ndarray]
    grouped_densities: list[numpy.ndarray]
    comp_labels: list[str]

    def __post_init__(self) -> None:
        ## container validation
        type_utils.ensure_sequence(
            var_obj=self.grouped_bin_centers,
            valid_containers=(list, tuple),
            var_name="grouped_bin_centers",
            seq_length=len(self.comp_labels),
        )
        type_utils.ensure_sequence(
            var_obj=self.grouped_densities,
            valid_containers=(list, tuple),
            var_name="grouped_densities",
            seq_length=len(self.comp_labels),
        )
        ## validate each comp-array
        for (bin_centers, densities) in zip(self.grouped_bin_centers, self.grouped_densities):
            array_utils.ensure_array(array=bin_centers)
            array_utils.ensure_array(array=densities)
            array_utils.ensure_1d(array=bin_centers)
            array_utils.ensure_1d(array=densities)
            array_utils.ensure_same_shape(
                array_a=bin_centers,
                array_b=densities,
            )

    @property
    def num_comps(
        self,
    ) -> int:
        return len(self.comp_labels)

    @property
    def is_scalar(
        self,
    ) -> bool:
        return self.num_comps == 1

    def get_pdf(
        self,
        comp_index: int = 0,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        if comp_index < 0 or comp_index >= self.num_comps:
            raise IndexError(f"comp_index {comp_index} out of range [0, {self.num_comps - 1}]")
        return self.grouped_bin_centers[comp_index], self.grouped_densities[comp_index]


##
## === LOADER
##


class ComputePDFs:

    def __init__(
        self,
        *,
        dataset_dirs: list[Path],
        field_name: str,
        field_loader: str,
        comps_to_plot: tuple[Axis, ...],
        num_bins: int,
    ):
        self.dataset_dirs = dataset_dirs
        self.field_name = field_name
        self.field_loader = field_loader
        self.comps_to_plot = comps_to_plot
        self.num_bins = num_bins

    @staticmethod
    def _estimate_pdf(
        *,
        field_data: numpy.ndarray,
        num_bins: int,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        pdf = compute_stats.estimate_pdf(values=field_data.ravel(), num_bins=num_bins)
        log10_densities = numpy.ma.log10(numpy.ma.masked_less_equal(pdf.densities, 0.0))
        return pdf.bin_centers, log10_densities

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

    def _build_vfield_pdf(
        self,
        field: field_types.VectorField,
    ) -> PDFData:
        if len(self.comps_to_plot) == 0:
            raise ValueError(f"Vector field '{self.field_name}' requires at least one component via -c")
        field_types.ensure_vfield(field)
        sim_time = self._get_sim_time(field=field)
        comp_names = sorted(self.comps_to_plot)
        comp_labels = [rf"$({self.field_name})_{{{comp_name}}}$" for comp_name in comp_names]
        grouped_bin_centers: list[numpy.ndarray] = []
        grouped_densities: list[numpy.ndarray] = []
        for comp_name in comp_names:
            comp_data = field.data[LOOKUP_AXIS_INDEX[comp_name]]
            bin_centers, densities = self._estimate_pdf(
                field_data=comp_data,
                num_bins=self.num_bins,
            )
            grouped_bin_centers.append(bin_centers)
            grouped_densities.append(densities)
        return PDFData(
            sim_time=sim_time,
            grouped_bin_centers=grouped_bin_centers,
            grouped_densities=grouped_densities,
            comp_labels=comp_labels,
        )

    def _build_sfield_pdf(
        self,
        field: field_types.ScalarField,
    ) -> PDFData:
        field_types.ensure_sfield(field)
        sim_time = self._get_sim_time(field=field)
        bin_centers, densities = self._estimate_pdf(
            field_data=field.data,
            num_bins=self.num_bins,
        )
        return PDFData(
            sim_time=sim_time,
            grouped_bin_centers=[bin_centers],
            grouped_densities=[densities],
            comp_labels=[self.field_name],
        )

    def run(
        self,
    ) -> list[PDFData]:
        field_pdfs: list[PDFData] = []
        for dataset_dir in self.dataset_dirs:
            with load_dataset.QuokkaDataset(dataset_dir=dataset_dir, verbose=False) as ds:
                loader_func = getattr(ds, self.field_loader)
                field = loader_func()
            if isinstance(field, field_types.ScalarField):
                pdf = self._build_sfield_pdf(field=field)
            elif isinstance(field, field_types.VectorField):
                pdf = self._build_vfield_pdf(field=field)
            else:
                raise ValueError(f"{self.field_name} is an unrecognised field type.")
            field_pdfs.append(pdf)
        field_pdfs.sort(key=lambda pdf: pdf.sim_time)
        return field_pdfs


##
## === RENDERER
##


class RenderPDFs:

    def __init__(
        self,
        *,
        dataset_dirs: list[Path],
        fig_dir: Path,
        field_name: str,
        comps_to_plot: tuple[Axis, ...],
        cmap_name: str,
        field_loader: str,
        num_bins: int,
        verbose: bool = False,
    ):
        self.dataset_dirs = dataset_dirs
        self.fig_dir = Path(fig_dir)
        self.field_name = field_name
        self.comps_to_plot = comps_to_plot
        self.cmap_name = cmap_name
        self.field_loader = field_loader
        self.num_bins = int(num_bins)
        self.verbose = bool(verbose)

    @staticmethod
    def _style_axs(
        *,
        axs_grid,
        comp_labels: list[str],
    ) -> None:
        for comp_index, label in enumerate(comp_labels):
            ax = axs_grid[0][comp_index]
            ax.set_xlabel(rf"$x \equiv$ {label}")
            if comp_index == 0:
                ax.set_ylabel(r"$\log_{10}\big(p(x)\big)$")

    @staticmethod
    def _plot_snapshot(
        *,
        axs_grid,
        pdf_data: PDFData,
        color,
    ) -> None:
        for comp_index in range(pdf_data.num_comps):
            ax = axs_grid[0][comp_index]
            x_values, y_values = pdf_data.get_pdf(comp_index)
            ax.step(x_values, y_values, where="mid", lw=2.0, color=color, zorder=comp_index + 1)

    @staticmethod
    def _plot_series(
        *,
        axs_grid,
        field_pdfs: list[PDFData],
        cmap_name: str,
    ) -> None:
        cmap, norm = add_color.create_cmap(
            cmap_name=cmap_name,
            cmin=0.25,
            vmin=0,
            vmax=max(
                0,
                len(field_pdfs) - 1,
            ),
        )
        for series_index, pdf_data in enumerate(field_pdfs):
            color = cmap(norm(series_index))
            RenderPDFs._plot_snapshot(axs_grid=axs_grid, pdf_data=pdf_data, color=color)
        add_color.add_cbar_from_cmap(
            ax=axs_grid[-1][-1],
            label=r"snapshot index",
            cmap=cmap,
            norm=norm,
            side="right",
            ax_percentage=0.05,
        )

    def run(self) -> None:
        compute_pdfs = ComputePDFs(
            dataset_dirs=self.dataset_dirs,
            field_name=self.field_name,
            field_loader=self.field_loader,
            comps_to_plot=self.comps_to_plot,
            num_bins=self.num_bins,
        )
        field_pdfs = compute_pdfs.run()
        if not field_pdfs:
            return
        num_cols = field_pdfs[0].num_comps
        fig, axs_grid = utils.create_figure(
            num_rows=1,
            num_cols=num_cols,
            add_cbar_space=len(field_pdfs) > 1,
        )
        if len(field_pdfs) == 1:
            self._plot_snapshot(
                axs_grid=axs_grid,
                pdf_data=field_pdfs[0],
                color="black",
            )
        else:
            self._plot_series(
                axs_grid=axs_grid,
                field_pdfs=field_pdfs,
                cmap_name=self.cmap_name,
            )
        self._style_axs(
            axs_grid=axs_grid,
            comp_labels=field_pdfs[0].comp_labels,
        )
        fig_path = self.fig_dir / f"{self.field_name}_pdfs.png"
        plot_manager.save_figure(
            fig=fig,
            fig_path=fig_path,
            verbose=self.verbose,
        )


##
## === ORCHESTRATOR
##


class PlotInterface:

    def __init__(
        self,
        *,
        input_dir: Path,
        fields_to_plot: tuple[str, ...] | list[str] | None,
        comps_to_plot: tuple[Axis, ...] | list[Axis] | None,
        num_bins: int = 15,
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
        self.input_dir = Path(input_dir)
        self.fields_to_plot = type_utils.as_tuple(seq_obj=fields_to_plot)
        self.comps_to_plot = type_utils.as_tuple(seq_obj=comps_to_plot)
        self.num_bins = int(num_bins)
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
            renderer = RenderPDFs(
                dataset_dirs=dataset_dirs,
                fig_dir=fig_dir,
                field_name=field_name,
                comps_to_plot=self.comps_to_plot,
                cmap_name=field_meta["cmap"],
                field_loader=field_meta["loader"],
                num_bins=self.num_bins,
                verbose=self.verbose,
            )
            renderer.run()


##
## === PROGRAM MAIN
##


def main():
    args = utils.get_user_args()
    plotter = PlotInterface(
        input_dir=args.dir,
        fields_to_plot=args.fields,
        comps_to_plot=args.comps,
        num_bins=15,
        verbose=True,
    )
    plotter.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
