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
from ww_quokka_sims.sim_io import load_dataset
from utils import helpers

##
## === DATA TYPES
##

Axis = Literal["x", "y", "z"]

LOOKUP_AXIS_INDEX: dict[Axis, int] = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class PlotArgs:
    fig_dir: Path
    dataset_dirs: list[Path]
    field_name: str
    components_to_plot: list[Axis]  # ignored for scalars
    field_loader: str
    cmap_name: str
    num_bins: int = 15
    verbose: bool = False


@dataclass(frozen=True)
class PDFData:
    sim_time: float
    grouped_bin_centers: numpy.ndarray
    grouped_densities: numpy.ndarray
    comp_labels: list[str]

    def get_pdf(
        self,
        comp_index: int,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        return (
            self.grouped_bin_centers[comp_index],
            self.grouped_densities[comp_index],
        )

    @property
    def num_comps(
        self,
    ) -> int:
        return len(self.comp_labels)


##
## === HELPERS
##


def _estimate_pdf(
    field_values: numpy.ndarray,
    num_bins: int,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    pdf = compute_stats.estimate_pdf(
        values=field_values.ravel(),
        num_bins=num_bins,
    )
    return pdf.bin_centers, pdf.density


def _style_axes(
    axs_grid,
    comp_labels: list[str],
) -> None:
    for comp_index in range(len(comp_labels)):
        ax = axs_grid[0][comp_index]
        ax.set_xlabel(comp_labels[comp_index])
        if comp_index == 0:
            ax.set_ylabel(r"$p$")


##
## === LOAD PDFS
##


def load_field_pdfs(
    plot_args: PlotArgs,
) -> list[PDFData]:
    field_pdfs: list[PDFData] = []
    for dataset_dir in plot_args.dataset_dirs:
        with load_dataset.QuokkaDataset(dataset_dir=dataset_dir, verbose=False) as ds:
            field_loader = getattr(ds, plot_args.field_loader)
            field = field_loader()
        sim_time = float(field.sim_time)
        if isinstance(field, field_types.VectorField):
            if len(plot_args.components_to_plot) == 0:
                raise ValueError(
                    f"Vector field '{plot_args.field_name}' requires at least one component via -c",
                )
            comp_names = sorted(plot_args.components_to_plot)
            comp_labels = [field.labels[LOOKUP_AXIS_INDEX[comp_name]] for comp_name in comp_names]
            grouped_bin_centers = numpy.empty((len(comp_names), ), dtype=object)
            grouped_densities = numpy.empty((len(comp_names), ), dtype=object)
            for comp_index, comp_name in enumerate(comp_names):
                data_comp = field.data[LOOKUP_AXIS_INDEX[comp_name]]
                bin_centers, pdf_density = _estimate_pdf(
                    field_values=data_comp,
                    num_bins=plot_args.num_bins,
                )
                grouped_bin_centers[comp_index] = bin_centers
                grouped_densities[comp_index] = pdf_density
            field_pdfs.append(
                PDFData(
                    sim_time=sim_time,
                    grouped_bin_centers=grouped_bin_centers,
                    grouped_densities=grouped_densities,
                    comp_labels=comp_labels,
                ),
            )
        elif isinstance(field, field_types.ScalarField):
            bin_centers, pdf_density = _estimate_pdf(
                field_values=field.data,
                num_bins=plot_args.num_bins,
            )
            grouped_bin_centers = numpy.array([bin_centers], dtype=object)
            grouped_densities = numpy.array([pdf_density], dtype=object)
            field_pdfs.append(
                PDFData(
                    sim_time=sim_time,
                    grouped_bin_centers=grouped_bin_centers,
                    grouped_densities=grouped_densities,
                    comp_labels=[field.label],
                ),
            )
        else:
            raise ValueError(f"{plot_args.field_name} is an unrecognised field type.")
    field_pdfs.sort(key=lambda field_pdf: field_pdf.sim_time)
    return field_pdfs


##
## === PLOTTING
##


def _plot_snapshot(
    *,
    axs_grid,
    pdf_data: PDFData,
    color,
) -> None:
    for comp_index in range(pdf_data.num_comps):
        ax = axs_grid[0][comp_index]
        x_values, y_density = pdf_data.get_pdf(comp_index)
        ax.step(
            x_values,
            y_density,
            where="mid",
            lw=2.0,
            color=color,
            zorder=comp_index + 1,
        )


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
        vmax=max(0,
                 len(field_pdfs) - 1),
    )
    for series_index, pdf_data in enumerate(field_pdfs):
        color = cmap(norm(series_index))
        _plot_snapshot(
            axs_grid=axs_grid,
            pdf_data=pdf_data,
            color=color,
        )
    add_color.add_cbar_from_cmap(
        ax=axs_grid[-1][-1],
        label=r"dump index",
        cmap=cmap,
        norm=norm,
        side="right",
        ax_percentage=0.05,
    )


def _plot_field(
    plot_args: PlotArgs,
) -> None:
    field_pdfs = load_field_pdfs(plot_args)
    if not field_pdfs: return
    num_cols = field_pdfs[0].num_comps
    fig, axs_grid = helpers.create_axes_grid(
        num_rows=1,
        num_cols=num_cols,
    )
    if len(field_pdfs) == 1:
        _plot_snapshot(
            axs_grid=axs_grid,
            pdf_data=field_pdfs[0],
            color="black",
        )
    else:
        _plot_series(
            axs_grid=axs_grid,
            field_pdfs=field_pdfs,
            cmap_name=plot_args.cmap_name,
        )
    _style_axes(
        axs_grid=axs_grid,
        comp_labels=field_pdfs[0].comp_labels,
    )
    fig_path = plot_args.fig_dir / f"{plot_args.field_name}_pdfs.png"
    plot_manager.save_figure(
        fig=fig,
        fig_path=fig_path,
        verbose=plot_args.verbose,
    )


##
## === OPERATOR
##


class Plotter:

    VALID_FIELDS = {
        "rho": {
            "loader": "load_density_sfield",
            "cmap": "Greys",
        },
        "vel": {
            "loader": "load_velocity_vfield",
            "cmap": "Oranges",
        },
        "mag": {
            "loader": "load_magnetic_vfield",
            "cmap": "Blues",
        },
        "Etot": {
            "loader": "load_total_energy_sfield",
            "cmap": "cividis",
        },
        "Ekin": {
            "loader": "load_kinetic_energy_sfield",
            "cmap": "magma",
        },
        "Emag": {
            "loader": "load_magnetic_energy_density_sfield",
            "cmap": "plasma",
        },
        "Eint": {
            "loader": "load_internal_energy_sfield",
            "cmap": "magma",
        },
        "pressure": {
            "loader": "load_pressure_sfield",
            "cmap": "Purples",
        },
        "divb": {
            "loader": "load_div_b_sfield",
            "cmap": "bwr",
        },
    }

    def __init__(
        self,
        *,
        input_dir: Path,
        fields_to_plot: list[str],
        components_to_plot: list[Axis],
        verbose: bool = True,
        num_bins: int = 15,
    ):
        valid_fields = set(self.VALID_FIELDS.keys())
        if not fields_to_plot or not set(fields_to_plot).issubset(valid_fields):
            raise ValueError(f"Provide fields via -f from: {sorted(valid_fields)}")
        valid_axes = {"x", "y", "z"}
        ## default to all components (if not provided)
        if not components_to_plot:
            components_to_plot = ["x", "y", "z"]
        elif not set(components_to_plot).issubset(valid_axes):
            raise ValueError("Provide one or more components (via -c) from: x, y, z")
        self.input_dir = Path(input_dir)
        self.fields_to_plot = fields_to_plot
        self.components_to_plot = components_to_plot  # ignored for scalars
        self.verbose = bool(verbose)
        self.num_bins = int(num_bins)

    def run(
        self,
    ) -> None:
        dataset_dirs = helpers.resolve_dataset_dirs(self.input_dir)
        if not dataset_dirs:
            return
        fig_dir = dataset_dirs[0].parent
        for field_name in self.fields_to_plot:
            field_meta = self.VALID_FIELDS[field_name]
            plot_args = PlotArgs(
                fig_dir=Path(fig_dir),
                dataset_dirs=dataset_dirs,
                field_name=field_name,
                components_to_plot=self.components_to_plot,
                field_loader=field_meta["loader"],
                cmap_name=field_meta["cmap"],
                num_bins=self.num_bins,
                verbose=self.verbose,
            )
            _plot_field(plot_args)


##
## === MAIN PROGRAM
##


def main():
    args = helpers.get_user_input()
    plotter = Plotter(
        input_dir=args.dir,
        fields_to_plot=args.fields,
        components_to_plot=args.components,
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
