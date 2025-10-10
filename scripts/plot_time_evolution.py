#!/usr/bin/env python3
# plot_time_evolution.py
#
## { SCRIPT
##
## === DEPENDENCIES
##

import numpy
from pathlib import Path
from dataclasses import dataclass
from jormi.ww_io import io_manager
from jormi.utils import parallel_utils
from jormi.ww_plots import plot_manager, plot_data, annotate_axis
from jormi.ww_fields import field_types, field_operators
from ww_quokka_sims.sim_io import load_dataset
from utils import helpers

##
## === DATA TYPES
##


@dataclass(frozen=True)
class LoaderArgs:
    dataset_dir: Path
    field_name: str
    loader_name: str
    verbose: bool


@dataclass(frozen=True)
class PlotterArgs:
    fig_dir: Path
    time_series: list[float]
    data_series: list[float]
    field_name: str
    color: str
    verbose: bool


##
## === PLOTTING
##


def _load_snapshot(
    loader_args: LoaderArgs,
) -> tuple[float, float]:
    with load_dataset.QuokkaDataset(dataset_dir=loader_args.dataset_dir, verbose=loader_args.verbose) as ds:
        domain = ds.load_domain()
        loader = getattr(ds, loader_args.loader_name)
        field = loader()  # expect ScalarField
    if not isinstance(field, field_types.ScalarField):
        raise ValueError(f"{loader_args.field_name} is not a scalar field (got {type(field).__name__}).")
    sim_time = float(field.sim_time)
    vi_quantity = field_operators.compute_sfield_volume_integral(
        sfield=field,
        domain=domain,
    )
    return (sim_time, vi_quantity)


def _plot_evolution(
    plotter_args: PlotterArgs,
) -> None:
    fig, ax = plot_manager.create_figure()
    ax.plot(
        plotter_args.time_series,
        plotter_args.data_series,
        color=plotter_args.color,
        marker="o",
        ms=6,
        ls="-",
        lw=1.5,
    )
    ax.set_xlabel("time")
    ax.set_ylabel(plotter_args.field_name)
    fig_name = f"{plotter_args.field_name}_time_evolution.png"
    fig_path = plotter_args.fig_dir / fig_name
    plot_manager.save_figure(
        fig=fig,
        fig_path=fig_path,
        verbose=plotter_args.verbose,
    )


##
## === OPERATOR CLASS
##


class Plotter:

    VALID_FIELDS = {
        "rho": {
            "loader": "load_density_sfield",
            "color": "black",
        },
        "Etot": {
            "loader": "load_total_energy_sfield",
            "color": "black",
        },
        "Eint": {
            "loader": "load_internal_energy_sfield",
            "color": "goldenrod",
        },
        "Ekin": {
            "loader": "load_kinetic_energy_sfield",
            "color": "darkorange",
        },
        "Emag": {
            "loader": "load_magnetic_energy_density_sfield",
            "color": "red",
        },
        "pressure": {
            "loader": "load_pressure_sfield",
            "color": "purple",
        },
        "divb": {
            "loader": "load_div_b_sfield",
            "color": "orange",
        },
    }

    def __init__(
        self,
        *,
        input_dir: Path,
        fields_to_plot: list[str],
        use_parallel: bool = True,
    ):
        valid_fields = set(self.VALID_FIELDS.keys())
        if (not fields_to_plot) or (not set(fields_to_plot).issubset(valid_fields)):
            raise ValueError(f"Provide one or more fields to plot (via -f) from: {sorted(valid_fields)}")
        self.input_dir = Path(input_dir)
        self.fields_to_plot = fields_to_plot
        self.use_parallel = bool(use_parallel)

    def run(self) -> None:
        dataset_dirs = helpers.resolve_dataset_dirs(self.input_dir)
        if not dataset_dirs: return
        dataset_dirs = sorted(dataset_dirs)
        fig_dir = dataset_dirs[0].parent
        for field_name in self.fields_to_plot:
            field_meta = self.VALID_FIELDS[field_name]
            loader_name = field_meta["loader"]
            color = field_meta["color"]
            grouped_loader_args: list[LoaderArgs] = [
                LoaderArgs(
                    dataset_dir=Path(dataset_dir),
                    field_name=field_name,
                    loader_name=loader_name,
                    verbose=False,
                ) for dataset_dir in dataset_dirs
            ]
            if not grouped_loader_args: continue
            if self.use_parallel and len(grouped_loader_args) > 5:
                results = parallel_utils.run_in_parallel(
                    func=_load_snapshot,
                    grouped_args=grouped_loader_args,
                    timeout_seconds=120,
                    show_progress=True,
                    enable_plotting=True,
                )
            else:
                results = [_load_snapshot(args) for args in grouped_loader_args]
            results = sorted(results, key=lambda data_pair: data_pair[0])
            sim_times = [sim_time for (sim_time, _) in results]
            vi_quantities = [vi_quantity for (_, vi_quantity) in results]
            _plot_evolution(
                PlotterArgs(
                    fig_dir=Path(fig_dir),
                    time_series=sim_times,
                    data_series=vi_quantities,
                    field_name=field_name,
                    color=color,
                    verbose=True,
                ),
            )


##
## === MAIN PROGRAM
##


def main():
    args = helpers.get_user_input()
    plotter = Plotter(
        input_dir=args.dir,
        fields_to_plot=args.fields,
        use_parallel=True,
    )
    plotter.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
