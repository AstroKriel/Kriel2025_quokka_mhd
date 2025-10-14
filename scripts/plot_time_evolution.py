## { SCRIPT

##
## === DEPENDENCIES
##

from pathlib import Path
from dataclasses import dataclass
from jormi.utils import parallel_utils
from jormi.ww_plots import plot_manager
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
    field_loader: str
    verbose: bool


@dataclass(frozen=True)
class DataPoint:
    sim_time: float
    vi_quantity: float


@dataclass(frozen=True)
class DataSeries:
    sim_times: list[float]
    vi_quantities: list[float]


@dataclass(frozen=True)
class PlotterArgs:
    fig_dir: Path
    data_series: DataSeries
    field_name: str
    color: str
    verbose: bool


##
## === PLOTTING
##


def _load_snapshot(
    loader_args: LoaderArgs,
) -> DataPoint:
    with load_dataset.QuokkaDataset(dataset_dir=loader_args.dataset_dir, verbose=loader_args.verbose) as ds:
        uniform_domain = ds.load_domain_details()
        field_loader = getattr(ds, loader_args.field_loader)
        field = field_loader()  # expect ScalarField
    if not isinstance(field, field_types.ScalarField):
        raise ValueError(f"{loader_args.field_name} is not a scalar field (got {type(field).__name__}).")
    assert field.sim_time is not None
    sim_time = float(field.sim_time)
    vi_quantity = float(
        field_operators.compute_sfield_volume_integral(
            sfield=field,
            uniform_domain=uniform_domain,
        ),
    )
    return DataPoint(
        sim_time=sim_time,
        vi_quantity=vi_quantity,
    )


def _plot_evolution(
    plotter_args: PlotterArgs,
) -> None:
    fig, axs_grid = plot_manager.create_figure()
    ax = axs_grid[0, 0]
    ax.plot(
        plotter_args.data_series.sim_times,
        plotter_args.data_series.vi_quantities,
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
        "Ekin": {
            "loader": "load_kinetic_energy_sfield",
            "color": "darkorange",
        },
        "Ekin_div": {
            "loader": "load_div_kinetic_energy_sfield",
            "color": "darkorange",
        },
        "Ekin_sol": {
            "loader": "load_sol_kinetic_energy_sfield",
            "color": "darkorange",
        },
        "Emag": {
            "loader": "load_magnetic_energy_density_sfield",
            "color": "red",
        },
        "Eint": {
            "loader": "load_internal_energy_sfield",
            "color": "sandybrown",
        },
        "pressure": {
            "loader": "load_pressure_sfield",
            "color": "purple",
        },
        "divb": {
            "loader": "load_divb_sfield",
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
            field_loader = field_meta["loader"]
            color = field_meta["color"]
            grouped_loader_args: list[LoaderArgs] = [
                LoaderArgs(
                    dataset_dir=Path(dataset_dir),
                    field_name=field_name,
                    field_loader=field_loader,
                    verbose=False,
                ) for dataset_dir in dataset_dirs
            ]
            if not grouped_loader_args: continue
            if self.use_parallel and len(grouped_loader_args) > 5:
                data_points: list[DataPoint] = parallel_utils.run_in_parallel(
                    func=_load_snapshot,
                    grouped_args=grouped_loader_args,
                    timeout_seconds=120,
                    show_progress=True,
                    enable_plotting=True,
                )
            else:
                data_points: list[DataPoint] = [_load_snapshot(args) for args in grouped_loader_args]
            data_points = sorted(data_points, key=lambda data_point: data_point.sim_time)
            data_series = DataSeries(
                sim_times=[data_point.sim_time for data_point in data_points],
                vi_quantities=[data_point.vi_quantity for data_point in data_points],
            )
            _plot_evolution(
                PlotterArgs(
                    fig_dir=Path(fig_dir),
                    data_series=data_series,
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
