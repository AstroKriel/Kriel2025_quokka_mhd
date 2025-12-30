## { SCRIPT

##
## === DEPENDENCIES
##

import argparse
from pathlib import Path

from jormi.ww_types import type_checks
from jormi.utils import list_utils
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager

from plot_vi_evolution import DataSeries, LoadDataSeries

import utils

##
## === OPERATOR CLASSES
##


class RenderComparisonPlot:

    def __init__(
        self,
        *,
        fig_dir: Path,
        field_name: str,
        color: str,
        label_dir_1: str,
        label_dir_2: str,
        marker_dir_1: str = "o",
        marker_dir_2: str = "s",
    ):
        self.fig_dir = Path(fig_dir)
        self.field_name = field_name
        self.color = color
        self.label_dir_1 = str(label_dir_1)
        self.label_dir_2 = str(label_dir_2)
        self.marker_dir_1 = str(marker_dir_1)
        self.marker_dir_2 = str(marker_dir_2)

    def run(
        self,
        *,
        data_series_1: DataSeries,
        data_series_2: DataSeries,
    ) -> None:
        x_array_1, y_array_1 = data_series_1.get_sorted_arrays()
        x_array_2, y_array_2 = data_series_2.get_sorted_arrays()
        if (x_array_1.size == 0) and (x_array_2.size == 0):
            raise RuntimeError(
                "No data found for either directory.\n"
                f"dir_1 ({self.label_dir_1}): empty DataSeries\n"
                f"dir_2 ({self.label_dir_2}): empty DataSeries",
            )
        if x_array_1.size == 0:
            raise RuntimeError(
                "No data found for dir_1.\n"
                f"dir_1 ({self.label_dir_1}): empty DataSeries\n"
                f"dir_2 ({self.label_dir_2}): {x_array_2.size} points",
            )
        if x_array_2.size == 0:
            raise RuntimeError(
                "No data found for dir_2.\n"
                f"dir_1 ({self.label_dir_1}): {x_array_1.size} points\n"
                f"dir_2 ({self.label_dir_2}): empty DataSeries",
            )
        fig, ax = plot_manager.create_figure()
        ax.plot(
            x_array_1,
            y_array_1,
            color=self.color,
            marker=self.marker_dir_1,
            ms=6,
            ls="-",
            lw=1.5,
            label=self.label_dir_1,
        )
        ax.plot(
            x_array_2,
            y_array_2,
            color=self.color,
            marker=self.marker_dir_2,
            ms=6,
            ls="-",
            lw=1.5,
            label=self.label_dir_2,
        )
        ax.set_xlabel("time")
        ax.set_ylabel(self.field_name)
        ax.legend(loc="best")
        fig_path = self.fig_dir / f"{self.field_name}_time_comparison.png"
        plot_manager.save_figure(
            fig=fig,
            fig_path=fig_path,
            verbose=True,
        )


class ScriptInterface:

    def __init__(
        self,
        *,
        dir_1: Path,
        dir_2: Path,
        dataset_tag: str,
        fields_to_plot: list[str],
        out_dir: Path,
    ):
        type_checks.ensure_nonempty_string(
            param=dataset_tag,
            param_name="dataset_tag",
        )
        io_manager.does_directory_exist(
            directory=dir_1,
            raise_error=True,
        )
        io_manager.does_directory_exist(
            directory=dir_2,
            raise_error=True,
        )
        io_manager.does_directory_exist(
            directory=out_dir,
            raise_error=True,
        )
        self.dir_1 = Path(dir_1)
        self.dir_2 = Path(dir_2)
        self.fig_dir = Path(out_dir)
        valid_fields = set(utils.QUOKKA_FIELD_LOOKUP.keys())
        if (not fields_to_plot) or (not set(fields_to_plot).issubset(valid_fields)):
            raise ValueError(f"Provide one or more fields to plot (via -f) from: {sorted(valid_fields)}")
        self.dataset_tag = dataset_tag
        self.fields_to_plot = list(fields_to_plot)

    def run(
        self,
    ) -> None:
        dataset_dirs_1 = utils.resolve_dataset_dirs(
            input_dir=self.dir_1,
            dataset_tag=self.dataset_tag,
        )
        dataset_dirs_2 = utils.resolve_dataset_dirs(
            input_dir=self.dir_2,
            dataset_tag=self.dataset_tag,
        )
        if not dataset_dirs_1:
            raise RuntimeError(f"No dataset directories resolved for dir_1: {self.dir_1} (tag={self.dataset_tag!r})")
        if not dataset_dirs_2:
            raise RuntimeError(f"No dataset directories resolved for dir_2: {self.dir_2} (tag={self.dataset_tag!r})")
        label_dir_1 = self.dir_1.name
        label_dir_2 = self.dir_2.name
        for field_name in self.fields_to_plot:
            field_meta = utils.QUOKKA_FIELD_LOOKUP[field_name]
            load_data_series_1 = LoadDataSeries(
                dataset_dirs=dataset_dirs_1,
                field_name=field_name,
                field_loader=field_meta["loader"],
                use_parallel=True,
            )
            load_data_series_2 = LoadDataSeries(
                dataset_dirs=dataset_dirs_2,
                field_name=field_name,
                field_loader=field_meta["loader"],
                use_parallel=True,
            )
            data_series_1 = load_data_series_1.run()
            data_series_2 = load_data_series_2.run()
            render_comparison_plot = RenderComparisonPlot(
                fig_dir=self.fig_dir,
                field_name=field_name,
                color=field_meta["color"],
                label_dir_1=label_dir_1,
                label_dir_2=label_dir_2,
                marker_dir_1="o",
                marker_dir_2="s",
            )
            render_comparison_plot.run(
                data_series_1=data_series_1,
                data_series_2=data_series_2,
            )


##
## === ARGPARSE
##


def _get_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir-1",
        "-d1",
        type=lambda path: Path(path).expanduser().resolve(),
        required=True,
        help="First directory to compare.",
    )
    parser.add_argument(
        "--dir-2",
        "-d2",
        type=lambda path: Path(path).expanduser().resolve(),
        required=True,
        help="Second directory to compare.",
    )
    parser.add_argument(
        "--tag",
        "-t",
        type=str,
        required=True,
        help="Dataset tag to resolve within each directory.",
    )
    field_list = list_utils.as_string(elems=sorted(utils.QUOKKA_FIELD_LOOKUP.keys()))
    parser.add_argument(
        "--fields",
        "-f",
        nargs="+",
        default=None,
        help=f"List of (vector and/or scalar) fields to plot. Options: {field_list}",
    )
    parser.add_argument(
        "--out",
        type=lambda path: Path(path).expanduser().resolve(),
        required=True,
        help="Output directory for figures.",
    )
    return parser.parse_args()


##
## === PROGRAM MAIN
##


def main():
    user_args = _get_user_args()
    script_interface = ScriptInterface(
        dir_1=user_args.dir_1,
        dir_2=user_args.dir_2,
        dataset_tag=user_args.tag,
        fields_to_plot=user_args.fields,
        out_dir=user_args.out,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
