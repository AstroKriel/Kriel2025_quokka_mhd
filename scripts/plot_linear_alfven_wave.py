##
## === DEPENDENCIES
##

import math
import numpy
from pathlib import Path
from dataclasses import dataclass
from jormi.ww_fields import field_operators
from ww_quokka_sims.sim_io import load_dataset
from utils import helpers

##
## === HELPERS
##


@dataclass(frozen=True)
class AdvectionSummary:
    fraction_of_cells_used: float
    median_speed: float
    mean_speed_magnitude_weighted: float
    direction_unit_vector: tuple[float, float, float]


def _safe_unit_vector(
    vector: numpy.ndarray,
) -> numpy.ndarray:
    vector_norm = float(numpy.linalg.norm(vector))
    return vector / vector_norm if vector_norm > 0.0 else numpy.array(
        [numpy.nan, numpy.nan, numpy.nan],
        dtype=numpy.float64,
    )


def _summarize_advection_field(
    advection_vector_field: numpy.ndarray,  # shape (3, n_cells_x, n_cells_y, n_cells_z)
    valid_mask: numpy.ndarray,  # shape (n_cells_x, n_cells_y, n_cells_z)
) -> AdvectionSummary:
    assert advection_vector_field.shape[0] == 3 and valid_mask.shape == advection_vector_field.shape[1:]
    component_x = advection_vector_field[0][valid_mask]
    component_y = advection_vector_field[1][valid_mask]
    component_z = advection_vector_field[2][valid_mask]
    if component_x.size == 0:
        return AdvectionSummary(
            fraction_of_cells_used=0.0,
            median_speed=0.0,
            mean_speed_magnitude_weighted=0.0,
            direction_unit_vector=(float("nan"), float("nan"), float("nan")),
        )
    local_speed = numpy.sqrt(
        component_x * component_x + component_y * component_y + component_z * component_z,
    )
    median_speed = float(numpy.median(local_speed))
    weights = local_speed
    weighted_x = float(numpy.sum(weights * component_x))
    weighted_y = float(numpy.sum(weights * component_y))
    weighted_z = float(numpy.sum(weights * component_z))
    direction_unit = _safe_unit_vector(numpy.array([weighted_x, weighted_y, weighted_z], dtype=numpy.float64))
    mean_speed_magnitude_weighted = float(numpy.sum(weights * local_speed) / numpy.sum(weights))
    fraction_of_cells_used = float(component_x.size) / float(numpy.prod(valid_mask.shape))
    return AdvectionSummary(
        fraction_of_cells_used,
        median_speed,
        mean_speed_magnitude_weighted,
        (float(direction_unit[0]), float(direction_unit[1]), float(direction_unit[2])),
    )


def _compute_advection_for_scalar(
    scalar_next: numpy.ndarray,
    scalar_prev: numpy.ndarray,
    delta_time: float,
    domain_lengths: tuple[float, float, float],
    gradient_norm2_epsilon: float = 1e-14,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
  Returns:
    advection_vector_field : shape (3, n_cells_x, n_cells_y, n_cells_z)
    valid_mask             : shape (n_cells_x, n_cells_y, n_cells_z) True where valid
  """
    # time_derivative = ( s^{n+1} - s^n ) / delta_t
    time_derivative = (scalar_next - scalar_prev) / float(delta_time)

    # gradient_field[i] = \partial_i s
    gradient_field = field_operators.compute_sfield_gradient(
        sfield=scalar_next,
        domain_lengths=domain_lengths,
    )  # shape (3, n_cells_x, n_cells_y, n_cells_z)

    # gradient_norm_squared = \sum_i ( \partial_i s )^2
    gradient_norm_squared = numpy.sum(gradient_field * gradient_field, axis=0)

    # mask cells where gradient_norm_squared > epsilon and all arrays finite
    valid_mask = (gradient_norm_squared > gradient_norm2_epsilon) & numpy.isfinite(time_derivative)
    for gradient_component_index in range(3):
        valid_mask &= numpy.isfinite(gradient_field[gradient_component_index])

    # formula: c_i = -( \partial_t s / ( \partial_j s \partial_j s ) ) * ( \partial_i s )
    advection_vector_field = numpy.zeros_like(gradient_field)
    inverse_gradient_norm_squared = numpy.zeros_like(gradient_norm_squared)
    inverse_gradient_norm_squared[valid_mask] = 1.0 / gradient_norm_squared[valid_mask]
    scaling_field = -(time_derivative * inverse_gradient_norm_squared)

    for component_index in range(3):
        advection_vector_field[component_index] = scaling_field * gradient_field[component_index]

    return advection_vector_field, valid_mask


##
## === OPERATOR CLASS
##


class Plotter:
    ## probe advection speed and direction from B-field snapshots; print summaries only

    def __init__(
        self,
        input_dir: Path,
        gradient_norm2_epsilon: float = 1e-14,
    ):
        self.input_dir = Path(input_dir)
        self.gradient_norm2_epsilon = float(gradient_norm2_epsilon)

    def run(
        self,
    ) -> None:
        dataset_dirs, is_single_dataset = self._resolve_dataset_dirs(self.input_dir)
        if is_single_dataset or len(dataset_dirs) < 2:
            raise ValueError("we need at least two time snapshots to estimate advection speed and direction")

        for previous_dataset_dir, next_dataset_dir in zip(dataset_dirs[:-1], dataset_dirs[1:]):
            self._process_pair(previous_dataset_dir, next_dataset_dir)

    def _process_pair(
        self,
        previous_dataset_dir: Path,
        next_dataset_dir: Path,
    ) -> None:
        ## load both snapshots
        with load_dataset.QuokkaDataset(
                dataset_dir=previous_dataset_dir,
                verbose=False,
        ) as dataset_previous:
            vfield_previous = dataset_previous.load_magnetic_field()
            domain = dataset_previous.load_domain()
        with load_dataset.QuokkaDataset(dataset_dir=next_dataset_dir, verbose=False) as dataset_next:
            vfield_next = dataset_next.load_magnetic_field()

        ## compute delta_t
        simulation_time_previous = float(vfield_previous.sim_time)
        simulation_time_next = float(vfield_next.sim_time)
        delta_time = simulation_time_next - simulation_time_previous
        if not (delta_time > 0.0 and math.isfinite(delta_time)):
            raise ValueError(
                f"invalid delta_t between {previous_dataset_dir.name} and {next_dataset_dir.name}: delta_t={delta_time}",
            )

        ## per-component advection vectors and summaries
        component_labels = vfield_previous.component_labels
        advection_vector_sum = numpy.zeros_like(vfield_previous.data)
        valid_masks: list[numpy.ndarray] = []
        per_component_summary: dict[str, AdvectionSummary] = {}

        for component_index, component_name in enumerate(component_labels):
            scalar_prev = numpy.asarray(vfield_previous.data[component_index], dtype=numpy.float64)
            scalar_next = numpy.asarray(vfield_next.data[component_index], dtype=numpy.float64)

            advection_vector_field, valid_mask = _compute_advection_for_scalar(
                scalar_next=scalar_next,
                scalar_prev=scalar_prev,
                delta_time=delta_time,
                domain_lengths=domain.domain_lengths,
                gradient_norm2_epsilon=self.gradient_norm2_epsilon,
            )

            per_component_summary[component_name] = _summarize_advection_field(
                advection_vector_field,
                valid_mask,
            )
            advection_vector_sum += advection_vector_field
            valid_masks.append(valid_mask)

        ## combined: use union of masks and sum of component advection vectors
        combined_valid_mask = numpy.logical_or.reduce(valid_masks)
        combined_summary = _summarize_advection_field(advection_vector_sum, combined_valid_mask)

        ## pretty print
        print(f"[{previous_dataset_dir.name} -> {next_dataset_dir.name}] delta_t={delta_time:.3e}")
        for component_name in component_labels:
            summary = per_component_summary[component_name]
            dir_x, dir_y, dir_z = summary.direction_unit_vector
            print(
                f"  {component_name:>2s}: used={summary.fraction_of_cells_used:6.2%}  "
                f"median|c|={summary.median_speed:.3e}  mean_w|c|={summary.mean_speed_magnitude_weighted:.3e}  "
                f"dir≈({dir_x:+.3f},{dir_y:+.3f},{dir_z:+.3f})",
            )
        dir_x, dir_y, dir_z = combined_summary.direction_unit_vector
        print(
            f"all: used={combined_summary.fraction_of_cells_used:6.2%}  "
            f"median|c|={combined_summary.median_speed:.3e}  mean_w|c|={combined_summary.mean_speed_magnitude_weighted:.3e}  "
            f"dir≈({dir_x:+.3f},{dir_y:+.3f},{dir_z:+.3f})",
        )

    @staticmethod
    def _resolve_dataset_dirs(
        input_dir: Path,
    ) -> tuple[list[Path], bool]:
        ## determine whether input is a single dataset dir (plt...) or a sim dir
        input_dir = Path(input_dir)
        if "plt" in input_dir.name:
            return [input_dir], True
        dataset_dirs = helpers.get_latest_dataset_dirs(sim_dir=input_dir)
        assert len(dataset_dirs) != 0
        return dataset_dirs, False


##
## === MAIN PROGRAM
##


def main():
    ## parse user inputs and run
    args = helpers.get_user_input()
    plotter = Plotter(
        input_dir=args.dir,
        gradient_norm2_epsilon=1e-14,
    )
    plotter.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()
