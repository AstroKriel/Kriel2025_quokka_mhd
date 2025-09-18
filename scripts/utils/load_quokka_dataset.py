## { MODULE

##
## === DEPENDENCIES
##

import numpy
from pathlib import Path
from dataclasses import dataclass
from yt.loaders import load as yt_load

##
## === FUNCTIONS
##


@dataclass(frozen=True)
class UniformDomain:
    periodicity: tuple[bool, bool, bool]
    resolution: tuple[int, int, int]
    domain_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]

    @property
    def cell_widths(
        self,
    ) -> tuple[float, float, float]:
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.domain_bounds
        n_cells_x, n_cells_y, n_cells_z = self.resolution
        return (
            (x_max - x_min) / n_cells_x,
            (y_max - y_min) / n_cells_y,
            (z_max - z_min) / n_cells_z,
        )

    @property
    def domain_lengths(
        self,
    ) -> tuple[float, float, float]:
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.domain_bounds
        return (
            x_max - x_min,
            y_max - y_min,
            z_max - z_min,
        )


@dataclass(frozen=True)
class VectorField:
    data: numpy.ndarray
    sim_time: float
    component_labels: tuple[str, str, str]


##
## === OPERATOR CLASS
##


class QuokkaDataset:

    MAGNETIC_FIELD_KEYS_TO_INDEX: dict[tuple[str, str], int] = {
        ("boxlib", "x-BField"): 0,
        ("boxlib", "y-BField"): 1,
        ("boxlib", "z-BField"): 2,
    }

    def __init__(
        self,
        dataset_dir: str | Path,
        keep_open: bool = False,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.keep_open = keep_open
        self.dataset = None
        self.sim_time = None
        self.covering_grid = None

    def _open_dataset(
        self,
    ) -> None:
        if self.dataset is None:
            self.dataset = yt_load(str(self.dataset_dir))
            self.sim_time = float(self.dataset.current_time)

    def _close_dataset_if_needed(
        self,
    ) -> None:
        if not self.keep_open and self.dataset is not None:
            self.dataset.close()
            self.dataset = None
            self.covering_grid = None

    def __enter__(
        self,
    ):
        self._open_dataset()
        return self

    def __exit__(
        self,
        _exception_type,
        _exception_value,
        _exception_traceback,
    ):
        self._close_dataset_if_needed()

    def _get_covering_grid(
        self,
    ):
        self._open_dataset()
        assert self.dataset is not None
        if self.covering_grid is None:
            self.covering_grid = self.dataset.covering_grid(
                level=0,
                left_edge=self.dataset.domain_left_edge,
                dims=self.dataset.domain_dimensions,
            )
        return self.covering_grid

    def load_domain(
        self,
    ) -> UniformDomain:
        self._open_dataset()
        assert self.dataset is not None
        x_min, y_min, z_min = (float(value) for value in self.dataset.domain_left_edge)
        x_max, y_max, z_max = (float(value) for value in self.dataset.domain_right_edge)
        n_cells_x, n_cells_y, n_cells_z = (int(num_cells) for num_cells in self.dataset.domain_dimensions)
        is_periodic_x, is_periodic_y, is_periodic_z = (
            bool(is_periodic) for is_periodic in self.dataset.periodicity
        )
        periodicity = (is_periodic_x, is_periodic_y, is_periodic_z)
        resolution = (n_cells_x, n_cells_y, n_cells_z)
        domain_bounds = ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        domain_info = UniformDomain(
            periodicity=periodicity,
            resolution=resolution,
            domain_bounds=domain_bounds,
        )
        self._close_dataset_if_needed()
        return domain_info

    def load_sim_time(
        self,
    ) -> float:
        self._open_dataset()
        assert self.sim_time is not None
        simulation_time = self.sim_time
        self._close_dataset_if_needed()
        return simulation_time

    def load_magnetic_field(
        self,
    ) -> VectorField:
        self._open_dataset()
        assert self.dataset is not None
        assert self.sim_time is not None
        available_fields = set(self.dataset.field_list)
        required_fields = set(self.MAGNETIC_FIELD_KEYS_TO_INDEX.keys())
        missing_fields = required_fields - available_fields
        if missing_fields:
            self._close_dataset_if_needed()
            raise RuntimeError(f"Missing magnetic-field components: {missing_fields}")
        covering_grid = self._get_covering_grid()
        components_by_index: dict[int, numpy.ndarray] = {}
        for field_key, component_index in self.MAGNETIC_FIELD_KEYS_TO_INDEX.items():
            field_data = numpy.asarray(covering_grid[field_key], dtype=numpy.float64)
            if field_data.ndim != 3:
                self._close_dataset_if_needed()
                raise ValueError(f"Expected 3-D array for {field_key}, got shape {field_data.shape}")
            components_by_index[component_index] = field_data
        component_arrays = [components_by_index[index] for index in range(3)]
        magnetic_field_data = numpy.ascontiguousarray(
            numpy.stack(component_arrays, axis=0),  # (3, n_cells_x, n_cells_y, n_cells_z)
        )
        vector_field = VectorField(
            data=magnetic_field_data,
            sim_time=float(self.sim_time),
            component_labels=("Bx", "By", "Bz"),
        )
        self._close_dataset_if_needed()
        return vector_field


## } MODULE
