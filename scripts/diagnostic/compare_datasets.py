##
## === DEPENDENCIES
##

import numpy
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from jormi.utils import type_utils
from jormi.ww_io import log_manager
from ww_quokka_sims.sim_io import load_dataset
import utils

##
## === HELPER FUNCTION
##


def get_user_args():
    parser = argparse.ArgumentParser(
        description="Compare two Quokka (BoxLib) data-directories (IN vs REF).",
    )
    parser.add_argument(
        "--dir-in",
        "-d1",
        type=Path,
        required=True,
        help="Path to the first data-directory.",
    )
    parser.add_argument(
        "--dir-ref",
        "-d2",
        type=Path,
        required=True,
        help="Path to the second (reference) data-directory.",
    )
    parser.add_argument(
        "--preview-limit",
        "-N",
        type=int,
        default=100,
        help="Maximum number of index locations to preview.",
    )
    return parser.parse_args()


##
## === DATA ACCESS LAYER
##


class DatasetView:

    def __init__(
        self,
        *,
        dataset_dir: Path,
    ):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()

    def get_field_keys(
        self,
    ) -> set[load_dataset.FieldKey]:
        with load_dataset.QuokkaDataset(dataset_dir=self.dataset_dir, verbose=False) as ds:
            field_keys = ds.list_available_field_keys()
        return set(field_keys)

    def load_sfield(
        self,
        field_key: load_dataset.FieldKey,
    ) -> numpy.ndarray:
        with load_dataset.QuokkaDataset(dataset_dir=self.dataset_dir, verbose=False) as ds:
            sarray = ds._load_sfield_data(field_key=field_key)
        return numpy.ascontiguousarray(sarray, dtype=numpy.float64)


##
## === FIELD COMPARISON LAYER
##


@dataclass(frozen=True)
class FieldComparison:
    same_shape: bool
    shape_in: tuple[int, ...]
    shape_ref: tuple[int, ...]
    diff_indices_preview: list[tuple[int, int, int]] = field(default_factory=list)
    num_diffs: int = 0


class CompareFields:

    def __init__(
        self,
        *,
        sarray_in: numpy.ndarray,
        sarray_ref: numpy.ndarray,
        preview_limit: int,
    ):
        self.sarray_in = sarray_in
        self.sarray_ref = sarray_ref
        self.preview_limit = preview_limit

    def _get_diff_mask(
        self,
    ) -> numpy.ndarray:
        ## Note: NaNs are naturally different (NaN == NaN -> False)
        diff_value_mask = numpy.not_equal(self.sarray_in, self.sarray_ref)
        both_zero_mask = (self.sarray_in == 0.0) & (self.sarray_ref == 0.0)
        diff_sign_mask = numpy.signbit(self.sarray_in) ^ numpy.signbit(self.sarray_ref)
        mismatched_signed_zeros_mask = both_zero_mask & diff_sign_mask
        return diff_value_mask | mismatched_signed_zeros_mask

    def _get_diff_indices(
        self,
        diff_mask: numpy.ndarray,
    ) -> list[tuple[int, int, int]]:
        if not numpy.any(diff_mask):
            return []
        diff_coords = numpy.argwhere(diff_mask)  # shape: (num_diffs, num_dims=3)
        assert diff_coords.shape[1] == 3
        return diff_coords.tolist()

    def run(
        self,
    ) -> FieldComparison:
        shape_in = self.sarray_in.shape
        shape_ref = self.sarray_ref.shape
        if shape_in != shape_ref:
            return FieldComparison(
                same_shape=False,
                shape_in=shape_in,
                shape_ref=shape_ref,
            )
        diff_mask = self._get_diff_mask()
        diff_indices = self._get_diff_indices(diff_mask)
        num_diffs = int(diff_mask.sum(dtype=numpy.int64))
        return FieldComparison(
            same_shape=True,
            shape_in=shape_in,
            shape_ref=shape_ref,
            diff_indices_preview=diff_indices[:self.preview_limit],
            num_diffs=num_diffs,
        )


##
## === DATASET COMPARISON LAYER
##


class CompareDatasets:

    def __init__(
        self,
        *,
        dataset_view_in: DatasetView,
        dataset_view_ref: DatasetView,
        preview_limit: int,
    ):
        self.dataset_view_in = dataset_view_in
        self.dataset_view_ref = dataset_view_ref
        self.preview_limit = preview_limit

    def get_shared_field_keys(
        self,
    ) -> list[load_dataset.FieldKey]:
        field_keys_in = self.dataset_view_in.get_field_keys()
        field_keys_ref = self.dataset_view_ref.get_field_keys()
        keys_missing_from_in = sorted(field_keys_ref - field_keys_in)
        keys_missing_from_ref = sorted(field_keys_in - field_keys_ref)
        shared_keys = sorted(field_keys_in & field_keys_ref)
        if len(keys_missing_from_in) > 0:
            log_manager.log_items(
                title="Fields from dir-REF missing from dir-IN",
                items=[str(field_key) for field_key in keys_missing_from_in],
                message=f"There are {len(keys_missing_from_in)} fields not found in dir-IN.",
                message_position="bottom",
            )
        if len(keys_missing_from_ref) > 0:
            log_manager.log_items(
                title="Fields from dir-IN missing from dir-REF",
                items=[str(field_key) for field_key in keys_missing_from_ref],
                message=f"There are {len(keys_missing_from_ref)} fields not found in dir-REF.",
                message_position="bottom",
            )
        if len(shared_keys) == 0:
            log_manager.log_items(
                title="Available fields in dir-IN",
                items=[str(field_key) for field_key in sorted(field_keys_in)],
                message=f"There are {len(field_keys_in)} available fields.",
                message_position="bottom",
            )
            log_manager.log_items(
                title="Available fields in dir-REF",
                items=[str(field_key) for field_key in sorted(field_keys_ref)],
                message=f"There are {len(field_keys_ref)} available fields.",
                message_position="bottom",
            )
            log_manager.log_error(
                text="There are no shared fields to compare.",
                notes={
                    "dir-in": str(self.dataset_view_in.dataset_dir),
                    "dir-ref": str(self.dataset_view_ref.dataset_dir),
                },
            )
            raise SystemExit(4)
        log_manager.log_summary(
            title="Fields",
            notes={
                "shared": len(shared_keys),
                "missing_from_in": len(keys_missing_from_in),
                "missing_from_ref": len(keys_missing_from_ref),
            },
        )
        return shared_keys

    def _compare_datasets(
        self,
        field_key: load_dataset.FieldKey,
    ) -> None:
        sarray_in = self.dataset_view_in.load_sfield(field_key=field_key)
        sarray_ref = self.dataset_view_ref.load_sfield(field_key=field_key)
        compare_fields = CompareFields(
            sarray_in=sarray_in,
            sarray_ref=sarray_ref,
            preview_limit=self.preview_limit,
        )
        field_comparison = compare_fields.run()
        if not field_comparison.same_shape:
            log_manager.log_error(
                text=f"[{field_key}] Shape mismatch (IN vs REF).",
                notes={
                    "IN-shape": field_comparison.shape_in,
                    "REF-shape": field_comparison.shape_ref,
                },
                message_position="top",
            )
            return
        if field_comparison.num_diffs == 0:
            log_manager.log_note(f"[{field_key}] IN == REF (identical).")
            return
        diff_label = "diff-indices"
        if field_comparison.num_diffs > self.preview_limit:
            diff_label += f" (preview {self.preview_limit}/{field_comparison.num_diffs})"
        log_manager.log_warning(
            text=f"[{field_key}] There are {field_comparison.num_diffs} cells that are different.",
            notes={
                "shape": field_comparison.shape_in,
                diff_label: field_comparison.diff_indices_preview,
            },
            message_position="top",
        )

    def run(
        self,
    ) -> None:
        field_keys = self.get_shared_field_keys()
        for field_key in field_keys:
            self._compare_datasets(field_key=field_key)


##
## === SCRIPT INTERFACE LAYER
##


class ScriptInterface:

    def __init__(
        self,
        *,
        dataset_dir_in: Path,
        dataset_dir_ref: Path,
        preview_limit: int,
    ):
        self.dataset_dir_in = Path(dataset_dir_in).expanduser().resolve()
        self.dataset_dir_ref = Path(dataset_dir_ref).expanduser().resolve()
        self.preview_limit = preview_limit
        self._validate_inputs()

    def _validate_inputs(
        self,
    ) -> None:
        type_utils.ensure_finite_int(var_obj=self.preview_limit)
        assert self.preview_limit > 0
        utils.ensure_looks_like_boxlib_dir(dataset_dir=self.dataset_dir_in)
        utils.ensure_looks_like_boxlib_dir(dataset_dir=self.dataset_dir_ref)

    def run(
        self,
    ) -> None:
        dataset_view_in = DatasetView(dataset_dir=self.dataset_dir_in)
        dataset_view_ref = DatasetView(dataset_dir=self.dataset_dir_ref)
        compare_datasets = CompareDatasets(
            dataset_view_in=dataset_view_in,
            dataset_view_ref=dataset_view_ref,
            preview_limit=self.preview_limit,
        )
        compare_datasets.run()


##
## === PROGRAM MAIN
##


def main():
    args = get_user_args()
    script_interface = ScriptInterface(
        dataset_dir_in=args.dir_in,
        dataset_dir_ref=args.dir_ref,
        preview_limit=args.preview_limit,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()
