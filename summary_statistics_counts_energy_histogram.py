#!/usr/bin/env python3
"""Summarize simulated light detections for later analysis.

The script checks each light dataset, groups detections by location and energy,
and summarizes how often different detection counts occur. It saves the
summaries, the matching input parameters, and a record of the run settings.

This portfolio version uses ellipses and placeholder names for selected
research settings and identifiers pending publication of the journal article.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import importlib
import json
import os
import pickle
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_EXECUTION_START = time.perf_counter()


# Print the total runtime at exit. This helps compare how long different worker
# counts or batch sizes take.
def report_execution_time() -> None:
    elapsed_seconds = time.perf_counter() - SCRIPT_EXECUTION_START
    print(f"Execution time: {elapsed_seconds:.2f} seconds", flush=True)


if __name__ == "__main__":
    atexit.register(report_execution_time)


# Match the notebook's thread settings. Set these before importing NumPy and
# other numerical libraries.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import astropy
import healpy as hp
import joblib
import numpy as np
import torch
from astropy import units as u
from joblib import Parallel, delayed


SCHEMA_VERSION = 1
SCRIPT_VERSION = "1.0.0"
DEFAULT_CASE = "tau_two_parameter"

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
OUTPUT_DIR = SCRIPT_DIR / "Data"


# Define which event datasets and parameter values to load, and which binning
# rules to apply. Return the paths and settings used throughout the run.
def get_case_configuration() -> dict[str, Any]:
    """Return all case-specific scientific settings and paths."""

    case_description = "tau two-parameter"
    filename_qualifier = "tau_2param"
    parameter_dimension = 2

    raw_photon_data_path = Path(
        "/home/users/user_name/Scripts for DGRB/DM model/"
        "Test Data/100 realizations of test data/"
        "Data/combined_test_data.pkl"
    )
    theta_path = Path(
        "/home/users/user_name/Scripts for DGRB/DM model/"
        "Test Data/100 realizations of test data/"
        "Data/theta_test.pt"
    )
    max_C_per_energy_bin_paths = [
        Path(
            "/home/phys/theory/user_name/Scripts for DGRB/DM Model/"
            "Training Data/server1/"
            "Summary Statistics/max_C_per_energy_bin_global_max_server1_part_1.npy"
        ),
        Path(
            "/home/phys/theory/user_name/Scripts for DGRB/DM Model/"
            "Training Data/server1/"
            "Summary Statistics/max_C_per_energy_bin_global_max_server1_part_2.npy"
        ),
        Path(
            "/home/phys/theory/user_name/Scripts for DGRB/DM Model/"
            "Training Data/server2/server2/"
            "Summary Statistics/max_C_per_energy_bin_global_max_server2.npy"
        ),
        Path(
            "/home/phys/theory/user_name/Scripts for DGRB/DM Model/"
            "Training Data/server3/"
            "Summary Statistics/max_C_per_energy_bin_global_max_server3.npy"
        ),
    ]

    center_mask = 60  # deg
    lat_mask = 30  # deg
    N_Ebins = ...
    Ebinspace = "log"
    N_countbins = ...
    countbinspace = "linear"
    N_side = ...

    energy_range = [1_000, 100_000]  # MeV
    max_radius = ...  # kpc
    exposure = ...  # cm^2 yr
    flux_cut = ...  # photons / cm^2 / s
    angular_cut_degrees = ...
    lat_cut_degrees = ...
    luminosity_range = np.array([..., ...], dtype=np.float64)

    return {
        "case": DEFAULT_CASE,
        "case_description": case_description,
        "filename_qualifier": filename_qualifier,
        "parameter_dimension": parameter_dimension,
        "raw_photon_data_path": raw_photon_data_path,
        "theta_path": theta_path,
        "max_C_per_energy_bin_paths": max_C_per_energy_bin_paths,
        "center_mask": center_mask,
        "lat_mask": lat_mask,
        "N_Ebins": N_Ebins,
        "Ebinspace": Ebinspace,
        "N_countbins": N_countbins,
        "countbinspace": countbinspace,
        "N_side": N_side,
        "energy_range": energy_range,
        "max_radius": max_radius,
        "exposure": exposure,
        "flux_cut": flux_cut,
        "angular_cut_degrees": angular_cut_degrees,
        "lat_cut_degrees": lat_cut_degrees,
        "luminosity_range": luminosity_range,
    }


# Derive the expected histogram shape from the count and energy bins. Add the
# source path and setup settings for the routines that build the features.
def add_shared_configuration(
    case_configuration: dict[str, Any],
) -> dict[str, Any]:
    """Add settings derived from the case configuration."""

    configuration = dict(case_configuration)

    aegis_source_directory = Path(
        "/home/users/user_name/Scripts for DGRB/DGRB"
    )
    aegis_source_path = aegis_source_directory / "aegis.py"

    counts_energy_histogram_shape = (
        configuration["N_countbins"],
        configuration["N_Ebins"],
    )
    energy_range_gen = configuration["energy_range"]
    parameter_range_aegis = [[], []]
    abundance_luminosity_and_spectrum_list: list[Any] = []
    source_class_list: list[Any] = []

    configuration.update(
        {
            "aegis_source_directory": aegis_source_directory,
            "aegis_source_path": aegis_source_path,
            "counts_energy_histogram_shape": counts_energy_histogram_shape,
            "energy_range_gen": energy_range_gen,
            "parameter_range_aegis": parameter_range_aegis,
            "abundance_luminosity_and_spectrum_list": (
                abundance_luminosity_and_spectrum_list
            ),
            "source_class_list": source_class_list,
        }
    )
    return configuration


# Check that a required dataset or supporting file is present. If the path does
# not point to a regular file, report its role and location.
def require_regular_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")


# Create a checksum that can identify a file's exact contents across runs. Read
# the file in chunks to limit memory use when checking large inputs.
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Record the path, size, and modification time of a source file. Add a content
# checksum when requested so later runs can check whether the file changed.
def file_record(path: Path, *, include_sha256: bool) -> dict[str, Any]:
    require_regular_file(path, "Required file")
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        record["sha256"] = sha256_file(path)
    return record


# Load the event datasets and check that every event has two coordinates and
# one energy value. Require numeric arrays and record each dataset's event
# count so differences in dataset size are visible.
def load_and_validate_raw_photon_data(
    path: Path,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Load the trusted pickle and validate structure without fixing photon N."""

    require_regular_file(path, "Raw-photon input")
    with path.open("rb") as handle:
        data_list = pickle.load(handle)

    if not isinstance(data_list, list):
        raise TypeError(f"Expected {path} to contain a list, got {type(data_list)}")
    if not data_list:
        raise ValueError(f"Raw-photon input contains no datasets: {path}")

    photon_counts: list[int] = []
    for index, photon_info in enumerate(data_list):
        if not isinstance(photon_info, dict):
            raise TypeError(
                f"data_list[{index}] is {type(photon_info)}, expected a dictionary"
            )

        missing_keys = {"angles", "energies"} - set(photon_info)
        if missing_keys:
            raise KeyError(
                f"data_list[{index}] is missing keys: {sorted(missing_keys)}"
            )

        angles = photon_info["angles"]
        energies = photon_info["energies"]
        if not isinstance(angles, np.ndarray):
            raise TypeError(
                f"data_list[{index}]['angles'] is {type(angles)}, expected ndarray"
            )
        if not isinstance(energies, np.ndarray):
            raise TypeError(
                f"data_list[{index}]['energies'] is {type(energies)}, expected ndarray"
            )
        if angles.ndim != 2 or angles.shape[1] != 2:
            raise ValueError(
                f"data_list[{index}]['angles'] must have shape (N, 2), "
                f"got {angles.shape}"
            )
        if energies.ndim != 1:
            raise ValueError(
                f"data_list[{index}]['energies'] must have shape (N,), "
                f"got {energies.shape}"
            )
        if angles.shape[0] != energies.shape[0]:
            raise ValueError(
                f"data_list[{index}] has {angles.shape[0]} angle rows but "
                f"{energies.shape[0]} energies"
            )
        if not np.issubdtype(angles.dtype, np.number):
            raise TypeError(
                f"data_list[{index}]['angles'] must be numeric, got {angles.dtype}"
            )
        if not np.issubdtype(energies.dtype, np.number):
            raise TypeError(
                f"data_list[{index}]['energies'] must be numeric, got {energies.dtype}"
            )

        photon_counts.append(int(energies.shape[0]))

    return data_list, photon_counts


# Match the supplied model parameter values to the feature rows. Check the
# number of parameters and require finite values. Repeat a shared parameter row
# or keep the supplied rows in input order.
def load_and_align_theta(
    path: Path,
    *,
    num_datasets: int,
    parameter_dimension: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Infer single-theta or one-theta-per-dataset mode and align rows."""

    require_regular_file(path, "Theta input")
    theta_object = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(theta_object):
        raise TypeError(f"Expected {path} to contain a tensor, got {type(theta_object)}")

    theta_array = theta_object.detach().cpu().numpy()
    original_shape = list(theta_array.shape)
    original_dtype = str(theta_array.dtype)

    if theta_array.ndim == 1:
        if theta_array.shape[0] != parameter_dimension:
            raise ValueError(
                f"Expected one theta with width {parameter_dimension}, "
                f"got shape {theta_array.shape}"
            )
        theta_rows = theta_array.reshape(1, parameter_dimension)
    elif theta_array.ndim == 2:
        if theta_array.shape[1] != parameter_dimension:
            raise ValueError(
                f"Expected theta width {parameter_dimension}, got {theta_array.shape}"
            )
        theta_rows = theta_array
    else:
        raise ValueError(
            "Theta must have shape (d,), (1, d), or (n, d); "
            f"got {theta_array.shape}"
        )

    if not np.issubdtype(theta_rows.dtype, np.number):
        raise TypeError(f"Theta must be numeric, got dtype {theta_rows.dtype}")
    if not np.all(np.isfinite(theta_rows)):
        raise ValueError("Theta contains non-finite values")

    num_thetas = int(theta_rows.shape[0])
    if num_thetas == 1:
        theta_mode = "single_theta_repeated"
        aligned_theta = np.repeat(theta_rows, num_datasets, axis=0)
    elif num_thetas == num_datasets:
        theta_mode = "one_theta_per_dataset"
        aligned_theta = np.array(theta_rows, copy=True)
    else:
        raise ValueError(
            f"Theta file contains {num_thetas} theta rows, while raw input contains "
            f"{num_datasets} datasets. Expected either 1 or {num_datasets}."
        )

    expected_shape = (num_datasets, parameter_dimension)
    if aligned_theta.shape != expected_shape:
        raise ValueError(
            f"Expected aligned theta shape {expected_shape}, got {aligned_theta.shape}"
        )

    details: dict[str, Any] = {
        "mode": theta_mode,
        "parameter_dimension": parameter_dimension,
        "source_shape": original_shape,
        "source_dtype": original_dtype,
        "num_source_thetas": num_thetas,
        "saved_shape": list(aligned_theta.shape),
        "saved_dtype": str(aligned_theta.dtype),
    }
    if num_thetas == 1:
        details["single_theta_value"] = theta_rows[0].tolist()

    return aligned_theta, details


# Set an upper count limit for each energy band using the largest value across
# the training files. Check that each file has one valid count per band and
# record the original values for traceability.
def load_max_C_per_energy_bin(
    paths: list[Path],
    *,
    N_Ebins: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load and elementwise-max the training C_max arrays."""

    if not paths:
        raise ValueError("No C_max source paths were configured")

    arrays: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    expected_shape = (N_Ebins,)

    for path in paths:
        require_regular_file(path, "C_max input")
        array = np.load(path, allow_pickle=False)
        if array.shape != expected_shape:
            raise ValueError(
                f"Expected C_max shape {expected_shape} for {path}, got {array.shape}"
            )
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"C_max array must be numeric: {path} ({array.dtype})")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"C_max array contains non-finite values: {path}")
        if np.any(array < 0):
            raise ValueError(f"C_max array contains negative values: {path}")
        if not np.allclose(array, np.rint(array), rtol=0.0, atol=1e-12):
            raise ValueError(f"C_max array is not integer-valued: {path}")

        arrays.append(array)
        record = file_record(path, include_sha256=True)
        record.update(
            {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "values": array.tolist(),
            }
        )
        records.append(record)

    max_C_per_Ebin = np.maximum.reduce(arrays)
    if max_C_per_Ebin.shape != expected_shape:
        raise ValueError(
            f"Expected combined C_max shape {expected_shape}, "
            f"got {max_C_per_Ebin.shape}"
        )

    return max_C_per_Ebin, records


class NotebookCompatibleAegis:
    """Expose the notebook's array-only API without changing its numerics."""

    # Keep a reference to the configured AEGIS engine so the count maps and
    # histograms use the same processing settings.
    def __init__(self, engine: Any) -> None:
        self._engine = engine

    # Standardize the output used in the feature calculation. If a routine
    # returns both the data and its bin edges, extract the data and return it
    # as a NumPy array.
    @staticmethod
    def _array_only(result: Any, method_name: str) -> Any:
        if isinstance(result, tuple):
            if not result:
                raise ValueError(f"{method_name} returned an empty tuple")
            result = result[0]
        return np.asarray(result)

    # Build a table with one row per selected sky pixel and one column per
    # energy band. Each cell gives the number of events in that pixel and band.
    def get_roi_map_summary(self, *args: Any, **kwargs: Any) -> Any:
        result = self._engine.get_roi_map_summary(*args, **kwargs)
        return self._array_only(result, "get_roi_map_summary")

    # Count how many selected pixels fall into each photon-count bin. Use the
    # supplied bin limits and spacing and return the histogram as a NumPy
    # array.
    def get_counts_histogram_from_roi_map(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        result = self._engine.get_counts_histogram_from_roi_map(*args, **kwargs)
        return self._array_only(result, "get_counts_histogram_from_roi_map")


# Set up the count-map and histogram routines with the configured ranges and
# units. Verify the source file so the run uses the intended AEGIS
# implementation.
def build_aegis_object(configuration: dict[str, Any]) -> NotebookCompatibleAegis:
    """Import the configured AEGIS source and build the notebook's engine."""

    source_directory = configuration["aegis_source_directory"]
    source_path = configuration["aegis_source_path"]
    require_regular_file(source_path, "AEGIS source")

    source_directory_text = str(source_directory)
    while source_directory_text in sys.path:
        sys.path.remove(source_directory_text)
    sys.path.insert(0, source_directory_text)

    existing_aegis_module = sys.modules.get("aegis")
    if existing_aegis_module is not None:
        existing_module_file = getattr(existing_aegis_module, "__file__", None)
        if (
            existing_module_file is None
            or Path(existing_module_file).resolve() != source_path.resolve()
        ):
            del sys.modules["aegis"]
    importlib.invalidate_caches()
    aegis_module = importlib.import_module("aegis")

    imported_path = Path(aegis_module.__file__).resolve()
    if imported_path != source_path.resolve():
        raise ImportError(
            f"Imported AEGIS from {imported_path}, expected {source_path.resolve()}"
        )

    angular_cut = configuration["angular_cut_degrees"] * u.deg.to("rad")
    angular_cut_gen = min(angular_cut * 1.5, np.pi)
    lat_cut = configuration["lat_cut_degrees"] * u.deg.to("rad")
    lat_cut_gen = lat_cut * 0.5

    engine = aegis_module.aegis(
        configuration["abundance_luminosity_and_spectrum_list"],
        configuration["source_class_list"],
        configuration["parameter_range_aegis"],
        configuration["energy_range"],
        configuration["luminosity_range"],
        configuration["max_radius"],
        configuration["exposure"],
        angular_cut,
        lat_cut,
        configuration["flux_cut"],
        energy_range_gen=configuration["energy_range_gen"],
        verbose=False,
    )
    engine.angular_cut_gen, engine.lat_cut_gen = angular_cut_gen, lat_cut_gen
    engine.GC_to_earth = 0
    return NotebookCompatibleAegis(engine)


# Build one feature row by counting how many selected pixels fall into each
# count bin within each energy band. Check that each band accounts for every
# selected pixel, then flatten the table with energy bands varying fastest.
def compute_summary_statistics(
    photon_info: dict[str, Any],
    my_AEGIS: NotebookCompatibleAegis,
    center_mask: float,
    lat_mask: float,
    N_Ebins: int,
    Ebinspace: str,
    N_countbins: int,
    countbinspace: str,
    N_side: int,
    max_C_per_Ebin: np.ndarray,
) -> np.ndarray:
    """Compute one flattened counts-energy histogram using notebook logic."""

    mincount_2D_hist, maxcount_2D_hist = (
        np.zeros(N_Ebins),
        max_C_per_Ebin,
    )

    N_pix = 12 * N_side**2
    pix_i = np.linspace(0, N_pix - 1, N_pix, dtype="int")
    roi_pix_i = np.where(
        np.logical_and(
            hp.rotator.angdist(
                np.array([np.pi / 2, 0]),
                hp.pix2ang(N_side, pix_i),
            )
            >= center_mask * u.deg.to("rad"),
            np.abs(np.pi / 2 - hp.pix2ang(N_side, pix_i)[0])
            >= lat_mask * u.deg.to("rad"),
        )
    )[0]

    roi_map_2D_unclipped = my_AEGIS.get_roi_map_summary(
        photon_info=photon_info,
        N_side=N_side,
        N_Ebins=N_Ebins,
        Ebinspace=Ebinspace,
        roi_pix_i=roi_pix_i,
    )
    expected_roi_map_shape = (roi_pix_i.size, N_Ebins)
    if roi_map_2D_unclipped.shape != expected_roi_map_shape:
        raise ValueError(
            f"Expected ROI map shape {expected_roi_map_shape}, "
            f"got {roi_map_2D_unclipped.shape}"
        )

    # Preserve the overflow policy: clip per energy bin at the
    # training-data C_max, so overflow accumulates in the final count bin.
    roi_map_2D = np.minimum(roi_map_2D_unclipped, maxcount_2D_hist)

    counts_energy_hist = np.zeros((N_countbins, N_Ebins))
    for i in range(N_Ebins):
        hist_2d = my_AEGIS.get_counts_histogram_from_roi_map(
            roi_map_2D[:, [i]],
            mincount=mincount_2D_hist[i],
            maxcount=maxcount_2D_hist[i],
            N_countbins=N_countbins,
            countbinspace=countbinspace,
        )
        expected_histogram_shape = (N_countbins, 1)
        if hist_2d.shape != expected_histogram_shape:
            raise ValueError(
                f"Expected histogram shape {expected_histogram_shape} for "
                f"energy bin i={i}, got {hist_2d.shape}"
            )
        if hist_2d.sum() != roi_pix_i.size:
            raise ValueError(
                f"Sanity check failed for energy bin i={i}: "
                f"sum(hist_2d)={hist_2d.sum()} vs "
                f"roi_pix_i.size={roi_pix_i.size}"
            )

        counts_energy_hist[:, i] = hist_2d.ravel()

    return counts_energy_hist.flatten()


# Build a feature matrix with one row per dataset in input order. Process
# datasets in parallel batches and check that each row has the expected number
# of finite, non-negative entries.
def compute_all_histograms(
    data_list: list[dict[str, Any]],
    *,
    my_AEGIS: NotebookCompatibleAegis,
    configuration: dict[str, Any],
    max_C_per_Ebin: np.ndarray,
    num_workers: int,
    batch_size: int,
) -> np.ndarray:
    """Compute and stack one flattened histogram per input dictionary."""

    num_data = len(data_list)
    histogram_shape = tuple(configuration["counts_energy_histogram_shape"])
    expected_feature_count = int(np.prod(histogram_shape))
    counts_energy_histograms = np.empty(
        (num_data, expected_feature_count), dtype=np.float32
    )

    for start in range(0, num_data, batch_size):
        end = min(start + batch_size, num_data)
        print(f"Processing realizations [{start}:{end}) ...", flush=True)

        batch_results = Parallel(n_jobs=num_workers, backend="threading")(
            delayed(compute_summary_statistics)(
                photon_info,
                my_AEGIS,
                configuration["center_mask"],
                configuration["lat_mask"],
                configuration["N_Ebins"],
                configuration["Ebinspace"],
                configuration["N_countbins"],
                configuration["countbinspace"],
                configuration["N_side"],
                max_C_per_Ebin,
            )
            for photon_info in data_list[start:end]
        )
        batch_histograms = np.stack(batch_results, axis=0).astype(np.float32)

        expected_batch_shape = (end - start, expected_feature_count)
        if batch_histograms.shape != expected_batch_shape:
            raise ValueError(
                f"Expected batch shape {expected_batch_shape}, "
                f"got {batch_histograms.shape}"
            )
        if not np.all(np.isfinite(batch_histograms)):
            raise ValueError("Computed histograms contain non-finite values")
        if np.any(batch_histograms < 0):
            raise ValueError("Computed histograms contain negative values")

        counts_energy_histograms[start:end, :] = batch_histograms
        del batch_results, batch_histograms
        gc.collect()

    expected_output_shape = (num_data, expected_feature_count)
    if counts_energy_histograms.shape != expected_output_shape:
        raise ValueError(
            f"Expected output shape {expected_output_shape}, "
            f"got {counts_energy_histograms.shape}"
        )

    return counts_energy_histograms


# Use the case label and dataset count to name the feature matrix, matching
# parameter table, and run record so readers can identify the files that belong
# together.
def output_paths(
    *,
    filename_qualifier: str,
    num_datasets: int,
) -> dict[str, Path]:
    suffix = f"{filename_qualifier}_{num_datasets}_test_data"
    return {
        "histograms": OUTPUT_DIR / f"counts_energy_histograms_{suffix}.npy",
        "theta": OUTPUT_DIR / f"theta_{suffix}.npy",
        "metadata": OUTPUT_DIR / f"metadata_{suffix}.json",
    }


# Count the pixels that pass both selection rules. Record that total in the run
# metadata as the number of pixels represented in each energy band's histogram.
def roi_pixel_count(configuration: dict[str, Any]) -> int:
    """Compute the ROI size for metadata using the notebook's mask."""

    N_side = configuration["N_side"]
    N_pix = 12 * N_side**2
    pix_i = np.linspace(0, N_pix - 1, N_pix, dtype="int")
    roi_pix_i = np.where(
        np.logical_and(
            hp.rotator.angdist(
                np.array([np.pi / 2, 0]),
                hp.pix2ang(N_side, pix_i),
            )
            >= configuration["center_mask"] * u.deg.to("rad"),
            np.abs(np.pi / 2 - hp.pix2ang(N_side, pix_i)[0])
            >= configuration["lat_mask"] * u.deg.to("rad"),
        )
    )[0]
    return int(roi_pix_i.size)


# Document how the feature matrix was produced. Record input checksums,
# parameter alignment, bin settings, software versions, and expected output
# shapes so later runs can be compared.
def build_comparison_metadata(
    *,
    configuration: dict[str, Any],
    photon_counts: list[int],
    aligned_theta: np.ndarray,
    theta_details: dict[str, Any],
    max_C_per_Ebin: np.ndarray,
    max_C_source_records: list[dict[str, Any]],
    paths: dict[str, Path],
) -> dict[str, Any]:
    """Build stable metadata used to decide whether overwrite is compatible."""

    num_datasets = len(photon_counts)
    histogram_shape = list(configuration["counts_energy_histogram_shape"])
    feature_count = int(np.prod(histogram_shape))

    return {
        "schema_version": SCHEMA_VERSION,
        "script": {
            "path": str(SCRIPT_PATH),
            "version": SCRIPT_VERSION,
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "case": {
            "identifier": configuration["case"],
            "description": configuration["case_description"],
            "filename_qualifier": configuration["filename_qualifier"],
            "parameter_dimension": configuration["parameter_dimension"],
        },
        "inputs": {
            "raw_photon_data": file_record(
                configuration["raw_photon_data_path"], include_sha256=True
            ),
            "theta": file_record(configuration["theta_path"], include_sha256=True),
            "num_datasets": num_datasets,
            "photon_counts_per_dataset": photon_counts,
            "raw_dictionary_keys": ["angles", "energies"],
            "raw_value_shapes": {
                "angles": "(N_i, 2)",
                "energies": "(N_i,)",
                "N_i": "variable per dataset",
            },
        },
        "theta_alignment": theta_details,
        "C_max": {
            "sources": max_C_source_records,
            "elementwise_max": max_C_per_Ebin.tolist(),
            "shape": list(max_C_per_Ebin.shape),
            "dtype": str(max_C_per_Ebin.dtype),
        },
        "histogram": {
            "center_mask_degrees": configuration["center_mask"],
            "latitude_mask_degrees": configuration["lat_mask"],
            "N_Ebins": configuration["N_Ebins"],
            "Ebinspace": configuration["Ebinspace"],
            "N_countbins": configuration["N_countbins"],
            "countbinspace": configuration["countbinspace"],
            "N_side": configuration["N_side"],
            "roi_pixel_count": roi_pixel_count(configuration),
            "matrix_shape_count_by_energy": histogram_shape,
            "flatten_order": "C",
            "flattened_feature_count": feature_count,
            "stacked_output_shape": [num_datasets, feature_count],
            "stacked_output_dtype": "float32",
            "overflow_policy": "clip each energy column at its training C_max",
        },
        "aegis": {
            "source_directory": str(configuration["aegis_source_directory"]),
            "source_file": file_record(
                configuration["aegis_source_path"], include_sha256=True
            ),
            "adapter": (
                "If an AEGIS method returns (array, bin_edges), use array only; "
                "array-only returns pass through unchanged."
            ),
            "energy_range_MeV": list(configuration["energy_range"]),
            "energy_range_gen_MeV": list(configuration["energy_range_gen"]),
            "max_radius_kpc": configuration["max_radius"],
            "exposure_cm2_yr": configuration["exposure"],
            "flux_cut_photons_cm2_s": configuration["flux_cut"],
            "angular_cut_degrees": configuration["angular_cut_degrees"],
            "latitude_cut_degrees": configuration["lat_cut_degrees"],
            "luminosity_range": configuration["luminosity_range"].tolist(),
            "parameter_range_aegis": configuration["parameter_range_aegis"],
            "abundance_luminosity_and_spectrum_list": configuration[
                "abundance_luminosity_and_spectrum_list"
            ],
            "source_class_list": configuration["source_class_list"],
            "GC_to_earth": 0,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "healpy": hp.__version__,
            "astropy": astropy.__version__,
            "joblib": joblib.__version__,
        },
        "artifacts": {
            "histograms": {
                "path": str(paths["histograms"]),
                "shape": [num_datasets, feature_count],
                "dtype": "float32",
            },
            "theta": {
                "path": str(paths["theta"]),
                "shape": list(aligned_theta.shape),
                "dtype": str(aligned_theta.dtype),
            },
            "metadata": {"path": str(paths["metadata"])},
        },
    }


# Put the run record into a consistent format before hashing it. Sort the JSON
# keys so the fingerprint depends on the recorded values and field names
# regardless of their order.
def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


# Compute a SHA-256 fingerprint of the run record. Use it to check whether two
# runs have the same recorded inputs and settings.
def metadata_fingerprint(comparison_metadata: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(comparison_metadata)).hexdigest()


# Compare the current run record with the one saved alongside earlier results.
# Return the outcome and a reason so the caller can decide whether to replace
# those results.
def existing_metadata_matches(
    metadata_path: Path,
    current_comparison_metadata: dict[str, Any],
) -> tuple[bool, str]:
    if not metadata_path.is_file():
        return False, f"Metadata file is missing: {metadata_path}"

    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            saved_metadata = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"Could not read existing metadata: {exc}"

    saved_comparison_metadata = saved_metadata.get("comparison_metadata")
    if not isinstance(saved_comparison_metadata, dict):
        return False, "Existing metadata has no valid comparison_metadata object"

    current_fingerprint = metadata_fingerprint(current_comparison_metadata)
    saved_fingerprint = metadata_fingerprint(saved_comparison_metadata)
    if saved_fingerprint == current_fingerprint:
        return True, f"Matching metadata fingerprint: {current_fingerprint}"

    return (
        False,
        "Metadata fingerprints differ: "
        f"saved={saved_fingerprint}, current={current_fingerprint}",
    )


# Protect earlier results before starting a new computation. Proceed
# automatically when writing new files or when the saved metadata matches. Ask
# for explicit confirmation for other replacements.
def approve_computation_or_overwrite(
    *,
    paths: dict[str, Path],
    comparison_metadata: dict[str, Any],
) -> bool:
    """Apply the user's metadata-aware overwrite policy before computation."""

    existing_paths = [path for path in paths.values() if path.exists()]
    if not existing_paths:
        return True

    print("Existing output artifact(s) detected:")
    for path in existing_paths:
        print(f"  {path}")

    matches, reason = existing_metadata_matches(
        paths["metadata"], comparison_metadata
    )
    print(reason)
    if matches:
        print("Metadata match. Recomputing and replacing the relevant artifacts.")
        return True

    print(
        "WARNING: Existing metadata does not match the current configuration.\n"
        "If you overwrite, the existing files for this case will change."
    )
    if not sys.stdin.isatty():
        print("No interactive terminal is available; overwrite was not approved.")
        print("Nothing was computed.")
        return False

    try:
        response = input("Overwrite the existing artifacts? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nOverwrite was not approved. Nothing was computed.")
        return False

    if response in {"y", "yes"}:
        print("Overwrite approved.")
        return True

    print("Overwrite was not approved. Nothing was computed.")
    return False


# Write the feature matrix or parameter table to a temporary NumPy file. Finish
# writing and sync it to disk before it can replace the final output.
def stage_npy(path: Path, array: np.ndarray) -> Path:
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


# Write the run record as readable JSON in a temporary file. Finish writing and
# sync it to disk so it is ready to save alongside the results.
def stage_json(path: Path, value: dict[str, Any]) -> Path:
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


# Save the feature matrix, corresponding parameter table, and run record.
# Prepare all three files first and place the metadata last as the completion
# marker.
def save_output_artifacts(
    *,
    paths: dict[str, Path],
    counts_energy_histograms: np.ndarray,
    aligned_theta: np.ndarray,
    completed_metadata: dict[str, Any],
) -> None:
    """Stage every artifact fully, then replace targets with metadata last."""

    staged_paths: dict[str, Path] = {}
    try:
        staged_paths["histograms"] = stage_npy(
            paths["histograms"], counts_energy_histograms
        )
        staged_paths["theta"] = stage_npy(paths["theta"], aligned_theta)
        staged_paths["metadata"] = stage_json(
            paths["metadata"], completed_metadata
        )

        os.replace(staged_paths["histograms"], paths["histograms"])
        os.replace(staged_paths["theta"], paths["theta"])
        # Metadata is the commit marker and is replaced only after both arrays.
        os.replace(staged_paths["metadata"], paths["metadata"])
    finally:
        for temporary_path in staged_paths.values():
            temporary_path.unlink(missing_ok=True)


# Read the command-line options for the case, worker count, and batch size.
# Apply defaults when options are omitted.
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute notebook-compatible counts-energy histograms for a list "
            "of raw-photon dictionaries."
        )
    )
    parser.add_argument(
        "--case",
        default=DEFAULT_CASE,
        choices=(DEFAULT_CASE,),
        help="Scientific case identifier (defaults to tau_two_parameter).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=36,
        help="Number of Joblib worker threads (default: 36; -1 uses all CPUs).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of realizations processed per batch (default: 100).",
    )
    return parser.parse_args()


# Run the analysis from raw event data to saved features. Validate the inputs,
# align the parameter rows, and check whether earlier results can be replaced.
# Then build and save the feature matrix with its run record.
def main() -> int:
    arguments = parse_arguments()
    if arguments.num_workers == 0 or arguments.num_workers < -1:
        raise ValueError("--num-workers must be -1 or a positive integer")
    if arguments.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")

    configuration = add_shared_configuration(get_case_configuration())

    print(f"Selected case: {configuration['case']}")
    print(
        "Histogram settings: "
        f"center_mask={configuration['center_mask']} deg, "
        f"lat_mask={configuration['lat_mask']} deg, "
        f"N_Ebins={configuration['N_Ebins']}, "
        f"Ebinspace={configuration['Ebinspace']}, "
        f"N_countbins={configuration['N_countbins']}, "
        f"countbinspace={configuration['countbinspace']}, "
        f"N_side={configuration['N_side']}"
    )
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Loading raw-photon data: {configuration['raw_photon_data_path']}")
    data_list, photon_counts = load_and_validate_raw_photon_data(
        configuration["raw_photon_data_path"]
    )
    num_datasets = len(data_list)
    print(f"Loaded {num_datasets} raw-photon datasets.")

    aligned_theta, theta_details = load_and_align_theta(
        configuration["theta_path"],
        num_datasets=num_datasets,
        parameter_dimension=configuration["parameter_dimension"],
    )
    if aligned_theta.shape[0] != num_datasets:
        raise ValueError(
            "The number of aligned theta rows does not match the number of datasets"
        )
    print(
        f"Theta mode: {theta_details['mode']}; "
        f"aligned shape: {aligned_theta.shape}"
    )

    max_C_per_Ebin, max_C_source_records = load_max_C_per_energy_bin(
        configuration["max_C_per_energy_bin_paths"],
        N_Ebins=configuration["N_Ebins"],
    )
    print(f"Combined C_max per energy bin: {max_C_per_Ebin.tolist()}")

    paths = output_paths(
        filename_qualifier=configuration["filename_qualifier"],
        num_datasets=num_datasets,
    )
    print("Building comparison metadata and input fingerprints ...", flush=True)
    comparison_metadata = build_comparison_metadata(
        configuration=configuration,
        photon_counts=photon_counts,
        aligned_theta=aligned_theta,
        theta_details=theta_details,
        max_C_per_Ebin=max_C_per_Ebin,
        max_C_source_records=max_C_source_records,
        paths=paths,
    )

    # This gate is deliberately before AEGIS initialization and before any
    # counts-energy histogram is computed.
    if not approve_computation_or_overwrite(
        paths=paths,
        comparison_metadata=comparison_metadata,
    ):
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    my_AEGIS = build_aegis_object(configuration)
    counts_energy_histograms = compute_all_histograms(
        data_list,
        my_AEGIS=my_AEGIS,
        configuration=configuration,
        max_C_per_Ebin=max_C_per_Ebin,
        num_workers=arguments.num_workers,
        batch_size=arguments.batch_size,
    )

    expected_histogram_shape = (
        num_datasets,
        int(np.prod(configuration["counts_energy_histogram_shape"])),
    )
    if counts_energy_histograms.shape != expected_histogram_shape:
        raise ValueError(
            f"Expected saved histogram shape {expected_histogram_shape}, "
            f"got {counts_energy_histograms.shape}"
        )
    if counts_energy_histograms.dtype != np.float32:
        raise TypeError(
            "Expected stacked histogram dtype float32, "
            f"got {counts_energy_histograms.dtype}"
        )

    completed_metadata = {
        "comparison_fingerprint_sha256": metadata_fingerprint(
            comparison_metadata
        ),
        "comparison_metadata": comparison_metadata,
        "run_metadata": {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "num_workers": arguments.num_workers,
            "batch_size": arguments.batch_size,
            "histogram_shape": list(counts_energy_histograms.shape),
            "histogram_dtype": str(counts_energy_histograms.dtype),
            "theta_shape": list(aligned_theta.shape),
            "theta_dtype": str(aligned_theta.dtype),
        },
    }

    save_output_artifacts(
        paths=paths,
        counts_energy_histograms=counts_energy_histograms,
        aligned_theta=aligned_theta,
        completed_metadata=completed_metadata,
    )

    print("Saved counts-energy histograms:")
    print(f"  {paths['histograms']}  shape={counts_energy_histograms.shape}")
    print("Saved aligned theta values:")
    print(f"  {paths['theta']}  shape={aligned_theta.shape}")
    print(f"Saved metadata:\n  {paths['metadata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
