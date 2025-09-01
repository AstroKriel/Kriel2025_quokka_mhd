## { SCRIPT


import os
import sys
import numpy
from pathlib import Path

from yt.loaders import load as yt_load
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager, add_color, plot_styler
from jormi.parallelism import independent_tasks

DEFAULT_DATA_DIR = Path(
  # "~/Documents/Codes/asgard/mimir/kriel_2025_quokka_mhd/sims/weak/AlfvenWaveLinear/fcvel_ld04_ro3_rk2_cfl0.3/N8_Nbo8_Nbl8_bopr1_mpir1"
  # "~/Documents/Codes/asgard/mimir/kriel_2025_quokka_mhd/sims/weak/AlfvenWaveLinear/fcvel_ld04_ro3_rk2_cfl0.3/N16_Nbo8_Nbl8_bopr1_mpir8"
  # "~/Documents/Codes/asgard/mimir/kriel_2025_quokka_mhd/sims/weak/AlfvenWaveLinear/fcvel_ld04_ro3_rk2_cfl0.3/N32_Nbo8_Nbl8_bopr1_mpir96"
  "~/Documents/Codes/asgard/mimir/kriel_2025_quokka_mhd/sims/weak/AlfvenWaveLinear/fcvel_ld04_ro3_rk2_cfl0.3/N48_Nbo8_Nbl8_bopr1_mpir240"
).expanduser()

FIELD_NAME   = ("boxlib", "z-BField")  # Bx
PROFILE_AXIS = 0  # 0:x, 1:y, 2:z  (we want x)
available_procs = (os.cpu_count() or 1)
capped_procs = min(available_procs, 24)
NUM_PROCS    = max(1, capped_procs - 1)
USE_TEX      = False
DARK_MODE    = True

def find_data_paths(directory: Path) -> list[Path]:
  data_paths = [
    path for path in directory.iterdir()
    if all([path.is_dir(), "plt" in path.name, "old" not in path.name])
  ]
  data_paths.sort(key=lambda path: int(path.name.split("plt")[1]))
  return data_paths

def get_line_profile(
    arr  : numpy.ndarray,
    axis : int,
  ) -> numpy.ndarray:
  """Keep the requested axis, take midpoints along the other two."""
  nx, ny, nz = arr.shape
  ix, iy, iz = nx // 2, ny // 2, nz // 2
  if axis == 0: return arr[:,  iy, iz]
  if axis == 1: return arr[ix,  :,  iz]
  if axis == 2: return arr[ix,  iy,  :]
  raise ValueError(f"Invalid axis: {axis}")

def _axis_centers(ds, axis: int) -> numpy.ndarray:
  domain_x0  = float(ds.domain_left_edge [axis])
  domain_x1  = float(ds.domain_right_edge[axis])
  num_cells  = int(ds.domain_dimensions[axis])
  cell_width = (domain_x1 - domain_x0) / num_cells
  return domain_x0 + (numpy.arange(num_cells, dtype=numpy.float64) + 0.5) * cell_width

def worker_extract_profile(
    data_path : str,
  ) -> tuple[float, numpy.ndarray, numpy.ndarray] | None:
  ds = yt_load(data_path)
  sim_time = float(ds.current_time)
  x_centers = _axis_centers(ds, PROFILE_AXIS)
  cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
  data_cube = numpy.asarray(cg[FIELD_NAME], dtype=numpy.float32)
  profile = get_line_profile(data_cube, PROFILE_AXIS).astype(numpy.float32, copy=False)
  try:
    ds.close()
  except Exception:
    pass
  if sim_time > 1: return None
  return sim_time, x_centers, profile

def plot_profiles(data_dir: Path):
  print(f"Looking at: {data_dir}")
  output_dir = data_dir / "profiles"
  io_manager.init_directory(output_dir)
  data_paths = find_data_paths(data_dir)
  if not data_paths:
    raise SystemExit(f"No data_paths found in {data_dir}")
  grouped_args = [
    (str(path),)
    for path in data_paths
  ]
  print(f"[extract] {len(grouped_args)} profiles with {NUM_PROCS} workers...")
  results = independent_tasks.run_in_parallel(
    func            = worker_extract_profile,
    grouped_args    = grouped_args,
    num_workers     = NUM_PROCS,
    timeout_seconds = 300,
    show_progress   = True,
  )
  sim_times = [
    results[0]
    for results in results
    if results is not None
  ]
  x_centers = next(
    (
      results[1]
      for results in results
      if results is not None
    ), None
  )
  profiles = [
    results[2]
    for results in results
    if results is not None
  ]
  if x_centers is None or not profiles:
    raise SystemExit("No profiles extracted.")
  fig, ax = plot_manager.create_figure(fig_scale=1.25)
  cmap, norm = add_color.create_cmap(
    cmap_name = "Blues",
    vmin = float(min(sim_times)),
    vmax = float(max(sim_times)),
  )
  for sim_time, profile in zip(sim_times, profiles):
    ax.plot(x_centers, profile, lw=1.5, color=cmap(norm(sim_time)))
  add_color.add_cbar_from_cmap(
    ax    = ax,
    cmap  = cmap,
    norm  = norm,
    label = "time",
    side  = "right",
  )
  ax.set_xlabel(r"$x$")
  ax.set_ylabel(r"$\delta b_z$")
  png_path = output_dir / "alfven_wave_profile.png"
  plot_manager.save_figure(fig, png_path)

def main():
  data_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_DATA_DIR
  if not data_dir.exists():
    print(f"Error: data directory does not exist: {data_dir}")
    sys.exit(1)
  if DARK_MODE: plot_styler.apply_theme_globally(theme="dark")
  plot_profiles(data_dir)

if __name__ == "__main__":
  main()

## } SCRIPT