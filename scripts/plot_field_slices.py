import os
import sys
import numpy
from pathlib import Path
from typing import Tuple, List

from yt.loaders import load as yt_load
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager, plot_data
from jormi.parallelism import independent_tasks

DEFAULT_DATA_DIR = Path(
  "~/Documents/Codes/asgard/mimir/kriel_2025_quokka_mhd/sims/weak/FieldLoop/fcvel_ld04_ro5_rk2_cfl0.3/N32_Nbo8_Nbl8_bopr1_mpir96"
).expanduser()

FIELD_NAME   = ("boxlib", "x-BField")
SLICE_AXIS   = 2 # 0:x, 1:y, 2:z
available_procs = (os.cpu_count() or 1)
capped_procs = min(available_procs, 24)
NUM_PROCS    = max(1, capped_procs - 1)
ONLY_ANIMATE = False
USE_TEX      = True
DARK_MODE    = True

def find_data_paths(directory: Path) -> List[Path]:
  data_paths = [
    path
    for path in directory.iterdir()
    if all([
      path.is_dir(),
      "plt" in path.name,
      "old" not in path.name
    ])
  ]
  data_paths.sort(
    key = lambda path: int(
      path.name.split("plt")[1]
    )
  )
  return data_paths

def get_mid_slice(
    arr  : numpy.ndarray,
    axis : int,
  ) -> numpy.ndarray:
  return numpy.take(arr, arr.shape[axis] // 2, axis=axis)

def extract_slice(
    data_path : str,
    npy_path  : str,
  ) -> Tuple[float, float, float]:
  ## force a headless backend per process
  import matplotlib
  matplotlib.use("Agg", force=True)
  ds = yt_load(data_path)
  sim_time = float(ds.current_time)
  cg = ds.covering_grid(
    level     = 0,
    left_edge = ds.domain_left_edge,
    dims      = ds.domain_dimensions,
  )
  data_cube = numpy.asarray(cg[FIELD_NAME], dtype=numpy.float32)
  data_slice = get_mid_slice(data_cube, SLICE_AXIS)
  numpy.save(npy_path, data_slice)
  try:
    ds.close()
  except Exception:
    pass
  return sim_time, float(data_slice.min()), float(data_slice.max())

def render_frame(
    npy_path    : str,
    png_path    : str,
    frame_title : str,
    min_value   : float,
    max_value   : float,
    slice_plane : str,
    use_tex     : bool = False,
  ) -> str:
  import matplotlib
  matplotlib.use("Agg", force=True)
  import matplotlib as mpl
  if DARK_MODE:
    mpl.rcParams.update({
      "figure.facecolor": "#0b0b0e",
      "axes.facecolor":   "#0b0b0e",
      "savefig.facecolor":"#0b0b0e",
      "axes.edgecolor":   "#e6e6e6",
      "axes.labelcolor":  "#e6e6e6",
      "text.color":       "#e6e6e6",
      "xtick.color":      "#cfcfd2",
      "ytick.color":      "#cfcfd2",
      "grid.color":       "#2e2e35",
    })
    try:
      import matplotlib.pyplot as _plt
      _plt.style.use("dark_background")
    except Exception:
      pass
  if use_tex:
    ## isolate latex cache per process
    import os, tempfile
    os.environ["TEXMFOUTPUT"] = tempfile.mkdtemp(prefix="mpltex_")
    matplotlib.rcParams["text.usetex"] = True
  else:
    matplotlib.rcParams["text.usetex"] = False
  import matplotlib.pyplot as plt
  field_slice = numpy.load(npy_path, mmap_mode="r")
  fig, ax = plot_manager.create_figure(fig_scale=1.25)
  plot_data.plot_sfield_slice(
    ax           = ax,
    field_slice  = field_slice,
    axis_bounds  = (-1, 1, -1, 1),
    cbar_bounds  = (min_value, max_value),
    cmap_name    = "cmr.iceburn",
    add_colorbar = True,
    cbar_label   = r"$\delta b_z$",
    cbar_side    = "right",
  )
  ax.set_title(frame_title)
  ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
  ax.set_xticks(ticks)
  ax.set_yticks(ticks)
  ax.set_xticklabels(str(v) for v in ticks)
  ax.set_yticklabels(str(v) for v in ticks)
  ax.set_xlabel(f"{slice_plane[0]} axis")
  ax.set_ylabel(f"{slice_plane[1]} axis")
  for _ax in fig.axes:
    _ax.tick_params(colors="#cfcfd2")
    _ax.xaxis.label.set_color("#e6e6e6")
    _ax.yaxis.label.set_color("#e6e6e6")
    _ax.title.set_color("#e6e6e6")
    for _spine in _ax.spines.values():
      _spine.set_color("#e6e6e6")
  fig.savefig(png_path, dpi=150)
  plt.close(fig)
  return png_path

def build_frames_with_parallel(data_dir: Path):
  print(f"Looking at: {data_dir}")
  output_dir = data_dir / "frames"
  npy_dir = output_dir / "_npy_slices"
  io_manager.init_directory(output_dir)
  io_manager.init_directory(npy_dir)
  data_paths = find_data_paths(data_dir)
  if not data_paths:
    raise SystemExit(f"No data_paths found in {data_dir}")
  slice_plane = ["yz", "xz", "xy"][SLICE_AXIS]
  print(f"Slice plane: {slice_plane}")
  png_paths = [
    output_dir / f"frame_{frame_index:05d}_{slice_plane}_plane.png"
    for frame_index in range(len(data_paths))
  ]
  all_frames_are_rendered = all(
    png_path.exists()
    for png_path in png_paths
  )
  if not ONLY_ANIMATE or not all_frames_are_rendered:
    npy_paths = [
      npy_dir / f"slice_{frame_index:05d}.npy"
      for frame_index in range(len(data_paths))
    ]
    grouped_args = [
      (str(data_path), str(npy_path))
      for data_path, npy_path in zip(data_paths, npy_paths)
    ]
    print(f"[Phase 1] Extracting slices...")
    extract_results = independent_tasks.run_in_parallel(
      func            = extract_slice,
      grouped_args    = grouped_args,
      num_workers     = NUM_PROCS,
      timeout_seconds = 300,
      show_progress   = True,
    )
    sim_times  = [ sim_time for (sim_time, _, _) in extract_results ]
    local_mins = [ min_value for (_, min_value, _) in extract_results ]
    local_maxs = [ max_value for (_, _, max_value) in extract_results ]
    min_value = float(min(local_mins))
    max_value = float(max(local_maxs))
    print(f"Global color limits: min={min_value:.6g}, max={max_value:.6g}")
    frame_titles = [
      f"{data_path.name}: t = {t:.3f}"
      for data_path, t in zip(data_paths, sim_times)
    ]
    render_args = [
      (str(npy_path), str(png_path), frame_title, min_value, max_value, slice_plane, USE_TEX)
      for npy_path, png_path, frame_title in zip(npy_paths, png_paths, frame_titles)
    ]
    print(f"[Phase 2] Rendering frames...")
    _ = independent_tasks.run_in_parallel(
      func            = render_frame,
      grouped_args    = render_args,
      num_workers     = NUM_PROCS,
      timeout_seconds = 300,
      show_progress   = True,
      enable_plotting = True,
    )
    print(f"Frames saved under: {output_dir}")
  mp4_path = output_dir / f"animated_{slice_plane}_plane.mp4"
  print(f"[Phase 3] Writing MP4...")
  plot_manager.animate_pngs_to_mp4(
    frames_dir = output_dir,
    mp4_path   = mp4_path,
    pattern    = f"frame_%05d_{slice_plane}_plane.png",
    fps        = 30,
  )

def main():
  data_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_DATA_DIR
  if not data_dir.exists():
    print(f"Error: data directory does not exist: {data_dir}")
    sys.exit(1)
  build_frames_with_parallel(data_dir)

if __name__ == "__main__":
  main()

## .