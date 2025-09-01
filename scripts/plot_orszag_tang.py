#!/usr/bin/env python3
import os
import sys
import gc
import numpy
from pathlib import Path

from yt.loaders import load as yt_load
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager, plot_data

DEFAULT_DATA_DIR = Path(
    "/scratch/jh2/nk7952/quokka/sims/weak/OrszagTang/fcvel_ld04_ro3_rk2_cfl0.3/N512_Nbo64_Nbl64_bopr1_mpir528/",
)

FIELD_NAME = ("boxlib", "x-BField")
SLICE_AXIS = 2  # 0:x, 1:y, 2:z
USE_TEX = False  # keep simple/headless


def find_data_paths(directory: Path) -> list[Path]:
    data_paths = [
        d for d in directory.iterdir()
        if d.is_dir() and ("plt" in d.name) and ("old" not in d.name)
    ]
    data_paths.sort(key=lambda d: int(d.name.split("plt")[1]))
    return data_paths


def get_mid_slice(arr: numpy.ndarray, axis: int) -> numpy.ndarray:
    return numpy.take(arr, arr.shape[axis] // 2, axis=axis)


def render_one_frame(
    plotfile_path: Path,
    frame_png: Path,
    slice_plane: str,
    use_tex: bool = False,
    auto_cbar: bool = True,
) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["text.usetex"] = bool(use_tex)

    import matplotlib.pyplot as plt

    ds = None
    data3d = None
    slice2d = None
    try:
        ds = yt_load(str(plotfile_path))
        sim_time = float(ds.current_time)

        # read only what we need; keep dtype small
        cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
        data3d = numpy.asarray(cg[FIELD_NAME], dtype=numpy.float32)

        slice2d = get_mid_slice(data3d, SLICE_AXIS)

        # per-frame color range: no global vmin/vmax
        cbar_bounds: tuple[float, float] | None
        if auto_cbar:
            vmin = float(slice2d.min())
            vmax = float(slice2d.max())
            cbar_bounds = (vmin, vmax)
        else:
            cbar_bounds = None  # let plotting util auto-range if it supports that

        fig, ax = plot_manager.create_figure(fig_scale=1.25)
        plot_data.plot_sfield_slice(
          ax           = ax,
          field_slice  = slice2d,
          axis_bounds  = (-1, 1, -1, 1),
          cbar_bounds  = cbar_bounds,   # per-frame scaling
          cmap_name    = "cmr.iceburn",
          add_colorbar = True,
          cbar_label   = "",
          cbar_side    = "right",
        )

        title = f"{plotfile_path.name}: t = {sim_time:.3f}"
        ax.set_title(title)
        ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(str(v) for v in ticks)
        ax.set_yticklabels(str(v) for v in ticks)
        ax.set_xlabel(f"{slice_plane[0]} axis")
        ax.set_ylabel(f"{slice_plane[1]} axis")

        frame_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(frame_png, dpi=150)
        plt.close(fig)

    finally:
        # aggressively free memory as soon as the frame is written
        try:
            if ds is not None:
                try:
                    ds.close()
                except Exception:
                    pass
                # best-effort: drop references to data that yt may keep
                try:
                    ds.index.clear_all_data()  # safe if available
                except Exception:
                    pass
        except Exception:
            pass

        del slice2d
        del data3d
        del ds
        gc.collect()


def build_frames_serial(data_dir: Path) -> None:
    print(f"Looking at: {data_dir}")
    output_dir = data_dir / "frames"
    io_manager.init_directory(output_dir)

    data_paths = find_data_paths(data_dir)
    if not data_paths:
        raise SystemExit(f"No data_paths found in {data_dir}")

    slice_plane = ["yz", "xz", "xy"][SLICE_AXIS]
    print(f"Slice plane: {slice_plane}")

    # generate frames one-by-one, skipping existing
    for i, p in enumerate(data_paths):
        frame_png = output_dir / f"frame_{i:05d}_{slice_plane}_plane.png"
        if frame_png.exists():
            print(f"[skip] {frame_png.name} (already exists)")
            continue
        print(f"[render] {p.name} -> {frame_png.name}")
        render_one_frame(
            plotfile_path=p,
            frame_png=frame_png,
            slice_plane=slice_plane,
            use_tex=USE_TEX,
            auto_cbar=True,  # per-frame cmap range
        )

    # write mp4 at the end (no need for tmp npy cache)
    mp4_path = output_dir / f"animated_{slice_plane}_plane.mp4"
    print(f"[mp4] Writing {mp4_path.name} ...")
    plot_manager.animate_pngs_to_mp4(
        frames_dir=output_dir,
        mp4_path=mp4_path,
        pattern=f"frame_%05d_{slice_plane}_plane.png",
        fps=30,
    )
    print(f"Done. Frames under: {output_dir}")


def main():
    data_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    if not data_dir.exists():
        print(f"Error: data directory does not exist: {data_dir}")
        sys.exit(1)
    build_frames_serial(data_dir)


if __name__ == "__main__":
    main()
