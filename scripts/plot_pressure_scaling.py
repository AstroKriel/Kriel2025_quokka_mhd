#!/usr/bin/env python3
# pressure_scaling.py

## { SCRIPT
##
## === DEPENDENCIES
##

import numpy
from pathlib import Path
from ww_quokka_sims.sim_io import load_dataset
from jormi.ww_fields import field_operators
from jormi.ww_plots import plot_manager, annotate_axis
from jormi.ww_io import log_manager
from utils import helpers

##
## === HELPERS
##


def get_final_dataset_dir(sim_dir: Path) -> Path | None:
    dataset_dirs = helpers.get_latest_dataset_dirs(sim_dir)
    return dataset_dirs[-1] if dataset_dirs else None


##
## === MAIN PROGRAM
##


def main():
    args = helpers.get_user_input()
    root_dir = args.dir
    measured_Machs: list[float] = []
    measured_jumps: list[float] = []
    for target_mach in [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]:
        sim_dir = root_dir / f"Mach={target_mach}"
        if not sim_dir.is_dir():
            log_manager.log_warning(f"Skipping: missing folder {sim_dir}")
            continue
        dataset_dir = get_final_dataset_dir(sim_dir)
        if dataset_dir is None:
            log_manager.log_warning(f"Skipping: no *plt* under {sim_dir}")
            continue
        gamma = 5.0 / 3.0
        with load_dataset.QuokkaDataset(dataset_dir=dataset_dir, verbose=False) as ds:
            pressure_sfield = ds.load_pressure_sfield(gamma=gamma)
            density_sfield = ds.load_density_sfield()
            vel_vfield = ds.load_velocity_vfield()
        bg_density = float(numpy.median(density_sfield.data))
        bg_pressure = float(numpy.median(pressure_sfield.data))
        pressure_jump = field_operators.compute_array_rms(pressure_sfield.data - bg_pressure)
        vel_sfield = field_operators.compute_vfield_magnitude(vel_vfield)
        rms_vel = field_operators.compute_sfield_rms(vel_sfield)
        sound_speed = float(numpy.sqrt(gamma * bg_pressure / bg_density))
        measured_Machs.append(rms_vel / sound_speed)
        measured_jumps.append(pressure_jump)
        log_manager.log_context(
            title=f"Read dataset from {sim_dir.name}",
            message=f"{dataset_dir.name} is the last dataset",
            notes={
                "rms_vel": f"{rms_vel:.6e}",
                "pressure_jump": f"{pressure_jump:.6e}",
            },
            show_time=True,
            message_position="top",
        )
    if not measured_Machs:
        log_manager.log_warning("No data found to plot.")
        return
    index_order = numpy.argsort(numpy.asarray(measured_Machs))
    log10_speeds = numpy.log10(numpy.asarray(measured_Machs, dtype=float)[index_order])
    log10_jumps = numpy.log10(numpy.asarray(measured_jumps, dtype=float)[index_order])
    fig, ax = plot_manager.create_figure()
    ax.plot(log10_speeds, log10_jumps, ls="", marker="o", ms=8, color="black")
    a0 = log10_jumps[0] - (2 * log10_speeds[0])
    ax.plot(log10_speeds, a0 + 2 * log10_speeds, ls=":", color="black")
    ax.set_xlabel(r"$\mathcal{M} \equiv v_{\rm rms} / \sqrt{\gamma p_0 / \rho_0}$")
    ax.set_ylabel(r"$\langle (p - p_0)^2 \rangle^{1/2}$")
    annotate_axis.add_text(
        ax=ax,
        x_pos=0.05,
        y_pos=0.95,
        label="HLLD",
        x_alignment="left",
        y_alignment="top",
        fontsize=20,
        font_color="black",
    )
    fig_path = root_dir / "pressure_scaling.png"
    plot_manager.save_figure(
        fig=fig,
        fig_path=fig_path,
        verbose=True,
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
