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


def get_far_field_bg_states(
    domain,
    density_sfield,
    pressure_sfield,
    min_mask_radius: float = 2.5,
) -> tuple[float, float]:
    (x_bounds, y_bounds, _) = domain.domain_bounds
    x_centers, y_centers, _ = domain.cell_centers
    (num_cells_x, num_cells_y, num_cells_z) = domain.resolution
    x_center = 0.5 * (x_bounds[0] + x_bounds[1])
    y_center = 0.5 * (y_bounds[0] + y_bounds[1])

    X, Y = numpy.meshgrid(x_centers, y_centers, indexing="ij")
    radius = numpy.sqrt((X - x_center)**2 + (Y - y_center)**2)
    mask_xy = radius >= min_mask_radius
    mask = numpy.broadcast_to(mask_xy[..., numpy.newaxis], (num_cells_x, num_cells_y, num_cells_z))
    bg_density = float(numpy.mean(density_sfield.data[mask]))
    bg_pressure = float(numpy.mean(pressure_sfield.data[mask]))
    return bg_density, bg_pressure


def get_analytic_pressure_soln(
    domain,
    bg_density: float,
    bg_pressure: float,
    v_max: float,
) -> numpy.ndarray:
    (x_bounds, y_bounds, _) = domain.domain_bounds
    x_centers, y_centers, _ = domain.cell_centers
    (num_cells_x, num_cells_y, num_cells_z) = domain.resolution
    x_vortex_center = 0.5 * (x_bounds[0] + x_bounds[1])
    y_vortex_center = 0.5 * (y_bounds[0] + y_bounds[1])
    x_mg, y_mg = numpy.meshgrid(x_centers, y_centers, indexing="ij")
    radius_sq = (x_mg - x_vortex_center)**2 + (y_mg - y_vortex_center)**2
    pressure_xy = bg_pressure - 0.5 * bg_density * (v_max**2) * numpy.exp(1.0 - radius_sq)
    return numpy.broadcast_to(pressure_xy[..., numpy.newaxis], (num_cells_x, num_cells_y, num_cells_z))


##
## === MAIN PROGRAM
##


def main():
    args = helpers.get_user_input()
    root_dir = args.dir
    measured_Machs: list[float] = []
    measured_jumps: list[float] = []
    for target_mach in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
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
            domain = ds.load_domain()
            pressure_sfield = ds.load_pressure_sfield(gamma=gamma)
            density_sfield = ds.load_density_sfield()
            vel_vfield = ds.load_velocity_vfield()
        bg_density, bg_pressure = get_far_field_bg_states(
            domain=domain,
            density_sfield=density_sfield,
            pressure_sfield=pressure_sfield,
            min_mask_radius=2.5,
        )
        vel_sfield = field_operators.compute_vfield_magnitude(vel_vfield)
        rms_vel = field_operators.compute_sfield_rms(vel_sfield)
        v_max = float(numpy.max(vel_sfield.data))
        sound_speed = float(numpy.sqrt(gamma * bg_pressure / bg_density))
        analytic_pressure = get_analytic_pressure_soln(
            domain=domain,
            bg_density=bg_density,
            bg_pressure=bg_pressure,
            v_max=v_max,
        )
        rms_pressure_jump = field_operators.compute_array_rms(pressure_sfield.data - analytic_pressure)
        measured_Machs.append(rms_vel / sound_speed)
        measured_jumps.append(rms_pressure_jump)
        log_manager.log_context(
            title=f"Read dataset from {sim_dir.name}",
            message=f"{dataset_dir.name} is the last dataset",
            notes={
                "rms_vel": f"{rms_vel:.6e}",
                "v_max": f"{v_max:.6e}",
                "rms_pressure_jump": f"{rms_pressure_jump:.6e}",
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
    intercept_linear = log10_jumps[0] - log10_speeds[0]
    intercept_quadratic = log10_jumps[-1] - (2 * log10_speeds[-1])
    ax.plot(log10_speeds, intercept_linear + log10_speeds, ls=":", color="black")
    ax.plot(log10_speeds, intercept_quadratic + 2 * log10_speeds, ls="--", color="black")
    ax.set_xlabel(r"$\mathcal{M} \equiv v_{\rm rms} / \sqrt{\gamma p_0 / \rho_0}$")
    ax.set_ylabel(r"$\langle (p - p_{\rm exact})^2 \rangle^{1/2}$")
    annotate_axis.add_text(
        ax=ax,
        x_pos=0.05,
        y_pos=0.95,
        label=str(root_dir).split("/")[-1],
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
