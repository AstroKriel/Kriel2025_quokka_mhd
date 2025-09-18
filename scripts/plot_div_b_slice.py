## { SCRIPT

##
## === DEPENDENCIES
##

import numpy
from jormi.ww_plots import plot_manager, plot_data
from jormi.ww_fields import field_operators
from utils import helpers, load_quokka_dataset


##
## === HELPERS
##


def _compute_xy_axis_bounds(
  domain,
):
  ## return (xmin, xmax, ymin, ymax) in code units for xy slices
  (x0, x1), (y0, y1), _ = domain.domain_bounds
  return (x0, x1, y0, y1)


def _plot_divb_mid_z_slice(
  ax,
  vfield_b,
  domain,
):
  ## compute div(b), take a z-mid slice, and render with colorbar
  z_mid = vfield_b.data.shape[-1] // 2
  sfield_div_b = field_operators.compute_vfield_divergence(
    vfield=vfield_b.data,
    domain_lengths=domain.domain_lengths,
  )
  max_value = numpy.max(numpy.abs(sfield_div_b))
  plot_data.plot_sfield_slice(
    ax=ax,
    field_slice=sfield_div_b[..., z_mid],
    axis_bounds=_compute_xy_axis_bounds(domain),
    add_colorbar=True,
    cmap_name="cmr.seasons",
    cbar_bounds=(-max_value, max_value),
    cbar_label=r"$\nabla\cdot\vec{b}$",
    cbar_side="right",
  )


##
## === OPERATOR CLASS
##


class Plotter:
  ## wrapper for plotting a mid-plane (z) slice of ∇·b for one or two datasets

  def __init__(
    self,
    input_dir,
  ):
    self.input_dir = input_dir

  def run(
    self,
  ) -> None:
    ## main routine: resolve datasets, branch, style, save
    dataset_dirs, single_snapshot = self._resolve_dataset_dirs(self.input_dir)
    if single_snapshot:
      fig, ax = plot_manager.create_figure()
      self._plot_single(ax, dataset_dirs[0])
    else:
      fig, axs = plot_manager.create_figure(num_rows=2)
      self._plot_series(axs, dataset_dirs)
    self._save(fig, out_name="div_b_slice.png")

  def _plot_single(
    self,
    ax,
    dataset_dir,
  ) -> None:
    ## single snapshot: one mid-z slice
    with load_quokka_dataset.QuokkaDataset(dataset_dir=dataset_dir) as ds:
      vfield_b = ds.load_magnetic_field()
      domain = ds.load_domain()
    _plot_divb_mid_z_slice(ax, vfield_b, domain)

  def _plot_series(
    self,
    axs,
    dataset_dirs,
  ) -> None:
    ## series: show first and last snapshot as two rows
    first_last = (dataset_dirs[0], dataset_dirs[-1])
    for ax, dataset_dir in zip(axs.flat, first_last):
      with load_quokka_dataset.QuokkaDataset(dataset_dir=dataset_dir) as ds:
        vfield_b = ds.load_magnetic_field()
        domain = ds.load_domain()
      _plot_divb_mid_z_slice(ax, vfield_b, domain)

  @staticmethod
  def _resolve_dataset_dirs(
    input_dir,
  ):
    ## determine whether input is a single snapshot or a series
    if "plt" in input_dir.name:
      dataset_dirs = [input_dir]
      return dataset_dirs, True
    dataset_dirs = helpers.get_latest_dataset_dirs(sim_dir=input_dir)
    assert len(dataset_dirs) != 0
    return dataset_dirs, (len(dataset_dirs) == 1)

  def _save(
    self,
    fig,
    out_name: str,
  ) -> None:
    ## save figure to input directory
    fig_path = self.input_dir / out_name
    plot_manager.save_figure(fig=fig, fig_path=fig_path)


##
## === MAIN PROGRAM
##


def main():
  ## parse user inputs and run
  args = helpers.get_user_input()
  plotter = Plotter(input_dir=args.dir)
  plotter.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
  main()

## } SCRIPT
