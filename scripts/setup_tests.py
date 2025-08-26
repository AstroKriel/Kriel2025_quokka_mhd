import math
from typing import Any, Union
from pathlib import Path
from jormi.ww_io import io_manager, shell_manager, json_files
from jormi.ww_jobs import pbs_job_manager

## macOS
QUOKKA_DIR = Path("/Users/necoturb/Documents/Codes/quokka")
DATA_DIR = Path(__file__).parent.parent
PLOTTING_UV_PROJECT = Path("/Users/necoturb/Documents/Codes/asgard/sindri/python/ww_quokka_sims")

# ## gadi
# QUOKKA_DIR = Path("/g/data1b/jh2/nk7952/quokka/")
# DATA_DIR = Path("/scratch/jh2/nk7952/quokka")
# PLOTTING_UV_PROJECT = Path("/home/586/nk7952/analysis/ww_quokka_sims")

## setup params
NUM_PROCS_PER_NODE = 48
EXECUTE_JOB = False

PROBLEM_NAME = "AlfvenWaveLinear"
SCALING_MODE = "weak"
SAVE_DATA = True
CELLS_PER_BLOCK_DIM = 2 ** 5 # 16

USE_FC_VELOCITY_FOR_EMF = 0
EMF_AVE_SCHEME = "LD04" # "BalsaraSpicer" or "LD04"
INTERP_ORDER = 2 # 1, 2, 3, or 5
RK_ORDER = 2 # 1 or 2
CFL = 0.3

QUOKKA_PROBLEM_SET = {
  "AlfvenWaveLinear": {
    "exe": "test_alfven_wave_linear",
    "in": "alfven_wave_linear.in"
  },
  "AlfvenWaveCircular": {
    "exe": "test_alfven_wave_circular",
    "in": "alfven_wave_circular.in"
  },
  "FastWave": {
    "exe": "test_fast_wave",
    "in": "fast_wave.in"
  },
  "CurrentSheet": {
    "exe": "test_current_sheet",
    "in": "current_sheet.in"
  },
  "FieldLoop": {
    "exe": "test_field_loop",
    "in": "field_loop.in"
  },
  "OrszagTang": {
    "exe": "test_orszag_tang",
    "in": "orszag_tang.in",
    "stop_time": 3
  }
}

def is_power_of_two(value: int) -> bool:
  # return value > 0 and (value & (value - 1)) == 0 # bitwise check
  exponent = math.log2(value)
  return abs(exponent - round(exponent)) < 1e-7 # single precision

def get_domain_params(
    *,
    cells_per_block_dim   : int, # quantisation of work (AMReX blocking-factor)
    blocks_per_sim_dim    : int, # domain decomposition
    blocks_per_box_dim    : int, # amount of communication within rank
    boxes_per_rank        : int, # amount of work taken on by each rank
    num_procs_per_node    : int, # resources available per node
    round_to_nearest_node : bool = False, # allow idle ranks to fill full nodes
  ) -> dict[str, Union[int, float, str]]:
  if not is_power_of_two(cells_per_block_dim):
    raise ValueError("`cells_per_block_dim` must be a power of 2.")
  if not isinstance(blocks_per_sim_dim, int) or blocks_per_sim_dim < 1:
    raise ValueError("`blocks_per_sim_dim` must be a positive integer.")
  if not isinstance(blocks_per_box_dim, int) or blocks_per_box_dim < 1:
    raise ValueError("`blocks_per_box_dim` must be a positive integer.")
  if not isinstance(boxes_per_rank, int) or boxes_per_rank < 1:
    raise ValueError("`boxes_per_rank` must be a positive integer.")
  if not isinstance(num_procs_per_node, int) or num_procs_per_node < 1:
    raise ValueError("`blocks_per_sim_dim` must be a positive integer.")
  if blocks_per_sim_dim % blocks_per_box_dim != 0:
    raise ValueError(
      f"`blocks_per_sim_dim = {blocks_per_sim_dim}` is not divisible by "
      f"`blocks_per_box_dim = {blocks_per_box_dim}` for exact tiling."
    )
  boxes_per_sim_dim = blocks_per_sim_dim // blocks_per_box_dim
  total_boxes = boxes_per_sim_dim ** 3
  if total_boxes % boxes_per_rank != 0:
    raise ValueError(
      f"`total_boxes = {total_boxes}` is not divisible by your chosen "
      f"`boxes_per_rank = {boxes_per_rank}`."
    )
  cells_per_sim_dim    = cells_per_block_dim * blocks_per_sim_dim
  cells_per_box_dim    = cells_per_block_dim * blocks_per_box_dim
  total_cells_per_box  = cells_per_box_dim ** 3
  total_cells_per_rank = boxes_per_rank * total_cells_per_box
  mpi_ranks_w_work     = total_boxes // boxes_per_rank
  if mpi_ranks_w_work <= num_procs_per_node:
    ## when using less than a single node
    mpi_ranks_requested = mpi_ranks_w_work
  else:
    ## multi-node jobs should use whole nodes
    if mpi_ranks_w_work % num_procs_per_node == 0:
      ## already using entirity of nodes
      mpi_ranks_requested = mpi_ranks_w_work
    else:
      ## when using a partial node
      if not round_to_nearest_node:
        raise ValueError(
          "Violated use of full-node: "
          f"`mpi_ranks_w_work={mpi_ranks_w_work}` is not a multiple of {num_procs_per_node}."
        )
      ## round up to a whole node (will lead to a few idle ranks)
      mpi_ranks_requested = math.ceil(mpi_ranks_w_work / num_procs_per_node) * num_procs_per_node
  mpi_ranks_idle = mpi_ranks_requested - mpi_ranks_w_work
  nodes_used = math.ceil(mpi_ranks_requested / num_procs_per_node)
  mpi_rank_utilisation = 100 * mpi_ranks_w_work / mpi_ranks_requested
  return {
    ## AMReX params
    "amr.max_level"         : 0,
    "amr.n_cell"            : f"{cells_per_sim_dim} {cells_per_sim_dim} {cells_per_sim_dim}",
    "amr.max_grid_size"     : cells_per_box_dim,
    "amr.blocking_factor_x" : cells_per_block_dim,
    "amr.blocking_factor_y" : cells_per_block_dim,
    "amr.blocking_factor_z" : cells_per_block_dim,
    ## knobs we can tweak
    "cells_per_block_dim"   : cells_per_block_dim,
    "blocks_per_sim_dim"    : blocks_per_sim_dim,
    "blocks_per_box_dim"    : blocks_per_box_dim,
    "boxes_per_rank"        : boxes_per_rank,
    ## domain decomposition
    "cells_per_sim_dim"     : cells_per_sim_dim,
    "boxes_per_sim_dim"     : boxes_per_sim_dim,
    "cells_per_box_dim"     : cells_per_box_dim,
    "total_boxes"           : total_boxes,
    ## workload
    "total_cells_per_box"   : total_cells_per_box,
    "total_cells_per_rank"  : total_cells_per_rank,
    "mpi_ranks_requested"   : mpi_ranks_requested,
    "mpi_ranks_w_work"      : mpi_ranks_w_work,
    "mpi_ranks_idle"        : mpi_ranks_idle,
    "nodes_used"            : nodes_used,
    "mpi_rank_utilisation"  : mpi_rank_utilisation,
    "num_procs_per_node"    : num_procs_per_node,
  }

def get_sim_params(
    domain_params : dict[str, Any],
    stop_time     : int | None = None,
  ) -> dict[str, Any]:
  sim_params = {
    ## scheme setup
    "cfl"                          : CFL,
    "hydro.use_dual_energy"        : 0,
    "hydro.rk_integrator_order"    : RK_ORDER,
    "hydro.reconstruction_order"   : INTERP_ORDER,
    "mhd.emf_reconstruction_order" : INTERP_ORDER,
    "mhd.use_fc_velocity_for_emf"  : USE_FC_VELOCITY_FOR_EMF,
    "mhd.emf_averaging_method"     : EMF_AVE_SCHEME,
    ## domain setup
    "amr.max_level"                : 0, # uniform grid
    "amr.n_cell"                   : domain_params["amr.n_cell"],
    "amr.max_grid_size"            : domain_params["amr.max_grid_size"],
    "amr.blocking_factor_x"        : domain_params["amr.blocking_factor_x"],
    "amr.blocking_factor_y"        : domain_params["amr.blocking_factor_y"],
    "amr.blocking_factor_z"        : domain_params["amr.blocking_factor_z"],
    ## output interval
    "plotfile_interval"            : 10 if SAVE_DATA else -1,
    "checkpoint_interval"          : -1,
  }
  if stop_time is not None:
    sim_params["stop_time"] = stop_time
  return sim_params

def get_scheme_label(sim_params: dict) -> str:
  emf_method     = "fcvel" if USE_FC_VELOCITY_FOR_EMF else "fs"
  emf_ave_scheme = sim_params["mhd.emf_averaging_method"].lower()
  spatial_order  = "ro{}".format(sim_params["hydro.reconstruction_order"])
  time_order     = "rk{}".format(sim_params["hydro.rk_integrator_order"])
  cfl            = "cfl{:.1f}".format(sim_params["cfl"])
  return "_".join([ emf_method, emf_ave_scheme, spatial_order, time_order, cfl ])

def get_domain_label(domain_params: dict[str, Any]) -> str:
  cells_per_sim_dim   = "N{}".format(domain_params["cells_per_sim_dim"])
  cells_per_box_dim   = "Nbo{}".format(domain_params["cells_per_box_dim"])
  cells_per_block_dim = "Nbl{}".format(domain_params["cells_per_block_dim"])
  boxes_per_rank      = "bopr{}".format(domain_params["boxes_per_rank"])
  mpi_ranks_requested = "mpir{}".format(domain_params["mpi_ranks_requested"])
  return "_".join([ cells_per_sim_dim, cells_per_box_dim, cells_per_block_dim, boxes_per_rank, mpi_ranks_requested ])

def adjust_input_file(
    file_path    : Path,
    sim_params : dict[str, Any],
  ) -> None:
  def _replace_or_add(_file_lines, _key, _value):
    for line_index, line_content in enumerate(_file_lines):
      if line_content.startswith("#"): continue
      if line_content.strip().startswith(f"{_key}"):
        _file_lines[line_index] = f"{_key} = {_value}"
        return
    _file_lines.append(f"{_key} = {_value}")
  with file_path.open("r", encoding="utf-8") as fp:
    file_lines = fp.read().splitlines()
  for key, value in sim_params.items():
    _replace_or_add(file_lines, key, value)
  with file_path.open("w") as fp:
    fp.write("\n".join(file_lines) + "\n")

def setup_problem(
    scaling_mode    : str,
    problem_name    : str,
    domain_params   : dict[str, Any],
    queued_job_tags : list[str],
  ):
  problem_files   = QUOKKA_PROBLEM_SET[problem_name]
  exe_file_name   = problem_files["exe"]
  input_file_name = problem_files["in"]
  stop_time       = problem_files.get("stop_time", None)
  ## source paths
  source_exe_file_dir    = QUOKKA_DIR / "build" / "src" / "problems" / problem_name
  source_input_file_dir  = QUOKKA_DIR / "inputs"
  source_exe_file_path   = source_exe_file_dir / exe_file_name
  source_input_file_path = source_input_file_dir / input_file_name
  ## check that the executable exists
  if not source_exe_file_path.exists():
    raise FileNotFoundError(f"Error: Executable could not be found under: {source_exe_file_path}")
  if not source_input_file_path.exists():
    raise FileNotFoundError(f"Error: Parameter file could not be found under: {source_input_file_path}")
  ## collect simulation parameter details
  sim_params   = get_sim_params(domain_params, stop_time)
  scheme_label = get_scheme_label(sim_params)
  domain_label = get_domain_label(domain_params)
  job_tag      = f"{problem_name}_{scheme_label}_{domain_label}"
  ## target folder
  target_problem_dir = DATA_DIR / "sims" / scaling_mode / problem_name / scheme_label / domain_label
  io_manager.init_directory(target_problem_dir)
  ## save generated parameter files
  json_files.save_dict_to_json_file(
    file_path  = target_problem_dir / "domain_params.json",
    input_dict = domain_params,
    overwrite  = True
  )
  json_files.save_dict_to_json_file(
    file_path  = target_problem_dir / "sim_params.json",
    input_dict = sim_params,
    overwrite  = True
  )
  ## copy problem executable and input files over
  io_manager.copy_file(
    directory_from = source_exe_file_dir,
    directory_to   = target_problem_dir,
    file_name      = exe_file_name,
    overwrite      = True
  )
  io_manager.copy_file(
    directory_from = source_input_file_dir,
    directory_to   = target_problem_dir,
    file_name      = input_file_name,
    overwrite      = True
  )
  ## adjust input file parameters
  problem_input_file = target_problem_dir / input_file_name
  adjust_input_file(problem_input_file, sim_params)
  ## create job
  if job_tag in queued_job_tags:
    print(f"Skipping. Simulation is already queued: {job_tag}")
    return
  mpi_ranks = int(domain_params["mpi_ranks_requested"])
  plotting_script_path = PLOTTING_UV_PROJECT / "scripts/plot_field_slices.py"
  job_path = pbs_job_manager.create_pbs_job_script(
    system_name        = "gadi",
    directory          = target_problem_dir,
    file_name          = f"job.sh",
    prep_command       = "source ~/modules_quokka",
    main_command       = f"mpirun -np {mpi_ranks} {exe_file_name} {input_file_name}",
    post_command       = (
      f'uv run --project "{PLOTTING_UV_PROJECT}" '
      f'python "{plotting_script_path}" "{target_problem_dir}"'
    ),
    always_run_post    = True,
    tag_name           = job_tag,
    queue_name         = "normal", # "rsaa",
    compute_group_name = "jh2", # "mk27",
    num_procs          = mpi_ranks,
    wall_time_hours    = 2,
    storage_group_name = "jh2",
    email_address      = "neco.kriel@anu.edu.au",
    email_on_start     = False,
    email_on_finish    = False,
    verbose            = True,
  )
  if EXECUTE_JOB:
    shell_manager.execute_shell_command(
      command           = f"qsub {job_path}",
      working_directory = target_problem_dir,
      timeout_seconds   = 30,
    )
  print(f"Finished setting up under: {target_problem_dir}")

def main():
  if EXECUTE_JOB:
    queued_jobs = pbs_job_manager.get_list_of_queued_jobs() or []
  else: queued_jobs = []
  queued_job_tags = [
    job_tag
    for _, job_tag in queued_jobs
  ]
  if "weak" in SCALING_MODE.lower():
    ## work per rank stays the same, but total work increases
    ## - fix `blocks_per_box_dim`
    ## - fix `boxes_per_rank`
    ## - increase `blocks_per_sim_dim`
    ## - where `cells_per_block_dim` is fixed (to keep things fair)
    for blocks_per_sim_dim in [1, 2, 3]:
      ## required: (blocks_per_sim_dim / blocks_per_box_dim) ** 3 % (boxes_per_rank * num_procs_per_node) == 0
      domain_params = get_domain_params(
        cells_per_block_dim   = CELLS_PER_BLOCK_DIM,
        blocks_per_sim_dim    = blocks_per_sim_dim,
        blocks_per_box_dim    = 1,
        boxes_per_rank        = 1,
        num_procs_per_node    = NUM_PROCS_PER_NODE,
        round_to_nearest_node = True
      )
      setup_problem(
        scaling_mode    = SCALING_MODE,
        problem_name    = PROBLEM_NAME,
        domain_params   = domain_params,
        queued_job_tags = queued_job_tags,
      )
      print(" ")
  elif "strong" in SCALING_MODE.lower():
    ## total work stays the same, but the work per rank decreases until communication overhead bites
    ## - fix `blocks_per_sim_dim`
    ## - fix `blocks_per_box_dim`
    ## - decrease `boxes_per_rank` (to increase number of workers)
    ## - where `cells_per_block_dim` is fixed (to keep things fair)
    for boxes_per_rank in [8, 4, 2, 1]:
      domain_params = get_domain_params(
        cells_per_block_dim   = CELLS_PER_BLOCK_DIM,
        blocks_per_sim_dim    = 16,
        blocks_per_box_dim    = 1,
        boxes_per_rank        = boxes_per_rank,
        num_procs_per_node    = NUM_PROCS_PER_NODE,
        round_to_nearest_node = True
      )
      setup_problem(
        scaling_mode    = SCALING_MODE,
        problem_name    = PROBLEM_NAME,
        domain_params   = domain_params,
        queued_job_tags = queued_job_tags,
      )
      print(" ")
  else: raise ValueError(f"`scaling_mode` must be `weak` or `strong` not `{SCALING_MODE}`.")

if __name__ == "__main__":
  main()

## .