from typing import Any
from pathlib import Path
from jormi.ww_io import io_manager, shell_manager
from jormi.ww_jobs import pbs_job_manager

QUOKKA_DIR = Path("/Users/necoturb/Documents/Codes/quokka") # macOS
# QUOKKA_DIR = Path("/g/data1b/jh2/nk7952/quokka/") # gadi
PAPER_DIR = Path(__file__).resolve().parent.parent
EXECUTE_JOB = False

QUOKKA_PROBLEM_SET = {
  "AlfvenWaveLinear": {
    "exe": "test_alfven_wave_linear",
    "input": "alfven_wave_linear.in"
  },
  "AlfvenWaveCircular": {
    "exe": "test_alfven_wave_circular",
    "input": "alfven_wave_circular.in"
  },
  "FastWave": {
    "exe": "test_fast_wave",
    "input": "fast_wave.in"
  },
  "CurrentSheet": {
    "exe": "test_current_sheet",
    "input": "current_sheet.in"
  },
  "FieldLoop": {
    "exe": "test_field_loop",
    "input": "field_loop.in"
  },
}

def get_input_params(nres_per_dim: int) -> dict:
  return {
    "plotfile_interval"            : 10,
    "checkpoint_interval"          : -1,
    "amr.n_cell"                   : f"{nres_per_dim} {nres_per_dim} {nres_per_dim}",
    "amr.max_level"                : 0,
    "amr.max_grid_size"            : nres_per_dim,
    "amr.blocking_factor_x"        : nres_per_dim,
    "amr.blocking_factor_y"        : nres_per_dim,
    "amr.blocking_factor_z"        : nres_per_dim,
    "cfl"                          : 0.3,
    "hydro.use_dual_energy"        : 0,
    "hydro.rk_integrator_order"    : 2, # 1 or 2
    "hydro.reconstruction_order"   : 5, # 1, 2, 3, or 5
    "mhd.emf_reconstruction_order" : 5, # 1, 2, 3, or 5
    "mhd.emf_averaging_method"     : "LD04", # BalsaraSpicer or LD04
  }

def get_setup_label(params: dict) -> str:
  cfl_str          = "cfl={:.1f}".format(params["cfl"])
  nres_str         = "nres={}".format(params["amr.max_grid_size"])
  rk_str           = "rk{}".format(params["hydro.rk_integrator_order"])
  interp_order_str = "interp-order={}".format(params["hydro.reconstruction_order"])
  ave_scheme_str   = params["mhd.emf_averaging_method"].lower()
  return "_".join([cfl_str, nres_str, rk_str, interp_order_str, ave_scheme_str])

def adjust_input_file(
    file_path    : Path,
    setup_params : dict[str, Any],
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
  for key, value in setup_params.items():
    _replace_or_add(file_lines, key, value)
  with file_path.open("w") as fp:
    fp.write("\n".join(file_lines) + "\n")

def setup_problem(
    build_name      : str,
    problem_name    : str,
    nres_per_dim    : int,
    num_procs       : int,
    queued_job_tags : list[str]
  ):
  problem_files = QUOKKA_PROBLEM_SET[problem_name]
  exe_file_name = problem_files["exe"]
  input_file_name = problem_files["input"]
  ## source paths
  source_exe_file_dir = QUOKKA_DIR / "build" / build_name / "src" / "problems" / problem_name
  source_input_file_dir = QUOKKA_DIR / "inputs"
  source_exe_file_path = source_exe_file_dir / exe_file_name
  source_input_file_path = source_input_file_dir / input_file_name
  ## check that the executable exists
  if not source_exe_file_path.exists():
    raise FileNotFoundError(f"Error: Executable could not be found under: {source_exe_file_path}")
  if not source_input_file_path.exists():
    raise FileNotFoundError(f"Error: Parameter file could not be found under: {source_input_file_path}")
  ## collect simulation parameter details
  setup_params = get_input_params(nres_per_dim)
  setup_label = get_setup_label(setup_params)
  ## target folder and files
  target_problem_dir = PAPER_DIR / "sims" / build_name / problem_name / setup_label
  ## create the target folder
  io_manager.init_directory(target_problem_dir)
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
  adjust_input_file(problem_input_file, setup_params)
  ## create job
  job_tag = setup_label
  if job_tag in queued_job_tags: return
  job_path = pbs_job_manager.create_pbs_job_script(
    system_name        = "gadi",
    directory          = target_problem_dir,
    file_name          = f"job.sh",
    command            = f". ~/modules_quokka\nmpirun -np {num_procs} {exe_file_name} {input_file_name}",
    tag_name           = job_tag,
    queue_name         = "normal", # "rsaa",
    compute_group_name = "jh2", # "mk27",
    num_procs          = num_procs,
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
  build_name = "fs_scheme"
  problem_name = "AlfvenWaveLinear"
  queued_jobs = pbs_job_manager.get_list_of_queued_jobs()
  queued_job_tags = [
    job_tag
    for _, job_tag in queued_jobs
  ]
  for nres_exponent in range(3, 7):
    nres_per_dim = 2**nres_exponent
    setup_problem(
      build_name      = build_name,
      problem_name    = problem_name,
      nres_per_dim    = nres_per_dim,
      num_procs       = 8,
      queued_job_tags = queued_job_tags,
    )
    print(" ")

if __name__ == "__main__":
  main()

## .