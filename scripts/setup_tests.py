from pathlib import Path
from jormi.ww_io import io_manager

QUOKKA_DIR = Path("/Users/necoturb/Documents/Codes/quokka")
PAPER_DIR = Path("/Users/necoturb/Documents/Codes/Asgard/mimir/kriel_2025_quokka_mhd")

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

def adjust_input_file(file_path, setup_params):
  def replace_or_add(_file_contents, key, value):
    for line_index, line_content in enumerate(_file_contents):
      if line_content.strip().startswith(f"{key}"):
        _file_contents[line_index] = f"{key} = {value}"
        return
    _file_contents.append(f"{key} = {value}")
  with file_path.open("r") as f:
    file_contents = f.read().splitlines()
  for key, value in setup_params.items():
    replace_or_add(file_contents, key, value)
  with file_path.open("w") as f:
    f.write("\n".join(file_contents) + "\n")

def setup_problem(
    build_name   : str,
    problem_name : str,
    nres_per_dim : int
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
  target_problem_dir = PAPER_DIR / "sim_outputs" / build_name / problem_name / setup_label
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
  print(f"Finished setting up under: {target_problem_dir}")

def main():
  build_name = "fs_scheme"
  problem_name = "AlfvenWaveLinear"
  for nres_exponent in range(3, 7):
    nres_per_dim = 2**nres_exponent
    setup_problem(build_name, problem_name, nres_per_dim)
    print(" ")

if __name__ == "__main__":
  main()

## .