## { SCRIPT

##
## === DEPENDENCIES
##

from pathlib import Path
from ww_quokka_sims.sim_io import load_dataset
from jormi.ww_io import log_manager
import utils

##
## === OPERATOR CLASS
##


class ScriptInterface:

    def __init__(
        self,
        *,
        dataset_dir: Path,
        dataset_tag: str,
    ):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.dataset_tag = dataset_tag
        if not utils.looks_like_boxlib_dir(dataset_dir=self.dataset_dir):
            log_manager.log_error(
                text="Provided dataset does not appear to be a valid BoxLib-like plotfile.",
                notes={
                    "Path": self.dataset_dir,
                    "Expected": "both a `Header` file and `Level_0` directory",
                },
            )
            raise SystemExit(2)

    def run(
        self,
    ) -> None:
        with load_dataset.QuokkaDataset(dataset_dir=self.dataset_dir, verbose=True) as ds:
            ds.list_available_field_keys()


##
## === PROGRAM MAIN
##


def main():
    user_args = utils.get_user_args()
    script_interface = ScriptInterface(
        dataset_dir=user_args.dir,
        dataset_tag=user_args.tag,
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
